"""
src/analytics/matcher.py

Семантический поиск похожих названий товаров через embedding-векторы.
Используется при сохранении чека чтобы предложить объединить новый товар
с уже известным каноническим именем.

Будущая правка имён: при переименовании канонического имени пользователем
нужно вызвать crud.rename_item(old, new) который перепишет все таблицы
и пересчитает embedding для нового имени.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

from src.db.crud import SessionLocal, get_all_item_embeddings, upsert_item_embedding
from src.llm.client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    new_name: str          # имя из нового чека
    canonical_name: str    # каноническое имя из БД
    similarity: float      # косинусное сходство 0..1


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ItemMatcher:
    """
    Находит семантически похожие названия товаров из базы для новых имён из чека.
    """

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    def find_matches(
        self,
        new_names: list[str],
        threshold: float = 0.88,
    ) -> list[MatchResult]:
        """
        Для каждого нового имени ищет наиболее похожее каноническое имя в БД.
        Возвращает только совпадения выше порога и исключает точные совпадения
        (они и так одинаковые, объединять не нужно).

        Алгоритм:
        1. Загружаем все известные embeddings из БД
        2. Считаем embeddings для новых имён (batch)
        3. Для каждого нового ищем ближайший канонический вектор
        4. Возвращаем пары выше порога
        """
        if not new_names:
            return []

        known = get_all_item_embeddings()
        if not known:
            logger.info("ItemMatcher: база embeddings пуста, совпадений нет")
            return []

        # Исключаем имена которые уже есть в базе как каноническое
        known_names = {entry["item_name"] for entry in known}
        names_to_check = [n for n in new_names if n not in known_names]
        if not names_to_check:
            return []

        # Batch embed новых имён
        try:
            new_vectors = self.client.embed(names_to_check)
        except OllamaError as exc:
            logger.warning("Ошибка при получении embeddings: %s", exc)
            return []

        # Косинусное сходство: каждое новое имя vs каждое известное
        results: list[MatchResult] = []
        for new_name, new_vec in zip(names_to_check, new_vectors):
            best_sim = 0.0
            best_canonical = ""
            for entry in known:
                sim = cosine_similarity(new_vec, entry["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_canonical = entry["item_name"]

            if best_sim >= threshold and best_canonical:
                results.append(MatchResult(
                    new_name=new_name,
                    canonical_name=best_canonical,
                    similarity=round(best_sim, 3),
                ))
                logger.info(
                    "Совпадение: «%s» → «%s» (%.1f%%)",
                    new_name, best_canonical, best_sim * 100,
                )

        return results

    def add_new_items(self, names: list[str]) -> None:
        """
        Считает embeddings для новых канонических имён и сохраняет в БД.
        Вызывается при сохранении чека для имён без совпадений —
        они становятся новыми каноническими эталонами.
        """
        if not names:
            return
        try:
            vectors = self.client.embed(names)
        except OllamaError as exc:
            logger.warning("Не удалось сохранить embeddings: %s", exc)
            return

        for name, vec in zip(names, vectors):
            upsert_item_embedding(name, vec)
        logger.info("Сохранено embeddings: %d имён", len(names))
