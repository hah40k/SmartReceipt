"""
src/analytics/retriever.py

RAG-ретривер для советника по бюджету и семантического поиска.

Два типа чанков:
  "receipt" — один чек: магазин, дата, список товаров с суммами, сводка категорий
  "item"    — один товар: категория/подкатегория + вся история цен по магазинам

Индексирование вызывается из receipt_service после save_receipt():
  retriever.index_receipt(db_receipt)        — для нового чека
  retriever.index_items(new_canonical_names) — для новых/переименованных товаров

Два публичных метода поиска:
  retrieve(question, top_k=6)
    → list[str] — для советника: чистые тексты чанков без скора
  search(query, top_k=10, chunk_type, min_similarity)
    → list[SearchResult] — для UI поиска: гибридный скор, тип, ключ
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.analytics.matcher import cosine_similarity
from src.db.crud import (
    get_all_rag_chunks,
    get_price_history,
    upsert_rag_chunk,
)
from src.llm.client import OllamaClient, OllamaError

if TYPE_CHECKING:
    from src.db.models import ReceiptModel

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 6
SEARCH_MIN_SIMILARITY = 0.25  # гибридный скор ниже которого результат считается нерелевантным
KEYWORD_WEIGHT = 0.35          # вес keyword-составляющей (0.35 keyword + 0.65 cosine)

# Служебные слова, не несущие смысловой нагрузки для поиска
_STOP_WORDS = frozenset({
    "с", "в", "на", "по", "из", "за", "от", "и", "или", "это",
    "где", "когда", "что", "все", "мне", "я", "был", "было", "были",
    "есть", "нет", "для", "при", "под", "над", "без", "до",
})


@dataclass
class SearchResult:
    chunk_type: str    # "receipt" | "item"
    ref_key: str       # str(receipt.id) для чеков, item_name для товаров
    chunk_text: str    # человекочитаемый текст чанка
    similarity: float  # итоговый гибридный скор 0..1


# ---------------------------------------------------------------------------
# Keyword-составляющая гибридного поиска
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    """
    Примитивный стеммер для русского языка.
    Берёт первые max(3, len-2) символов — отсекает типичные падежные окончания:
      "пивом"  (6) → "пиво"  (4)  ✓  найдётся в "пиво × 3"
      "чеки"   (4) → "чек"   (3)  ✓  найдётся в "Чек от..."
      "молоко" (6) → "моло"  (4)  ✓
      "хлеба"  (5) → "хле"   (3)  ✓
    """
    return word[:max(3, len(word) - 2)]


def _keyword_score(query: str, chunk_text: str) -> float:
    """
    Доля слов запроса (после стемминга), найденных в тексте чанка.
    Возвращает 0..1. Стоп-слова и слова короче 3 букв игнорируются.

    Пример: "чеки с пивом"
      Слова для проверки: ["чеки", "пивом"]
      _stem("чеки") = "чек" → найдено в "Чек от..."  → +1
      _stem("пивом") = "пиво" → найдено в "Состав: пиво × 3"  → +1
      score = 2/2 = 1.0
    """
    text_lower = chunk_text.lower()
    words = [
        w for w in query.lower().split()
        if len(w) >= 3 and w not in _STOP_WORDS
    ]
    if not words:
        return 0.0
    matches = sum(1 for w in words if _stem(w) in text_lower)
    return matches / len(words)


class RagRetriever:
    """
    Семантический ретривер: хранит чанки, индексирует, ищет.

    Использует nomic-embed-text через OllamaClient.embed().
    Хранит чанки в таблице rag_chunks — отдельно от item_embeddings,
    так как здесь нужен богатый текст (не просто имя товара).
    """

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    # ------------------------------------------------------------------
    # Индексирование
    # ------------------------------------------------------------------

    def index_receipt(self, receipt: "ReceiptModel") -> None:
        """
        Строит текстовый чанк для чека и сохраняет с embedding.
        Вызывать сразу после save_receipt() — receipt.id уже присвоен.
        """
        text = self._build_receipt_text(receipt)
        ref_key = str(receipt.id)
        try:
            vec = self.client.embed([text])[0]
        except OllamaError as exc:
            logger.warning("RagRetriever: embed чека %s не удался: %s", receipt.id, exc)
            return
        upsert_rag_chunk("receipt", ref_key, text, vec)
        logger.debug("RAG: проиндексирован чек id=%s", receipt.id)

    def index_items(self, item_names: list[str]) -> None:
        """
        Строит чанки для товаров (по данным price_history) и сохраняет с embeddings.
        Вызывать после add_new_items() — чтобы price_history уже существовала.
        При переименовании товара вызвать с новым именем для пересоздания чанка.
        """
        if not item_names:
            return
        texts = [self._build_item_text(name) for name in item_names]
        try:
            vectors = self.client.embed(texts)
        except OllamaError as exc:
            logger.warning("RagRetriever: embed товаров не удался: %s", exc)
            return
        for name, text, vec in zip(item_names, texts, vectors):
            upsert_rag_chunk("item", name, text, vec)
        logger.info("RAG: проиндексировано товаров: %d", len(item_names))

    # ------------------------------------------------------------------
    # Поиск для советника
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
        """
        Возвращает top_k наиболее релевантных чанков для вопроса советника.
        Чистый косинусный поиск — для LLM-контекста скор не нужен.
        """
        chunks = get_all_rag_chunks()
        if not chunks:
            return []

        try:
            q_vec = self.client.embed([question])[0]
        except OllamaError as exc:
            logger.warning("RagRetriever: embed вопроса не удался: %s", exc)
            return []

        scored = [
            (cosine_similarity(q_vec, c["embedding"]), c["chunk_text"])
            for c in chunks
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[:top_k]
        if top:
            logger.info(
                "RAG retrieve: top-%d, similarity %.3f…%.3f",
                top_k, top[0][0], top[-1][0],
            )
        return [text for _, text in top]

    # ------------------------------------------------------------------
    # Семантический поиск для UI
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        chunk_type: str | None = None,
        min_similarity: float = SEARCH_MIN_SIMILARITY,
    ) -> list[SearchResult]:
        """
        Гибридный семантический поиск по базе чанков.

        Итоговый скор = (1 - KEYWORD_WEIGHT) × cosine_similarity
                      +      KEYWORD_WEIGHT  × keyword_score

        Зачем гибрид: эмбеддинг-модель кодирует весь чек в один вектор.
        Если в чеке 10 позиций и 3 из них — пиво, слово "пиво" получает
        низкий вес в итоговом векторе. Keyword-составляющая это компенсирует:
        буквальное вхождение слова в текст дает прямой буст к скору.

        chunk_type: "receipt" | "item" | None (оба типа)
        min_similarity: порог по итоговому гибридному скору.
        """
        chunks = get_all_rag_chunks()
        if not chunks:
            return []

        if chunk_type:
            chunks = [c for c in chunks if c["chunk_type"] == chunk_type]
        if not chunks:
            return []

        try:
            q_vec = self.client.embed([query])[0]
        except OllamaError as exc:
            logger.warning("RagRetriever.search: embed не удался: %s", exc)
            return []

        results = []
        for c in chunks:
            cos = cosine_similarity(q_vec, c["embedding"])
            kw = _keyword_score(query, c["chunk_text"])
            hybrid = round((1 - KEYWORD_WEIGHT) * cos + KEYWORD_WEIGHT * kw, 3)
            results.append(SearchResult(
                chunk_type=c["chunk_type"],
                ref_key=c["ref_key"],
                chunk_text=c["chunk_text"],
                similarity=hybrid,
            ))

        results.sort(key=lambda x: x.similarity, reverse=True)
        results = [r for r in results if r.similarity >= min_similarity]
        top = results[:top_k]

        if top:
            logger.info(
                "Semantic search: запрос=%r, результатов=%d, score %.3f…%.3f",
                query[:50], len(top), top[0].similarity, top[-1].similarity,
            )
        return top

    # ------------------------------------------------------------------
    # Построение текста чанка
    # ------------------------------------------------------------------

    @staticmethod
    def _build_receipt_text(receipt: "ReceiptModel") -> str:
        """
        Чанк чека: заголовок + список товаров + сводная строка состава.

        Сводная строка "Состав: пиво × 3, молоко, хлеб и выпечка" решает проблему
        dilution: повторяет ключевые категории/подкатегории в компактном виде,
        что усиливает их вес в эмбеддинге и помогает keyword-поиску.
        """
        header = (
            f"Чек от {receipt.date}, магазин: {receipt.store_name or '—'}, "
            f"итого: {receipt.total:.0f} руб."
        )
        parts = []
        content_counts: dict[str, int] = {}
        for item in receipt.items:
            sub = f", {item.subcategory}" if item.subcategory else ""
            qty = f"{item.quantity:.0f}×" if item.quantity != 1 else ""
            parts.append(
                f"{item.name} ({item.category}{sub}, "
                f"{qty}{item.price_per_unit:.0f} руб.)"
            )
            # Для сводки: подкатегория если есть, иначе категория
            key = item.subcategory if item.subcategory else item.category
            content_counts[key] = content_counts.get(key, 0) + 1

        items_line = "Товары: " + "; ".join(parts)
        summary_parts = [
            f"{k} × {v}" if v > 1 else k
            for k, v in sorted(content_counts.items(), key=lambda x: -x[1])
        ]
        summary_line = "Состав: " + ", ".join(summary_parts)

        return header + "\n" + items_line + "\n" + summary_line

    @staticmethod
    def _build_item_text(item_name: str) -> str:
        """
        Чанк товара: категория/подкатегория + история цен по магазинам.
        Использует price_history — поэтому вызывать после сохранения чека.
        """
        history = get_price_history(item_name)
        if not history:
            return f"Товар: {item_name} (история цен отсутствует)"

        h0 = history[0]
        sub = f", подкатегория: {h0.subcategory}" if h0.subcategory else ""
        header = f"Товар: {item_name}, категория: {h0.category}{sub}."

        # Группируем по магазину, берём последние 4 цены на магазин
        by_store: dict[str, list[str]] = {}
        for h in history:
            store = h.store_name or "—"
            by_store.setdefault(store, []).append(
                f"{h.price_per_unit:.0f} руб. ({h.date})"
            )

        store_parts = [
            f"{store}: {', '.join(prices[-4:])}"
            for store, prices in by_store.items()
        ]
        return header + "\nИстория цен: " + "; ".join(store_parts)