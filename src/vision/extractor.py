"""
src/vision/extractor.py
Принимает фото чека → возвращает валидированный объект Receipt.

Pipeline:
  1. chat_vision (9b) → сырой JSON (имена, категории, цены)
  2. chat_text   (4b) → нормализация имён + классификация подкатегорий за один вызов
  3. Pydantic         → валидация
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path

from pydantic import ValidationError

from config import OLLAMA_TEXT_MODEL  # noqa: F401  # оставлен для будущего использования
from src.llm.client import OllamaClient, OllamaError
from src.llm.prompts import (
    ENRICH_SYSTEM,
    ENRICH_USER,
    RECEIPT_EXTRACTION_SYSTEM,
    RECEIPT_EXTRACTION_USER,
)
from src.schemas.receipt import Receipt

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2


def _strip_category_prefix(subcategory: str) -> str:
    """
    Убирает префикс категории из подкатегории если модель вернула "категория: подкатегория".
    "алкоголь: пиво" → "пиво"
    "пиво"           → "пиво"
    """
    if ": " in subcategory:
        return subcategory.split(": ", 1)[-1].strip()
    return subcategory.strip()


class ExtractionError(Exception):
    """Ошибка извлечения данных из чека."""


class ReceiptExtractor:
    def __init__(self, client: OllamaClient | None = None) -> None:
        # Один клиент (9b) для Vision и для текстовых задач.
        # Не переключаемся между моделями — это вызывает swap и нестабильность Ollama.
        self.client = client or OllamaClient()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def extract_from_file(self, image_path: str | Path) -> Receipt:
        path = Path(image_path)
        if not path.exists():
            raise ExtractionError(f"Файл не найден: {path}")
        return self._extract_with_retry(self._encode_image(path))

    def extract_from_bytes(self, image_bytes: bytes) -> Receipt:
        return self._extract_with_retry(
            base64.b64encode(image_bytes).decode("utf-8")
        )

    # ------------------------------------------------------------------
    # Retry-обёртка
    # ------------------------------------------------------------------

    def _extract_with_retry(self, image_b64: str) -> Receipt:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("Vision extraction, попытка %d/%d", attempt, MAX_RETRIES)
                return self._extract(image_b64)
            except ExtractionError as exc:
                last_error = exc
                logger.warning("Попытка %d неудачна: %s", attempt, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
            except OllamaError as exc:
                raise ExtractionError(f"Ошибка Ollama: {exc}") from exc

        raise ExtractionError(
            f"Не удалось извлечь данные за {MAX_RETRIES} попытки. "
            f"Последняя ошибка: {last_error}"
        )

    # ------------------------------------------------------------------
    # Основной pipeline
    # ------------------------------------------------------------------

    def _extract(self, image_b64: str) -> Receipt:
        # Шаг 1: Vision (9b) → сырой JSON без subcategory
        # prefill="{" — модель сразу начинает писать JSON, минуя thinking
        try:
            raw_response = self.client.chat_vision(
                user_prompt=RECEIPT_EXTRACTION_USER,
                image_b64=image_b64,
                system_prompt=RECEIPT_EXTRACTION_SYSTEM,
                prefill="{",
            )
        except OllamaError as exc:
            raise ExtractionError(f"Ошибка Ollama: {exc}") from exc

        logger.debug("Сырой ответ модели (первые 500 символов):\n%s", raw_response[:500])
        logger.info("Vision ответ: %d символов", len(raw_response))

        json_str = self._parse_json_from_response(raw_response)

        # Шаг 2: Enrich (4b) → нормализация имён + подкатегории за один вызов
        json_str = self._enrich_items(json_str)

        # Шаг 3: валидация Pydantic
        try:
            receipt = Receipt.model_validate_json(json_str)
        except ValidationError as exc:
            logger.error("Pydantic ошибка валидации:\n%s", exc)
            raise ExtractionError(f"Невалидная схема JSON: {exc}") from exc

        logger.info(
            "Чек извлечён: магазин=%s, дата=%s, позиций=%d, итог=%.2f",
            receipt.store_name, receipt.date, len(receipt.items), receipt.total,
        )
        return receipt

    # ------------------------------------------------------------------
    # Enrich: нормализация + подкатегории за один текстовый вызов к 4b
    # ------------------------------------------------------------------

    def _enrich_items(self, json_str: str) -> str:
        """
        Отправляет список товаров в 4b-модель.
        Получает нормализованные имена и подкатегории за один вызов.
        При любой ошибке возвращает исходный JSON — чек не теряется.
        """
        try:
            data = json.loads(json_str)
            items = data.get("items", [])
            if not items:
                return json_str

            enrich_map = self._call_enrich(items)
            if not enrich_map:
                return json_str

            for item in items:
                orig = item.get("name", "")
                if orig in enrich_map:
                    entry = enrich_map[orig]
                    if entry.get("normalized"):
                        item["name"] = entry["normalized"]
                    if entry.get("subcategory"):
                        item["subcategory"] = entry["subcategory"]

            return json.dumps(data, ensure_ascii=False)

        except Exception as exc:
            logger.warning("Enrich не удался, используем оригинал: %s", exc)
            return json_str

    def _call_enrich(self, items: list[dict]) -> dict[str, dict]:
        """
        Вызывает 4b-модель с объединённым промптом.
        Возвращает {original_name: {normalized, subcategory}}.
        """
        items_text = "\n".join(
            f"- {item.get('name', '')} (категория: {item.get('category', '')})"
            for item in items
        )
        prompt = ENRICH_USER.format(items=items_text)

        # prefill="[" — модель сразу начинает писать массив, минуя thinking
        try:
            raw = self.client.chat_text(
                user_prompt=prompt,
                system_prompt=ENRICH_SYSTEM,
                prefill="[",
            )
        except OllamaError as exc:
            logger.warning("Ошибка 4b при enrich: %s", exc)
            return {}

        found = self._find_json(raw)
        if not found:
            thinking = self._extract_thinking(raw)
            if thinking:
                found = self._find_json(thinking)

        if not found:
            logger.warning("Enrich не вернул JSON. Ответ (%d симв.): %s", len(raw), raw[:300])
            return {}

        try:
            result = json.loads(found)
            if not isinstance(result, list):
                logger.warning("Enrich вернул не массив: %s", type(result))
                return {}
            return {
                entry["original"]: {
                    "normalized": entry.get("normalized", ""),
                    "subcategory": _strip_category_prefix(entry.get("subcategory", "")),
                }
                for entry in result
                if isinstance(entry, dict) and "original" in entry
            }
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Ошибка парсинга enrich JSON: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Парсинг JSON из ответа модели
    # ------------------------------------------------------------------

    def _parse_json_from_response(self, response: str) -> str:
        original = response
        thinking_content = self._extract_thinking(response)
        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        if text:
            found = self._find_json(text)
            if found:
                logger.debug("JSON найден вне thinking-блока")
                return found

        if thinking_content:
            found = self._find_json(thinking_content)
            if found:
                logger.warning("JSON найден внутри <think> блока")
                return found

        preview = original[:300].replace("\n", " ")
        raise ExtractionError(f"Модель не вернула JSON. Ответ: {preview!r}")

    @staticmethod
    def _extract_thinking(text: str) -> str:
        match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _find_json(text: str) -> str | None:
        text = text.strip()

        # Markdown-блок
        md_match = re.search(r"```(?:json)?\s*([{\[].*?[}\]])\s*```", text, re.DOTALL)
        if md_match:
            candidate = md_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Определяем, что идёт первым — массив или объект,
        # чтобы не вырезать кусок из середины массива
        obj_start = text.find("{")
        arr_start = text.find("[")

        if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
            searches = [("[", "]"), ("{", "}")]
        else:
            searches = [("{", "}"), ("[", "]")]

        for open_ch, close_ch in searches:
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass

        return None

    @staticmethod
    def _encode_image(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")