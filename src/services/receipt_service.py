"""
src/services/receipt_service.py

Сервисный слой для сохранения чеков и управления товарами.
Отделяет бизнес-логику от UI — функции здесь не знают о Gradio.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.db.crud import get_all_item_embeddings, rename_item, save_receipt
from src.schemas.receipt import Receipt

if TYPE_CHECKING:
    from src.analytics.matcher import ItemMatcher
    from src.analytics.retriever import RagRetriever
    from src.db.models import ReceiptModel

logger = logging.getLogger(__name__)


@dataclass
class SaveReceiptResult:
    db_receipt: "ReceiptModel"
    rename_count: int     # количество объединённых (renamed) товаров
    tracked_n: int        # количество позиций с отслеживанием цен
    total_n: int          # всего позиций в чеке


def save_receipt_with_context(
    receipt: Receipt,
    tracked_items: list[str],
    matches_state,                  # list[MatchResult] | None
    accepted_matches: list[str],
    store_name_override: str,
    matcher: "ItemMatcher",
    retriever: "RagRetriever",
) -> SaveReceiptResult:
    """
    Полный pipeline сохранения чека:

    1. Применяет ручное имя магазина (нормализуется как в Pydantic-валидаторе).
    2. Переименовывает товары согласно принятым embedding-совпадениям.
    3. Сохраняет чек в БД (с price_history для отслеживаемых позиций).
    4. Индексирует чек для RAG.
    5. Сохраняет embeddings для новых канонических имён.
    6. Индексирует новые товары для RAG.

    Принимает matcher и retriever как зависимости — легко тестировать через моки.
    """
    # Нормализуем имя магазина так же как Pydantic-валидатор Receipt
    store_override = (store_name_override or "").strip().title() or None
    if store_override:
        receipt.store_name = store_override

    # Строим карту переименований: new_name → canonical_name
    accepted_set = set(accepted_matches or [])
    rename_map: dict[str, str] = {}
    if matches_state:
        for m in matches_state:
            if m.new_name in accepted_set:
                rename_map[m.new_name] = m.canonical_name

    if rename_map:
        for item in receipt.items:
            if item.name in rename_map:
                item.name = rename_map[item.name]

    # Имена, которые не объединены → становятся новыми каноническими
    all_names = {item.name for item in receipt.items}
    new_canonical = all_names - set(rename_map.values())

    # Синхронизируем track_set с переименованиями
    track_set = set(tracked_items or [])
    for old, new in rename_map.items():
        if old in track_set:
            track_set.discard(old)
            track_set.add(new)

    db_receipt = save_receipt(receipt, track_items=track_set)

    # RAG: индексируем чек
    retriever.index_receipt(db_receipt)

    # Embeddings для новых канонических имён + RAG-чанки товаров
    # (index_items после save_receipt — price_history уже записана)
    matcher.add_new_items(list(new_canonical))
    retriever.index_items(list(new_canonical))

    return SaveReceiptResult(
        db_receipt=db_receipt,
        rename_count=len(rename_map),
        tracked_n=len(track_set),
        total_n=len(receipt.items),
    )


def merge_items(
    source: str,
    target: str,
    matcher: "ItemMatcher",
    retriever: "RagRetriever",
) -> int:
    """
    Объединяет два товара: все записи source переписываются под имя target.

    После слияния:
    - embedding source удалён (crud.rename_item)
    - если у target нет embedding — создаётся
    - RAG-чанк source удалён (crud.rename_item), пересоздаётся под target

    Возвращает количество затронутых позиций в receipt_items.
    """
    count = rename_item(source, target)

    # Если у целевого имени нет embedding — нужно создать
    existing_embeddings = {e["item_name"] for e in get_all_item_embeddings()}
    if target not in existing_embeddings:
        matcher.add_new_items([target])

    # Пересоздаём RAG-чанк (старый чанк source удалён в rename_item)
    retriever.index_items([target])

    logger.info("Товары объединены: «%s» → «%s», позиций: %d", source, target, count)
    return count
