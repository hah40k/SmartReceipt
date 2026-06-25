import logging
from datetime import date
from typing import Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from config import DB_URL
from src.db.models import Base, ItemEmbeddingModel, PriceHistoryModel, RagChunkModel, ReceiptItemModel, ReceiptModel
from src.schemas.receipt import Receipt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Движок и фабрика сессий
# ---------------------------------------------------------------------------

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Создаёт таблицы, если они ещё не существуют."""
    Base.metadata.create_all(bind=engine)
    logger.info("БД инициализирована: %s", DB_URL)


# ---------------------------------------------------------------------------
# Вспомогательный контекстный менеджер
# ---------------------------------------------------------------------------

def get_session() -> Session:
    return SessionLocal()


# ---------------------------------------------------------------------------
# Запись
# ---------------------------------------------------------------------------

def save_receipt(
    receipt: Receipt,
    track_items: set[str] | None = None,
) -> ReceiptModel:
    """
    Сохраняет объект Receipt (Pydantic) в базу.

    track_items:
      - None  → отслеживать все товары (обратная совместимость)
      - set() → не отслеживать ничего
      - {...} → отслеживать только перечисленные названия
    """
    with SessionLocal() as session:
        db_receipt = ReceiptModel(
            store_name=receipt.store_name,
            date=receipt.date,
            time=receipt.time,
            subtotal=receipt.subtotal,
            discount=receipt.discount,
            total=receipt.total,
            vat_total=receipt.vat_total,
            payment_method=receipt.payment_method,
        )

        for item in receipt.items:
            db_item = ReceiptItemModel(
                name=item.name,
                category=item.category,
                subcategory=item.subcategory,
                quantity=item.quantity,
                price_per_unit=item.price_per_unit,
                total_price=item.total_price,
                vat_rate=item.vat_rate,
            )
            db_receipt.items.append(db_item)

            # Добавляем в price_history только если товар выбран для отслеживания
            if track_items is None or item.name in track_items:
                _upsert_price_history(
                    session,
                    store_name=receipt.store_name,
                    item_name=item.name,
                    category=item.category,
                    subcategory=item.subcategory,
                    price_per_unit=item.price_per_unit,
                    purchase_date=receipt.date,
                )

        session.add(db_receipt)
        session.commit()
        session.refresh(db_receipt)

        tracked_n = len(track_items) if track_items is not None else len(receipt.items)
        logger.info(
            "Чек сохранён: id=%s, магазин=%s, сумма=%s, отслеживается позиций=%d",
            db_receipt.id, db_receipt.store_name, db_receipt.total, tracked_n,
        )
        return db_receipt


def _upsert_price_history(
    session: Session,
    store_name: Optional[str],
    item_name: str,
    category: str,
    subcategory: Optional[str],
    price_per_unit: float,
    purchase_date: date,
) -> None:
    """
    Upsert цены товара через нативный SQLite INSERT OR REPLACE.
    Надёжно работает в циклах внутри одной сессии — не зависит от flush.
    """
    stmt = (
        sqlite_insert(PriceHistoryModel)
        .values(
            store_name=store_name,
            item_name=item_name,
            category=category,
            subcategory=subcategory,
            price_per_unit=price_per_unit,
            date=purchase_date,
        )
        .on_conflict_do_update(
            index_elements=["store_name", "item_name", "date"],
            set_={
                "price_per_unit": price_per_unit,
                "category": category,
                "subcategory": subcategory,
            },
        )
    )
    session.execute(stmt)


# ---------------------------------------------------------------------------
# Управление отслеживанием цен
# ---------------------------------------------------------------------------

def enable_price_tracking_for_item(item_name: str) -> int:
    """
    Включает отслеживание цен для товара по данным уже сохранённых чеков.
    Проходит по всем ReceiptItemModel с таким именем и upsert-ит price_history.
    Возвращает количество добавленных/обновлённых записей.
    """
    with SessionLocal() as session:
        rows = session.execute(
            select(ReceiptItemModel, ReceiptModel)
            .join(ReceiptModel, ReceiptItemModel.receipt_id == ReceiptModel.id)
            .where(ReceiptItemModel.name == item_name)
            .order_by(ReceiptModel.date.asc())
        ).all()

        count = 0
        for item_row, receipt_row in rows:
            _upsert_price_history(
                session,
                store_name=receipt_row.store_name,
                item_name=item_row.name,
                category=item_row.category,
                subcategory=item_row.subcategory,
                price_per_unit=item_row.price_per_unit,
                purchase_date=receipt_row.date,
            )
            count += 1

        session.commit()
        logger.info(
            "Включено отслеживание для «%s»: добавлено/обновлено %d записей",
            item_name, count,
        )
        return count


def get_untracked_items() -> list[str]:
    """
    Возвращает список товаров из чеков, для которых ещё не включено отслеживание.
    (есть в receipt_items, но отсутствуют в price_history)
    """
    with SessionLocal() as session:
        tracked_subq = select(PriceHistoryModel.item_name).distinct().scalar_subquery()
        stmt = (
            select(ReceiptItemModel.name)
            .distinct()
            .where(ReceiptItemModel.name.not_in(tracked_subq))
            .order_by(ReceiptItemModel.name)
        )
        return list(session.scalars(stmt).all())


def get_all_item_names() -> list[str]:
    """Возвращает все уникальные имена товаров из чеков, отсортированные по алфавиту."""
    with SessionLocal() as session:
        stmt = select(ReceiptItemModel.name).distinct().order_by(ReceiptItemModel.name)
        return list(session.scalars(stmt).all())


# ---------------------------------------------------------------------------
# Чтение
# ---------------------------------------------------------------------------

def get_all_receipts(limit: int = 500) -> list[ReceiptModel]:
    with SessionLocal() as session:
        stmt = (
            select(ReceiptModel)
            .order_by(ReceiptModel.date.desc(), ReceiptModel.id.desc())
            .limit(limit)
        )
        return list(session.scalars(stmt).all())


def get_receipt_by_id(receipt_id: int) -> Optional[ReceiptModel]:
    with SessionLocal() as session:
        return session.get(ReceiptModel, receipt_id)


def get_receipts_by_date_range(start: date, end: date) -> list[ReceiptModel]:
    with SessionLocal() as session:
        stmt = (
            select(ReceiptModel)
            .where(ReceiptModel.date >= start, ReceiptModel.date <= end)
            .order_by(ReceiptModel.date.asc())
        )
        return list(session.scalars(stmt).all())


def get_price_history(item_name: str, store_name: Optional[str] = None) -> list[PriceHistoryModel]:
    with SessionLocal() as session:
        stmt = (
            select(PriceHistoryModel)
            .where(PriceHistoryModel.item_name == item_name)
            .order_by(PriceHistoryModel.date.asc())
        )
        if store_name:
            stmt = stmt.where(PriceHistoryModel.store_name == store_name)
        return list(session.scalars(stmt).all())


# ---------------------------------------------------------------------------
# Агрегация
# ---------------------------------------------------------------------------

def get_totals_by_category(start: Optional[date] = None, end: Optional[date] = None) -> list[dict]:
    with SessionLocal() as session:
        stmt = (
            select(
                ReceiptItemModel.category,
                func.sum(ReceiptItemModel.total_price).label("total"),
            )
            .join(ReceiptModel, ReceiptItemModel.receipt_id == ReceiptModel.id)
            .group_by(ReceiptItemModel.category)
        )
        if start:
            stmt = stmt.where(ReceiptModel.date >= start)
        if end:
            stmt = stmt.where(ReceiptModel.date <= end)
        rows = session.execute(stmt).all()
        return [{"category": row.category, "total": round(row.total, 2)} for row in rows]


def get_store_stats(store_name: str) -> dict:
    with SessionLocal() as session:
        stmt = select(
            func.count(ReceiptModel.id).label("count"),
            func.avg(ReceiptModel.total).label("avg_total"),
            func.max(ReceiptModel.total).label("max_total"),
        ).where(ReceiptModel.store_name == store_name)
        row = session.execute(stmt).one()
        return {
            "store_name": store_name,
            "count": row.count or 0,
            "avg_total": round(row.avg_total or 0, 2),
            "max_total": round(row.max_total or 0, 2),
        }


def clear_all_data() -> dict:
    """Удаляет все записи из всех таблиц. Только для разработки/тестирования."""
    with SessionLocal() as session:
        ph = session.query(PriceHistoryModel).delete()
        ri = session.query(ReceiptItemModel).delete()
        r  = session.query(ReceiptModel).delete()
        session.commit()
    logger.warning("База очищена: чеков=%d, позиций=%d, цен=%d", r, ri, ph)
    return {"receipts": r, "items": ri, "price_history": ph}


def upsert_item_embedding(item_name: str, embedding: list[float]) -> None:
    """Сохраняет или обновляет embedding для канонического имени товара."""
    import json as _json
    embedding_json = _json.dumps(embedding)
    with SessionLocal() as session:
        existing = session.execute(
            select(ItemEmbeddingModel).where(ItemEmbeddingModel.item_name == item_name)
        ).scalar_one_or_none()
        if existing:
            existing.embedding_json = embedding_json
        else:
            session.add(ItemEmbeddingModel(
                item_name=item_name,
                embedding_json=embedding_json,
            ))
        session.commit()


def get_all_item_embeddings() -> list[dict]:
    """
    Возвращает все записи из item_embeddings как список словарей
    {item_name: str, embedding: list[float]}.
    """
    import json as _json
    with SessionLocal() as session:
        rows = session.scalars(select(ItemEmbeddingModel)).all()
        return [
            {"item_name": r.item_name, "embedding": _json.loads(r.embedding_json)}
            for r in rows
        ]


def rename_item(old_name: str, new_name: str) -> int:
    """
    Переименовывает товар во всех таблицах: receipt_items, price_history, item_embeddings.
    RAG-чанк для старого имени тоже удаляется — retriever пересоздаст его под новым именем.
    Embedding для нового имени нужно пересчитать отдельно (вызвать matcher.add_new_items).
    Возвращает количество затронутых позиций в receipt_items.
    """
    with SessionLocal() as session:
        items_updated = session.execute(
            ReceiptItemModel.__table__.update()
            .where(ReceiptItemModel.name == old_name)
            .values(name=new_name)
        ).rowcount
        session.execute(
            PriceHistoryModel.__table__.update()
            .where(PriceHistoryModel.item_name == old_name)
            .values(item_name=new_name)
        )
        session.execute(
            ItemEmbeddingModel.__table__.delete()
            .where(ItemEmbeddingModel.item_name == old_name)
        )
        # Удаляем устаревший RAG-чанк — retriever.index_items(new_name) пересоздаст
        session.execute(
            RagChunkModel.__table__.delete()
            .where(RagChunkModel.chunk_type == "item")
            .where(RagChunkModel.ref_key == old_name)
        )
        session.commit()
    logger.info("Товар переименован: «%s» → «%s», позиций: %d", old_name, new_name, items_updated)
    return items_updated


# ---------------------------------------------------------------------------
# RAG-чанки
# ---------------------------------------------------------------------------

def upsert_rag_chunk(
    chunk_type: str,
    ref_key: str,
    chunk_text: str,
    embedding: list[float],
) -> None:
    """Сохраняет или обновляет RAG-чанк (тип + ключ уникальны вместе)."""
    import json as _json
    embedding_json = _json.dumps(embedding)
    with SessionLocal() as session:
        existing = session.execute(
            select(RagChunkModel).where(
                RagChunkModel.chunk_type == chunk_type,
                RagChunkModel.ref_key == ref_key,
            )
        ).scalar_one_or_none()
        if existing:
            existing.chunk_text = chunk_text
            existing.embedding_json = embedding_json
        else:
            session.add(RagChunkModel(
                chunk_type=chunk_type,
                ref_key=ref_key,
                chunk_text=chunk_text,
                embedding_json=embedding_json,
            ))
        session.commit()


def get_all_rag_chunks() -> list[dict]:
    """
    Возвращает все RAG-чанки как список словарей
    {chunk_type, ref_key, chunk_text, embedding: list[float]}.
    """
    import json as _json
    with SessionLocal() as session:
        rows = session.scalars(select(RagChunkModel)).all()
        return [
            {
                "chunk_type": r.chunk_type,
                "ref_key": r.ref_key,
                "chunk_text": r.chunk_text,
                "embedding": _json.loads(r.embedding_json),
            }
            for r in rows
        ]