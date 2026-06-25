from datetime import date, time
from typing import Optional

from sqlalchemy import (
    Date, Float, ForeignKey, Integer, String, Text, Time, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ReceiptModel(Base):
    """Заголовок чека."""
    __tablename__ = "receipts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    subtotal: Mapped[float] = mapped_column(Float, nullable=False)
    discount: Mapped[float] = mapped_column(Float, default=0.0)
    total: Mapped[float] = mapped_column(Float, nullable=False)
    vat_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    payment_method: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    items: Mapped[list["ReceiptItemModel"]] = relationship(
        "ReceiptItemModel",
        back_populates="receipt",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Receipt id={self.id} store={self.store_name!r} date={self.date} total={self.total}>"


class ReceiptItemModel(Base):
    """Позиция в чеке."""
    __tablename__ = "receipt_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    receipt_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("receipts.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    subcategory: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price_per_unit: Mapped[float] = mapped_column(Float, nullable=False)
    total_price: Mapped[float] = mapped_column(Float, nullable=False)
    vat_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    receipt: Mapped["ReceiptModel"] = relationship("ReceiptModel", back_populates="items")

    def __repr__(self) -> str:
        return f"<Item id={self.id} name={self.name!r} subcategory={self.subcategory!r} total={self.total_price}>"


class PriceHistoryModel(Base):
    """
    Денормализованная таблица для трекинга цен.
    Каждая запись — цена конкретного товара в конкретном магазине на конкретную дату.
    """
    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint("store_name", "item_name", "date", name="uq_price_per_day"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    item_name: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    subcategory: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    price_per_unit: Mapped[float] = mapped_column(Float, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<PriceHistory item={self.item_name!r} "
            f"subcategory={self.subcategory!r} "
            f"price={self.price_per_unit} date={self.date}>"
        )


class ItemEmbeddingModel(Base):
    """
    Хранит embedding-вектор для каждого канонического имени товара.
    item_name — нормализованное каноническое имя (эталон).
    embedding_json — вектор сериализованный как JSON-строка.
    При переименовании товара: запись удаляется, создаётся новая с новым именем и вектором.
    """
    __tablename__ = "item_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_name: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    embedding_json: Mapped[str] = mapped_column(String, nullable=False)

    def __repr__(self) -> str:
        return f"<ItemEmbedding item={self.item_name!r}>"


class RagChunkModel(Base):
    """
    Текстовые чанки с embedding-векторами для RAG-поиска в советнике.

    chunk_type = "receipt" → один чек (ref_key = str(receipt.id))
    chunk_type = "item"    → один товар с историей цен (ref_key = item_name)

    chunk_text — человекочитаемый текст, передаётся в LLM как контекст.
    embedding_json — вектор для косинусного поиска по вопросу пользователя.
    """
    __tablename__ = "rag_chunks"
    __table_args__ = (
        UniqueConstraint("chunk_type", "ref_key", name="uq_rag_chunk"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ref_key: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return f"<RagChunk type={self.chunk_type!r} ref={self.ref_key!r}>"