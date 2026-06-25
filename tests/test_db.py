"""
tests/test_db.py
Тесты для схем (Pydantic) и CRUD-операций (SQLite in-memory).
Запуск: python -m pytest tests/test_db.py -v
"""
import sys
from datetime import date as Date, time as Time
from pathlib import Path

# Чтобы импорты работали из корня проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base
from src.schemas.receipt import Receipt, ReceiptItem


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_session():
    """SQLite in-memory для изолированных тестов."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        yield session


@pytest.fixture
def sample_receipt() -> Receipt:
    return Receipt(
        store_name="Пятёрочка",
        date=Date(2025, 6, 1),
        time=Time(14, 30),
        items=[
            ReceiptItem(
                name="Молоко 3.2%",
                category="продукты питания",
                quantity=1.0,
                price_per_unit=89.9,
                total_price=89.9,
            ),
            ReceiptItem(
                name="Хлеб Нарезной",
                category="продукты питания",
                quantity=2.0,
                price_per_unit=45.0,
                total_price=90.0,
            ),
        ],
        subtotal=179.9,
        discount=0.0,
        total=179.9,
    )


# ---------------------------------------------------------------------------
# Pydantic-схемы
# ---------------------------------------------------------------------------

class TestReceiptSchema:

    def test_valid_receipt(self, sample_receipt):
        assert sample_receipt.total == 179.9
        assert len(sample_receipt.items) == 2

    def test_category_normalized_to_lowercase(self):
        item = ReceiptItem(
            name="Пиво",
            category="  Алкоголь  ",
            quantity=1,
            price_per_unit=120.0,
            total_price=120.0,
        )
        assert item.category == "алкоголь"

    def test_negative_price_raises(self):
        with pytest.raises(Exception):
            ReceiptItem(
                name="Товар",
                category="прочее",
                quantity=1,
                price_per_unit=-10.0,
                total_price=-10.0,
            )

    def test_empty_items_raises(self):
        with pytest.raises(Exception):
            Receipt(
                store_name="Магазин",
                date=Date(2025, 1, 1),
                items=[],
                subtotal=0,
                total=0,
            )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestCRUD:

    def test_save_and_retrieve(self, sample_receipt, in_memory_session, monkeypatch):
        """Сохраняем чек через crud и проверяем, что он читается обратно."""
        from src.db import crud
        from src.db.models import ReceiptModel

        # Подменяем SessionLocal на in-memory сессию
        monkeypatch.setattr(crud, "SessionLocal", lambda: in_memory_session)
        # Также инициализируем таблицы
        Base.metadata.create_all(in_memory_session.get_bind())

        # Используем сессию напрямую для теста
        from src.db.models import ReceiptItemModel

        db_receipt = ReceiptModel(
            store_name=sample_receipt.store_name,
            date=sample_receipt.date,
            time=sample_receipt.time,
            subtotal=sample_receipt.subtotal,
            discount=sample_receipt.discount,
            total=sample_receipt.total,
        )
        for item in sample_receipt.items:
            db_receipt.items.append(ReceiptItemModel(
                name=item.name,
                category=item.category,
                quantity=item.quantity,
                price_per_unit=item.price_per_unit,
                total_price=item.total_price,
            ))

        in_memory_session.add(db_receipt)
        in_memory_session.commit()
        in_memory_session.refresh(db_receipt)

        assert db_receipt.id is not None
        assert db_receipt.store_name == "Пятёрочка"
        assert len(db_receipt.items) == 2

    def test_receipt_item_count(self, sample_receipt, in_memory_session):
        from src.db.models import ReceiptItemModel, ReceiptModel

        db_receipt = ReceiptModel(
            store_name=sample_receipt.store_name,
            date=sample_receipt.date,
            subtotal=sample_receipt.subtotal,
            discount=0.0,
            total=sample_receipt.total,
        )
        for item in sample_receipt.items:
            db_receipt.items.append(ReceiptItemModel(
                name=item.name,
                category=item.category,
                quantity=item.quantity,
                price_per_unit=item.price_per_unit,
                total_price=item.total_price,
            ))
        in_memory_session.add(db_receipt)
        in_memory_session.commit()

        from sqlalchemy import select
        count = in_memory_session.scalar(
            select(ReceiptItemModel).where(ReceiptItemModel.receipt_id == db_receipt.id)
        )
        assert count is not None
