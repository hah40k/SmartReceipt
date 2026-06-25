"""
tests/test_analytics.py
Тесты аналитики — используют in-memory SQLite, без реального Ollama.
Запуск: python -m pytest tests/test_analytics.py -v
"""
from __future__ import annotations

import sys
from datetime import date as Date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, ReceiptItemModel, ReceiptModel, PriceHistoryModel
from src.schemas.receipt import Receipt, ReceiptItem
from src.analytics.analyzer import ReceiptAnalyzer


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(e)
    return e


@pytest.fixture
def session(engine):
    S = sessionmaker(bind=engine)
    with S() as s:
        yield s


@pytest.fixture
def analyzer(engine, monkeypatch):
    """Analyzer, у которого SessionLocal подменён на in-memory."""
    from src.analytics import analyzer as ana_mod
    from src.db import crud as crud_mod

    S = sessionmaker(bind=engine)
    monkeypatch.setattr(ana_mod, "SessionLocal", S)
    monkeypatch.setattr(crud_mod, "SessionLocal", S)
    return ReceiptAnalyzer()


def make_db_receipt(session, store: str, total: float, receipt_date: Date, items: list[dict]):
    r = ReceiptModel(
        store_name=store,
        date=receipt_date,
        subtotal=total,
        discount=0.0,
        total=total,
    )
    for it in items:
        r.items.append(ReceiptItemModel(
            name=it["name"],
            category=it["category"],
            quantity=it.get("qty", 1.0),
            price_per_unit=it["price"],
            total_price=it["price"] * it.get("qty", 1.0),
        ))
    session.add(r)
    session.commit()
    return r


def make_receipt_schema(store: str, total: float, receipt_date: Date, items: list[dict]) -> Receipt:
    return Receipt(
        store_name=store,
        date=receipt_date,
        items=[
            ReceiptItem(
                name=it["name"],
                category=it["category"],
                quantity=it.get("qty", 1.0),
                price_per_unit=it["price"],
                total_price=it["price"] * it.get("qty", 1.0),
            )
            for it in items
        ],
        subtotal=total,
        discount=0.0,
        total=total,
    )


# ---------------------------------------------------------------------------
# Тесты: аномалия — высокая сумма чека
# ---------------------------------------------------------------------------

class TestHighTotal:

    def test_no_anomaly_below_threshold(self, session, analyzer):
        store = "Пятёрочка"
        items = [{"name": "Молоко", "category": "молочные продукты", "price": 90.0}]
        for i in range(5):
            make_db_receipt(session, store, 500.0, Date(2025, 6, i + 1), items)

        receipt = make_receipt_schema(store, 800.0, Date(2025, 6, 10), items)
        report = analyzer.detect_anomalies(receipt)
        high_total = [a for a in report.anomalies if a.kind == "high_total"]
        assert len(high_total) == 0

    def test_anomaly_detected_when_double(self, session, analyzer):
        store = "Пятёрочка"
        items = [{"name": "Молоко", "category": "молочные продукты", "price": 90.0}]
        for i in range(5):
            make_db_receipt(session, store, 500.0, Date(2025, 6, i + 1), items)

        receipt = make_receipt_schema(store, 1200.0, Date(2025, 6, 10), items)
        report = analyzer.detect_anomalies(receipt)
        high_total = [a for a in report.anomalies if a.kind == "high_total"]
        assert len(high_total) == 1
        assert "1200" in high_total[0].message or "2" in high_total[0].message

    def test_no_anomaly_without_enough_history(self, session, analyzer):
        store = "Новый магазин"
        items = [{"name": "Хлеб", "category": "хлеб и выпечка", "price": 50.0}]
        make_db_receipt(session, store, 50.0, Date(2025, 6, 1), items)

        receipt = make_receipt_schema(store, 5000.0, Date(2025, 6, 2), items)
        report = analyzer.detect_anomalies(receipt)
        high_total = [a for a in report.anomalies if a.kind == "high_total"]
        assert len(high_total) == 0


# ---------------------------------------------------------------------------
# Тесты: трекинг цен / price spike
# ---------------------------------------------------------------------------

class TestPriceSpike:

    def test_price_spike_detected(self, session, analyzer):
        # Добавляем историю цен
        session.add(PriceHistoryModel(
            store_name="Магнит",
            item_name="Молоко 3.2%",
            category="молочные продукты",
            price_per_unit=80.0,
            date=Date(2025, 5, 1),
        ))
        session.commit()

        receipt = make_receipt_schema(
            "Магнит", 120.0, Date(2025, 6, 1),
            [{"name": "Молоко 3.2%", "category": "молочные продукты", "price": 120.0}]
        )
        report = analyzer.detect_anomalies(receipt)
        spikes = [a for a in report.anomalies if a.kind == "price_spike"]
        assert len(spikes) == 1
        assert "Молоко 3.2%" in spikes[0].message

    def test_no_spike_for_small_change(self, session, analyzer):
        session.add(PriceHistoryModel(
            store_name="Магнит",
            item_name="Хлеб нарезной",
            category="хлеб и выпечка",
            price_per_unit=50.0,
            date=Date(2025, 5, 1),
        ))
        session.commit()

        receipt = make_receipt_schema(
            "Магнит", 55.0, Date(2025, 6, 1),
            [{"name": "Хлеб нарезной", "category": "хлеб и выпечка", "price": 55.0}]
        )
        report = analyzer.detect_anomalies(receipt)
        spikes = [a for a in report.anomalies if a.kind == "price_spike"]
        assert len(spikes) == 0


# ---------------------------------------------------------------------------
# Тесты: агрегация по категориям
# ---------------------------------------------------------------------------

class TestCategoryTotals:

    def test_returns_sorted_by_total(self, session, analyzer):
        items = [
            {"name": "Пиво", "category": "алкоголь", "price": 200.0},
            {"name": "Молоко", "category": "молочные продукты", "price": 90.0},
        ]
        make_db_receipt(session, "Магазин", 290.0, Date(2025, 6, 1), items)

        totals = analyzer.get_category_totals()
        assert totals[0].category == "алкоголь"
        assert totals[0].total == 200.0
        assert totals[1].category == "молочные продукты"

    def test_percent_sums_to_100(self, session, analyzer):
        items = [
            {"name": "A", "category": "алкоголь", "price": 300.0},
            {"name": "B", "category": "еда", "price": 700.0},
        ]
        make_db_receipt(session, "Магазин", 1000.0, Date(2025, 6, 1), items)

        totals = analyzer.get_category_totals()
        total_pct = sum(ct.percent for ct in totals)
        assert abs(total_pct - 100.0) < 0.5
