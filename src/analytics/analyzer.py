"""
src/analytics/analyzer.py

Модуль аналитики:
- Детекция аномалий нового чека относительно истории
- Агрегация расходов по категориям / подкатегориям / периодам
- Трекинг цен на конкретные товары
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func, select

from src.db.crud import SessionLocal, get_receipts_by_date_range
from src.db.models import PriceHistoryModel, ReceiptItemModel, ReceiptModel
from src.schemas.receipt import Receipt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass-результаты
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    kind: str          # "high_total" | "price_spike" | "category_surge"
    message: str       # человекочитаемое описание
    severity: str      # "info" | "warning" | "critical"


@dataclass
class AnomalyReport:
    anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)

    def summary(self) -> str:
        if not self.anomalies:
            return "Всё в пределах нормы."
        return "\n".join(f"• {a.message}" for a in self.anomalies)


@dataclass
class CategoryTotal:
    category: str
    total: float
    percent: float


@dataclass
class SubcategoryTotal:
    category: str
    subcategory: str        # "прочее" если subcategory IS NULL
    total: float
    percent_of_category: float


@dataclass
class PricePoint:
    date: date
    price: float
    store_name: Optional[str]


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class ReceiptAnalyzer:
    """
    Аналитика по истории чеков.
    Все методы работают с БД через SessionLocal.
    """

    # Пороги для аномалий
    HIGH_TOTAL_MULTIPLIER = 2.0      # чек в 2× дороже среднего → warning
    PRICE_SPIKE_PERCENT = 20.0       # рост цены товара на 20%+ → warning
    CATEGORY_SURGE_PERCENT = 40.0    # рост категории за месяц на 40%+ → warning
    MIN_RECEIPTS_FOR_STATS = 3       # минимум чеков для сравнения

    # ---------------------------------------------------------------------------
    # Детекция аномалий
    # ---------------------------------------------------------------------------

    def detect_anomalies(self, receipt: Receipt) -> AnomalyReport:
        """
        Сравнивает новый чек с историей и возвращает отчёт об аномалиях.
        Вызывать ДО сохранения чека в БД.
        """
        report = AnomalyReport()

        report.anomalies.extend(self._check_high_total(receipt))
        report.anomalies.extend(self._check_price_spikes(receipt))
        report.anomalies.extend(self._check_category_surge(receipt))

        if report.has_anomalies:
            logger.info("Аномалии в чеке (%s): %d шт.", receipt.store_name, len(report.anomalies))
        return report

    def _check_high_total(self, receipt: Receipt) -> list[Anomaly]:
        """Проверяет, не превышает ли сумма чека средний в N раз."""
        if not receipt.store_name:
            return []

        with SessionLocal() as session:
            row = session.execute(
                select(
                    func.count(ReceiptModel.id).label("cnt"),
                    func.avg(ReceiptModel.total).label("avg"),
                ).where(ReceiptModel.store_name == receipt.store_name)
            ).one()

        if (row.cnt or 0) < self.MIN_RECEIPTS_FOR_STATS:
            return []

        avg = row.avg or 0
        if avg > 0 and receipt.total > avg * self.HIGH_TOTAL_MULTIPLIER:
            return [Anomaly(
                kind="high_total",
                message=(
                    f"Сумма чека {receipt.total:.0f} руб. — в "
                    f"{receipt.total / avg:.1f}× больше среднего по «{receipt.store_name}» "
                    f"({avg:.0f} руб.)"
                ),
                severity="warning",
            )]
        return []

    def _check_price_spikes(self, receipt: Receipt) -> list[Anomaly]:
        """Сравнивает цены товаров с последней известной ценой в истории."""
        anomalies: list[Anomaly] = []

        with SessionLocal() as session:
            for item in receipt.items:
                row = session.execute(
                    select(PriceHistoryModel.price_per_unit, PriceHistoryModel.date)
                    .where(PriceHistoryModel.item_name == item.name)
                    .order_by(PriceHistoryModel.date.desc())
                    .limit(1)
                ).one_or_none()

                if row is None:
                    continue

                old_price, old_date = row
                if old_price <= 0:
                    continue

                change_pct = (item.price_per_unit - old_price) / old_price * 100
                if change_pct >= self.PRICE_SPIKE_PERCENT:
                    anomalies.append(Anomaly(
                        kind="price_spike",
                        message=(
                            f"«{item.name}»: цена выросла с {old_price:.2f} до "
                            f"{item.price_per_unit:.2f} руб. "
                            f"(+{change_pct:.0f}% с {old_date})"
                        ),
                        severity="warning",
                    ))

        return anomalies

    def _check_category_surge(self, receipt: Receipt) -> list[Anomaly]:
        """Сравнивает расходы по категориям: этот месяц vs прошлый."""
        anomalies: list[Anomaly] = []
        today = receipt.date

        cur_start = today.replace(day=1)
        prev_end = cur_start - timedelta(days=1)
        prev_start = prev_end.replace(day=1)

        new_categories: dict[str, float] = {}
        for item in receipt.items:
            new_categories[item.category] = (
                new_categories.get(item.category, 0) + item.total_price
            )

        with SessionLocal() as session:
            for category, new_amount in new_categories.items():
                prev = session.scalar(
                    select(func.sum(ReceiptItemModel.total_price))
                    .join(ReceiptModel, ReceiptItemModel.receipt_id == ReceiptModel.id)
                    .where(
                        ReceiptItemModel.category == category,
                        ReceiptModel.date >= prev_start,
                        ReceiptModel.date <= prev_end,
                    )
                ) or 0.0

                cur = session.scalar(
                    select(func.sum(ReceiptItemModel.total_price))
                    .join(ReceiptModel, ReceiptItemModel.receipt_id == ReceiptModel.id)
                    .where(
                        ReceiptItemModel.category == category,
                        ReceiptModel.date >= cur_start,
                        ReceiptModel.date <= today,
                    )
                ) or 0.0

                total_cur = cur + new_amount
                if prev > 0:
                    change_pct = (total_cur - prev) / prev * 100
                    if change_pct >= self.CATEGORY_SURGE_PERCENT:
                        anomalies.append(Anomaly(
                            kind="category_surge",
                            message=(
                                f"Расходы на «{category}» в этом месяце "
                                f"{total_cur:.0f} руб. (+{change_pct:.0f}% к прошлому месяцу)"
                            ),
                            severity="info",
                        ))

        return anomalies

    # ---------------------------------------------------------------------------
    # Агрегация по категориям
    # ---------------------------------------------------------------------------

    def get_category_totals(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> list[CategoryTotal]:
        """
        Суммарные расходы по категориям за период.
        Возвращает список, отсортированный по убыванию суммы.
        """
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

        if not rows:
            return []

        grand_total = sum(r.total for r in rows)
        result = [
            CategoryTotal(
                category=r.category,
                total=round(r.total, 2),
                percent=round(r.total / grand_total * 100, 1) if grand_total else 0,
            )
            for r in rows
        ]
        return sorted(result, key=lambda x: x.total, reverse=True)

    def get_subcategory_totals(
        self,
        category: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> list[SubcategoryTotal]:
        """
        Суммарные расходы по подкатегориям за период.
        Если category передана — только по ней, иначе по всем.
        NULL-подкатегории объединяются в «прочее».
        Возвращает список, отсортированный по категории, затем по убыванию суммы.
        """
        with SessionLocal() as session:
            stmt = (
                select(
                    ReceiptItemModel.category,
                    func.coalesce(ReceiptItemModel.subcategory, "прочее").label("subcategory"),
                    func.sum(ReceiptItemModel.total_price).label("total"),
                )
                .join(ReceiptModel, ReceiptItemModel.receipt_id == ReceiptModel.id)
                .group_by(
                    ReceiptItemModel.category,
                    func.coalesce(ReceiptItemModel.subcategory, "прочее"),
                )
            )
            if category:
                stmt = stmt.where(ReceiptItemModel.category == category)
            if start:
                stmt = stmt.where(ReceiptModel.date >= start)
            if end:
                stmt = stmt.where(ReceiptModel.date <= end)

            rows = session.execute(stmt).all()

        if not rows:
            return []

        # Считаем итог по каждой категории для процентов
        cat_totals: dict[str, float] = defaultdict(float)
        for r in rows:
            cat_totals[r.category] += r.total

        result = [
            SubcategoryTotal(
                category=r.category,
                subcategory=r.subcategory,
                total=round(r.total, 2),
                percent_of_category=round(
                    r.total / cat_totals[r.category] * 100, 1
                ) if cat_totals[r.category] else 0,
            )
            for r in rows
        ]

        return sorted(result, key=lambda x: (x.category, -x.total))

    # ---------------------------------------------------------------------------
    # Трекинг расходов по времени
    # ---------------------------------------------------------------------------

    def get_daily_totals(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> list[dict]:
        with SessionLocal() as session:
            stmt = (
                select(
                    ReceiptModel.date,
                    func.sum(ReceiptModel.total).label("total"),
                )
                .group_by(ReceiptModel.date)
                .order_by(ReceiptModel.date.asc())
            )
            if start:
                stmt = stmt.where(ReceiptModel.date >= start)
            if end:
                stmt = stmt.where(ReceiptModel.date <= end)

            rows = session.execute(stmt).all()

        return [{"date": r.date, "total": round(r.total, 2)} for r in rows]

    def get_monthly_totals(self) -> list[dict]:
        with SessionLocal() as session:
            rows = session.execute(
                select(
                    func.strftime("%Y-%m", ReceiptModel.date).label("month"),
                    func.sum(ReceiptModel.total).label("total"),
                )
                .group_by("month")
                .order_by("month")
            ).all()

        return [{"month": r.month, "total": round(r.total, 2)} for r in rows]

    # ---------------------------------------------------------------------------
    # Трекинг цен
    # ---------------------------------------------------------------------------

    def get_price_history(
        self,
        item_name: str,
        store_name: Optional[str] = None,
    ) -> list[PricePoint]:
        with SessionLocal() as session:
            stmt = (
                select(PriceHistoryModel)
                .where(PriceHistoryModel.item_name == item_name)
                .order_by(PriceHistoryModel.date.asc())
            )
            if store_name:
                stmt = stmt.where(PriceHistoryModel.store_name == store_name)

            rows = session.scalars(stmt).all()

        return [
            PricePoint(date=r.date, price=r.price_per_unit, store_name=r.store_name)
            for r in rows
        ]

    def get_tracked_items(self) -> list[str]:
        with SessionLocal() as session:
            rows = session.execute(
                select(PriceHistoryModel.item_name)
                .distinct()
                .order_by(PriceHistoryModel.item_name)
            ).all()
        return [r[0] for r in rows]

    # ---------------------------------------------------------------------------
    # Сводка для советника
    # ---------------------------------------------------------------------------

    def build_purchase_summary(self, limit_days: int | None = None) -> str:
        """
        Формирует текстовую сводку истории покупок для передачи в LLM-советник.
        Включает детализацию по подкатегориям внутри каждой категории.

        limit_days=None — вся история (по умолчанию).
        limit_days=N    — последние N дней.
        """
        end = date.today()
        if limit_days is not None:
            start = end - timedelta(days=limit_days)
            receipts = get_receipts_by_date_range(start, end)
            period_label = f"последние {limit_days} дней ({start} — {end})"
        else:
            from src.db.crud import get_all_receipts
            receipts = get_all_receipts(limit=10000)
            start = None
            period_label = "всё время"

        if not receipts:
            return "История покупок пуста."

        cat_start = start if limit_days is not None else None
        cat_totals = self.get_category_totals(start=cat_start, end=end)
        subcat_totals = self.get_subcategory_totals(start=cat_start, end=end)

        # Группируем подкатегории по категории для удобного доступа
        subcat_by_cat: dict[str, list[SubcategoryTotal]] = defaultdict(list)
        for sc in subcat_totals:
            subcat_by_cat[sc.category].append(sc)

        lines: list[str] = [
            f"История покупок за {period_label}:",
            f"Всего чеков: {len(receipts)}",
            f"Общая сумма: {sum(r.total for r in receipts):.2f} руб.",
            "",
        ]

        if cat_totals:
            lines.append("Расходы по категориям (с детализацией по подкатегориям):")
            for ct in cat_totals:
                lines.append(f"  {ct.category}: {ct.total:.2f} руб. ({ct.percent}%)")

                # Подкатегории этой категории (уже отсортированы по убыванию суммы)
                subcats = subcat_by_cat.get(ct.category, [])
                # Показываем только если подкатегорий больше одной
                # или если единственная подкатегория не «прочее»
                if len(subcats) > 1 or (len(subcats) == 1 and subcats[0].subcategory != "прочее"):
                    for sc in subcats:
                        lines.append(
                            f"    • {sc.subcategory}: {sc.total:.2f} руб. "
                            f"({sc.percent_of_category}%)"
                        )
            lines.append("")

        lines.append("Последние чеки:")
        for r in receipts[:10]:
            lines.append(
                f"  {r.date} | {r.store_name or '?'} | {r.total:.2f} руб. "
                f"| {len(r.items)} позиций"
            )

        return "\n".join(lines)

    def build_period_report_data(
        self,
        cur_start: date,
        cur_end: date,
        prev_start: date,
        prev_end: date,
        with_subcategories: bool = True,
        significant_change_pct: float = 15.0,
    ) -> str:
        """
        Формирует структурированный текст для отчёта за период.
        Включает категории (и подкатегории если with_subcategories=True)
        с % изменением к предыдущему периоду.
        Передаётся напрямую в LLM для генерации краткой выжимки.
        """
        from src.db.crud import get_receipts_by_date_range

        cur_receipts = get_receipts_by_date_range(cur_start, cur_end)
        prev_receipts = get_receipts_by_date_range(prev_start, prev_end)

        if not cur_receipts:
            return f"Данных за период {cur_start} — {cur_end} нет."

        cur_total = sum(r.total for r in cur_receipts)
        prev_total = sum(r.total for r in prev_receipts)

        # Форматируем метки периодов
        def period_label(s: date, e: date) -> str:
            if s.day == 1 and e.day in (28, 29, 30, 31) and s.month == e.month:
                months_ru = [
                    "", "январь", "февраль", "март", "апрель", "май", "июнь",
                    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
                ]
                return f"{months_ru[s.month]} {s.year} ({s} — {e})"
            if s.month == 1 and s.day == 1 and e.month == 12 and e.day == 31:
                return f"{s.year} год ({s} — {e})"
            if s.year == e.year and s.month == 1 and s.day == 1:
                return f"{s.year} год (с начала года, {s} — {e})"
            return f"{s} — {e}"

        cur_label = period_label(cur_start, cur_end)
        prev_label = period_label(prev_start, prev_end)

        lines: list[str] = [
            f"Период: {cur_label}",
            f"Чеков: {len(cur_receipts)}, итого: {cur_total:.0f} руб.",
        ]

        if prev_receipts:
            total_change = (cur_total - prev_total) / prev_total * 100 if prev_total else 0
            flag = " ⚠️" if abs(total_change) >= significant_change_pct else ""
            lines.append(
                f"Сравнение с {prev_label} "
                f"(итого {prev_total:.0f} руб., "
                f"{total_change:+.0f}%{flag})"
            )
        else:
            lines.append(f"Предыдущий период ({prev_label}): данных нет")

        # Категории текущего и предыдущего периода
        cur_cats = self.get_category_totals(start=cur_start, end=cur_end)
        prev_cats = {ct.category: ct.total for ct in
                     self.get_category_totals(start=prev_start, end=prev_end)}

        lines.append("\nКатегории:")
        for ct in cur_cats:
            prev = prev_cats.get(ct.category, 0)
            if prev > 0:
                change = (ct.total - prev) / prev * 100
                flag = " ⚠️" if abs(change) >= significant_change_pct else ""
                cmp = f" | было {prev:.0f} руб. ({change:+.0f}%{flag})"
            elif prev_receipts:
                cmp = " | впервые в этот период"
            else:
                cmp = ""

            lines.append(f"- {ct.category}: {ct.total:.0f} руб. ({ct.percent}%){cmp}")

            if not with_subcategories:
                continue

            # Подкатегории
            cur_subcats = self.get_subcategory_totals(ct.category, cur_start, cur_end)
            prev_subcats = {sc.subcategory: sc.total for sc in
                            self.get_subcategory_totals(ct.category, prev_start, prev_end)}

            meaningful = [
                sc for sc in cur_subcats
                if sc.subcategory != "прочее" or len(cur_subcats) == 1
            ]
            for sc in meaningful:
                prev_sc = prev_subcats.get(sc.subcategory, 0)
                if prev_sc > 0:
                    sc_change = (sc.total - prev_sc) / prev_sc * 100
                    sc_flag = " ⚠️" if abs(sc_change) >= significant_change_pct else ""
                    sc_cmp = f" ({sc_change:+.0f}%{sc_flag})"
                else:
                    sc_cmp = ""
                lines.append(f"  • {sc.subcategory}: {sc.total:.0f} руб.{sc_cmp}")

        return "\n".join(lines)


# Категории еды для списка покупок (без бытовой химии, гигиены, табака)
FOOD_CATEGORIES = frozenset({
    "продукты питания", "молочные продукты", "мясо и птица",
    "рыба и морепродукты", "фрукты и овощи", "хлеб и выпечка",
    "заморозка", "снеки и сладости", "напитки", "алкоголь",
})


class ShoppingContextBuilder:
    """
    Собирает контекст для умного списка покупок.
    Отдельный класс чтобы не раздувать ReceiptAnalyzer.
    """

    def build(self) -> str:
        """
        Возвращает форматированный текст с:
        - паттерном потребления по пищевым категориям (14 дней)
        - ценами на конкретные товары по магазинам (7 дней)
        """
        today = date.today()
        two_weeks_ago = today - timedelta(days=14)
        one_week_ago = today - timedelta(days=7)

        lines: list[str] = []

        # --- Паттерн потребления (14 дней) ---
        with SessionLocal() as session:
            cat_rows = session.execute(
                select(
                    ReceiptItemModel.category,
                    func.count(
                        func.distinct(ReceiptModel.date)
                    ).label("occasions"),
                    func.sum(ReceiptItemModel.total_price).label("total"),
                    func.max(ReceiptModel.date).label("last_date"),
                )
                .join(ReceiptModel, ReceiptItemModel.receipt_id == ReceiptModel.id)
                .where(
                    ReceiptModel.date >= two_weeks_ago,
                    ReceiptItemModel.category.in_(FOOD_CATEGORIES),
                )
                .group_by(ReceiptItemModel.category)
                .order_by(func.sum(ReceiptItemModel.total_price).desc())
            ).all()

        if cat_rows:
            lines.append("ПАТТЕРН ПОТРЕБЛЕНИЯ (последние 14 дней, только еда):")
            for r in cat_rows:
                days_since = (today - r.last_date).days
                if r.occasions > 1:
                    interval = round(14 / r.occasions)
                    freq = f"примерно раз в {interval} дн."
                else:
                    freq = "1 раз за период"
                lines.append(
                    f"- {r.category}: куплено {r.occasions} раз, "
                    f"{r.total:.0f} руб., {freq}, "
                    f"последний раз {days_since} дн. назад"
                )
        else:
            lines.append(
                "ПАТТЕРН ПОТРЕБЛЕНИЯ: данных за последние 14 дней нет. "
                "Список будет основан только на ценах и профиле."
            )

        lines.append("")

        # --- Цены на товары (7 дней) ---
        # Читаем из receipt_items JOIN receipts, а не из price_history:
        # price_history содержит только товары с включённым отслеживанием,
        # что приводило к пропуску регулярно покупаемых, но неотслеживаемых товаров.
        with SessionLocal() as session:
            price_rows = session.execute(
                select(
                    ReceiptItemModel.name.label("item_name"),
                    ReceiptModel.store_name,
                    ReceiptItemModel.price_per_unit,
                    ReceiptModel.date,
                )
                .join(ReceiptModel, ReceiptItemModel.receipt_id == ReceiptModel.id)
                .where(
                    ReceiptModel.date >= one_week_ago,
                    ReceiptItemModel.category.in_(FOOD_CATEGORIES),
                )
                .order_by(
                    ReceiptItemModel.name,
                    ReceiptModel.date.desc(),
                )
            ).all()

        if price_rows:
            # Для каждого товара берём последнюю цену в каждом магазине
            seen: set[tuple] = set()
            item_prices: dict[str, list[str]] = defaultdict(list)
            for r in price_rows:
                key = (r.item_name, r.store_name or "—")
                if key not in seen:
                    seen.add(key)
                    item_prices[r.item_name].append(
                        f"{r.store_name or '—'}: {r.price_per_unit:.0f} руб."
                    )

            lines.append("ЦЕНЫ НА ТОВАРЫ (последние 7 дней):")
            for item_name, prices in item_prices.items():
                lines.append(f"- {item_name}: {', '.join(prices)}")
        else:
            lines.append("ЦЕНЫ НА ТОВАРЫ: данных за последнюю неделю нет.")

        return "\n".join(lines)