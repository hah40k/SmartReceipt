"""
src/llm/reporter.py

Генератор периодических отчётов о расходах.

Два типа отчётов:
  monthly_rolling   — последние 30 дней vs предыдущие 30 дней
  monthly_calendar  — прошлый завершённый календарный месяц vs месяц до него
  yearly            — текущий календарный год vs предыдущий полный год

Каждый отчёт = SQL-агрегаты из ReceiptAnalyzer → форматированный текст →
LLM генерирует краткую выжимку (3-5 предложений).
"""
from __future__ import annotations

import calendar
import logging
import re
from calendar import monthrange
from datetime import date, timedelta

from src.analytics.analyzer import ReceiptAnalyzer
from src.llm.client import OllamaClient, OllamaError
from src.llm.prompts import (
    MONTHLY_REPORT_USER,
    REPORT_SYSTEM,
    YEARLY_REPORT_USER,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Генерирует краткие текстовые отчёты о расходах через LLM.
    Данные берёт из ReceiptAnalyzer (SQL), текст генерирует OllamaClient.
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        analyzer: ReceiptAnalyzer | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.analyzer = analyzer or ReceiptAnalyzer()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def monthly_rolling_report(self) -> str:
        """
        Отчёт за последние 30 дней vs предыдущие 30 дней (31-60 дней назад).
        Включает подкатегории.
        """
        today = date.today()
        cur_end = today
        cur_start = today - timedelta(days=29)
        prev_end = cur_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=29)

        return self._generate(
            cur_start, cur_end, prev_start, prev_end,
            with_subcategories=True,
            prompt_template=MONTHLY_REPORT_USER,
            report_kind="месяц (30 дней)",
        )

    def monthly_calendar_report(self) -> str:
        """
        Отчёт за прошлый завершённый календарный месяц vs месяц до него.
        Если сейчас первый день месяца — берём позапрошлый как «прошлый завершённый».
        Включает подкатегории.
        """
        today = date.today()

        # Прошлый завершённый месяц
        first_of_current = today.replace(day=1)
        cur_end = first_of_current - timedelta(days=1)     # последний день прошлого месяца
        cur_start = cur_end.replace(day=1)                 # первый день прошлого месяца

        # Месяц до него
        prev_end = cur_start - timedelta(days=1)
        prev_start = prev_end.replace(day=1)

        return self._generate(
            cur_start, cur_end, prev_start, prev_end,
            with_subcategories=True,
            prompt_template=MONTHLY_REPORT_USER,
            report_kind="календарный месяц",
        )

    def yearly_report(self) -> str:
        """
        Отчёт за текущий календарный год (с 1 января по сегодня)
        vs прошлый полный год. Без подкатегорий — только основные категории.
        """
        today = date.today()
        cur_start = date(today.year, 1, 1)
        cur_end = today

        prev_start = date(today.year - 1, 1, 1)
        prev_end = date(today.year - 1, 12, 31)

        return self._generate(
            cur_start, cur_end, prev_start, prev_end,
            with_subcategories=False,
            prompt_template=YEARLY_REPORT_USER,
            report_kind="год",
        )

    # ------------------------------------------------------------------
    # Внутренняя логика
    # ------------------------------------------------------------------

    def _generate(
        self,
        cur_start: date,
        cur_end: date,
        prev_start: date,
        prev_end: date,
        with_subcategories: bool,
        prompt_template: str,
        report_kind: str,
    ) -> str:
        period_data = self.analyzer.build_period_report_data(
            cur_start, cur_end,
            prev_start, prev_end,
            with_subcategories=with_subcategories,
        )

        if period_data.startswith("Данных за период"):
            return f"⚠️ {period_data}"

        prompt = prompt_template.format(period_data=period_data)

        logger.info("ReportGenerator: запрос отчёта «%s»", report_kind)
        try:
            raw = self.client.chat_text(
                user_prompt=prompt,
                system_prompt=REPORT_SYSTEM,
            )
        except OllamaError as exc:
            logger.error("Ошибка при генерации отчёта: %s", exc)
            return f"❌ Ошибка при обращении к модели: {exc}"

        result = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        logger.info("ReportGenerator: отчёт «%s» сгенерирован (%d симв.)", report_kind, len(result))
        return result
