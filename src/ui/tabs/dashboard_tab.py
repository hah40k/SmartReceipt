"""
src/ui/tabs/dashboard_tab.py

Вкладка «Дашборд»:
  Plotly-графики расходов, трекинг цен, включение отслеживания,
  слияние дублирующихся товаров, периодические отчёты.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import gradio as gr
import plotly.graph_objects as go

from src.db.crud import (
    clear_all_data,
    enable_price_tracking_for_item,
    get_all_item_names,
    get_all_receipts,
    get_untracked_items,
)
from src.services.receipt_service import merge_items
from src.ui.singletons import (
    _saved_hashes,
    get_analyzer,
    get_matcher,
    get_reporter,
    get_retriever,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _empty_figure(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=text, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color="#888"),
    )
    fig.update_layout(
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font_color="#e0e0e0", height=280,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Обработчики событий
# ---------------------------------------------------------------------------

def build_dashboard(period: str):
    today = date.today()
    period_map = {"Неделя": 7, "Месяц": 30, "3 месяца": 90}
    days = period_map.get(period)
    start = (today - timedelta(days=days)) if days else None

    analyzer = get_analyzer()

    cat_totals = analyzer.get_category_totals(start=start, end=today)
    if cat_totals:
        fig_cat = go.Figure(go.Bar(
            x=[ct.total for ct in cat_totals],
            y=[ct.category for ct in cat_totals],
            orientation="h",
            marker_color="#4F8EF7",
            text=[f"{ct.total:.0f} ₽ ({ct.percent}%)" for ct in cat_totals],
            textposition="outside",
        ))
        fig_cat.update_layout(
            title="Расходы по категориям",
            xaxis_title="Сумма, руб.",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=180, r=100, t=50, b=40),
            height=max(280, len(cat_totals) * 36 + 100),
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font_color="#e0e0e0",
        )
    else:
        fig_cat = _empty_figure("Нет данных — сохраните хотя бы один чек")

    daily = analyzer.get_daily_totals(start=start, end=today)
    if daily:
        fig_daily = go.Figure(go.Scatter(
            x=[d["date"] for d in daily],
            y=[d["total"] for d in daily],
            mode="lines+markers",
            line=dict(color="#4F8EF7", width=2),
            marker=dict(size=6, color="#F7A44F"),
            fill="tozeroy",
            fillcolor="rgba(79,142,247,0.15)",
        ))
        fig_daily.update_layout(
            title="По дням",
            xaxis_title="Дата", yaxis_title="Руб.",
            margin=dict(l=60, r=40, t=50, b=40), height=280,
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font_color="#e0e0e0",
        )
    else:
        fig_daily = _empty_figure("Нет данных по дням")

    monthly = analyzer.get_monthly_totals()
    if monthly:
        fig_monthly = go.Figure(go.Bar(
            x=[m["month"] for m in monthly],
            y=[m["total"] for m in monthly],
            marker_color="#F7A44F",
            text=[f"{m['total']:.0f} ₽" for m in monthly],
            textposition="outside",
        ))
        fig_monthly.update_layout(
            title="По месяцам",
            xaxis_title="Месяц", yaxis_title="Руб.",
            margin=dict(l=60, r=40, t=50, b=40), height=280,
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font_color="#e0e0e0",
        )
    else:
        fig_monthly = _empty_figure("Нет данных по месяцам")

    receipts = get_all_receipts(limit=1000)
    if receipts:
        total_sum = sum(r.total for r in receipts)
        stat_md = (
            f"**Чеков:** {len(receipts)}  |  "
            f"**Всего потрачено:** {total_sum:,.0f} ₽  |  "
            f"**Средний чек:** {total_sum / len(receipts):,.0f} ₽"
        )
    else:
        stat_md = "_Чеков пока нет. Загрузите и сохраните первый чек._"

    return fig_cat, fig_daily, fig_monthly, stat_md


def build_price_chart(item_name: str):
    if not item_name or not item_name.strip():
        return _empty_figure("Выберите товар из списка")
    points = get_analyzer().get_price_history(item_name.strip())
    if not points:
        return _empty_figure(f"Нет истории цен для «{item_name}»")
    fig = go.Figure(go.Scatter(
        x=[p.date for p in points],
        y=[p.price for p in points],
        mode="lines+markers",
        line=dict(color="#A44FF7", width=2),
        marker=dict(size=8, color="#F74F4F"),
        text=[p.store_name or "?" for p in points],
        hovertemplate="%{x}<br>%{y:.2f} ₽<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Цены: {item_name}",
        xaxis_title="Дата", yaxis_title="Цена, руб.",
        margin=dict(l=60, r=40, t=50, b=40), height=280,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e", font_color="#e0e0e0",
    )
    return fig


def get_tracked_items_list() -> str:
    items = get_analyzer().get_tracked_items()
    if not items:
        return "_Нет отслеживаемых товаров (нужно сохранить хотя бы один чек)_"
    return "**Отслеживаемые товары:**\n" + "\n".join(f"• {i}" for i in items[:30])


def refresh_all(period: str):
    f1, f2, f3, stat = build_dashboard(period)
    tracked_items = get_analyzer().get_tracked_items()
    tracked_md = get_tracked_items_list()
    untracked = get_untracked_items()
    all_names = get_all_item_names()
    return (
        f1, f2, f3, stat, tracked_md,
        gr.update(choices=untracked, value=[]),
        gr.update(choices=tracked_items),
        gr.update(choices=all_names, value=None),
        gr.update(choices=all_names, value=None),
    )


def enable_tracking_fn(selected_items: list[str]):
    if not selected_items:
        return "⚠️ Выберите хотя бы один товар", gr.update(), gr.update()
    msgs = []
    for name in selected_items:
        n = enable_price_tracking_for_item(name)
        msgs.append(f"«{name}» ({n} зап.)")
    untracked = get_untracked_items()
    tracked_md = get_tracked_items_list()
    return (
        "✅ Включено отслеживание: " + ", ".join(msgs),
        gr.update(choices=untracked, value=[]),
        tracked_md,
    )


def merge_items_fn(source: str, target: str):
    """
    UI-обёртка над receipt_service.merge_items.
    Возвращает статус + обновлённые списки обоих dropdown.
    """
    if not source or not target:
        return "⚠️ Выберите оба товара", gr.update(), gr.update()
    if source == target:
        return "⚠️ Исходное и целевое имена совпадают", gr.update(), gr.update()

    try:
        count = merge_items(source, target, matcher=get_matcher(), retriever=get_retriever())
        all_names = get_all_item_names()
        return (
            f"✅ «{source}» → «{target}»: перенесено {count} позиций",
            gr.update(choices=all_names, value=None),
            gr.update(choices=all_names, value=None),
        )
    except Exception as exc:
        logger.exception("Ошибка при объединении товаров")
        return f"❌ Ошибка: {exc}", gr.update(), gr.update()


def clear_database_fn():
    _saved_hashes.clear()
    res = clear_all_data()
    return (
        f"🗑 База очищена — удалено: {res['receipts']} чеков, "
        f"{res['items']} позиций, {res['price_history']} записей цен"
    )


def reindex_all_fn():
    """
    Пересчитывает все embeddings и переиндексирует RAG.

    Нужно запускать при:
    - смене модели OLLAMA_EMBED_MODEL в config.py
    - изменении формата чанков в _build_receipt_text()
    - первом добавлении RAG для старых чеков

    Пересчитывает три вещи:
    1. item_embeddings — используются ItemMatcher при дедупликации товаров
    2. RAG-чанки чеков — используются поиском и советником
    3. RAG-чанки товаров — используются поиском и советником
    """
    all_names = get_all_item_names()
    receipts = get_all_receipts(limit=10000)

    if not receipts and not all_names:
        yield "⚠️ База пуста — нечего переиндексировать."
        return

    # Шаг 1: item_embeddings (для ItemMatcher)
    if all_names:
        yield f"⏳ Пересчитываю embeddings товаров ({len(all_names)} шт.)..."
        get_matcher().add_new_items(all_names)  # upsert — перезапишет старые векторы

    # Шаг 2: RAG-чанки чеков
    if receipts:
        yield f"⏳ Переиндексирую чеки ({len(receipts)} шт.)..."
        retriever = get_retriever()
        for receipt in receipts:
            retriever.index_receipt(receipt)

    # Шаг 3: RAG-чанки товаров
    if all_names:
        yield f"⏳ Переиндексирую товары ({len(all_names)} шт.)..."
        get_retriever().index_items(all_names)

    parts = []
    if receipts:
        parts.append(f"{len(receipts)} чеков")
    if all_names:
        parts.append(f"{len(all_names)} товаров")
    yield "✅ Переиндексировано: " + ", ".join(parts)


def generate_monthly_report_fn(report_type: str):
    yield "⏳ Собираю данные и генерирую отчёт..."
    if report_type == "Последние 30 дней":
        result = get_reporter().monthly_rolling_report()
    else:
        result = get_reporter().monthly_calendar_report()
    yield result


def generate_yearly_report_fn():
    yield "⏳ Собираю данные и генерирую отчёт..."
    yield get_reporter().yearly_report()


# ---------------------------------------------------------------------------
# Сборка UI
# ---------------------------------------------------------------------------

def build_dashboard_tab() -> None:
    """Строит вкладку «Дашборд» и подключает обработчики событий."""
    with gr.Tab("📊 Дашборд"):
        with gr.Row():
            period_radio = gr.Radio(
                choices=["Неделя", "Месяц", "3 месяца", "Всё время"],
                value="Месяц",
                label="Период",
            )
            refresh_btn = gr.Button("🔄 Обновить", variant="secondary")
            reindex_btn = gr.Button("🔁 Переиндексировать", variant="secondary", size="sm")
            clear_db_btn = gr.Button("🗑 Очистить БД", variant="stop", size="sm")

        clear_db_status = gr.Markdown()
        reindex_status = gr.Markdown()
        stats_md = gr.Markdown("_Нажмите «Обновить» для загрузки данных_")
        fig_cat_out = gr.Plot(label="По категориям")

        with gr.Row():
            fig_daily_out = gr.Plot(label="По дням")
            fig_monthly_out = gr.Plot(label="По месяцам")

        gr.Markdown("---\n### 📈 Трекинг цен")
        with gr.Row():
            tracked_items_md = gr.Markdown()
            with gr.Column():
                price_item_input = gr.Dropdown(
                    choices=[],
                    label="Товар (нажмите «Обновить» чтобы загрузить список)",
                )
                price_btn = gr.Button("Показать динамику цен")
        price_chart_out = gr.Plot()

        gr.Markdown("---\n### ➕ Включить отслеживание для сохранённых товаров")
        gr.Markdown(
            "_Товары, которые попали в базу без отслеживания. "
            "Выберите нужные — история цен будет восстановлена из уже сохранённых чеков._"
        )
        with gr.Row():
            untracked_dropdown = gr.Dropdown(
                choices=[],
                multiselect=True,
                label="Неотслеживаемые товары (нажмите «Обновить» чтобы загрузить список)",
                scale=4,
            )
            enable_tracking_btn = gr.Button(
                "✅ Включить отслеживание",
                variant="primary",
                scale=1,
            )
        enable_status = gr.Markdown()

        gr.Markdown("---\n### ✏️ Объединить товары")
        gr.Markdown(
            "_Исправьте опечатку или объедините дубликаты. "
            "Все записи «Что» будут перенесены под имя «Во что», "
            "история цен сохранится. Нажмите «Обновить» чтобы загрузить список._"
        )
        with gr.Row():
            merge_source = gr.Dropdown(
                choices=[],
                label="Что (исходное имя — будет удалено)",
                scale=2,
            )
            merge_target = gr.Dropdown(
                choices=[],
                label="Во что (каноническое имя — останется)",
                scale=2,
            )
            merge_btn = gr.Button("🔀 Объединить", variant="primary", scale=1)
        merge_status = gr.Markdown()

        gr.Markdown("---\n### 📋 Отчёты")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### За месяц")
                monthly_type_radio = gr.Radio(
                    choices=["Последние 30 дней", "Прошлый календарный месяц"],
                    value="Последние 30 дней",
                    label="Период",
                )
                monthly_report_btn = gr.Button(
                    "📝 Сформировать месячный отчёт", variant="primary"
                )
                monthly_report_out = gr.Markdown(label="Отчёт за месяц", show_label=True)
            with gr.Column():
                gr.Markdown("#### За год")
                gr.Markdown(
                    "_Текущий календарный год (с 1 января по сегодня) "
                    "в сравнении с прошлым полным годом._"
                )
                yearly_report_btn = gr.Button(
                    "📝 Сформировать годовой отчёт", variant="primary"
                )
                yearly_report_out = gr.Markdown(label="Отчёт за год", show_label=True)

        # Подключаем события
        refresh_btn.click(
            fn=refresh_all,
            inputs=[period_radio],
            outputs=[
                fig_cat_out, fig_daily_out, fig_monthly_out,
                stats_md, tracked_items_md,
                untracked_dropdown, price_item_input,
                merge_source, merge_target,
            ],
        )
        price_btn.click(
            fn=build_price_chart,
            inputs=[price_item_input],
            outputs=[price_chart_out],
        )
        enable_tracking_btn.click(
            fn=enable_tracking_fn,
            inputs=[untracked_dropdown],
            outputs=[enable_status, untracked_dropdown, tracked_items_md],
        )
        merge_btn.click(
            fn=merge_items_fn,
            inputs=[merge_source, merge_target],
            outputs=[merge_status, merge_source, merge_target],
        )
        monthly_report_btn.click(
            fn=generate_monthly_report_fn,
            inputs=[monthly_type_radio],
            outputs=[monthly_report_out],
        )
        yearly_report_btn.click(
            fn=generate_yearly_report_fn,
            inputs=[],
            outputs=[yearly_report_out],
        )
        clear_db_btn.click(
            fn=clear_database_fn,
            outputs=[clear_db_status],
        )
        reindex_btn.click(
            fn=reindex_all_fn,
            outputs=[reindex_status],
        )