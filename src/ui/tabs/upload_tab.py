"""
src/ui/tabs/upload_tab.py

Вкладка «Загрузить чек»:
  фото → Vision → таблица товаров + аномалии →
  embedding-совпадения → галочки отслеживания → сохранить.
"""
from __future__ import annotations

import io
import logging

import gradio as gr

from config import SIMILARITY_THRESHOLD
from src.services.receipt_service import save_receipt_with_context
from src.ui.singletons import (
    _saved_hashes,
    get_analyzer,
    get_extractor,
    get_matcher,
    get_retriever,
)
from src.vision.extractor import ExtractionError

logger = logging.getLogger(__name__)

_AUTO_TRACK_KEYWORDS = ["яйц", "молок", "хлеб"]


def _should_auto_track(item_name: str) -> bool:
    """Автоматически отмечаем базовые продукты для отслеживания цен."""
    low = item_name.lower()
    return any(kw in low for kw in _AUTO_TRACK_KEYWORDS)


def _receipt_hash(receipt) -> int:
    """Лёгкий хэш чека для защиты от двойного сохранения."""
    return hash((receipt.store_name, receipt.date, receipt.total, len(receipt.items)))


def process_receipt(image):
    """
    Генератор: фото → Vision → таблица + галочки отслеживания + embedding-совпадения.

    Выходы (9): items_table, summary_out, anomaly_out,
                receipt_state, save_btn, track_checkboxes,
                matches_state, matches_md, store_name_input
    """
    _no_data = (
        gr.update(), "⬆️ Загрузите фото чека", "", None,
        gr.update(interactive=False), gr.update(choices=[], value=[]),
        None, "", gr.update(value=""),
    )

    if image is None:
        yield _no_data
        return

    yield (
        gr.update(), "⏳ Отправляю в Qwen Vision...",
        "", None, gr.update(interactive=False), gr.update(choices=[], value=[]),
        None, "", gr.update(value=""),
    )

    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        receipt = get_extractor().extract_from_bytes(buf.getvalue())
    except ExtractionError as exc:
        yield (
            gr.update(), f"❌ Ошибка распознавания:\n\n{exc}", "", None,
            gr.update(interactive=False), gr.update(choices=[], value=[]),
            None, "", gr.update(value=""),
        )
        return
    except Exception as exc:
        logger.exception("Неожиданная ошибка при обработке чека")
        yield (
            gr.update(), f"❌ Неожиданная ошибка: {exc}", "", None,
            gr.update(interactive=False), gr.update(choices=[], value=[]),
            None, "", gr.update(value=""),
        )
        return

    items_data = [
        [
            item.name, item.category, item.subcategory or "—",
            item.quantity, f"{item.price_per_unit:.2f}", f"{item.total_price:.2f}",
        ]
        for item in receipt.items
    ]

    summary_lines = [
        f"🏪 **Магазин:** {receipt.store_name or '—'}",
        f"📅 **Дата:** {receipt.date}" + (f"  🕐 {receipt.time}" if receipt.time else ""),
        f"💳 **Оплата:** {receipt.payment_method or '—'}",
        f"🧾 **Позиций:** {len(receipt.items)}",
        f"💰 **Итого:** {receipt.total:.2f} руб."
        + (f"  (скидка: {receipt.discount:.2f} руб.)" if receipt.discount else ""),
    ]
    summary_md = "\n\n".join(summary_lines)

    report = get_analyzer().detect_anomalies(receipt)
    anomaly_md = (
        "### ⚠️ Замечены аномалии\n\n" + report.summary()
        if report.has_anomalies
        else "### ✅ Всё в пределах нормы"
    )

    item_names = [item.name for item in receipt.items]
    auto_checked = [n for n in item_names if _should_auto_track(n)]

    # Сначала отдаём распознанный чек, пока ищем совпадения
    yield (
        items_data, summary_md, anomaly_md, receipt,
        gr.update(interactive=False), gr.update(choices=item_names, value=auto_checked),
        None, "⏳ Ищу похожие товары в базе...",
        gr.update(value=receipt.store_name or ""),
    )

    matches = get_matcher().find_matches(item_names, threshold=SIMILARITY_THRESHOLD)

    if matches:
        lines = [
            "### 🔗 Найдены похожие товары\n",
            "_Возможно это одни и те же товары. Отметьте что объединить при сохранении._\n",
        ]
        for m in matches:
            lines.append(
                f"- **«{m.new_name}»** → «{m.canonical_name}» ({m.similarity * 100:.0f}%)"
            )
        matches_md = "\n".join(lines)
    else:
        matches_md = ""

    yield (
        items_data, summary_md, anomaly_md, receipt,
        gr.update(interactive=True), gr.update(choices=item_names, value=auto_checked),
        matches, matches_md,
        gr.update(value=receipt.store_name or ""),
    )


def save_confirmed_receipt(
    receipt_state,
    tracked_items,
    matches_state,
    accepted_matches,
    store_name_override,
):
    """
    UI-обёртка над receipt_service.save_receipt_with_context.
    Форматирует сообщение об успехе/ошибке для отображения в Gradio.
    """
    if receipt_state is None:
        return "❌ Нет данных. Сначала распознайте чек."

    h = _receipt_hash(receipt_state)
    if h in _saved_hashes:
        return "⚠️ Этот чек уже сохранён."

    try:
        result = save_receipt_with_context(
            receipt=receipt_state,
            tracked_items=tracked_items,
            matches_state=matches_state,
            accepted_matches=accepted_matches,
            store_name_override=store_name_override,
            matcher=get_matcher(),
            retriever=get_retriever(),
        )
        _saved_hashes.add(h)

        rename_note = f" | объединено: {result.rename_count} товаров" if result.rename_count else ""
        if result.tracked_n < result.total_n:
            tracking_note = f"отслеживается {result.tracked_n} из {result.total_n} позиций"
        else:
            tracking_note = f"все {result.total_n} позиций отслеживаются"

        return (
            f"✅ Чек сохранён (ID: {result.db_receipt.id}) — "
            f"{result.db_receipt.total:.2f} руб. | {tracking_note}{rename_note}"
        )
    except Exception as exc:
        logger.exception("Ошибка сохранения")
        return f"❌ Ошибка при сохранении: {exc}"


def build_upload_tab() -> None:
    """Строит вкладку «Загрузить чек» и подключает обработчики событий."""
    with gr.Tab("📤 Загрузить чек"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Фото чека", height=420)
                scan_btn = gr.Button("🔍 Распознать чек", variant="primary")

            with gr.Column(scale=2):
                summary_out = gr.Markdown("_Загрузите фото и нажмите «Распознать»_")
                anomaly_out = gr.Markdown()
                store_name_input = gr.Textbox(
                    label="🏪 Магазин",
                    placeholder=(
                        "Заполняется автоматически. "
                        "Исправьте если не распознан или неверный."
                    ),
                    interactive=True,
                )
                items_table = gr.Dataframe(
                    headers=["Товар", "Категория", "Подкатегория", "Кол-во", "Цена/ед.", "Сумма"],
                    datatype=["str", "str", "str", "number", "str", "str"],
                    label="Позиции чека",
                    interactive=False,
                )

                # Секция embedding-совпадений
                matches_md = gr.Markdown()
                accepted_matches = gr.CheckboxGroup(
                    choices=[],
                    label="🔗 Объединить с известными товарами (снимите галочку если не согласны)",
                )

                track_checkboxes = gr.CheckboxGroup(
                    choices=[],
                    label="🔍 Отслеживать цены (яйца, молоко, хлеб — выбраны автоматически)",
                )
                save_btn = gr.Button(
                    "💾 Сохранить в базу",
                    variant="secondary",
                    interactive=False,
                )
                save_status = gr.Markdown()

        receipt_state = gr.State(None)
        matches_state = gr.State(None)

        scan_btn.click(
            fn=process_receipt,
            inputs=[image_input],
            outputs=[
                items_table, summary_out, anomaly_out,
                receipt_state, save_btn, track_checkboxes,
                matches_state, matches_md, store_name_input,
            ],
        )

        def update_accepted_matches(matches):
            if not matches:
                return gr.update(choices=[], value=[], visible=False)
            choices = [m.new_name for m in matches]
            return gr.update(choices=choices, value=choices, visible=True)

        matches_state.change(
            fn=update_accepted_matches,
            inputs=[matches_state],
            outputs=[accepted_matches],
        )

        save_btn.click(
            fn=save_confirmed_receipt,
            inputs=[
                receipt_state, track_checkboxes,
                matches_state, accepted_matches, store_name_input,
            ],
            outputs=[save_status],
        )
