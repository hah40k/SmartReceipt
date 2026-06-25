"""
src/ui/tabs/search_tab.py

Вкладка «Поиск» — семантический поиск по истории чеков и товаров.

Пользователь вводит запрос в свободной форме:
  «когда я последний раз покупал что-то к завтраку»
  «чеки с большими тратами на алкоголь»
  «где я покупал молоко и почём»

Ответ — список релевантных чанков (чеки и/или товары), отсортированных
по косинусному сходству с запросом. Никакого LLM — только embeddings + поиск.
"""
from __future__ import annotations

import logging

import gradio as gr

from src.analytics.retriever import SearchResult
from src.ui.singletons import get_retriever

logger = logging.getLogger(__name__)

# Минимальный скор для отображения (синхронизирован с SEARCH_MIN_SIMILARITY в retriever.py)
_MIN_SCORE_DISPLAY = 0.30

_EXAMPLES = [
    "когда я последний раз покупал молоко",
    "чеки с алкоголем",
    "где дешевле всего покупали хлеб",
    "крупные покупки в Пятёрочке",
    "что покупал на прошлой неделе",
    "товары категории мясо",
]


# ---------------------------------------------------------------------------
# Форматирование результатов
# ---------------------------------------------------------------------------

def _score_bar(similarity: float) -> str:
    """Визуальная полоска релевантности из 8 символов."""
    filled = round(similarity * 8)
    return "█" * filled + "░" * (8 - filled)


def _format_results(results: list[SearchResult], query: str) -> str:
    if not results:
        return (
            f"_По запросу **«{query}»** ничего не найдено._\n\n"
            "Попробуйте другую формулировку или убедитесь что чеки сохранены и проиндексированы."
        )

    receipt_count = sum(1 for r in results if r.chunk_type == "receipt")
    item_count = sum(1 for r in results if r.chunk_type == "item")

    counts = []
    if receipt_count:
        counts.append(f"чеков: {receipt_count}")
    if item_count:
        counts.append(f"товаров: {item_count}")

    header = f"### Найдено по запросу «{query}» — {', '.join(counts)}\n"
    parts = [header]

    for i, r in enumerate(results, 1):
        icon = "🧾" if r.chunk_type == "receipt" else "🏷️"
        bar = _score_bar(r.similarity)
        pct = int(r.similarity * 100)

        parts.append(
            f"**{i}. {icon} {bar} {pct}%**\n\n"
            f"{r.chunk_text}"
        )

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Обработчик поиска
# ---------------------------------------------------------------------------

def run_search(query: str, filter_type: str, top_k: int) -> str:
    """Выполняет семантический поиск и возвращает отформатированный Markdown."""
    query = query.strip()
    if not query:
        return "_Введите запрос._"

    chunk_type_map = {
        "Чеки и товары": None,
        "Только чеки": "receipt",
        "Только товары": "item",
    }
    chunk_type = chunk_type_map.get(filter_type)

    results = get_retriever().search(query, top_k=int(top_k), chunk_type=chunk_type)
    return _format_results(results, query)


# ---------------------------------------------------------------------------
# Сборка UI
# ---------------------------------------------------------------------------

def build_search_tab() -> None:
    """Строит вкладку «Поиск» и подключает обработчики событий."""
    with gr.Tab("🔍 Поиск"):
        gr.Markdown(
            "Ищите по истории чеков в свободной форме — без точных дат и названий.\n\n"
            "_Примеры: «когда покупал молоко», «чеки с алкоголем», "
            "«где дешевле хлеб», «крупные покупки на прошлой неделе»_"
        )

        with gr.Row():
            query_input = gr.Textbox(
                placeholder="Введите запрос...",
                show_label=False,
                scale=5,
            )
            search_btn = gr.Button("🔍 Найти", variant="primary", scale=1)

        with gr.Row():
            filter_radio = gr.Radio(
                choices=["Чеки и товары", "Только чеки", "Только товары"],
                value="Чеки и товары",
                label="Что искать",
                scale=3,
            )
            top_k_slider = gr.Slider(
                minimum=3,
                maximum=20,
                value=8,
                step=1,
                label="Максимум результатов",
                scale=2,
            )

        gr.Examples(
            examples=_EXAMPLES,
            inputs=[query_input],
            label="Примеры запросов",
        )

        results_out = gr.Markdown(
            "_Введите запрос и нажмите «Найти»._",
            label="Результаты",
        )

        search_btn.click(
            fn=run_search,
            inputs=[query_input, filter_radio, top_k_slider],
            outputs=[results_out],
        )
        query_input.submit(
            fn=run_search,
            inputs=[query_input, filter_radio, top_k_slider],
            outputs=[results_out],
        )
