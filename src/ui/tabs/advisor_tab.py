"""
src/ui/tabs/advisor_tab.py

Вкладка «Советник» с двумя подвкладками:
  - «Аналитик бюджета» — чат по истории чеков с RAG
  - «Список покупок»   — профиль покупателя + чат для составления списка
"""
from __future__ import annotations

import logging

import gradio as gr

from src.llm.prompts import SHOPPING_PROFILE_PLACEHOLDER
from src.ui.singletons import (
    get_advisor,
    get_advisor_session,
    get_shopping,
    get_shopping_session,
)

logger = logging.getLogger(__name__)

# Плейсхолдер, который ShoppingAssistant.load_profile() возвращает при отсутствии файла
_EMPTY_PROFILE_SENTINEL = "Профиль не заполнен. Составь список на основе паттерна покупок."


# ---------------------------------------------------------------------------
# Обработчики советника
# ---------------------------------------------------------------------------

def chat_with_advisor(user_message: str, history: list):
    if not user_message.strip():
        yield history, ""
        return

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": "⏳ Думаю..."},
    ]
    yield history, ""

    answer = get_advisor().ask(user_message, get_advisor_session())
    history[-1]["content"] = answer
    yield history, ""


def clear_advisor_chat():
    get_advisor_session().clear()
    return [], ""


# ---------------------------------------------------------------------------
# Обработчики списка покупок
# ---------------------------------------------------------------------------

def save_shopping_profile_fn(profile_text: str):
    if not profile_text.strip():
        return "⚠️ Профиль пустой — не сохранён."
    get_shopping().save_profile(profile_text)
    return "✅ Профиль сохранён."


def chat_with_shopping(user_message: str, history: list):
    if not user_message.strip():
        yield history, ""
        return

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": "⏳ Анализирую покупки и составляю список..."},
    ]
    yield history, ""

    answer = get_shopping().ask(user_message, get_shopping_session())
    history[-1]["content"] = answer
    yield history, ""


def clear_shopping_chat():
    get_shopping_session().clear()
    return [], ""


# ---------------------------------------------------------------------------
# Сборка UI
# ---------------------------------------------------------------------------

def build_advisor_tab() -> None:
    """Строит вкладку «Советник» с двумя подвкладками и подключает события."""
    with gr.Tab("💬 Советник"):
        with gr.Tabs():

            # ----------------------------------------------------------------
            # Подвкладка: Аналитик бюджета
            # ----------------------------------------------------------------
            with gr.Tab("🧠 Аналитик бюджета"):
                gr.Markdown(
                    "Спросите о своих расходах — советник отвечает на основе истории чеков.\n\n"
                    "_Примеры: «Сколько я трачу на еду?», "
                    "«Какой алкоголь я покупаю?», «На чём сэкономить?»_"
                )
                chatbot = gr.Chatbot(height=420, label="")
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ваш вопрос...",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("➤ Отправить", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑 Очистить", variant="secondary", scale=1)

                send_btn.click(
                    fn=chat_with_advisor,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                chat_input.submit(
                    fn=chat_with_advisor,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                clear_btn.click(
                    fn=clear_advisor_chat,
                    outputs=[chatbot, chat_input],
                )

            # ----------------------------------------------------------------
            # Подвкладка: Умный список покупок
            # ----------------------------------------------------------------
            with gr.Tab("🛒 Список покупок"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 👤 Ваш профиль покупателя")

                        # Подставляем сохранённый профиль; плейсхолдер оставляем пустым
                        _saved_profile = get_shopping().load_profile()
                        _profile_value = (
                            "" if _saved_profile == _EMPTY_PROFILE_SENTINEL else _saved_profile
                        )

                        profile_input = gr.Textbox(
                            label="Профиль",
                            placeholder=SHOPPING_PROFILE_PLACEHOLDER,
                            lines=8,
                            value=_profile_value,
                        )
                        profile_save_btn = gr.Button(
                            "💾 Сохранить профиль", variant="secondary"
                        )
                        profile_status = gr.Markdown()

                    with gr.Column(scale=2):
                        gr.Markdown("### 🛒 Список покупок")
                        gr.Markdown(
                            "_Попросите составить список — ИИ учтёт паттерн покупок, "
                            "актуальные цены и ваши предпочтения по магазинам._\n\n"
                            "_Примеры: «Составь список на сегодня», "
                            "«Где дешевле купить молоко?», «Добавь хлеб в список»_"
                        )
                        shopping_chatbot = gr.Chatbot(height=380, label="")
                        with gr.Row():
                            shopping_input = gr.Textbox(
                                placeholder="Составь список покупок...",
                                show_label=False,
                                scale=5,
                            )
                            shopping_send_btn = gr.Button(
                                "➤ Отправить", variant="primary", scale=1
                            )
                            shopping_clear_btn = gr.Button(
                                "🗑 Очистить", variant="secondary", scale=1
                            )

                profile_save_btn.click(
                    fn=save_shopping_profile_fn,
                    inputs=[profile_input],
                    outputs=[profile_status],
                )
                shopping_send_btn.click(
                    fn=chat_with_shopping,
                    inputs=[shopping_input, shopping_chatbot],
                    outputs=[shopping_chatbot, shopping_input],
                )
                shopping_input.submit(
                    fn=chat_with_shopping,
                    inputs=[shopping_input, shopping_chatbot],
                    outputs=[shopping_chatbot, shopping_input],
                )
                shopping_clear_btn.click(
                    fn=clear_shopping_chat,
                    outputs=[shopping_chatbot, shopping_input],
                )
