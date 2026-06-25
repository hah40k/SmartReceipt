"""
tests/test_advisor.py
Тесты советника с mock-клиентом — без реального Ollama.
Запуск: python -m pytest tests/test_advisor.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

from src.llm.advisor import BudgetAdvisor, AdvisorSession


def make_advisor(mock_response: str) -> BudgetAdvisor:
    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response

    mock_analyzer = MagicMock()
    mock_analyzer.build_purchase_summary.return_value = (
        "История покупок за последние 90 дней:\n"
        "Всего чеков: 5\n"
        "Общая сумма: 3500.00 руб.\n"
        "Расходы по категориям:\n"
        "  продукты питания: 2000.00 руб. (57.1%)\n"
        "  алкоголь: 1500.00 руб. (42.9%)"
    )

    return BudgetAdvisor(client=mock_client, analyzer=mock_analyzer)


class TestAdvisorSession:

    def test_session_stores_messages(self):
        session = AdvisorSession()
        session.add_user("Привет")
        session.add_assistant("Привет! Чем могу помочь?")
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"

    def test_session_clear(self):
        session = AdvisorSession()
        session.add_user("Вопрос")
        session.clear()
        assert len(session.messages) == 0

    def test_to_ollama_messages_format(self):
        session = AdvisorSession()
        session.add_user("Сколько я трачу?")
        msgs = session.to_ollama_messages()
        assert msgs[0] == {"role": "user", "content": "Сколько я трачу?"}


class TestBudgetAdvisor:

    def test_ask_returns_clean_answer(self):
        advisor = make_advisor("В среднем вы тратите 700 руб. в день.")
        session = AdvisorSession()
        answer = advisor.ask("Сколько я трачу в день?", session)
        assert "700" in answer

    def test_thinking_blocks_stripped(self):
        raw = "<think>Считаю расходы...</think>\nВы тратите 700 руб. в день."
        advisor = make_advisor(raw)
        session = AdvisorSession()
        answer = advisor.ask("Сколько?", session)
        assert "<think>" not in answer
        assert "700" in answer

    def test_session_updated_after_ask(self):
        advisor = make_advisor("Ответ советника.")
        session = AdvisorSession()
        advisor.ask("Мой вопрос", session)
        assert len(session.messages) == 2
        assert session.messages[0].content == "Мой вопрос"
        assert session.messages[1].content == "Ответ советника."

    def test_history_limited_to_recent(self):
        """Советник не уходит в бесконечный контекст — берёт последние 8 сообщений."""
        advisor = make_advisor("Ок.")
        session = AdvisorSession()

        # Имитируем длинную историю
        for i in range(10):
            session.add_user(f"Вопрос {i}")
            session.add_assistant(f"Ответ {i}")

        advisor.ask("Новый вопрос", session)

        # Проверяем что chat вызван с разумным числом сообщений
        call_args = advisor.client.chat.call_args[0][0]
        # system + ≤8 исторических + 1 новый = не более 10
        assert len(call_args) <= 10

    def test_ollama_error_returns_message(self):
        from src.llm.client import OllamaError
        mock_client = MagicMock()
        mock_client.chat.side_effect = OllamaError("Нет соединения")
        mock_analyzer = MagicMock()
        mock_analyzer.build_purchase_summary.return_value = "Пусто"

        advisor = BudgetAdvisor(client=mock_client, analyzer=mock_analyzer)
        session = AdvisorSession()
        answer = advisor.ask("Вопрос", session)
        assert "Ошибка" in answer
