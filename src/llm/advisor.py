"""
src/llm/advisor.py

ИИ-советник по бюджету.
Получает вопрос пользователя + контекст из базы → отвечает на основе реальных данных.

Контекст формируется двумя уровнями:
  1. build_brief_summary()  — краткая общая статистика (суммы по категориям, период)
  2. RagRetriever.retrieve() — топ-N релевантных чанков (конкретные чеки и товары)

Если retriever не передан — fallback на полную build_purchase_summary() (старое поведение).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.analytics.analyzer import ReceiptAnalyzer
from src.llm.client import OllamaClient, OllamaError
from src.llm.prompts import ADVISOR_SYSTEM, ADVISOR_USER_TEMPLATE

if TYPE_CHECKING:
    from src.analytics.retriever import RagRetriever

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str    # "user" | "assistant"
    content: str


@dataclass
class AdvisorSession:
    """Хранит историю диалога в рамках одной сессии Gradio."""
    messages: list[ChatMessage] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append(ChatMessage(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=text))

    def to_ollama_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        self.messages.clear()


class BudgetAdvisor:
    """
    Советник по бюджету.

    При наличии retriever каждый вопрос обрабатывается через RAG:
      - Краткая общая статистика из build_brief_summary() (ориентация по суммам)
      - Топ-N релевантных чанков из retrieve() (конкретные чеки, история цен)
    Это позволяет советнику точно отвечать на специфические вопросы
    без передачи всей истории покупок в контекст.
    """

    def __init__(
            self,
            client: OllamaClient | None = None,
            analyzer: ReceiptAnalyzer | None = None,
            retriever: "RagRetriever | None" = None,
            history_days: int | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.analyzer = analyzer or ReceiptAnalyzer()
        self.retriever = retriever
        self.history_days = history_days

    def ask(self, question: str, session: AdvisorSession) -> str:
        """
        Задаёт вопрос советнику.
        Обновляет session на месте (добавляет user + assistant сообщения).
        Возвращает текст ответа.
        """
        purchase_context = self._build_context(question)

        user_content = ADVISOR_USER_TEMPLATE.format(
            purchase_history=purchase_context,
            question=question,
        )

        messages: list[dict] = [{"role": "system", "content": ADVISOR_SYSTEM}]
        recent = session.messages[-8:]
        for msg in recent:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_content})

        logger.info("Советник: вопрос — %s...", question[:60])

        try:
            raw = self.client.chat(messages)
        except OllamaError as exc:
            logger.error("Ошибка советника: %s", exc)
            return f"Ошибка при обращении к модели: {exc}"

        answer = self._clean_response(raw)

        session.add_user(question)
        session.add_assistant(answer)

        logger.info("Советник: ответ — %s...", answer[:80])
        return answer

    def _build_context(self, question: str) -> str:
        """
        Формирует контекст для вопроса.

        Всегда включает build_purchase_summary() — SQL-агрегаты по категориям
        и подкатегориям, нужны для точных ответов на вопросы типа
        «сколько потратил на пиво за всё время».

        Если retriever доступен — дополнительно добавляет RAG-чанки
        с конкретными чеками и историей цен по теме вопроса.
        """
        summary = self.analyzer.build_purchase_summary(self.history_days)

        if self.retriever is None:
            return summary

        chunks = self.retriever.retrieve(question)
        if not chunks:
            return summary

        rag_context = "\n\n---\n".join(chunks)
        logger.info("RAG: передаём %d чанков в контекст", len(chunks))
        return (
            f"СВОДКА ПОКУПОК:\n{summary}\n\n"
            f"ДЕТАЛИ ПО ТЕМЕ ВОПРОСА:\n{rag_context}"
        )

    @staticmethod
    def _clean_response(text: str) -> str:
        """Убирает <think>...</think> блоки из ответа Qwen."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()