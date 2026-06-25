"""
src/llm/shopping.py

Умный помощник по составлению списка покупок.

Логика:
  - Профиль пользователя (частота, приоритеты магазинов) читается из файла
  - Контекст из БД: паттерн потребления (14 дней) + цены (7 дней)
  - LLM составляет список с учётом удобства магазинов и известных цен
  - Чат позволяет уточнять список после первого ответа
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.analytics.analyzer import ShoppingContextBuilder
from src.llm.client import OllamaClient, OllamaError
from src.llm.prompts import SHOPPING_SYSTEM, SHOPPING_USER_TEMPLATE

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE = "Профиль не заполнен. Составь список на основе паттерна покупок."


@dataclass
class ShoppingMessage:
    role: str    # "user" | "assistant"
    content: str


@dataclass
class ShoppingSession:
    """История диалога в рамках одной сессии списка покупок."""
    messages: list[ShoppingMessage] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append(ShoppingMessage(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(ShoppingMessage(role="assistant", content=text))

    def clear(self) -> None:
        self.messages.clear()


class ShoppingAssistant:
    """
    Помощник по составлению списка продуктов.

    Каждый вопрос получает:
    - профиль пользователя (из файла)
    - свежий контекст из БД (паттерн + цены)
    - историю текущего диалога (последние 6 сообщений)
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        profile_path: Path | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.context_builder = ShoppingContextBuilder()

        from config import SHOPPING_PROFILE_PATH
        self.profile_path = profile_path or SHOPPING_PROFILE_PATH

    # ------------------------------------------------------------------
    # Профиль
    # ------------------------------------------------------------------

    def load_profile(self) -> str:
        try:
            text = self.profile_path.read_text(encoding="utf-8").strip()
            return text if text else _DEFAULT_PROFILE
        except FileNotFoundError:
            return _DEFAULT_PROFILE

    def save_profile(self, text: str) -> None:
        self.profile_path.write_text(text.strip(), encoding="utf-8")
        logger.info("Профиль покупателя сохранён: %d символов", len(text))

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def ask(self, question: str, session: ShoppingSession) -> str:
        """
        Отвечает на вопрос с учётом профиля и данных из БД.
        Обновляет session на месте.
        """
        profile = self.load_profile()
        shopping_context = self.context_builder.build()

        user_content = SHOPPING_USER_TEMPLATE.format(
            profile=profile,
            shopping_context=shopping_context,
            question=question,
        )

        messages: list[dict] = [{"role": "system", "content": SHOPPING_SYSTEM}]

        # История диалога — последние 6 сообщений (3 пары)
        # Не включаем контекст из прошлых сообщений — он обновляется каждый раз
        for msg in session.messages[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_content})

        logger.info("ShoppingAssistant: запрос — %s...", question[:60])

        try:
            raw = self.client.chat(messages)
        except OllamaError as exc:
            logger.error("Ошибка ShoppingAssistant: %s", exc)
            return f"❌ Ошибка при обращении к модели: {exc}"

        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        session.add_user(question)
        session.add_assistant(answer)

        logger.info("ShoppingAssistant: ответ — %s...", answer[:80])
        return answer
