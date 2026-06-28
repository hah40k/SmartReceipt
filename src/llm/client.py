import json
import logging
from typing import Any

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Ошибка при обращении к Ollama."""


class OllamaClient:
    """
    Лёгкий HTTP-клиент для Ollama /api/chat.
    Поддерживает текстовые и vision-запросы (base64-изображения).
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._chat_url = f"{self.base_url}/api/chat"

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def chat(
            self,
            messages: list[dict[str, Any]],
            disable_thinking: bool = False,
            prefill: str | None = None,
            model: str | None = None,  # <- новый параметр последним
    ) -> str:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if disable_thinking:
            options["think"] = False

        send_messages = messages
        if prefill:
            send_messages = messages + [{"role": "assistant", "content": prefill}]

        payload = {
            "model": model or self.model,  # <- единственное изменение здесь
            "messages": send_messages,
            "stream": False,
            "options": options,
        }

        logger.debug("OllamaClient.chat → %d msg(s), model=%s, prefill=%r",
                     len(messages), self.model, prefill)

        try:
            response = requests.post(
                self._chat_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                f"Не удалось подключиться к Ollama по адресу {self.base_url}. "
                "Убедитесь, что сервис запущен."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError(
                f"Ollama не ответил за {self.timeout} секунд."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise OllamaError(
                f"Ollama вернул ошибку {response.status_code}: {response.text}"
            ) from exc

        data = response.json()
        msg = data["message"]
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")

        logger.debug(
            "Ollama ответ: content=%d симв., thinking=%d симв.",
            len(content), len(thinking),
        )

        # Собираем финальный текст.
        # Новые версии Ollama возвращают размышления в message.thinking,
        # а message.content содержит только финальный ответ.
        # Оборачиваем thinking в теги — парсер extractor.py умеет их вырезать.
        if thinking and not content.strip():
            result = f"<think>{thinking}</think>"
        elif thinking:
            result = f"<think>{thinking}</think>{content}"
        else:
            result = content

        # Если использовался prefill — prepend его к ответу,
        # чтобы парсер получил полный валидный JSON/массив
        if prefill and not result.startswith(prefill):
            result = prefill + result

        return result

    def chat_text(
            self,
            user_prompt: str,
            system_prompt: str | None = None,
            disable_thinking: bool = False,
            prefill: str | None = None,
            model: str | None = None,  # <- добавить
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.chat(
            messages,
            disable_thinking=disable_thinking,
            prefill=prefill,
            model=model,  # <- передать дальше
        )

    def chat_vision(
        self,
        user_prompt: str,
        image_b64: str,
        system_prompt: str | None = None,
        prefill: str | None = None,
    ) -> str:
        """
        Vision-запрос: текст + изображение в base64.
        think=false НЕ передаётся — несовместимо с vision в этой версии Ollama.
        Prefill используется для принудительного начала JSON-вывода.
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": user_prompt,
            "images": [image_b64],
        })
        return self.chat(messages, disable_thinking=False, prefill=prefill)

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Проверяет доступность Ollama (GET /api/tags)."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list[str]:
        """Возвращает список загруженных моделей."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.exceptions.RequestException as exc:
            raise OllamaError("Не удалось получить список моделей") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Генерирует embedding-векторы для списка текстов.
        Использует отдельную лёгкую модель (nomic-embed-text).
        Возвращает список векторов в том же порядке что и входной список.
        """
        from config import OLLAMA_EMBED_MODEL
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": OLLAMA_EMBED_MODEL, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"Ошибка при получении embeddings: {exc}") from exc

    @staticmethod
    def extract_json(text: str) -> str:
        """
        Извлекает JSON из ответа модели.
        Обрабатывает случай, когда модель оборачивает JSON в ```json ... ```.
        """
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        return text