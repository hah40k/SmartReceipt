"""
main.py — точка входа SmartReceipt.

  python main.py          → запуск Gradio UI
  python main.py --test   → smoke test (без UI)
"""
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def smoke_test() -> None:
    """Проверяет, что ядро собирается, Ollama доступен и Vision работает."""
    from src.db.crud import init_db
    from src.llm.client import OllamaClient

    logger.info("=== Инициализация БД ===")
    init_db()

    logger.info("=== Проверка Ollama ===")
    client = OllamaClient()
    if not client.is_available():
        logger.error("Ollama недоступен. Запустите: ollama serve")
        sys.exit(1)

    models = client.list_models()
    logger.info("Доступные модели: %s", models)

    logger.info("=== Тестовый текстовый запрос ===")
    response = client.chat_text("Ответь одним словом: работаешь?")
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    logger.info("Ответ модели: %s", clean or "(thinking mode)")

    receipts_dir = Path("receipts")
    images = (
        list(receipts_dir.glob("*.jpg"))
        + list(receipts_dir.glob("*.jpeg"))
        + list(receipts_dir.glob("*.png"))
    )
    if images:
        logger.info("=== Vision-тест на файле: %s ===", images[0].name)
        from src.vision.extractor import ReceiptExtractor
        extractor = ReceiptExtractor(client=client)
        receipt = extractor.extract_from_file(images[0])
        logger.info(
            "Результат: магазин=%s, дата=%s, позиций=%d, итог=%.2f руб.",
            receipt.store_name, receipt.date, len(receipt.items), receipt.total,
        )
        for item in receipt.items:
            logger.info("  [%s] %s — %.2f руб.", item.category, item.name, item.total_price)
    else:
        logger.info("=== Vision-тест пропущен (нет фото в receipts/) ===")

    logger.info("=== Smoke test пройден ✓ ===")


def run_ui() -> None:
    from src.ui.app import create_app
    app = create_app()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    if "--test" in sys.argv:
        smoke_test()
    else:
        run_ui()
