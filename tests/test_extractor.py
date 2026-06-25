"""
tests/test_extractor.py
Тесты Vision-экстрактора без реального Ollama (mock).
Запуск: python -m pytest tests/test_extractor.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock

from src.vision.extractor import ReceiptExtractor, ExtractionError
from src.schemas.receipt import Receipt


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

VALID_JSON_RESPONSE = """{
  "store_name": "Пятёрочка",
  "date": "2025-06-01",
  "time": "14:30:00",
  "items": [
    {
      "name": "Молоко 3.2%",
      "category": "молочные продукты",
      "quantity": 1.0,
      "price_per_unit": 89.90,
      "total_price": 89.90,
      "vat_rate": null
    },
    {
      "name": "Хлеб нарезной",
      "category": "хлеб и выпечка",
      "quantity": 2.0,
      "price_per_unit": 45.00,
      "total_price": 90.00,
      "vat_rate": null
    }
  ],
  "subtotal": 179.90,
  "discount": 0.0,
  "total": 179.90,
  "vat_total": null,
  "payment_method": "карта"
}"""

THINKING_WRAPPED_RESPONSE = """<think>
Смотрю на чек, вижу позиции...
</think>
{
  "store_name": "Магнит",
  "date": "2025-07-15",
  "time": null,
  "items": [
    {
      "name": "Кефир 1%",
      "category": "молочные продукты",
      "quantity": 1.0,
      "price_per_unit": 65.00,
      "total_price": 65.00,
      "vat_rate": null
    }
  ],
  "subtotal": 65.00,
  "discount": 0.0,
  "total": 65.00,
  "vat_total": null,
  "payment_method": null
}"""

MARKDOWN_WRAPPED_RESPONSE = """```json
{
  "store_name": "Лента",
  "date": "2025-08-01",
  "time": "10:00:00",
  "items": [
    {
      "name": "Сыр Российский",
      "category": "молочные продукты",
      "quantity": 0.3,
      "price_per_unit": 500.00,
      "total_price": 150.00,
      "vat_rate": null
    }
  ],
  "subtotal": 150.00,
  "discount": 0.0,
  "total": 150.00,
  "vat_total": null,
  "payment_method": "карта"
}
```"""


def make_extractor(mock_response: str) -> ReceiptExtractor:
    """Создаёт экстрактор с замоканным Ollama-клиентом."""
    mock_client = MagicMock()
    mock_client.chat_vision.return_value = mock_response
    return ReceiptExtractor(client=mock_client)


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------

class TestJsonParsing:

    def test_clean_json(self):
        extractor = make_extractor(VALID_JSON_RESPONSE)
        receipt = extractor._extract("fake_b64")
        assert isinstance(receipt, Receipt)
        assert receipt.store_name == "Пятёрочка"
        assert len(receipt.items) == 2
        assert receipt.total == 179.90

    def test_thinking_tags_stripped(self):
        extractor = make_extractor(THINKING_WRAPPED_RESPONSE)
        receipt = extractor._extract("fake_b64")
        assert receipt.store_name == "Магнит"
        assert len(receipt.items) == 1

    def test_markdown_block_stripped(self):
        extractor = make_extractor(MARKDOWN_WRAPPED_RESPONSE)
        receipt = extractor._extract("fake_b64")
        assert receipt.store_name == "Лента"
        assert receipt.items[0].name == "Сыр Российский"

    def test_invalid_json_raises(self):
        extractor = make_extractor("Извините, я не могу распознать чек.")
        with pytest.raises(ExtractionError):
            extractor._extract("fake_b64")

    def test_invalid_schema_raises(self):
        # JSON есть, но не соответствует схеме Receipt
        bad_json = '{"foo": "bar"}'
        extractor = make_extractor(bad_json)
        with pytest.raises(ExtractionError):
            extractor._extract("fake_b64")


class TestReceiptFields:

    def test_category_normalized(self):
        extractor = make_extractor(VALID_JSON_RESPONSE)
        receipt = extractor._extract("fake_b64")
        for item in receipt.items:
            assert item.category == item.category.lower()

    def test_payment_method_preserved(self):
        extractor = make_extractor(VALID_JSON_RESPONSE)
        receipt = extractor._extract("fake_b64")
        assert receipt.payment_method == "карта"

    def test_null_time_allowed(self):
        extractor = make_extractor(THINKING_WRAPPED_RESPONSE)
        receipt = extractor._extract("fake_b64")
        assert receipt.time is None
