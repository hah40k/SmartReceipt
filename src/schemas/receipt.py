import logging

from datetime import date as Date, time as Time
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


class ReceiptItem(BaseModel):
    name: str
    category: str
    subcategory: str | None = None
    quantity: float
    price_per_unit: float
    total_price: float
    vat_rate: float | None = None

    @field_validator("quantity", "price_per_unit", "total_price")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Значение не может быть отрицательным")
        return round(v, 2)

    @field_validator("category")
    @classmethod
    def normalize_category(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("subcategory")
    @classmethod
    def normalize_subcategory(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        # Убираем префикс категории если модель вернула "категория: подкатегория"
        if ": " in v:
            v = v.split(": ", 1)[-1].strip()
        return v or None


class Receipt(BaseModel):
    store_name: str | None = None
    date: Date
    time: Time | None = None

    @field_validator("store_name", mode="before")
    @classmethod
    def normalize_store_name(cls, v) -> str | None:
        if not v:
            return None
        normalized = str(v).strip()
        return normalized.title() if normalized else None

    @field_validator("date", mode="before")
    @classmethod
    def fallback_to_today(cls, v) -> Date:
        if v is None:
            import datetime
            logger.warning("Дата не найдена в чеке, используем сегодня")
            return datetime.date.today()
        return v
    items: list[ReceiptItem]
    subtotal: float
    discount: float = 0.0
    total: float
    vat_total: float | None = None
    payment_method: str | None = None

    @field_validator("subtotal", "total")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Сумма не может быть отрицательной")
        return round(v, 2)

    @field_validator("discount")
    @classmethod
    def discount_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Скидка не может быть отрицательной")
        return round(v, 2)

    @model_validator(mode="after")
    def items_not_empty(self) -> "Receipt":
        if not self.items:
            raise ValueError("Список товаров не может быть пустым")
        return self


# Вспомогательная схема для ответа API
class OllamaMessage(BaseModel):
    role: str
    content: str


class OllamaRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = False
    options: dict = {}


class OllamaResponse(BaseModel):
    model: str
    message: OllamaMessage
    done: bool