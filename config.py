from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RECEIPTS_DIR = BASE_DIR / "receipts"

DATA_DIR.mkdir(exist_ok=True)
RECEIPTS_DIR.mkdir(exist_ok=True)

# База данных
DB_PATH = DATA_DIR / "budget.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Профиль покупателя (для умного списка покупок)
SHOPPING_PROFILE_PATH = DATA_DIR / "shopping_profile.txt"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:9b"       # для Vision (распознавание фото)
OLLAMA_TEXT_MODEL = "qwen3.5:4b"  # для текстовых задач: нормализация + классификация
OLLAMA_EMBED_MODEL = "nomic-embed-text-v2-moe"  # мультиязычная модель с русским датасетом
# Альтернатива: "bge-m3" — также хорошо для русского, можно сравнить качество
OLLAMA_TIMEOUT = 180
OLLAMA_ENRICH_MODEL = "receipt-extractor"

# LLM-параметры
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 8192

# Embedding matching
SIMILARITY_THRESHOLD = 0.88  # порог косинусного сходства для авто-предложения совпадения