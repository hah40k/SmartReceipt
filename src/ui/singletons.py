"""
src/ui/singletons.py

Ленивые синглтоны компонентов приложения.
Все объекты создаются при первом обращении и разделяются между вкладками.

Жизненный цикл:
  - OllamaClient, Extractor, Analyzer, Matcher, Retriever — живут всё время работы
  - AdvisorSession, ShoppingSession — сессии диалогов, очищаются кнопкой «Очистить»
  - _saved_hashes — защита от двойного сохранения, очищается вместе с БД
"""
from __future__ import annotations

from src.analytics.analyzer import ReceiptAnalyzer
from src.analytics.matcher import ItemMatcher
from src.analytics.retriever import RagRetriever
from src.llm.advisor import AdvisorSession, BudgetAdvisor
from src.llm.client import OllamaClient
from src.llm.reporter import ReportGenerator
from src.llm.shopping import ShoppingAssistant, ShoppingSession
from src.vision.extractor import ReceiptExtractor

# Хэши чеков, сохранённых в этой сессии — защита от двойного сохранения.
# Изменяется из upload_tab и dashboard_tab (clear), поэтому живёт здесь как
# единственный экземпляр, на который оба модуля держат ссылку.
_saved_hashes: set[int] = set()

_client: OllamaClient | None = None
_extractor: ReceiptExtractor | None = None
_analyzer: ReceiptAnalyzer | None = None
_matcher: ItemMatcher | None = None
_retriever: RagRetriever | None = None
_advisor: BudgetAdvisor | None = None
_advisor_session: AdvisorSession | None = None
_reporter: ReportGenerator | None = None
_shopping: ShoppingAssistant | None = None
_shopping_session: ShoppingSession | None = None


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


def get_extractor() -> ReceiptExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ReceiptExtractor(client=get_client())
    return _extractor


def get_analyzer() -> ReceiptAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ReceiptAnalyzer()
    return _analyzer


def get_matcher() -> ItemMatcher:
    global _matcher
    if _matcher is None:
        _matcher = ItemMatcher(client=get_client())
    return _matcher


def get_retriever() -> RagRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RagRetriever(client=get_client())
    return _retriever


def get_advisor() -> BudgetAdvisor:
    global _advisor
    if _advisor is None:
        _advisor = BudgetAdvisor(
            client=get_client(),
            analyzer=get_analyzer(),
            retriever=get_retriever(),
        )
    return _advisor


def get_advisor_session() -> AdvisorSession:
    global _advisor_session
    if _advisor_session is None:
        _advisor_session = AdvisorSession()
    return _advisor_session


def get_reporter() -> ReportGenerator:
    global _reporter
    if _reporter is None:
        _reporter = ReportGenerator(client=get_client(), analyzer=get_analyzer())
    return _reporter


def get_shopping() -> ShoppingAssistant:
    global _shopping
    if _shopping is None:
        _shopping = ShoppingAssistant(client=get_client())
    return _shopping


def get_shopping_session() -> ShoppingSession:
    global _shopping_session
    if _shopping_session is None:
        _shopping_session = ShoppingSession()
    return _shopping_session
