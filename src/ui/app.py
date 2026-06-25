"""
src/ui/app.py

Точка сборки Gradio-приложения SmartReceipt.
Три вкладки собираются из отдельных модулей в src/ui/tabs/.
"""
from __future__ import annotations

import logging

import gradio as gr

from src.db.crud import init_db
from src.ui.tabs.advisor_tab import build_advisor_tab
from src.ui.tabs.dashboard_tab import build_dashboard_tab
from src.ui.tabs.search_tab import build_search_tab
from src.ui.tabs.upload_tab import build_upload_tab

logger = logging.getLogger(__name__)


def create_app() -> gr.Blocks:
    init_db()

    with gr.Blocks(title="SmartReceipt") as app:
        gr.Markdown("# 🧾 SmartReceipt — локальный ИИ-ассистент по расходам")
        build_upload_tab()
        build_dashboard_tab()
        build_search_tab()
        build_advisor_tab()

    return app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = create_app()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    )