"""Public backend exports for the standalone modular project."""

from backend.services import (
    ask_question,
    get_app_context,
    get_dashboard_data,
    get_dataset_overview,
    get_insurer_analytics,
    get_insurer_summary,
    predict_review,
    run_eda,
    run_error_analysis,
    run_preprocessing,
    run_supervised_pipeline,
    run_theme_and_embedding_pipeline,
    search_reviews,
)

__all__ = [
    "ask_question",
    "get_app_context",
    "get_dashboard_data",
    "get_dataset_overview",
    "get_insurer_analytics",
    "get_insurer_summary",
    "predict_review",
    "run_eda",
    "run_error_analysis",
    "run_preprocessing",
    "run_supervised_pipeline",
    "run_theme_and_embedding_pipeline",
    "search_reviews",
]
