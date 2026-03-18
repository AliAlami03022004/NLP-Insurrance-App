"""Central orchestration layer used by the modular frontend."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.config import (
    EMBEDDINGS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SUPERVISED_DIR,
    TABLES_DIR,
    THEMES_DIR,
    ensure_project_dirs,
)
from backend.data_loader import (
    dataset_overview,
    load_processed_reviews_dataset,
    load_raw_reviews_dataset,
)
from backend.eda import run_eda as run_eda_reports
from backend.embeddings import build_embedding_artifacts, export_embedding_artifacts
from backend.evaluation import run_phase5_error_analysis
from backend.modeling import run_supervised_benchmark
from backend.preprocessing import prepare_text_record, run_preprocessing_pipeline
from backend.rag import (
    extractive_qa,
    hybrid_rag_answer,
    rag_generative_answer,
    rag_generative_answer_ollama,
    rag_summary_answer,
)
from backend.search import (
    InferenceResources,
    keyword_search,
    load_inference_resources,
    local_token_explanation,
    predict_stars_and_sentiment,
    predict_theme,
    semantic_search,
)
from backend.themes import export_theme_artifacts, generate_theme_reports, run_hybrid_theme_pipeline


PHASE3_DATASET_PATH = PROCESSED_DATA_DIR / "clean_reviews_phase3.csv"


def reset_runtime_caches() -> None:
    """Clear cached resources after local regeneration."""
    get_processed_reviews.cache_clear()
    get_inference_resources.cache_clear()


def _write_column_summary_csv(df: pd.DataFrame) -> Path:
    summary_rows: list[dict[str, Any]] = []
    for column in df.columns:
        summary_rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": float(df[column].isna().mean() * 100),
                "n_unique": int(df[column].nunique(dropna=True)),
                "sample_value": str(df[column].dropna().iloc[0]) if df[column].dropna().shape[0] else "",
            }
        )
    out_path = REPORTS_DIR / "phase2_column_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def run_preprocessing(force: bool = False) -> dict[str, Any]:
    """Build the clean dataset locally from modular raw Excel files."""
    ensure_project_dirs()
    clean_csv_path = PROCESSED_DATA_DIR / "clean_reviews.csv"
    report_path = REPORTS_DIR / "phase2_preprocessing_report.json"
    if clean_csv_path.exists() and report_path.exists() and not force:
        clean_df = load_processed_reviews_dataset(phase3=False)
        column_summary_path = REPORTS_DIR / "phase2_column_summary.csv"
        if not column_summary_path.exists():
            column_summary_path = _write_column_summary_csv(clean_df)
        return {
            "clean_reviews_csv": str(clean_csv_path),
            "preprocessing_report": str(report_path),
            "column_summary_csv": str(column_summary_path),
            "rows": int(len(clean_df)),
        }

    raw_df = load_raw_reviews_dataset()
    output_paths = run_preprocessing_pipeline(raw_df)
    clean_df = load_processed_reviews_dataset(phase3=False)
    column_summary_path = _write_column_summary_csv(clean_df)
    reset_runtime_caches()
    return {
        **output_paths,
        "column_summary_csv": str(column_summary_path),
        "rows": int(len(clean_df)),
    }


def run_eda(force: bool = False) -> dict[str, Any]:
    """Generate EDA tables and plots from the local clean dataset."""
    ensure_project_dirs()
    if force or not (TABLES_DIR / "eda_star_distribution.csv").exists():
        if not (PROCESSED_DATA_DIR / "clean_reviews.csv").exists():
            run_preprocessing(force=False)
        clean_df = load_processed_reviews_dataset(phase3=False)
        eda_outputs = run_eda_reports(clean_df, FIGURES_DIR, TABLES_DIR, REPORTS_DIR)
        return eda_outputs
    return {
        "summary_report": str(REPORTS_DIR / "phase2_eda_summary.md"),
        "tables_dir": str(TABLES_DIR),
        "figures_dir": str(FIGURES_DIR),
    }


def run_theme_and_embedding_pipeline(force: bool = False) -> dict[str, Any]:
    """Create local phase 3 dataset, theme artifacts, and embedding artifacts."""
    ensure_project_dirs()
    phase3_ready = all(
        path.exists()
        for path in [
            PHASE3_DATASET_PATH,
            THEMES_DIR / "hybrid_theme_artifacts.joblib",
            THEMES_DIR / "hybrid_theme_metadata.json",
            EMBEDDINGS_DIR / "review_embeddings.npy",
            EMBEDDINGS_DIR / "embedding_metadata.json",
            EMBEDDINGS_DIR / "review_nn_index.joblib",
            EMBEDDINGS_DIR / "word2vec.model",
        ]
    )
    if force or not phase3_ready:
        if not (PROCESSED_DATA_DIR / "clean_reviews.csv").exists():
            run_preprocessing(force=False)
        clean_df = load_processed_reviews_dataset(phase3=False)
        theme_artifacts = run_hybrid_theme_pipeline(clean_df)
        scored_df = theme_artifacts.scored_df
        scored_df.to_csv(PHASE3_DATASET_PATH, index=False, encoding="utf-8-sig")

        export_theme_artifacts(
            artifacts=theme_artifacts,
            model_path=THEMES_DIR / "hybrid_theme_artifacts.joblib",
            metadata_path=THEMES_DIR / "hybrid_theme_metadata.json",
        )
        theme_reports = generate_theme_reports(
            scored_df=scored_df,
            figures_dir=FIGURES_DIR,
            tables_dir=TABLES_DIR,
            reports_dir=REPORTS_DIR,
        )

        embedding_artifacts = build_embedding_artifacts(scored_df)
        embedding_reports = export_embedding_artifacts(
            artifacts=embedding_artifacts,
            scored_df=scored_df,
            model_dir=EMBEDDINGS_DIR,
            tables_dir=TABLES_DIR,
            figures_dir=FIGURES_DIR,
            reports_dir=REPORTS_DIR,
        )
        reset_runtime_caches()
        return {
            "clean_reviews_phase3_csv": str(PHASE3_DATASET_PATH),
            "theme_model_path": str(THEMES_DIR / "hybrid_theme_artifacts.joblib"),
            "theme_metadata_path": str(THEMES_DIR / "hybrid_theme_metadata.json"),
            **theme_reports,
            **embedding_reports,
        }

    return {
        "clean_reviews_phase3_csv": str(PHASE3_DATASET_PATH),
        "theme_model_path": str(THEMES_DIR / "hybrid_theme_artifacts.joblib"),
        "theme_metadata_path": str(THEMES_DIR / "hybrid_theme_metadata.json"),
        "embedding_metadata_json": str(EMBEDDINGS_DIR / "embedding_metadata.json"),
    }


def run_supervised_pipeline(force: bool = False) -> dict[str, Any]:
    """Train and compare local supervised models using modular artifacts."""
    ensure_project_dirs()
    supervised_ready = all(
        path.exists()
        for path in [
            SUPERVISED_DIR / "stars_best_model.joblib",
            SUPERVISED_DIR / "sentiment_best_model.joblib",
            TABLES_DIR / "phase4_model_comparison_stars.csv",
            TABLES_DIR / "phase4_model_comparison_sentiment.csv",
            REPORTS_DIR / "phase4_modeling_summary.md",
        ]
    )
    if force or not supervised_ready:
        if not PHASE3_DATASET_PATH.exists() or not (EMBEDDINGS_DIR / "review_embeddings.npy").exists():
            run_theme_and_embedding_pipeline(force=False)
        phase3_df = load_processed_reviews_dataset(phase3=True)
        review_embeddings = np.load(EMBEDDINGS_DIR / "review_embeddings.npy")
        embedding_metadata = json.loads((EMBEDDINGS_DIR / "embedding_metadata.json").read_text(encoding="utf-8"))
        outputs = run_supervised_benchmark(
            phase3_df=phase3_df,
            review_embeddings=review_embeddings,
            embedding_backend=str(embedding_metadata.get("embedding_backend", "unknown")),
            word2vec_path=EMBEDDINGS_DIR / "word2vec.model",
            models_dir=SUPERVISED_DIR,
            tables_dir=TABLES_DIR,
            figures_dir=FIGURES_DIR,
            reports_dir=REPORTS_DIR,
        )
        reset_runtime_caches()
        return outputs
    return {
        "stars_results_path": str(TABLES_DIR / "phase4_model_comparison_stars.csv"),
        "sentiment_results_path": str(TABLES_DIR / "phase4_model_comparison_sentiment.csv"),
        "summary_path": str(REPORTS_DIR / "phase4_modeling_summary.md"),
    }


def run_error_analysis(force: bool = False) -> dict[str, Any]:
    """Generate local phase 5 interpretation, SHAP-lite, and anomaly outputs."""
    ensure_project_dirs()
    phase5_ready = all(
        path.exists()
        for path in [
            REPORTS_DIR / "phase5_error_analysis.md",
            TABLES_DIR / "phase5_top_anomalies.csv",
            TABLES_DIR / "phase5_shap_lite_examples_stars.csv",
            TABLES_DIR / "phase5_shap_lite_examples_sentiment.csv",
        ]
    )
    if force or not phase5_ready:
        if not (SUPERVISED_DIR / "stars_best_model.joblib").exists():
            run_supervised_pipeline(force=False)
        phase3_df = load_processed_reviews_dataset(phase3=True)
        review_embeddings = np.load(EMBEDDINGS_DIR / "review_embeddings.npy")
        return run_phase5_error_analysis(
            phase3_df=phase3_df,
            review_embeddings=review_embeddings,
            models_dir=SUPERVISED_DIR,
            tables_dir=TABLES_DIR,
            figures_dir=FIGURES_DIR,
            reports_dir=REPORTS_DIR,
        )
    return {"phase5_report": str(REPORTS_DIR / "phase5_error_analysis.md")}


def ensure_inference_assets() -> None:
    """Guarantee the minimal local artifacts needed by the app."""
    required_paths = [
        PHASE3_DATASET_PATH,
        EMBEDDINGS_DIR / "review_embeddings.npy",
        EMBEDDINGS_DIR / "review_nn_index.joblib",
        EMBEDDINGS_DIR / "embedding_metadata.json",
        THEMES_DIR / "hybrid_theme_artifacts.joblib",
        SUPERVISED_DIR / "stars_best_model.joblib",
        SUPERVISED_DIR / "sentiment_best_model.joblib",
        SUPERVISED_DIR / "stars_tfidf_logreg_model.joblib",
        SUPERVISED_DIR / "sentiment_tfidf_logreg_model.joblib",
    ]
    if all(path.exists() for path in required_paths):
        return
    run_theme_and_embedding_pipeline(force=False)
    run_supervised_pipeline(force=False)


@lru_cache(maxsize=1)
def get_inference_resources() -> InferenceResources:
    ensure_inference_assets()
    return load_inference_resources(PROJECT_ROOT)


@lru_cache(maxsize=1)
def get_processed_reviews() -> pd.DataFrame:
    if PHASE3_DATASET_PATH.exists():
        return load_processed_reviews_dataset(phase3=True)
    if (PROCESSED_DATA_DIR / "clean_reviews.csv").exists():
        try:
            run_theme_and_embedding_pipeline(force=False)
        except Exception:
            return load_processed_reviews_dataset(phase3=False)
        if PHASE3_DATASET_PATH.exists():
            return load_processed_reviews_dataset(phase3=True)
        return load_processed_reviews_dataset(phase3=False)
    run_preprocessing(force=False)
    try:
        run_theme_and_embedding_pipeline(force=False)
    except Exception:
        return load_processed_reviews_dataset(phase3=False)
    if PHASE3_DATASET_PATH.exists():
        return load_processed_reviews_dataset(phase3=True)
    return load_processed_reviews_dataset(phase3=False)


def get_raw_reviews() -> pd.DataFrame:
    return load_raw_reviews_dataset()


def get_dataset_overview() -> dict[str, Any]:
    return dataset_overview(get_processed_reviews())


def get_app_context() -> dict[str, Any]:
    df = get_processed_reviews()
    return {
        "rows": int(len(df)),
        "insurers": sorted(df.get("assureur", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "themes": sorted(df.get("theme_primary", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "artifacts_ready": all(
            path.exists()
            for path in [
                PHASE3_DATASET_PATH,
                EMBEDDINGS_DIR / "review_embeddings.npy",
                THEMES_DIR / "hybrid_theme_artifacts.joblib",
                SUPERVISED_DIR / "stars_best_model.joblib",
            ]
        ),
    }


def predict_review(text: str) -> dict[str, Any]:
    resources = get_inference_resources()
    prediction = predict_stars_and_sentiment(text, resources)
    theme = predict_theme(text, resources)
    star_expl = local_token_explanation(text, resources.star_explainer_model, top_n=10)
    sentiment_expl = local_token_explanation(text, resources.sentiment_explainer_model, top_n=10)
    neighbors = semantic_search(text, resources, top_k=6)
    return {
        "prepared_text": prepare_text_record(text),
        "prediction": prediction,
        "theme": theme,
        "star_explanation": star_expl,
        "sentiment_explanation": sentiment_expl,
        "neighbors": neighbors,
    }


def search_reviews(
    query: str,
    mode: str = "semantic",
    top_k: int = 10,
    insurer: str | None = None,
    theme: str | None = None,
    note: int | None = None,
) -> pd.DataFrame:
    resources = get_inference_resources()
    if mode == "keyword":
        return keyword_search(query, resources, top_k=top_k, filter_assureur=insurer, filter_theme=theme, filter_note=note)
    return semantic_search(query, resources, top_k=top_k, filter_assureur=insurer, filter_theme=theme, filter_note=note)


def ask_question(
    question: str,
    insurer: str | None = None,
    theme: str | None = None,
    note: int | None = None,
    top_k_docs: int = 14,
    top_k_sentences: int = 5,
    min_reviews_for_ranking: int = 120,
    generative_backend: str = "hf",
    generative_model: str = "google/flan-t5-base",
    ollama_base_url: str = "http://localhost:11434",
) -> dict[str, Any]:
    resources = get_inference_resources()
    qa_output = extractive_qa(
        question=question,
        resources=resources,
        top_k_docs=top_k_docs,
        top_k_sentences=top_k_sentences,
        filter_assureur=insurer,
        filter_theme=theme,
        filter_note=note,
    )
    template_answer = rag_summary_answer(question=question, qa_output=qa_output)
    hybrid_answer = hybrid_rag_answer(
        question=question,
        resources=resources,
        qa_output=qa_output,
        min_reviews=min_reviews_for_ranking,
        top_n=8,
        filter_theme=theme,
        filter_note=note,
    )
    if generative_backend.lower() == "ollama":
        generative_answer = rag_generative_answer_ollama(
            question=question,
            qa_output=qa_output,
            model_name=generative_model,
            max_evidence_sentences=top_k_sentences,
            base_url=ollama_base_url,
        )
    else:
        generative_answer = rag_generative_answer(
            question=question,
            qa_output=qa_output,
            model_name=generative_model,
            max_evidence_sentences=top_k_sentences,
        )
    return {
        "template_answer": template_answer,
        "hybrid_answer": hybrid_answer,
        "generative_answer": generative_answer,
        "qa_output": qa_output,
    }


def get_insurer_analytics(split_filter: str = "All", min_reviews: int = 150) -> dict[str, pd.DataFrame]:
    df = get_processed_reviews().copy()
    if split_filter != "All" and "type" in df.columns:
        df = df[df["type"].astype(str).eq(split_filter)]

    insurer_stats = (
        df.groupby("assureur", as_index=False)
        .agg(review_count=("type", "size"), avg_star=("note", "mean"))
        .sort_values("review_count", ascending=False)
    )
    insurer_stats = insurer_stats[insurer_stats["review_count"] >= min_reviews]

    theme_stats = (
        df[df["note"].notna()]
        .groupby("theme_primary", as_index=False)
        .agg(avg_star=("note", "mean"), reviews=("note", "size"))
    )
    insurer_theme_stats = (
        df[df["note"].notna()]
        .groupby(["assureur", "theme_primary"], as_index=False)
        .agg(avg_star=("note", "mean"), reviews=("note", "size"))
    )
    return {
        "insurer_stats": insurer_stats,
        "theme_stats": theme_stats,
        "insurer_theme_stats": insurer_theme_stats,
        "raw": df,
    }


def get_insurer_summary(insurer: str, max_examples: int = 3) -> dict[str, Any]:
    """Build a grading-oriented insurer summary from local review evidence."""
    df = get_processed_reviews().copy()
    if "assureur" not in df.columns:
        return {
            "summary_text": "Insurer information is not available in this dataset.",
            "theme_table": pd.DataFrame(),
            "positive_examples": pd.DataFrame(),
            "negative_examples": pd.DataFrame(),
            "review_count": 0,
            "avg_star": np.nan,
        }

    subset = df[df["assureur"].astype(str).eq(insurer)].copy()
    if subset.empty:
        return {
            "summary_text": f"No reviews were found for insurer `{insurer}`.",
            "theme_table": pd.DataFrame(),
            "positive_examples": pd.DataFrame(),
            "negative_examples": pd.DataFrame(),
            "review_count": 0,
            "avg_star": np.nan,
        }

    subset["note_num"] = pd.to_numeric(subset.get("note"), errors="coerce")
    review_count = int(len(subset))
    avg_star = float(subset["note_num"].mean()) if subset["note_num"].notna().any() else np.nan
    sentiment_counts = subset.get("sentiment_label", pd.Series(dtype=str)).fillna("unknown").value_counts(normalize=True)
    theme_table = (
        subset.groupby("theme_primary", as_index=False)
        .agg(review_count=("theme_primary", "size"), avg_star=("note_num", "mean"))
        .sort_values(["review_count", "avg_star"], ascending=[False, False])
    )

    top_themes = theme_table["theme_primary"].head(3).tolist() if not theme_table.empty else []
    strongest_themes = theme_table.sort_values(["avg_star", "review_count"], ascending=[False, False]).head(2)["theme_primary"].tolist() if not theme_table.empty else []
    weakest_themes = theme_table.sort_values(["avg_star", "review_count"], ascending=[True, False]).head(2)["theme_primary"].tolist() if not theme_table.empty else []

    display_cols = [col for col in ["note", "theme_primary", "text_clean_corrected"] if col in subset.columns]
    positive_examples = (
        subset[subset["note_num"] >= 4]
        .sort_values(["note_num", "theme_confidence"], ascending=[False, False])
        .loc[:, display_cols]
        .head(max_examples)
        .reset_index(drop=True)
    )
    negative_examples = (
        subset[subset["note_num"] <= 2]
        .sort_values(["note_num", "theme_confidence"], ascending=[True, False])
        .loc[:, display_cols]
        .head(max_examples)
        .reset_index(drop=True)
    )

    positive_share = float(sentiment_counts.get("positive", 0.0))
    neutral_share = float(sentiment_counts.get("neutral", 0.0))
    negative_share = float(sentiment_counts.get("negative", 0.0))

    summary_lines = [
        f"**{insurer}** is represented by **{review_count:,}** reviews.",
    ]
    if np.isfinite(avg_star):
        summary_lines.append(f"The average rating is **{avg_star:.2f}/5**.")
    if top_themes:
        summary_lines.append(f"The most discussed themes are **{', '.join(top_themes)}**.")
    summary_lines.append(
        f"Sentiment mix is approximately **{positive_share*100:.0f}% positive**, "
        f"**{neutral_share*100:.0f}% neutral**, and **{negative_share*100:.0f}% negative**."
    )
    if strongest_themes:
        summary_lines.append(f"Best-perceived themes are **{', '.join(strongest_themes)}**.")
    if weakest_themes:
        summary_lines.append(f"Most problematic themes are **{', '.join(weakest_themes)}**.")

    return {
        "summary_text": "\n\n".join(summary_lines),
        "theme_table": theme_table.reset_index(drop=True),
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "review_count": review_count,
        "avg_star": avg_star,
    }


def get_model_performance_tables() -> dict[str, pd.DataFrame]:
    if not (TABLES_DIR / "phase4_model_comparison_stars.csv").exists():
        run_supervised_pipeline(force=False)
    return {
        "stars": pd.read_csv(TABLES_DIR / "phase4_model_comparison_stars.csv"),
        "sentiment": pd.read_csv(TABLES_DIR / "phase4_model_comparison_sentiment.csv"),
    }


def get_model_reports() -> dict[str, str]:
    return {
        "phase4_summary": get_report_text("phase4_modeling_summary.md"),
        "stars_report": get_report_text("phase4_classification_report_stars.txt"),
        "sentiment_report": get_report_text("phase4_classification_report_sentiment.txt"),
    }


def get_error_analysis_tables() -> dict[str, pd.DataFrame]:
    run_error_analysis(force=False)
    outputs: dict[str, pd.DataFrame] = {}
    for name in [
        "phase5_predictions_stars_test.csv",
        "phase5_predictions_sentiment_test.csv",
        "phase5_top_anomalies.csv",
        "phase5_shap_lite_examples_stars.csv",
        "phase5_shap_lite_examples_sentiment.csv",
    ]:
        path = TABLES_DIR / name
        outputs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return outputs


def get_report_text(report_name: str) -> str:
    report_path = REPORTS_DIR / report_name
    if not report_path.exists() and report_name.startswith("phase5_"):
        run_error_analysis(force=False)
    if not report_path.exists() and report_name.startswith("phase4_"):
        run_supervised_pipeline(force=False)
    if not report_path.exists():
        return ""
    return report_path.read_text(encoding="utf-8")


def get_shap_lite_examples(task_name: str) -> pd.DataFrame:
    path = TABLES_DIR / f"phase5_shap_lite_examples_{task_name}.csv"
    if not path.exists():
        run_error_analysis(force=False)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_anomaly_examples(limit: int = 50) -> pd.DataFrame:
    path = TABLES_DIR / "phase5_top_anomalies.csv"
    if not path.exists():
        run_error_analysis(force=False)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).head(limit)


def get_project_metadata() -> dict[str, Any]:
    embedding_meta_path = EMBEDDINGS_DIR / "embedding_metadata.json"
    metadata: dict[str, Any] = {}
    if embedding_meta_path.exists():
        metadata = json.loads(embedding_meta_path.read_text(encoding="utf-8"))
    return {
        "project_root": str(PROJECT_ROOT),
        "outputs_dir": str(OUTPUTS_DIR),
        "models_dir": str(MODELS_DIR),
        "embedding_metadata": metadata,
    }


def get_dashboard_data() -> dict[str, Any]:
    overview = get_dataset_overview()
    analytics = get_insurer_analytics(split_filter="All", min_reviews=150)
    return {
        "overview": overview,
        "top_insurers": analytics["insurer_stats"].head(20),
        "theme_stats": analytics["theme_stats"],
    }
