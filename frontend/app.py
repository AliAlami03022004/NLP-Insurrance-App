"""Minimal Streamlit frontend for the modularized NLP project."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


MODULAR_ROOT = Path(__file__).resolve().parents[1]
if str(MODULAR_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULAR_ROOT))

from backend.services import (
    ask_question,
    get_anomaly_examples,
    get_dataset_overview,
    get_insurer_analytics,
    get_insurer_summary,
    get_model_performance_tables,
    get_processed_reviews,
    get_project_metadata,
    get_report_text,
    get_shap_lite_examples,
    predict_review,
    search_reviews,
)


def _dict_to_df(values: dict[str, float], label_col: str, value_col: str) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(columns=[label_col, value_col])
    return (
        pd.DataFrame([{label_col: key, value_col: float(val)} for key, val in values.items()])
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )


def render_prediction_page() -> None:
    st.header("Prediction and Explanation")
    text = st.text_area(
        "Enter a review",
        height=160,
        value="Customer service answered quickly but the claim reimbursement took too long.",
    )
    if not st.button("Run Prediction", type="primary"):
        return

    result = predict_review(text)
    pred = result["prediction"]
    theme = result["theme"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Stars", str(pred["star_prediction"]))
    c2.metric("Sentiment", pred["sentiment_prediction"])
    c3.metric("Theme", theme["theme_primary"])

    tab_probs, tab_expl, tab_neighbors = st.tabs(["Probabilities", "Explanation", "Similar Reviews"])

    with tab_probs:
        star_probs_df = _dict_to_df(pred.get("star_probabilities", {}), "star_label", "probability")
        sent_probs_df = _dict_to_df(pred.get("sentiment_probabilities", {}), "sentiment_label", "probability")
        theme_scores_df = _dict_to_df(theme.get("theme_scores", {}), "theme", "score")

        if not star_probs_df.empty:
            st.subheader("Star probabilities")
            st.plotly_chart(px.bar(star_probs_df, x="star_label", y="probability"), use_container_width=True)
        if not sent_probs_df.empty:
            st.subheader("Sentiment probabilities")
            st.plotly_chart(px.bar(sent_probs_df, x="sentiment_label", y="probability"), use_container_width=True)
        if not theme_scores_df.empty:
            st.subheader("Theme scores")
            st.plotly_chart(px.bar(theme_scores_df.head(7), x="theme", y="score"), use_container_width=True)

    with tab_expl:
        st.subheader("Why the model predicted this")
        st.write("Prepared text used by the backend:")
        st.code(result["prepared_text"]["text_for_classical_ml"][:600] or result["prepared_text"]["text_clean_corrected"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Star explanation (SHAP-lite style)**")
            st.dataframe(pd.DataFrame(result["star_explanation"]), use_container_width=True)
        with c2:
            st.markdown("**Sentiment explanation (SHAP-lite style)**")
            st.dataframe(pd.DataFrame(result["sentiment_explanation"]), use_container_width=True)

    with tab_neighbors:
        st.subheader("Nearest reviews from the dataset")
        st.dataframe(result["neighbors"], use_container_width=True)


def render_search_page(df) -> None:
    st.header("Search and Information Retrieval")
    mode = st.radio("Mode", ["semantic", "keyword"], horizontal=True)
    query = st.text_input("Query", value="claim delay reimbursement")
    insurer_options = [None] + sorted(df["assureur"].dropna().astype(str).unique().tolist())
    theme_options = [None]
    if "theme_primary" in df.columns:
        theme_options += sorted(df["theme_primary"].dropna().astype(str).unique().tolist())
    c1, c2, c3 = st.columns(3)
    insurer = c1.selectbox("Insurer", options=insurer_options, format_func=lambda x: x or "All")
    theme = c2.selectbox("Theme", options=theme_options, format_func=lambda x: x or "All")
    note = c3.selectbox("Stars", options=[None, 1, 2, 3, 4, 5], format_func=lambda x: "All" if x is None else str(x))
    top_k = st.slider("Top K", 5, 30, 10)
    if st.button("Run Search", type="primary"):
        results = search_reviews(query, mode=mode, top_k=top_k, insurer=insurer, theme=theme, note=note)
        st.dataframe(results, use_container_width=True)


def render_analytics_page(df: pd.DataFrame) -> None:
    st.header("Summary and Insurer Analytics")
    analytics = get_insurer_analytics(split_filter="All", min_reviews=150)
    insurer_stats = analytics["insurer_stats"]
    theme_stats = analytics["theme_stats"]
    insurer_theme_stats = analytics["insurer_theme_stats"]

    insurer_options = sorted(df["assureur"].dropna().astype(str).unique().tolist())
    selected_insurer = st.selectbox("Select an insurer to summarize", insurer_options)
    insurer_summary = get_insurer_summary(selected_insurer)

    s1, s2 = st.columns(2)
    s1.metric("Selected insurer reviews", f"{insurer_summary['review_count']:,}")
    avg_star = insurer_summary["avg_star"]
    s2.metric("Selected insurer avg stars", "n/a" if pd.isna(avg_star) else f"{avg_star:.2f}/5")
    st.markdown("### Generated insurer summary")
    st.markdown(insurer_summary["summary_text"])

    if not insurer_summary["theme_table"].empty:
        st.markdown("### Theme performance for the selected insurer")
        st.dataframe(insurer_summary["theme_table"], use_container_width=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("### Positive evidence")
        st.dataframe(insurer_summary["positive_examples"], use_container_width=True)
    with p2:
        st.markdown("### Negative evidence")
        st.dataframe(insurer_summary["negative_examples"], use_container_width=True)

    if not insurer_stats.empty:
        st.markdown("### Global insurer metrics")
        st.plotly_chart(
            px.bar(insurer_stats.head(20), x="assureur", y="review_count", title="Top insurers by volume"),
            use_container_width=True,
        )
    else:
        st.info("No insurer analytics are available yet.")

    if not theme_stats.empty:
        st.markdown("### Global theme metrics")
        st.plotly_chart(
            px.bar(theme_stats, x="theme_primary", y="avg_star", color="reviews", title="Average stars by theme"),
            use_container_width=True,
        )
    else:
        st.info("No theme analytics are available yet.")

    selected_theme_stats = insurer_theme_stats[insurer_theme_stats["assureur"].astype(str).eq(selected_insurer)].copy()
    if not selected_theme_stats.empty:
        st.markdown("### Selected insurer average stars by theme")
        st.plotly_chart(
            px.bar(
                selected_theme_stats.sort_values("avg_star", ascending=False),
                x="theme_primary",
                y="avg_star",
                color="reviews",
                title=f"{selected_insurer} - average stars by theme",
            ),
            use_container_width=True,
        )


def render_rag_page() -> None:
    st.header("RAG and QA")
    question = st.text_area("Question", value="Which is the best insurer overall by stars?", height=120)
    backend = st.radio("Generative backend", ["hf", "ollama"], horizontal=True)
    if backend == "hf":
        model_name = st.selectbox("HF model", ["google/flan-t5-base", "google/flan-t5-small", "google/flan-t5-large"])
        ollama_url = "http://localhost:11434"
    else:
        model_name = st.text_input("Ollama model", value="llama3.1:8b")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

    if st.button("Run RAG", type="primary"):
        result = ask_question(
            question,
            generative_backend=backend,
            generative_model=model_name,
            ollama_base_url=ollama_url,
        )

        st.subheader("Extractive QA")
        st.markdown(f"**Best answer sentence:** {result['qa_output']['best_answer']}")
        if not result["qa_output"]["answer_sentences"].empty:
            st.dataframe(result["qa_output"]["answer_sentences"], use_container_width=True)

        st.subheader("Template-grounded RAG")
        st.markdown(result["template_answer"])

        st.subheader("Hybrid routed answer")
        st.markdown(result["hybrid_answer"]["answer"])
        if not result["hybrid_answer"]["ranking_table"].empty:
            st.dataframe(result["hybrid_answer"]["ranking_table"], use_container_width=True)

        st.subheader("Generative RAG")
        st.markdown(result["generative_answer"]["answer"])
        if result["generative_answer"].get("error"):
            st.warning(result["generative_answer"]["error"])

        with st.expander("Retrieved review evidence"):
            st.dataframe(result["qa_output"]["retrieved_reviews"], use_container_width=True)


def render_diagnostics_page() -> None:
    st.header("Diagnostics")
    overview = get_dataset_overview()
    metadata = get_project_metadata()
    insurer_count = overview.get("insurers", overview.get("n_insurers", 0))
    theme_count = overview.get("themes", overview.get("n_themes", 0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{overview['rows']:,}")
    c2.metric("Insurers", str(insurer_count))
    c3.metric("Themes", str(theme_count))
    st.write("Embedding metadata:", metadata["embedding_metadata"])
    st.subheader("SHAP-lite examples")
    st.dataframe(get_shap_lite_examples("sentiment").head(10), use_container_width=True)
    st.subheader("Anomaly examples")
    st.dataframe(get_anomaly_examples(10), use_container_width=True)
    report_text = get_report_text("phase5_error_analysis.md")
    if report_text:
        st.subheader("Phase 5 report")
        st.markdown(report_text)


def main() -> None:
    st.set_page_config(page_title="NLP_ProjetV2 Modular Frontend", layout="wide")
    st.title("NLP_ProjetV2 Modular Frontend")
    st.caption("UI-only layer calling backend services.")

    df = get_processed_reviews()
    page = st.sidebar.radio(
        "Navigation",
        ["Prediction and Explanation", "Summary and Analytics", "Search and Retrieval", "RAG and QA", "Diagnostics"],
    )

    if page == "Prediction and Explanation":
        render_prediction_page()
    elif page == "Search and Retrieval":
        render_search_page(df)
    elif page == "Summary and Analytics":
        render_analytics_page(df)
    elif page == "RAG and QA":
        render_rag_page()
    else:
        render_diagnostics_page()


if __name__ == "__main__":
    main()
