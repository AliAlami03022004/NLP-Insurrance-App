"""Hybrid theme detection pipeline for insurance reviews."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


THEME_LEXICON: dict[str, list[str]] = {
    "Pricing": [
        "price",
        "pricing",
        "cost",
        "expensive",
        "cheap",
        "premium",
        "tariff",
        "rate",
        "quote",
        "euros",
        "payment",
        "refund",
        "increase",
        "fee",
    ],
    "Coverage": [
        "coverage",
        "covered",
        "guarantee",
        "warranty",
        "policy",
        "contract",
        "clause",
        "option",
        "protection",
        "benefits",
        "excluded",
        "deductible",
    ],
    "Enrollment": [
        "enroll",
        "subscription",
        "sign",
        "signup",
        "join",
        "member",
        "application",
        "documents",
        "registration",
        "inscription",
        "underwrite",
        "new contract",
    ],
    "Customer Service": [
        "customer service",
        "advisor",
        "support",
        "call center",
        "agent",
        "phone",
        "response",
        "email",
        "contact",
        "listening",
        "staff",
        "reception",
        "helpful",
        "rude",
    ],
    "Claims Processing": [
        "claim",
        "claims",
        "reimbursement",
        "compensation",
        "file",
        "case",
        "processing",
        "delay",
        "expert",
        "approval",
        "refused",
        "refund",
        "accident",
        "damage",
    ],
    "Cancellation": [
        "cancel",
        "termination",
        "terminate",
        "resiliation",
        "unsubscribe",
        "close contract",
        "notice period",
        "withdrawal",
        "stop contract",
        "switch insurer",
    ],
    "Other": ["other", "general", "miscellaneous"],
}


THEME_DESCRIPTIONS: dict[str, str] = {
    "Pricing": "Discussion about prices, premiums, costs, billing, value for money and tariff changes.",
    "Coverage": "Discussion about policy terms, coverage limits, guarantees, exclusions and protections.",
    "Enrollment": "Discussion about subscription, onboarding, registration, membership and signup paperwork.",
    "Customer Service": "Discussion about interactions with advisors, responsiveness, communication quality and support.",
    "Claims Processing": "Discussion about claims, reimbursements, delays, refusals, compensation and case handling.",
    "Cancellation": "Discussion about cancellation, termination procedures, notice periods and contract closure.",
    "Other": "Reviews that do not clearly fit one major insurance theme.",
}


@dataclass
class ThemePipelineArtifacts:
    """Container for theme detection outputs and reusable models."""

    scored_df: pd.DataFrame
    metadata: dict[str, Any]
    vectorizer: TfidfVectorizer
    prototype_matrix: np.ndarray
    topic_vectorizer: TfidfVectorizer
    nmf_model: NMF
    topic_to_theme: dict[int, str]


def _normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = str(text).lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _theme_keyword_patterns(theme_lexicon: dict[str, list[str]]) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for theme, keywords in theme_lexicon.items():
        escaped: list[str] = []
        for keyword in keywords:
            keyword = keyword.strip().lower()
            if not keyword:
                continue
            escaped_keyword = re.escape(keyword).replace(r"\ ", r"\s+")
            escaped.append(escaped_keyword)
        pattern = r"\b(?:%s)\b" % "|".join(sorted(set(escaped)))
        patterns[theme] = re.compile(pattern, flags=re.IGNORECASE)
    return patterns


def _keyword_score_matrix(texts: pd.Series, theme_lexicon: dict[str, list[str]]) -> pd.DataFrame:
    patterns = _theme_keyword_patterns(theme_lexicon)
    word_len = texts.str.split().str.len().replace(0, 1).astype(float)
    scores = {}
    for theme, pattern in patterns.items():
        count = texts.str.count(pattern).astype(float)
        scores[theme] = count / np.sqrt(word_len)
    keyword_df = pd.DataFrame(scores).fillna(0.0)
    return keyword_df


def _prototype_similarity(
    texts: pd.Series,
    theme_lexicon: dict[str, list[str]],
    theme_descriptions: dict[str, str],
) -> tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    themes = list(theme_lexicon.keys())
    prototype_docs: list[str] = []
    for theme in themes:
        lexicon = " ".join(theme_lexicon[theme])
        desc = theme_descriptions.get(theme, "")
        prototype_docs.append(f"{theme} {desc} {lexicon}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=50000,
        stop_words="english",
    )
    corpus_for_fit = pd.concat([texts, pd.Series(prototype_docs)], ignore_index=True)
    vectorizer.fit(corpus_for_fit)
    text_matrix = vectorizer.transform(texts)
    prototype_matrix = vectorizer.transform(prototype_docs)
    sims = cosine_similarity(text_matrix, prototype_matrix)
    sim_df = pd.DataFrame(sims, columns=themes)
    return sim_df, vectorizer, prototype_matrix


def _topic_signal_from_nmf(
    texts: pd.Series,
    theme_lexicon: dict[str, list[str]],
    n_topics: int = 10,
) -> tuple[pd.DataFrame, TfidfVectorizer, NMF, dict[int, str], pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=10,
        max_df=0.9,
        max_features=20000,
        stop_words="english",
    )
    text_matrix = vectorizer.fit_transform(texts)
    nmf_model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)
    doc_topic = nmf_model.fit_transform(text_matrix)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_to_theme: dict[int, str] = {}
    topic_rows: list[dict[str, Any]] = []

    for topic_idx, topic_weights in enumerate(nmf_model.components_):
        top_term_indices = topic_weights.argsort()[::-1][:20]
        top_terms = feature_names[top_term_indices]
        top_term_set = set(top_terms.tolist())

        best_theme = "Other"
        best_overlap = -1
        for theme, keywords in theme_lexicon.items():
            normalized_keywords = {kw.lower() for kw in keywords}
            overlap = len(top_term_set.intersection(normalized_keywords))
            if overlap > best_overlap:
                best_overlap = overlap
                best_theme = theme

        topic_to_theme[topic_idx] = best_theme
        topic_rows.append(
            {
                "topic_id": topic_idx,
                "mapped_theme": best_theme,
                "keyword_overlap": best_overlap,
                "top_terms": ", ".join(top_terms[:12]),
            }
        )

    themes = list(theme_lexicon.keys())
    theme_topic_scores = pd.DataFrame(0.0, index=np.arange(len(texts)), columns=themes)
    for topic_idx, mapped_theme in topic_to_theme.items():
        theme_topic_scores[mapped_theme] += doc_topic[:, topic_idx]

    row_sum = theme_topic_scores.sum(axis=1).replace(0.0, 1.0)
    theme_topic_scores = theme_topic_scores.div(row_sum, axis=0)

    topic_mapping_df = pd.DataFrame(topic_rows)
    return theme_topic_scores, vectorizer, nmf_model, topic_to_theme, topic_mapping_df


def _rowwise_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
    max_values = df.max(axis=1).replace(0.0, 1.0)
    return df.div(max_values, axis=0)


def run_hybrid_theme_pipeline(
    clean_df: pd.DataFrame,
    text_col: str = "text_for_deep_learning",
    weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
) -> ThemePipelineArtifacts:
    """Run hybrid theme detection with keyword, prototype, and topic signals."""
    df = clean_df.copy()
    texts = df[text_col].map(_normalize_text)
    themes = list(THEME_LEXICON.keys())

    keyword_scores = _keyword_score_matrix(texts, THEME_LEXICON)
    prototype_scores, tfidf_vectorizer, prototype_matrix = _prototype_similarity(
        texts=texts,
        theme_lexicon=THEME_LEXICON,
        theme_descriptions=THEME_DESCRIPTIONS,
    )
    topic_scores, topic_vectorizer, nmf_model, topic_to_theme, topic_mapping_df = _topic_signal_from_nmf(
        texts=texts,
        theme_lexicon=THEME_LEXICON,
        n_topics=10,
    )

    keyword_norm = _rowwise_max_normalize(keyword_scores)
    prototype_norm = _rowwise_max_normalize(prototype_scores.clip(lower=0.0))
    topic_norm = _rowwise_max_normalize(topic_scores)

    w_proto, w_keyword, w_topic = weights
    fused = (w_proto * prototype_norm) + (w_keyword * keyword_norm) + (w_topic * topic_norm)

    max_score = fused.max(axis=1)
    primary_theme = fused.idxmax(axis=1)
    primary_theme = np.where(max_score < 0.25, "Other", primary_theme)

    sorted_scores = np.sort(fused.values, axis=1)
    second_best = sorted_scores[:, -2]
    second_theme = fused.apply(lambda row: row.nlargest(2).index[-1], axis=1)
    secondary_theme = np.where(
        (second_best >= 0.80 * max_score) & ((max_score - second_best) <= 0.18) & (second_best >= 0.25),
        second_theme,
        "",
    )

    df["theme_primary"] = primary_theme
    df["theme_confidence"] = max_score.round(4)
    df["theme_secondary"] = secondary_theme
    for theme in themes:
        df[f"theme_score_{theme.lower().replace(' ', '_')}"] = fused[theme].round(6)

    def _top_scores_to_json(row: pd.Series, top_k: int = 3) -> str:
        ordered = row.sort_values(ascending=False).head(top_k)
        payload = {k: float(v) for k, v in ordered.items()}
        return json.dumps(payload, ensure_ascii=False)

    df["theme_top3_scores_json"] = fused.apply(_top_scores_to_json, axis=1)

    metadata: dict[str, Any] = {
        "themes": themes,
        "weights": {"prototype": w_proto, "keyword": w_keyword, "topic": w_topic},
        "topic_to_theme": {str(k): v for k, v in topic_to_theme.items()},
        "text_column_used": text_col,
    }
    metadata["topic_mapping_table"] = topic_mapping_df.to_dict(orient="records")

    return ThemePipelineArtifacts(
        scored_df=df,
        metadata=metadata,
        vectorizer=tfidf_vectorizer,
        prototype_matrix=prototype_matrix,
        topic_vectorizer=topic_vectorizer,
        nmf_model=nmf_model,
        topic_to_theme=topic_to_theme,
    )


def export_theme_artifacts(
    artifacts: ThemePipelineArtifacts,
    model_path: Path,
    metadata_path: Path,
) -> None:
    """Persist reusable theme components for inference in Streamlit."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "theme_lexicon": THEME_LEXICON,
        "theme_descriptions": THEME_DESCRIPTIONS,
        "themes": artifacts.metadata["themes"],
        "weights": artifacts.metadata["weights"],
        "vectorizer": artifacts.vectorizer,
        "prototype_matrix": artifacts.prototype_matrix,
        "topic_vectorizer": artifacts.topic_vectorizer,
        "nmf_model": artifacts.nmf_model,
        "topic_to_theme": artifacts.topic_to_theme,
    }
    joblib.dump(payload, model_path)
    metadata_path.write_text(json.dumps(artifacts.metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_theme_reports(
    scored_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    reports_dir: Path,
) -> dict[str, str]:
    """Generate theme-level visualizations and summary tables."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    train_df = scored_df[scored_df["type"].eq("train")].copy()
    theme_order = (
        train_df["theme_primary"].value_counts().sort_values(ascending=False).index.tolist()
    )

    theme_counts = (
        train_df["theme_primary"].value_counts().rename_axis("theme_primary").reset_index(name="review_count")
    )
    theme_counts["share_pct"] = (theme_counts["review_count"] / len(train_df) * 100).round(2)
    theme_counts.to_csv(tables_dir / "phase3_theme_distribution_train.csv", index=False, encoding="utf-8-sig")

    mean_star_theme = (
        train_df.groupby("theme_primary", as_index=False)["note"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "review_count", "mean": "mean_star"})
    )
    mean_star_theme.to_csv(tables_dir / "phase3_mean_star_by_theme.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=theme_counts, x="theme_primary", y="review_count", order=theme_order, palette="Set2", ax=ax)
    ax.set_title("Theme Distribution (Train)")
    ax.set_xlabel("Detected Theme")
    ax.set_ylabel("Review Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_theme_distribution_train.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=mean_star_theme, x="theme_primary", y="mean_star", order=theme_order, palette="viridis", ax=ax)
    ax.set_title("Average Star by Detected Theme (Train)")
    ax.set_xlabel("Detected Theme")
    ax.set_ylabel("Average Stars")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_mean_star_by_theme.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top_insurers = train_df["assureur"].value_counts().head(12).index
    heat_df = (
        train_df[train_df["assureur"].isin(top_insurers)]
        .groupby(["assureur", "theme_primary"])
        .size()
        .reset_index(name="count")
    )
    pivot = heat_df.pivot(index="assureur", columns="theme_primary", values="count").fillna(0)
    pivot.to_csv(tables_dir / "phase3_insurer_theme_count_matrix.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot, cmap="YlGnBu", annot=False, ax=ax)
    ax.set_title("Theme Volume by Top Insurers (Train)")
    ax.set_xlabel("Theme")
    ax.set_ylabel("Insurer")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase3_insurer_theme_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Top terms per theme from classified reviews.
    rows: list[pd.DataFrame] = []
    for theme, subset in train_df.groupby("theme_primary"):
        texts = subset["text_for_classical_ml"].fillna("").astype(str)
        texts = texts[texts.str.strip().ne("")]
        if len(texts) < 40:
            continue
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5)
        matrix = vectorizer.fit_transform(texts)
        freqs = matrix.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        top_df = (
            pd.DataFrame({"theme_primary": theme, "term": vocab, "frequency": freqs})
            .sort_values("frequency", ascending=False)
            .head(20)
        )
        rows.append(top_df)
    top_terms_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    top_terms_df.to_csv(tables_dir / "phase3_top_terms_by_theme.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        "# Phase 3 Theme Detection Summary",
        "",
        f"- Train reviews analyzed: {len(train_df):,}.",
        f"- Dominant theme: {theme_counts.iloc[0]['theme_primary']} ({int(theme_counts.iloc[0]['review_count']):,} reviews).",
        "- Hybrid scoring used: prototype similarity + keyword coverage + NMF topic signal.",
    ]
    summary_path = reports_dir / "phase3_theme_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {"theme_summary_report": str(summary_path)}
