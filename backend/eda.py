"""Generate high-value EDA figures and tables for grading."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_star_distribution(train_df: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    counts = train_df["note"].value_counts().sort_index()
    percentages = (counts / counts.sum() * 100).round(2)
    table = pd.DataFrame({"count": counts, "percentage": percentages})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=table.index.astype(str), y=table["count"], color="#2b8cbe", ax=ax)
    ax.set_title("Star Distribution (Train Split)")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Number of Reviews")
    for idx, (count, pct) in enumerate(zip(table["count"], table["percentage"])):
        ax.text(idx, count, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    _save_plot(fig, figures_dir / "eda_star_distribution.png")
    return table.reset_index(names="note")


def _plot_top_insurers(train_df: pd.DataFrame, figures_dir: Path, top_n: int = 15) -> pd.DataFrame:
    insurer_stats = (
        train_df.groupby("assureur", as_index=False)
        .agg(review_count=("note", "size"), mean_star=("note", "mean"))
        .sort_values("review_count", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=insurer_stats, y="assureur", x="review_count", palette="Blues_r", ax=ax)
    ax.set_title(f"Top {top_n} Insurers by Review Volume")
    ax.set_xlabel("Review Count")
    ax.set_ylabel("Insurer")
    _save_plot(fig, figures_dir / "eda_top_insurers_review_volume.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    ordered = insurer_stats.sort_values("mean_star", ascending=False)
    sns.barplot(data=ordered, y="assureur", x="mean_star", palette="viridis", ax=ax)
    ax.set_title(f"Average Star Rating for Top {top_n} Insurers")
    ax.set_xlabel("Average Stars")
    ax.set_ylabel("Insurer")
    ax.set_xlim(0, 5)
    _save_plot(fig, figures_dir / "eda_top_insurers_mean_stars.png")

    return insurer_stats


def _plot_product_distribution(train_df: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    product_stats = (
        train_df.groupby("produit", as_index=False)
        .agg(review_count=("note", "size"), mean_star=("note", "mean"))
        .sort_values("review_count", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=product_stats, x="produit", y="review_count", palette="mako", ax=ax)
    ax.set_title("Review Volume by Insurance Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Review Count")
    ax.tick_params(axis="x", rotation=40)
    _save_plot(fig, figures_dir / "eda_product_review_volume.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=product_stats, x="produit", y="mean_star", palette="crest", ax=ax)
    ax.set_title("Average Stars by Insurance Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Average Stars")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=40)
    _save_plot(fig, figures_dir / "eda_product_mean_stars.png")
    return product_stats


def _plot_review_length(train_df: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    length_stats = (
        train_df.groupby("note", as_index=False)["text_word_len"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=train_df, x="note", y="text_word_len", color="#99d8c9", showfliers=False, ax=ax)
    ax.set_title("Review Length Distribution by Star Rating")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Words per Review")
    _save_plot(fig, figures_dir / "eda_review_length_by_star_boxplot.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=length_stats, x="note", y="mean", color="#fe9929", ax=ax)
    ax.set_title("Average Review Length by Star Rating")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Average Number of Words")
    _save_plot(fig, figures_dir / "eda_review_length_mean_by_star.png")
    return length_stats


def _build_top_ngrams(
    texts: pd.Series,
    ngram_range: tuple[int, int],
    top_n: int,
) -> pd.DataFrame:
    texts = texts.fillna("").astype(str)
    texts = texts[texts.str.strip().ne("")]
    if texts.empty:
        return pd.DataFrame(columns=["ngram", "frequency"])

    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=5)
    matrix = vectorizer.fit_transform(texts)
    freqs = matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    ngrams_df = (
        pd.DataFrame({"ngram": vocab, "frequency": freqs})
        .sort_values("frequency", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return ngrams_df


def _build_top_ngrams_by_sentiment(
    train_df: pd.DataFrame,
    text_col: str,
    ngram_range: tuple[int, int],
    top_n: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sentiment, subset in train_df.groupby("sentiment_label"):
        texts = subset[text_col].fillna("").astype(str)
        texts = texts[texts.str.strip().ne("")]
        if len(texts) < 20:
            continue
        top = _build_top_ngrams(texts=texts, ngram_range=ngram_range, top_n=top_n)
        top.insert(0, "sentiment_label", sentiment)
        rows.append(top)
    if not rows:
        return pd.DataFrame(columns=["sentiment_label", "ngram", "frequency"])
    return pd.concat(rows, ignore_index=True)


def _plot_sentiment_distribution(train_df: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    sentiment_counts = (
        train_df["sentiment_label"].value_counts().rename_axis("sentiment_label").reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=sentiment_counts, x="sentiment_label", y="count", palette="Set2", ax=ax)
    ax.set_title("Derived Sentiment Distribution (Train)")
    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Number of Reviews")
    _save_plot(fig, figures_dir / "eda_sentiment_distribution.png")
    return sentiment_counts


def run_eda(clean_df: pd.DataFrame, figures_dir: Path, tables_dir: Path, reports_dir: Path) -> dict[str, str]:
    """Execute core EDA and save outputs."""
    _ensure_dirs(figures_dir, tables_dir, reports_dir)
    sns.set_theme(style="whitegrid")

    train_df = clean_df[clean_df["type"].eq("train")].copy()

    star_table = _plot_star_distribution(train_df, figures_dir)
    insurer_table = _plot_top_insurers(train_df, figures_dir, top_n=15)
    product_table = _plot_product_distribution(train_df, figures_dir)
    length_table = _plot_review_length(train_df, figures_dir)
    sentiment_table = _plot_sentiment_distribution(train_df, figures_dir)

    unigram_global = _build_top_ngrams(train_df["text_for_classical_ml"], ngram_range=(1, 1), top_n=40)
    bigram_global = _build_top_ngrams(train_df["text_for_classical_ml"], ngram_range=(2, 2), top_n=40)
    trigram_global = _build_top_ngrams(train_df["text_for_classical_ml"], ngram_range=(3, 3), top_n=30)
    unigram_by_sentiment = _build_top_ngrams_by_sentiment(
        train_df=train_df,
        text_col="text_for_classical_ml",
        ngram_range=(1, 1),
        top_n=25,
    )
    bigram_by_sentiment = _build_top_ngrams_by_sentiment(
        train_df=train_df,
        text_col="text_for_classical_ml",
        ngram_range=(2, 2),
        top_n=20,
    )

    star_table.to_csv(tables_dir / "eda_star_distribution.csv", index=False, encoding="utf-8-sig")
    insurer_table.to_csv(tables_dir / "eda_insurer_metrics_top15.csv", index=False, encoding="utf-8-sig")
    product_table.to_csv(tables_dir / "eda_product_metrics.csv", index=False, encoding="utf-8-sig")
    length_table.to_csv(tables_dir / "eda_review_length_by_star.csv", index=False, encoding="utf-8-sig")
    sentiment_table.to_csv(tables_dir / "eda_sentiment_distribution.csv", index=False, encoding="utf-8-sig")
    unigram_global.to_csv(tables_dir / "eda_top_unigrams_global.csv", index=False, encoding="utf-8-sig")
    bigram_global.to_csv(tables_dir / "eda_top_bigrams_global.csv", index=False, encoding="utf-8-sig")
    trigram_global.to_csv(tables_dir / "eda_top_trigrams_global.csv", index=False, encoding="utf-8-sig")
    unigram_by_sentiment.to_csv(
        tables_dir / "eda_top_unigrams_by_sentiment.csv", index=False, encoding="utf-8-sig"
    )
    bigram_by_sentiment.to_csv(
        tables_dir / "eda_top_bigrams_by_sentiment.csv", index=False, encoding="utf-8-sig"
    )

    key_insights = [
        f"Train split size: {len(train_df):,} reviews.",
        f"Most represented star class: {int(star_table.loc[star_table['count'].idxmax(), 'note'])} stars.",
        f"Largest insurer by volume: {insurer_table.iloc[0]['assureur']} ({int(insurer_table.iloc[0]['review_count'])} reviews).",
        (
            f"Review length trend: mean words drops from "
            f"{length_table.loc[length_table['note'] == 1.0, 'mean'].values[0]:.1f} (1-star) to "
            f"{length_table.loc[length_table['note'] == 5.0, 'mean'].values[0]:.1f} (5-star)."
        ),
    ]

    summary_path = reports_dir / "phase2_eda_summary.md"
    summary_text = "# Phase 2 EDA Summary\n\n" + "\n".join(f"- {item}" for item in key_insights) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")

    return {
        "summary_report": str(summary_path),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }
