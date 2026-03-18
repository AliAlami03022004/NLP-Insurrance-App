"""Phase 5 error analysis and interpretation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _text_to_avg_word2vec_vectors(texts: pd.Series, w2v_model: Word2Vec) -> np.ndarray:
    dim = w2v_model.wv.vector_size
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for idx, text in enumerate(texts.fillna("").astype(str).tolist()):
        tokens = [tok for tok in text.split() if tok in w2v_model.wv]
        if tokens:
            vectors[idx] = np.mean([w2v_model.wv[tok] for tok in tokens], axis=0)
    return vectors


def _prepare_task_features(
    feature_family: str,
    train_df: pd.DataFrame,
    review_embeddings: np.ndarray,
    word2vec_model: Word2Vec,
) -> Any:
    if feature_family == "tfidf_text":
        return train_df["text_for_classical_ml"].fillna("").astype(str)
    if feature_family == "word2vec_dense":
        return _text_to_avg_word2vec_vectors(train_df["text_for_classical_ml"], word2vec_model)
    return review_embeddings


def _prediction_confidence(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs.max(axis=1)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim == 1:
            return np.abs(decision)
        return np.max(decision, axis=1)
    return np.full(shape=(len(X),), fill_value=np.nan, dtype=float)


def _build_confusion_pairs(y_true: np.ndarray, y_pred: np.ndarray, labels: list[Any], top_k: int = 6) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: list[dict[str, Any]] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append({"true_label": true_label, "pred_label": pred_label, "count": count})
    out = pd.DataFrame(pairs).sort_values("count", ascending=False).head(top_k)
    return out


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def _format_top_terms(contributions: list[dict[str, Any]], top_n: int = 8) -> str:
    if not contributions:
        return ""
    clipped = contributions[:top_n]
    return " | ".join(f"{item['term']} ({item['contribution']:.4f})" for item in clipped)


def _shap_lite_token_contributions(
    text_for_classical_ml: str,
    tfidf_logreg_model: Any,
    target_label: Any | None = None,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """Compute lightweight per-token contributions: tfidf_value * class_coefficient."""
    vectorizer = tfidf_logreg_model.named_steps["tfidf"]
    clf = tfidf_logreg_model.named_steps["clf"]
    x = vectorizer.transform([text_for_classical_ml])

    if target_label is None:
        target_label = tfidf_logreg_model.predict([text_for_classical_ml])[0]

    class_to_idx = {label: idx for idx, label in enumerate(clf.classes_)}
    class_idx = class_to_idx[target_label]

    feature_names = vectorizer.get_feature_names_out()
    row = x.tocoo()
    contributions: list[dict[str, Any]] = []
    for col, value in zip(row.col, row.data):
        coef = clf.coef_[class_idx, col]
        contrib = float(value * coef)
        if contrib > 0:
            contributions.append({"term": feature_names[col], "contribution": contrib})
    return sorted(contributions, key=lambda d: d["contribution"], reverse=True)[:top_n]


def _export_shap_lite_examples(
    task_df: pd.DataFrame,
    task_name: str,
    tfidf_logreg_model: Any,
    tables_dir: Path,
) -> Path:
    """Export SHAP-lite examples from high-confidence successes and failures."""
    failures = task_df[task_df["is_correct"].eq(0)].sort_values("confidence", ascending=False).head(40)
    corrects = task_df[task_df["is_correct"].eq(1)].sort_values("confidence", ascending=False).head(40)
    sample_df = pd.concat([failures, corrects], axis=0).reset_index(drop=True)
    sample_df["example_type"] = np.where(sample_df["is_correct"].eq(1), "good_prediction", "failure")

    rows: list[dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        text_for_model = str(row.get("text_for_classical_ml", "") or "")
        terms = _shap_lite_token_contributions(
            text_for_classical_ml=text_for_model,
            tfidf_logreg_model=tfidf_logreg_model,
            target_label=row["pred_label"],
            top_n=12,
        )
        rows.append(
            {
                "example_type": row["example_type"],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "confidence": float(row.get("confidence", np.nan)),
                "theme_primary": row.get("theme_primary", ""),
                "assureur": row.get("assureur", ""),
                "text_clean_corrected": row.get("text_clean_corrected", ""),
                "shap_lite_terms": _format_top_terms(terms, top_n=10),
            }
        )

    out_path = tables_dir / f"phase5_shap_lite_examples_{task_name}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def run_anomaly_detection(
    phase3_df: pd.DataFrame,
    review_embeddings: np.ndarray,
    tables_dir: Path,
    figures_dir: Path,
    contamination: float = 0.02,
) -> dict[str, Any]:
    """Detect atypical reviews with an unsupervised ensemble (IForest + LOF + kNN distance)."""
    base_df = phase3_df.reset_index(drop=True).copy()
    emb = np.asarray(review_embeddings, dtype=np.float32)
    if len(base_df) != len(emb):
        n = min(len(base_df), len(emb))
        base_df = base_df.iloc[:n].copy()
        emb = emb[:n]

    text_len = base_df["text_clean_corrected"].fillna("").astype(str).str.split().str.len().to_numpy(dtype=float)

    iforest = IsolationForest(
        n_estimators=220,
        contamination=contamination,
        random_state=42,
        n_jobs=1,
    )
    iforest.fit(emb)
    iforest_score = -iforest.score_samples(emb)

    pca_dim = min(50, emb.shape[1])
    emb_reduced = PCA(n_components=pca_dim, random_state=42).fit_transform(emb)
    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination=contamination,
        novelty=True,
        metric="euclidean",
    )
    lof.fit(emb_reduced)
    lof_score = -lof.decision_function(emb_reduced)

    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6)
    knn.fit(emb)
    knn_dist, _ = knn.kneighbors(emb, n_neighbors=6)
    knn_score = knn_dist[:, -1]

    iforest_pct = _percentile_rank(iforest_score)
    lof_pct = _percentile_rank(lof_score)
    knn_pct = _percentile_rank(knn_score)
    text_len_pct = _percentile_rank(text_len)

    ensemble_score = 0.45 * iforest_pct + 0.30 * lof_pct + 0.20 * knn_pct + 0.05 * text_len_pct
    threshold = float(np.quantile(ensemble_score, 0.98))
    is_anomaly = ensemble_score >= threshold

    out_df = pd.DataFrame(
        {
            "row_id": base_df.get("row_id", pd.Series(np.arange(len(base_df)))).astype(str),
            "type": base_df.get("type", ""),
            "assureur": base_df.get("assureur", ""),
            "produit": base_df.get("produit", ""),
            "note": base_df.get("note", np.nan),
            "sentiment_label": base_df.get("sentiment_label", ""),
            "theme_primary": base_df.get("theme_primary", ""),
            "text_word_len": text_len,
            "iforest_score": iforest_score,
            "lof_score": lof_score,
            "knn_distance_score": knn_score,
            "ensemble_anomaly_score": ensemble_score,
            "is_anomaly": is_anomaly.astype(int),
            "text_clean_corrected": base_df.get("text_clean_corrected", "").astype(str).str.slice(0, 360),
        }
    ).sort_values("ensemble_anomaly_score", ascending=False)

    scores_path = tables_dir / "phase5_anomaly_scores.csv"
    top_path = tables_dir / "phase5_top_anomalies.csv"
    out_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
    out_df.head(250).to_csv(top_path, index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(out_df["ensemble_anomaly_score"], bins=40, kde=True, color="#d7301f", ax=ax)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Anomaly Score Distribution (Unsupervised Ensemble)")
    ax.set_xlabel("Ensemble Anomaly Score")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_anomaly_score_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sample_plot = out_df.sample(min(4000, len(out_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=sample_plot,
        x="text_word_len",
        y="ensemble_anomaly_score",
        hue="is_anomaly",
        alpha=0.55,
        s=18,
        palette={0: "#74a9cf", 1: "#cb181d"},
        ax=ax,
    )
    ax.set_title("Anomaly Score vs Review Length")
    ax.set_xlabel("Review Length (words)")
    ax.set_ylabel("Ensemble Anomaly Score")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_anomaly_length_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    anomalies_df = out_df[out_df["is_anomaly"].eq(1)].copy()
    top_themes = anomalies_df["theme_primary"].astype(str).value_counts().head(5).to_dict()
    top_insurers = anomalies_df["assureur"].astype(str).value_counts().head(5).to_dict()

    return {
        "scores_path": str(scores_path),
        "top_anomalies_path": str(top_path),
        "anomaly_count": int(anomalies_df.shape[0]),
        "anomaly_rate": float(anomalies_df.shape[0] / max(1, out_df.shape[0])),
        "top_themes": top_themes,
        "top_insurers": top_insurers,
    }


def _save_task_plots(task_df: pd.DataFrame, task_name: str, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=task_df, x="confidence", hue="is_correct", bins=30, stat="density", common_norm=False, ax=ax)
    ax.set_title(f"{task_name.capitalize()} - Confidence Distribution (Correct vs Error)")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(figures_dir / f"phase5_confidence_distribution_{task_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    class_error = task_df.groupby("true_label", as_index=False)["is_correct"].mean()
    class_error["error_rate"] = 1.0 - class_error["is_correct"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=class_error, x="true_label", y="error_rate", color="#f03b20", ax=ax)
    ax.set_title(f"{task_name.capitalize()} - Error Rate by True Class")
    ax.set_xlabel("True Label")
    ax.set_ylabel("Error Rate")
    fig.tight_layout()
    fig.savefig(figures_dir / f"phase5_error_rate_by_class_{task_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=task_df, x="is_correct", y="text_word_len", showfliers=False, ax=ax)
    ax.set_title(f"{task_name.capitalize()} - Text Length vs Correctness")
    ax.set_xlabel("Correct Prediction (1=yes)")
    ax.set_ylabel("Review Length (words)")
    fig.tight_layout()
    fig.savefig(figures_dir / f"phase5_length_vs_correctness_{task_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_phase5_error_analysis(
    phase3_df: pd.DataFrame,
    review_embeddings: np.ndarray,
    models_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, str]:
    """Generate grading-oriented error analysis for stars and sentiment tasks."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_source = phase3_df[phase3_df["type"].eq("train")].copy()
    train_positions = train_source.index.to_numpy()
    train_df = train_source.reset_index(drop=True)
    train_embeddings = review_embeddings[train_positions]
    train_df["text_word_len"] = train_df["text_clean_corrected"].fillna("").astype(str).str.split().str.len()

    word2vec_model = Word2Vec.load(str(models_dir.parent / "embeddings" / "word2vec.model"))

    task_configs = [
        ("stars", train_df["note"].astype(int)),
        ("sentiment", train_df["sentiment_label"].astype(str)),
    ]
    shap_lite_paths: dict[str, str] = {}

    report_lines: list[str] = ["# Phase 5 Error Analysis and Interpretation", ""]

    for task_name, y_full in task_configs:
        split_info = json.loads((reports_dir / f"phase4_split_and_best_{task_name}.json").read_text(encoding="utf-8"))
        comparison_df = pd.read_csv(tables_dir / f"phase4_model_comparison_{task_name}.csv")
        best_row = comparison_df.iloc[0]
        best_model_name = str(best_row["model_name"])
        feature_family = str(best_row["feature_family"])

        idx_test = np.array(split_info["idx_test"], dtype=int)
        idx_val = np.array(split_info["idx_val"], dtype=int)

        best_model_path = models_dir / f"{task_name}_best_model.joblib"
        best_model = joblib.load(best_model_path)

        X_all = _prepare_task_features(
            feature_family=feature_family,
            train_df=train_df,
            review_embeddings=train_embeddings,
            word2vec_model=word2vec_model,
        )

        y_true_test = y_full.iloc[idx_test].to_numpy()
        y_pred_test = best_model.predict(X_all[idx_test] if not isinstance(X_all, pd.Series) else X_all.iloc[idx_test])
        y_true_val = y_full.iloc[idx_val].to_numpy()
        y_pred_val = best_model.predict(X_all[idx_val] if not isinstance(X_all, pd.Series) else X_all.iloc[idx_val])

        confidence_test = _prediction_confidence(
            best_model,
            X_all[idx_test] if not isinstance(X_all, pd.Series) else X_all.iloc[idx_test],
        )

        test_rows = train_df.iloc[idx_test].copy().reset_index(drop=True)
        task_df = pd.DataFrame(
            {
                "true_label": y_true_test,
                "pred_label": y_pred_test,
                "is_correct": (y_true_test == y_pred_test).astype(int),
                "confidence": confidence_test,
                "text_word_len": test_rows["text_word_len"].values,
                "text_for_classical_ml": test_rows["text_for_classical_ml"].values,
                "theme_primary": test_rows.get("theme_primary", "").values,
                "assureur": test_rows["assureur"].values,
                "produit": test_rows["produit"].values,
                "text_clean_corrected": test_rows["text_clean_corrected"].values,
            }
        )
        task_df["error_type"] = np.where(task_df["is_correct"] == 1, "correct", "error")

        task_df.to_csv(tables_dir / f"phase5_predictions_{task_name}_test.csv", index=False, encoding="utf-8-sig")
        task_df[task_df["is_correct"] == 1].sort_values("confidence", ascending=False).head(120).to_csv(
            tables_dir / f"phase5_good_predictions_{task_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        task_df[task_df["is_correct"] == 0].sort_values("confidence", ascending=False).head(180).to_csv(
            tables_dir / f"phase5_failure_examples_{task_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        error_by_theme = (
            task_df.groupby("theme_primary", as_index=False)["is_correct"].mean().rename(columns={"is_correct": "accuracy"})
        )
        error_by_theme["error_rate"] = 1.0 - error_by_theme["accuracy"]
        error_by_theme.to_csv(tables_dir / f"phase5_error_by_theme_{task_name}.csv", index=False, encoding="utf-8-sig")

        bins = [0, 20, 40, 80, 160, 5000]
        labels = ["0-20", "21-40", "41-80", "81-160", "160+"]
        task_df["length_bin"] = pd.cut(task_df["text_word_len"], bins=bins, labels=labels, include_lowest=True)
        err_len = (
            task_df.groupby("length_bin", as_index=False, observed=False)["is_correct"]
            .mean()
            .rename(columns={"is_correct": "accuracy"})
        )
        err_len["error_rate"] = 1.0 - err_len["accuracy"]
        err_len.to_csv(tables_dir / f"phase5_error_by_length_bin_{task_name}.csv", index=False, encoding="utf-8-sig")

        confusion_pairs = _build_confusion_pairs(
            y_true=y_true_test,
            y_pred=y_pred_test,
            labels=sorted(np.unique(y_true_test).tolist()),
            top_k=8,
        )
        confusion_pairs.to_csv(
            tables_dir / f"phase5_top_confusions_{task_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        _save_task_plots(task_df=task_df, task_name=task_name, figures_dir=figures_dir)

        tfidf_explainer_path = models_dir / f"{task_name}_tfidf_logreg_model.joblib"
        if tfidf_explainer_path.exists():
            tfidf_explainer = joblib.load(tfidf_explainer_path)
            shap_path = _export_shap_lite_examples(
                task_df=task_df,
                task_name=task_name,
                tfidf_logreg_model=tfidf_explainer,
                tables_dir=tables_dir,
            )
            shap_lite_paths[task_name] = str(shap_path)

        tfidf_feat_path = tables_dir / f"phase4_top_features_{task_name}_tfidf_logreg.csv"
        top_feat_text = "n/a"
        if tfidf_feat_path.exists():
            feat_df = pd.read_csv(tfidf_feat_path)
            top_feat_text = ", ".join(feat_df.head(8)["term"].tolist())

        report_lines.extend(
            [
                f"## {task_name.capitalize()}",
                "",
                f"- Best model: `{best_model_name}` ({feature_family}).",
                f"- Validation F1-macro: {float(best_row['val_f1_macro']):.4f}.",
                f"- Test F1-macro: {float(best_row['test_f1_macro']):.4f}.",
                f"- Test accuracy: {float(best_row['test_accuracy']):.4f}.",
                f"- Most frequent confusions: {confusion_pairs.head(3).to_dict(orient='records')}.",
                f"- Indicative discriminative terms from TF-IDF Logistic Regression: {top_feat_text}.",
                f"- SHAP-lite examples: `{shap_lite_paths.get(task_name, 'not_available')}`.",
                "",
            ]
        )

    anomaly_outputs = run_anomaly_detection(
        phase3_df=phase3_df,
        review_embeddings=review_embeddings,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        contamination=0.02,
    )
    report_lines.extend(
        [
            "## Unsupervised Anomaly Detection",
            "",
            "- Ensemble methods: Isolation Forest, Local Outlier Factor, and cosine kNN distance.",
            f"- Flagged anomalies: {anomaly_outputs['anomaly_count']} ({anomaly_outputs['anomaly_rate']:.2%} of corpus).",
            f"- Top anomaly themes: {anomaly_outputs['top_themes']}.",
            f"- Top anomaly insurers: {anomaly_outputs['top_insurers']}.",
            f"- Scores table: `{anomaly_outputs['scores_path']}`.",
            f"- Top anomalies table: `{anomaly_outputs['top_anomalies_path']}`.",
            "",
        ]
    )

    report_path = reports_dir / "phase5_error_analysis.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "phase5_report": str(report_path),
        "shap_lite_examples": shap_lite_paths,
        "anomaly_scores_path": anomaly_outputs["scores_path"],
        "top_anomalies_path": anomaly_outputs["top_anomalies_path"],
    }
