"""Supervised modeling pipeline for stars and sentiment tasks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_SEED = 42


@dataclass
class TaskBundle:
    """Container for one supervised task."""

    task_name: str
    y: pd.Series
    labels: list[Any]
    is_multiclass: bool = True


def _build_splits(y: pd.Series, random_seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_seed,
        stratify=y,
    )
    y_train_val = y.iloc[idx_train_val]
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=0.20,
        random_state=random_seed,
        stratify=y_train_val,
    )
    return idx_train, idx_val, idx_test


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _train_and_evaluate_model(
    model_name: str,
    estimator: BaseEstimator,
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], BaseEstimator, np.ndarray, np.ndarray]:
    start = time.time()
    estimator.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred_val = estimator.predict(X_val)
    y_pred_test = estimator.predict(X_test)
    infer_time = time.time() - start

    val_metrics = _metrics_dict(y_true=y_val, y_pred=y_pred_val)
    test_metrics = _metrics_dict(y_true=y_test, y_pred=y_pred_test)
    row = {
        "model_name": model_name,
        "train_time_sec": round(train_time, 3),
        "inference_time_sec": round(infer_time, 3),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    return row, estimator, y_pred_val, y_pred_test


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
    title: str,
    path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_classical_text_models() -> dict[str, BaseEstimator]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "tfidf_logreg": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=70000),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=70000),
                ),
                ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, dual="auto")),
            ]
        ),
        "tfidf_multinomial_nb": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=70000),
                ),
                ("clf", MultinomialNB(alpha=0.5)),
            ]
        ),
    }


def _text_to_avg_word2vec_vectors(texts: pd.Series, w2v_model: Word2Vec) -> np.ndarray:
    dim = w2v_model.wv.vector_size
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for idx, text in enumerate(texts.fillna("").astype(str).tolist()):
        tokens = [tok for tok in text.split() if tok in w2v_model.wv]
        if not tokens:
            continue
        vectors[idx] = np.mean([w2v_model.wv[tok] for tok in tokens], axis=0)
    return vectors


def _embedding_models() -> dict[str, BaseEstimator]:
    return {
        "word2vec_logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "word2vec_mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=35,
            early_stopping=True,
            random_state=RANDOM_SEED,
        ),
        "semantic_embed_logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    }


def run_supervised_benchmark(
    phase3_df: pd.DataFrame,
    review_embeddings: np.ndarray,
    embedding_backend: str,
    word2vec_path: Path,
    models_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate multiple supervised models for stars and sentiment."""
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    train_df = phase3_df[phase3_df["type"].eq("train")].copy()
    train_positions = train_df.index.to_numpy()
    train_df = train_df.reset_index(drop=True)
    X_text = train_df["text_for_classical_ml"].fillna("").astype(str)
    X_sent_embed = review_embeddings[train_positions]

    w2v_model = Word2Vec.load(str(word2vec_path))
    X_w2v = _text_to_avg_word2vec_vectors(X_text, w2v_model)

    tasks = [
        TaskBundle(task_name="stars", y=train_df["note"].astype(int), labels=sorted(train_df["note"].astype(int).unique())),
        TaskBundle(
            task_name="sentiment",
            y=train_df["sentiment_label"].astype(str),
            labels=sorted(train_df["sentiment_label"].astype(str).unique()),
        ),
    ]

    task_outputs: dict[str, dict[str, Any]] = {}

    for task in tasks:
        y = task.y.reset_index(drop=True)
        idx_train, idx_val, idx_test = _build_splits(y)

        y_train = y.iloc[idx_train].to_numpy()
        y_val = y.iloc[idx_val].to_numpy()
        y_test = y.iloc[idx_test].to_numpy()

        benchmark_rows: list[dict[str, Any]] = []
        saved_models: dict[str, BaseEstimator] = {}
        predictions_cache: dict[str, dict[str, np.ndarray]] = {}

        classical_models = _build_classical_text_models()
        for model_name, estimator in classical_models.items():
            row, fitted_model, y_pred_val, y_pred_test = _train_and_evaluate_model(
                model_name=model_name,
                estimator=estimator,
                X_train=X_text.iloc[idx_train],
                y_train=y_train,
                X_val=X_text.iloc[idx_val],
                y_val=y_val,
                X_test=X_text.iloc[idx_test],
                y_test=y_test,
            )
            row["feature_family"] = "tfidf_text"
            benchmark_rows.append(row)
            saved_models[model_name] = fitted_model
            predictions_cache[model_name] = {"val": y_pred_val, "test": y_pred_test}

        for explain_model_name in ["tfidf_logreg", "tfidf_linear_svm"]:
            if explain_model_name in saved_models:
                joblib.dump(
                    saved_models[explain_model_name],
                    models_dir / f"{task.task_name}_{explain_model_name}_model.joblib",
                )

        # Export interpretable top features from linear TF-IDF models for error analysis/reporting.
        for linear_name in ["tfidf_logreg", "tfidf_linear_svm"]:
            if linear_name not in saved_models:
                continue
            linear_model = saved_models[linear_name]
            clf = linear_model.named_steps["clf"]
            tfidf = linear_model.named_steps["tfidf"]
            feature_names = np.array(tfidf.get_feature_names_out())
            coefs = clf.coef_
            feat_rows: list[dict[str, Any]] = []
            for class_idx, class_label in enumerate(clf.classes_):
                top_idx = np.argsort(coefs[class_idx])[::-1][:25]
                for rank, feat_idx in enumerate(top_idx, start=1):
                    feat_rows.append(
                        {
                            "model_name": linear_name,
                            "class_label": class_label,
                            "rank": rank,
                            "term": feature_names[feat_idx],
                            "coefficient": float(coefs[class_idx][feat_idx]),
                        }
                    )
            pd.DataFrame(feat_rows).to_csv(
                tables_dir / f"phase4_top_features_{task.task_name}_{linear_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        embedding_models = _embedding_models()
        for model_name, estimator in embedding_models.items():
            if model_name.startswith("word2vec"):
                X_matrix = X_w2v
                family = "word2vec_dense"
            else:
                X_matrix = X_sent_embed
                family = f"sentence_embedding_{embedding_backend}"

            row, fitted_model, y_pred_val, y_pred_test = _train_and_evaluate_model(
                model_name=model_name,
                estimator=estimator,
                X_train=X_matrix[idx_train],
                y_train=y_train,
                X_val=X_matrix[idx_val],
                y_val=y_val,
                X_test=X_matrix[idx_test],
                y_test=y_test,
            )
            row["feature_family"] = family
            benchmark_rows.append(row)
            saved_models[model_name] = fitted_model
            predictions_cache[model_name] = {"val": y_pred_val, "test": y_pred_test}

        results_df = pd.DataFrame(benchmark_rows).sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
        results_path = tables_dir / f"phase4_model_comparison_{task.task_name}.csv"
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

        best_model_name = results_df.iloc[0]["model_name"]
        best_model = saved_models[best_model_name]
        best_pred_test = predictions_cache[best_model_name]["test"]
        best_pred_val = predictions_cache[best_model_name]["val"]

        model_save_path = models_dir / f"{task.task_name}_best_model.joblib"
        joblib.dump(best_model, model_save_path)

        _plot_confusion_matrix(
            y_true=y_test,
            y_pred=best_pred_test,
            labels=task.labels,
            title=f"{task.task_name.capitalize()} - Best Model ({best_model_name})",
            path=figures_dir / f"phase4_confusion_matrix_{task.task_name}.png",
        )

        report_text = classification_report(y_test, best_pred_test, zero_division=0)
        (reports_dir / f"phase4_classification_report_{task.task_name}.txt").write_text(report_text, encoding="utf-8")

        # Error analysis table.
        test_rows = train_df.iloc[idx_test].copy().reset_index(drop=True)
        err_mask = best_pred_test != y_test
        errors_df = test_rows.loc[err_mask, ["note", "sentiment_label", "assureur", "produit", "text_clean_corrected"]].copy()
        errors_df["true_label"] = y_test[err_mask]
        errors_df["pred_label"] = best_pred_test[err_mask]
        errors_df = errors_df.head(250)
        errors_df.to_csv(
            tables_dir / f"phase4_error_examples_{task.task_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        split_payload = {
            "idx_train": idx_train.tolist(),
            "idx_val": idx_val.tolist(),
            "idx_test": idx_test.tolist(),
            "best_model_name": best_model_name,
            "best_val_f1_macro": float(results_df.iloc[0]["val_f1_macro"]),
            "best_test_f1_macro": float(results_df.iloc[0]["test_f1_macro"]),
            "test_size": int(len(idx_test)),
            "val_size": int(len(idx_val)),
        }
        (reports_dir / f"phase4_split_and_best_{task.task_name}.json").write_text(
            json.dumps(split_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        task_outputs[task.task_name] = {
            "results_path": str(results_path),
            "best_model_name": best_model_name,
            "best_model_path": str(model_save_path),
            "best_test_f1_macro": float(results_df.iloc[0]["test_f1_macro"]),
            "best_val_f1_macro": float(results_df.iloc[0]["val_f1_macro"]),
            "classification_report_path": str(reports_dir / f"phase4_classification_report_{task.task_name}.txt"),
            "confusion_matrix_path": str(figures_dir / f"phase4_confusion_matrix_{task.task_name}.png"),
        }

    summary_lines = [
        "# Phase 4 Supervised Benchmark Summary",
        "",
        f"- Embedding backend for sentence-level features: {embedding_backend}.",
        f"- Stars best model: {task_outputs['stars']['best_model_name']} (test F1-macro={task_outputs['stars']['best_test_f1_macro']:.4f}).",
        f"- Sentiment best model: {task_outputs['sentiment']['best_model_name']} (test F1-macro={task_outputs['sentiment']['best_test_f1_macro']:.4f}).",
    ]
    summary_path = reports_dir / "phase4_modeling_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {"task_outputs": task_outputs, "summary_path": str(summary_path)}
