"""Reusable inference helpers for app prediction, themes, and search."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MULTISPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z]+")


def _pick_writable_st_cache_dir(project_root: Path) -> Path:
    """Choose a writable cache location for sentence-transformer downloads."""
    candidates: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "nlp_projetv2_st_cache")
    candidates.append(Path(tempfile.gettempdir()) / "nlp_projetv2_st_cache")
    candidates.append(project_root / ".hf_cache")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except Exception:
            continue
    return candidates[-1]


@dataclass
class InferenceResources:
    """Container for loaded models and data used by the app."""

    data: pd.DataFrame
    embeddings: np.ndarray
    nn_index: Any
    embedding_metadata: dict[str, Any]
    embedding_backend_payload: dict[str, Any]
    star_model: Any
    sentiment_model: Any
    star_explainer_model: Any
    sentiment_explainer_model: Any
    theme_payload: dict[str, Any]
    sentence_transformer_model: Any | None = None


def preprocess_text_for_inference(text: str) -> dict[str, str]:
    text_raw = text or ""
    text_basic = URL_PATTERN.sub(" ", text_raw)
    text_basic = text_basic.replace("\r", " ").replace("\n", " ").strip()
    text_basic = MULTISPACE_PATTERN.sub(" ", text_basic).strip()
    text_lower = text_basic.lower()
    tokens = TOKEN_PATTERN.findall(text_lower)
    text_for_classical_ml = " ".join(tokens)
    return {
        "text_raw": text_raw.strip(),
        "text_clean_basic": text_basic,
        "text_for_deep_learning": text_lower,
        "text_for_classical_ml": text_for_classical_ml,
    }


def load_inference_resources(project_root: Path) -> InferenceResources:
    phase3_path = project_root / "data" / "processed" / "clean_reviews_phase3.csv"
    models_dir = project_root / "models"

    data = pd.read_csv(phase3_path, low_memory=False)
    embeddings = np.load(models_dir / "embeddings" / "review_embeddings.npy")
    nn_index = joblib.load(models_dir / "embeddings" / "review_nn_index.joblib")
    embedding_metadata = json.loads((models_dir / "embeddings" / "embedding_metadata.json").read_text(encoding="utf-8"))
    embedding_backend_payload = joblib.load(models_dir / "embeddings" / "embedding_backend_payload.joblib")

    star_model = joblib.load(models_dir / "supervised" / "stars_best_model.joblib")
    sentiment_model = joblib.load(models_dir / "supervised" / "sentiment_best_model.joblib")
    star_explainer_model = joblib.load(models_dir / "supervised" / "stars_tfidf_logreg_model.joblib")
    sentiment_explainer_model = joblib.load(models_dir / "supervised" / "sentiment_tfidf_logreg_model.joblib")
    theme_payload = joblib.load(models_dir / "themes" / "hybrid_theme_artifacts.joblib")

    sentence_transformer_model = None
    if embedding_metadata.get("embedding_backend", "") == "sentence_transformer":
        model_name = embedding_backend_payload.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        cache_dir = _pick_writable_st_cache_dir(project_root)
        from sentence_transformers import SentenceTransformer

        sentence_transformer_model = SentenceTransformer(model_name, cache_folder=str(cache_dir))

    return InferenceResources(
        data=data,
        embeddings=embeddings,
        nn_index=nn_index,
        embedding_metadata=embedding_metadata,
        embedding_backend_payload=embedding_backend_payload,
        star_model=star_model,
        sentiment_model=sentiment_model,
        star_explainer_model=star_explainer_model,
        sentiment_explainer_model=sentiment_explainer_model,
        theme_payload=theme_payload,
        sentence_transformer_model=sentence_transformer_model,
    )


def embed_text(text_for_deep_learning: str, resources: InferenceResources) -> np.ndarray:
    backend = resources.embedding_metadata.get("embedding_backend", "")
    if backend == "sentence_transformer":
        if resources.sentence_transformer_model is None:
            raise ValueError("SentenceTransformer model is not loaded in inference resources.")
        dense = resources.sentence_transformer_model.encode(
            [text_for_deep_learning],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return dense
    if backend == "tfidf_svd_fallback":
        vectorizer = resources.embedding_backend_payload["vectorizer"]
        svd = resources.embedding_backend_payload["svd"]
        tfidf = vectorizer.transform([text_for_deep_learning])
        dense = svd.transform(tfidf).astype(np.float32)
        norm = np.linalg.norm(dense, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        dense = dense / norm
        return dense
    raise ValueError(f"Unsupported embedding backend for app inference: {backend}")


def _predict_with_probabilities(model: Any, X: Any) -> tuple[Any, dict[str, float]]:
    pred = model.predict(X)[0]
    probs: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(X)[0]
        for label, prob in zip(model.classes_, prob_arr):
            probs[str(label)] = float(prob)
    return pred, probs


def _predict_with_auto_input(
    model: Any,
    cleaned_text: str,
    embedding_vector: np.ndarray,
) -> tuple[Any, dict[str, float]]:
    """Predict with either embedding or TF-IDF text input based on model compatibility."""
    # Try embedding-style input first (for dense models).
    try:
        return _predict_with_probabilities(model, embedding_vector)
    except Exception:
        pass

    # Fallback for TF-IDF / text pipelines.
    return _predict_with_probabilities(model, [cleaned_text])


def predict_stars_and_sentiment(text: str, resources: InferenceResources) -> dict[str, Any]:
    cleaned = preprocess_text_for_inference(text)
    emb = embed_text(cleaned["text_for_deep_learning"], resources)
    star_pred, star_probs = _predict_with_auto_input(resources.star_model, cleaned["text_for_classical_ml"], emb)
    sent_pred, sent_probs = _predict_with_auto_input(resources.sentiment_model, cleaned["text_for_classical_ml"], emb)
    return {
        "cleaned": cleaned,
        "star_prediction": int(star_pred) if str(star_pred).isdigit() else star_pred,
        "star_probabilities": star_probs,
        "sentiment_prediction": str(sent_pred),
        "sentiment_probabilities": sent_probs,
        "embedding_vector": emb,
    }


def _keyword_score_single(text: str, theme_lexicon: dict[str, list[str]]) -> dict[str, float]:
    words_count = max(1, len(text.split()))
    scores: dict[str, float] = {}
    for theme, keywords in theme_lexicon.items():
        count = 0
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"\b"
            count += len(re.findall(pattern, text.lower()))
        scores[theme] = count / np.sqrt(words_count)
    return scores


def predict_theme(text: str, resources: InferenceResources) -> dict[str, Any]:
    cleaned = preprocess_text_for_inference(text)
    payload = resources.theme_payload
    themes = payload["themes"]
    theme_lexicon = payload["theme_lexicon"]
    topic_to_theme = payload["topic_to_theme"]
    weights = payload["weights"]

    keyword_scores = _keyword_score_single(cleaned["text_for_deep_learning"], theme_lexicon)
    keyword_df = pd.DataFrame([keyword_scores], columns=themes).fillna(0.0)
    keyword_norm = keyword_df.div(keyword_df.max(axis=1).replace(0.0, 1.0), axis=0)

    vectorizer = payload["vectorizer"]
    prototype_matrix = payload["prototype_matrix"]
    vec = vectorizer.transform([cleaned["text_for_deep_learning"]])
    proto_scores = cosine_similarity(vec, prototype_matrix)
    proto_df = pd.DataFrame(proto_scores, columns=themes)
    proto_norm = proto_df.div(proto_df.max(axis=1).replace(0.0, 1.0), axis=0)

    topic_vectorizer = payload["topic_vectorizer"]
    nmf_model = payload["nmf_model"]
    topic_vec = topic_vectorizer.transform([cleaned["text_for_deep_learning"]])
    doc_topic = nmf_model.transform(topic_vec)[0]
    topic_theme_scores = {theme: 0.0 for theme in themes}
    for topic_idx, val in enumerate(doc_topic):
        mapped_theme = topic_to_theme.get(topic_idx, "Other")
        topic_theme_scores[mapped_theme] += float(val)
    topic_df = pd.DataFrame([topic_theme_scores], columns=themes)
    topic_norm = topic_df.div(topic_df.max(axis=1).replace(0.0, 1.0), axis=0)

    fused = (
        weights["prototype"] * proto_norm
        + weights["keyword"] * keyword_norm
        + weights["topic"] * topic_norm
    )
    fused_scores = {theme: float(fused.iloc[0][theme]) for theme in themes}

    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    primary_theme, primary_score = sorted_items[0]
    secondary_theme = ""
    if len(sorted_items) > 1:
        sec_theme, sec_score = sorted_items[1]
        if sec_score >= 0.8 * primary_score and (primary_score - sec_score) <= 0.18 and sec_score >= 0.25:
            secondary_theme = sec_theme
    if primary_score < 0.25:
        primary_theme = "Other"

    return {
        "theme_primary": primary_theme,
        "theme_secondary": secondary_theme,
        "theme_confidence": float(primary_score),
        "theme_scores": fused_scores,
    }


def local_token_explanation(text: str, tfidf_logreg_model: Any, target_label: Any | None = None, top_n: int = 10) -> list[dict[str, Any]]:
    cleaned = preprocess_text_for_inference(text)
    vectorizer = tfidf_logreg_model.named_steps["tfidf"]
    clf = tfidf_logreg_model.named_steps["clf"]
    x = vectorizer.transform([cleaned["text_for_classical_ml"]])

    if target_label is None:
        target_label = tfidf_logreg_model.predict([cleaned["text_for_classical_ml"]])[0]

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
    contributions = sorted(contributions, key=lambda d: d["contribution"], reverse=True)[:top_n]
    return contributions


def semantic_search(
    query_text: str,
    resources: InferenceResources,
    top_k: int = 10,
    filter_assureur: str | None = None,
    filter_theme: str | None = None,
    filter_note: int | None = None,
) -> pd.DataFrame:
    cleaned = preprocess_text_for_inference(query_text)
    query_emb = embed_text(cleaned["text_for_deep_learning"], resources)[0]

    data = resources.data.copy()
    mask = pd.Series(True, index=data.index)
    if filter_assureur and filter_assureur != "All":
        mask &= data["assureur"].astype(str).eq(filter_assureur)
    if filter_theme and filter_theme != "All":
        mask &= data["theme_primary"].astype(str).eq(filter_theme)
    if filter_note is not None:
        mask &= data["note"].fillna(-1).astype(float).eq(float(filter_note))
    subset_idx = data.index[mask].to_numpy()

    if len(subset_idx) == 0:
        return pd.DataFrame()

    subset_embeddings = resources.embeddings[subset_idx]
    sims = subset_embeddings @ query_emb.reshape(-1, 1)
    sims = sims.reshape(-1)
    top_local = np.argsort(-sims)[:top_k]
    top_idx = subset_idx[top_local]

    base_cols = ["row_id", "assureur", "produit", "type", "note", "theme_primary", "text_clean_corrected"]
    existing_cols = [col for col in base_cols if col in data.columns]
    out = data.loc[top_idx, existing_cols].copy()
    out["similarity"] = sims[top_local]
    out = out.sort_values("similarity", ascending=False).reset_index(drop=True)
    return out


def keyword_search(
    query: str,
    resources: InferenceResources,
    top_k: int = 50,
    filter_assureur: str | None = None,
    filter_theme: str | None = None,
    filter_note: int | None = None,
) -> pd.DataFrame:
    data = resources.data.copy()
    mask = data["text_clean_corrected"].fillna("").str.contains(query, case=False, regex=False)
    if filter_assureur and filter_assureur != "All":
        mask &= data["assureur"].astype(str).eq(filter_assureur)
    if filter_theme and filter_theme != "All":
        mask &= data["theme_primary"].astype(str).eq(filter_theme)
    if filter_note is not None:
        mask &= data["note"].fillna(-1).astype(float).eq(float(filter_note))

    base_cols = ["row_id", "assureur", "produit", "type", "note", "theme_primary", "text_clean_corrected"]
    existing_cols = [col for col in base_cols if col in data.columns]
    out = data.loc[mask, existing_cols].head(top_k)
    return out.reset_index(drop=True)
