"""Lightweight extractive RAG + QA utilities built on local review artifacts."""

from __future__ import annotations

import os
import re
import tempfile
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import PROJECT_ROOT
from backend.search import InferenceResources, semantic_search


SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")
TOKEN_RE = re.compile(r"[a-z]{2,}")

BEST_HINTS = ["best", "top", "highest", "meilleur", "meilleure", "top rated"]
WORST_HINTS = ["worst", "lowest", "pire", "moins bon", "badest"]
RANKING_HINTS = ["rank", "ranking", "classement", "compare", "comparison", "which insurer", "quelle assurance"]
INSURER_HINTS = ["insurer", "insurance", "assureur", "assurance", "company", "compagnie"]

THEME_ALIASES: dict[str, list[str]] = {
    "Pricing": ["price", "pricing", "tarif", "tarifs", "cost", "expensive", "cheap", "value"],
    "Coverage": ["coverage", "cover", "garantie", "guarantee", "protection"],
    "Enrollment": ["enroll", "subscription", "signup", "souscription", "contract", "contrat"],
    "Customer Service": ["customer service", "support", "service client", "advisor", "agent"],
    "Claims Processing": ["claim", "reimburse", "refund", "indemn", "sinistre", "remboursement"],
    "Cancellation": ["cancel", "cancellation", "terminate", "résiliation", "resiliation"],
}


def _pick_writable_cache_dir() -> Path:
    candidates: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "hf_cache_generative")
    candidates.append(Path(tempfile.gettempdir()) / "hf_cache_generative")
    candidates.append(PROJECT_ROOT / ".hf_cache_generative")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except Exception:
            continue

    # Last resort; from_pretrained will surface a clear error if this still fails.
    return candidates[-1]


def _sentiment_from_note(note_value: Any) -> str:
    try:
        note = float(note_value)
    except Exception:
        return "unknown"
    if note <= 2:
        return "negative"
    if note == 3:
        return "neutral"
    return "positive"


def _extract_candidate_sentences(retrieved_df: pd.DataFrame, max_sentences: int = 400) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in retrieved_df.iterrows():
        text = str(row.get("text_clean_corrected", "") or "").strip()
        if not text:
            continue
        parts = SENTENCE_SPLIT_RE.split(text)
        for sent in parts:
            sentence = sent.strip()
            if len(sentence) < 20:
                continue
            rows.append(
                {
                    "row_id": row.get("row_id", None),
                    "assureur": row.get("assureur", ""),
                    "theme_primary": row.get("theme_primary", ""),
                    "note": row.get("note", np.nan),
                    "doc_similarity": float(row.get("similarity", 0.0)),
                    "sentence": sentence,
                }
            )
            if len(rows) >= max_sentences:
                break
        if len(rows) >= max_sentences:
            break
    if not rows:
        return pd.DataFrame(columns=["row_id", "assureur", "theme_primary", "note", "doc_similarity", "sentence"])
    return pd.DataFrame(rows)


def extractive_qa(
    question: str,
    resources: InferenceResources,
    top_k_docs: int = 14,
    top_k_sentences: int = 5,
    filter_assureur: str | None = None,
    filter_theme: str | None = None,
    filter_note: int | None = None,
) -> dict[str, Any]:
    """Retrieve relevant reviews and rank answer sentences by semantic relevance."""
    retrieved = semantic_search(
        query_text=question,
        resources=resources,
        top_k=top_k_docs,
        filter_assureur=filter_assureur,
        filter_theme=filter_theme,
        filter_note=filter_note,
    )
    if retrieved.empty:
        return {
            "question": question,
            "retrieved_reviews": retrieved,
            "candidate_sentences": pd.DataFrame(),
            "answer_sentences": pd.DataFrame(),
            "best_answer": "No relevant reviews found for this question and filters.",
        }

    candidate_sentences = _extract_candidate_sentences(retrieved_df=retrieved)
    if candidate_sentences.empty:
        return {
            "question": question,
            "retrieved_reviews": retrieved,
            "candidate_sentences": candidate_sentences,
            "answer_sentences": pd.DataFrame(),
            "best_answer": "Relevant reviews were found, but no usable answer sentence could be extracted.",
        }

    docs = [question] + candidate_sentences["sentence"].astype(str).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vectorizer.fit_transform(docs)
    q_vec = mat[0:1]
    s_vec = mat[1:]
    sims = cosine_similarity(q_vec, s_vec).reshape(-1)

    lexical_question_tokens = set(TOKEN_RE.findall(question.lower()))
    lexical_overlap_scores = []
    for sentence in candidate_sentences["sentence"].astype(str):
        sent_tokens = set(TOKEN_RE.findall(sentence.lower()))
        overlap = len(lexical_question_tokens.intersection(sent_tokens))
        lexical_overlap_scores.append(float(overlap))
    lexical_overlap_scores = np.array(lexical_overlap_scores, dtype=float)
    if lexical_overlap_scores.max() > 0:
        lexical_overlap_scores = lexical_overlap_scores / lexical_overlap_scores.max()

    doc_sim = candidate_sentences["doc_similarity"].fillna(0.0).to_numpy(dtype=float)
    if doc_sim.max() > 0:
        doc_sim = doc_sim / doc_sim.max()

    final_score = 0.60 * sims + 0.25 * lexical_overlap_scores + 0.15 * doc_sim
    ranked_idx = np.argsort(-final_score)[:top_k_sentences]

    answer_df = candidate_sentences.iloc[ranked_idx].copy().reset_index(drop=True)
    answer_df["score"] = final_score[ranked_idx]
    answer_df = answer_df.sort_values("score", ascending=False).reset_index(drop=True)

    best_answer = answer_df.iloc[0]["sentence"] if not answer_df.empty else "No answer sentence extracted."

    return {
        "question": question,
        "retrieved_reviews": retrieved,
        "candidate_sentences": candidate_sentences,
        "answer_sentences": answer_df,
        "best_answer": best_answer,
    }


def rag_summary_answer(question: str, qa_output: dict[str, Any]) -> str:
    """Generate a compact grounded answer summary using retrieved evidence statistics."""
    retrieved = qa_output["retrieved_reviews"]
    answer_sentences = qa_output["answer_sentences"]
    if retrieved.empty:
        return "I could not find relevant review evidence for this question."

    n_docs = len(retrieved)
    notes = pd.to_numeric(retrieved.get("note", pd.Series(dtype=float)), errors="coerce")
    mean_note = notes.mean() if notes.notna().any() else np.nan
    sentiment_counts = notes.dropna().map(_sentiment_from_note).value_counts()
    theme_counts = retrieved.get("theme_primary", pd.Series(dtype=str)).astype(str).value_counts()
    insurer_counts = retrieved.get("assureur", pd.Series(dtype=str)).astype(str).value_counts()

    dominant_theme = theme_counts.index[0] if len(theme_counts) else "n/a"
    dominant_insurer = insurer_counts.index[0] if len(insurer_counts) else "n/a"
    neg_share = float(sentiment_counts.get("negative", 0) / max(1, sentiment_counts.sum()))
    pos_share = float(sentiment_counts.get("positive", 0) / max(1, sentiment_counts.sum()))

    evidence_lines = []
    for _, row in answer_sentences.head(3).iterrows():
        evidence_lines.append(f"- {row['sentence']}")

    answer_lines = [
        f"Based on {n_docs} semantically retrieved reviews, the dominant theme is {dominant_theme} and the most represented insurer is {dominant_insurer}.",
    ]
    if np.isfinite(mean_note):
        answer_lines.append(f"The mean observed rating in this evidence set is {mean_note:.2f}/5.")
    answer_lines.append(
        f"Sentiment mix in evidence is approximately {pos_share*100:.0f}% positive vs {neg_share*100:.0f}% negative (remaining mostly neutral)."
    )
    answer_lines.append("Most relevant extracted evidence:")
    answer_lines.extend(evidence_lines if evidence_lines else ["- No high-confidence evidence sentence available."])

    return "\n".join(answer_lines)


def _build_grounded_prompt(
    question: str,
    qa_output: dict[str, Any],
    max_evidence_sentences: int,
) -> tuple[str, str]:
    answer_sentences = qa_output.get("answer_sentences", pd.DataFrame())
    if answer_sentences.empty:
        return "", "No evidence sentences were retrieved, so generative RAG could not run."

    evidence_lines: list[str] = []
    for idx, (_, row) in enumerate(answer_sentences.head(max_evidence_sentences).iterrows(), start=1):
        evidence_lines.append(
            (
                f"[{idx}] insurer={row.get('assureur', '')}; theme={row.get('theme_primary', '')}; "
                f"note={row.get('note', '')}; sentence={str(row.get('sentence', '')).strip()}"
            )
        )

    prompt = (
        "You are a grounded assistant for insurance review analytics.\n"
        "Answer using only the provided evidence. If evidence is insufficient, say so explicitly.\n"
        "Keep the answer concise and factual.\n\n"
        f"Question: {question}\n\n"
        "Evidence:\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        "Answer:"
    )
    return prompt, ""


def _detect_question_intent(question: str) -> str:
    q = question.lower()
    has_best = any(h in q for h in BEST_HINTS)
    has_worst = any(h in q for h in WORST_HINTS)
    has_rank = any(h in q for h in RANKING_HINTS) or has_best or has_worst
    has_insurer = any(h in q for h in INSURER_HINTS)

    if has_rank and has_insurer and has_worst:
        return "worst_insurer"
    if has_rank and has_insurer and has_best:
        return "best_insurer"
    if has_rank and has_insurer:
        return "insurer_ranking"
    return "qualitative"


def _infer_theme_from_question(question: str, available_themes: list[str]) -> str | None:
    q = question.lower()
    for theme in available_themes:
        aliases = THEME_ALIASES.get(theme, [theme.lower()])
        if any(alias in q for alias in aliases):
            return theme
    return None


def _insurer_ranking_table(
    resources: InferenceResources,
    min_reviews: int = 120,
    theme_scope: str | None = None,
    note_scope: int | None = None,
) -> pd.DataFrame:
    df = resources.data.copy()
    if "note" not in df.columns or "assureur" not in df.columns:
        return pd.DataFrame()

    df["note_num"] = pd.to_numeric(df["note"], errors="coerce")
    df = df[df["note_num"].notna()].copy()
    if "type" in df.columns:
        df = df[df["type"].astype(str).eq("train") | df["type"].isna()].copy()
    if theme_scope:
        df = df[df.get("theme_primary", "").astype(str).eq(theme_scope)].copy()
    if note_scope is not None:
        df = df[df["note_num"].eq(float(note_scope))].copy()
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("assureur", as_index=False).agg(
        review_count=("note_num", "size"),
        mean_note=("note_num", "mean"),
        median_note=("note_num", "median"),
        std_note=("note_num", "std"),
    )
    neg_share = (
        df.assign(is_negative=df["note_num"] <= 2)
        .groupby("assureur", as_index=False)["is_negative"]
        .mean()
        .rename(columns={"is_negative": "negative_share"})
    )
    grouped = grouped.merge(neg_share, on="assureur", how="left")

    c = float(df["note_num"].mean())
    m = max(20, int(min_reviews))
    grouped["bayesian_score"] = (
        (grouped["review_count"] / (grouped["review_count"] + m)) * grouped["mean_note"]
        + (m / (grouped["review_count"] + m)) * c
    )
    grouped["std_note"] = grouped["std_note"].fillna(0.0)

    eligible = grouped[grouped["review_count"] >= m].copy()
    if eligible.empty:
        eligible = grouped[grouped["review_count"] >= max(10, m // 2)].copy()
    if eligible.empty:
        eligible = grouped.copy()

    return eligible.sort_values(["bayesian_score", "review_count"], ascending=[False, False]).reset_index(drop=True)


def hybrid_rag_answer(
    question: str,
    resources: InferenceResources,
    qa_output: dict[str, Any],
    min_reviews: int = 120,
    top_n: int = 5,
    filter_theme: str | None = None,
    filter_note: int | None = None,
) -> dict[str, Any]:
    """
    Route factual ranking questions to structured full-dataset analytics,
    and keep qualitative questions on retrieval-based RAG.
    """
    intent = _detect_question_intent(question)
    available_themes = sorted(resources.data.get("theme_primary", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    inferred_theme = _infer_theme_from_question(question, available_themes)
    theme_scope = filter_theme if filter_theme else inferred_theme

    if intent in {"best_insurer", "worst_insurer", "insurer_ranking"}:
        ranking_df = _insurer_ranking_table(
            resources=resources,
            min_reviews=min_reviews,
            theme_scope=theme_scope,
            note_scope=filter_note,
        )
        if ranking_df.empty:
            return {
                "mode": "analytics_routing",
                "intent": intent,
                "answer": "No insurer ranking can be computed from the available star-labeled data and filters.",
                "ranking_table": pd.DataFrame(),
                "theme_scope": theme_scope or "",
            }

        ranking_view = ranking_df.copy()
        if intent == "worst_insurer":
            ranking_view = ranking_view.sort_values(["bayesian_score", "review_count"], ascending=[True, False]).reset_index(drop=True)
            label = "worst"
        elif intent == "best_insurer":
            ranking_view = ranking_view.sort_values(["bayesian_score", "review_count"], ascending=[False, False]).reset_index(drop=True)
            label = "best"
        else:
            ranking_view = ranking_view.sort_values(["bayesian_score", "review_count"], ascending=[False, False]).reset_index(drop=True)
            label = "top"

        top_row = ranking_view.iloc[0]
        scope = f" within theme `{theme_scope}`" if theme_scope else ""
        answer = (
            f"Using full-dataset insurer ranking{scope} with minimum {min_reviews} reviews, "
            f"the {label} insurer is **{top_row['assureur']}** "
            f"(bayesian score {top_row['bayesian_score']:.2f}, mean stars {top_row['mean_note']:.2f}, "
            f"{int(top_row['review_count'])} reviews)."
        )
        return {
            "mode": "analytics_routing",
            "intent": intent,
            "answer": answer,
            "ranking_table": ranking_view.head(top_n),
            "theme_scope": theme_scope or "",
        }

    return {
        "mode": "retrieval_rag",
        "intent": intent,
        "answer": rag_summary_answer(question=question, qa_output=qa_output),
        "ranking_table": pd.DataFrame(),
        "theme_scope": theme_scope or "",
    }


@lru_cache(maxsize=2)
def _load_text2text_generator(model_name: str):
    """Lazily load and cache a small seq2seq model for grounded generation."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # On some Windows/OneDrive setups, default HF cache/Xet path can fail with permission issues.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    cache_dir = _pick_writable_cache_dir()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(cache_dir))
    return tokenizer, model


def rag_generative_answer(
    question: str,
    qa_output: dict[str, Any],
    model_name: str = "google/flan-t5-base",
    max_evidence_sentences: int = 5,
    max_new_tokens: int = 110,
) -> dict[str, Any]:
    """
    Generate a grounded answer from retrieved evidence using a small seq2seq model.
    Returns answer text plus metadata for app display.
    """
    prompt, early_error = _build_grounded_prompt(
        question=question,
        qa_output=qa_output,
        max_evidence_sentences=max_evidence_sentences,
    )
    if early_error:
        return {
            "answer": early_error,
            "model_name": model_name,
            "prompt": prompt,
            "error": "",
        }

    try:
        tokenizer, model = _load_text2text_generator(model_name)
        import torch

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=2,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if not generated_text:
            generated_text = "The model returned an empty generation."
        return {
            "answer": generated_text,
            "model_name": model_name,
            "prompt": prompt,
            "error": "",
        }
    except Exception as exc:
        return {
            "answer": (
                "Generative RAG is unavailable in this environment. "
                "Template-based RAG is still available."
            ),
            "model_name": model_name,
            "prompt": prompt,
            "error": str(exc),
        }


def rag_generative_answer_ollama(
    question: str,
    qa_output: dict[str, Any],
    model_name: str = "llama3.1:8b",
    max_evidence_sentences: int = 5,
    max_new_tokens: int = 140,
    temperature: float = 0.1,
    base_url: str = "http://localhost:11434",
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """Generate grounded answer through local Ollama REST API."""
    prompt, early_error = _build_grounded_prompt(
        question=question,
        qa_output=qa_output,
        max_evidence_sentences=max_evidence_sentences,
    )
    if early_error:
        return {
            "answer": early_error,
            "model_name": model_name,
            "prompt": prompt,
            "error": "",
        }

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
        },
    }
    request_url = f"{base_url.rstrip('/')}/api/generate"
    request_data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        request_url,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
        generated_text = str(parsed.get("response", "")).strip()
        if not generated_text:
            generated_text = "The Ollama model returned an empty generation."
        return {
            "answer": generated_text,
            "model_name": model_name,
            "prompt": prompt,
            "error": "",
        }
    except urlerror.URLError as exc:
        return {
            "answer": (
                "Ollama generative RAG is unavailable. "
                "Ensure Ollama is running and the model is pulled locally."
            ),
            "model_name": model_name,
            "prompt": prompt,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "answer": (
                "Ollama generative RAG failed in this environment. "
                "Template-based RAG is still available."
            ),
            "model_name": model_name,
            "prompt": prompt,
            "error": str(exc),
        }
