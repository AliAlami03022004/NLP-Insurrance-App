"""Generate standalone Phase 6 RAG and QA examples from local artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


MODULAR_ROOT = Path(__file__).resolve().parents[1]
if str(MODULAR_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULAR_ROOT))

from backend.config import REPORTS_DIR, TABLES_DIR, ensure_project_dirs
from backend.services import ask_question, ensure_inference_assets


DEFAULT_QUESTIONS = [
    "Which is the best insurer overall by stars?",
    "What do customers say about claims processing?",
    "How is customer service quality perceived?",
]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:90]


def main() -> None:
    ensure_project_dirs()
    ensure_inference_assets()

    qa_rows: list[dict[str, Any]] = []
    for question in DEFAULT_QUESTIONS:
        result = ask_question(
            question=question,
            generative_backend="hf",
            generative_model="google/flan-t5-small",
        )
        qa_rows.append(
            {
                "question": question,
                "template_answer": result["template_answer"],
                "hybrid_mode": result["hybrid_answer"]["mode"],
                "hybrid_intent": result["hybrid_answer"]["intent"],
                "hybrid_answer": result["hybrid_answer"]["answer"],
                "generative_answer": result["generative_answer"]["answer"],
                "generative_error": result["generative_answer"].get("error", ""),
            }
        )

        slug = _slugify(question)
        retrieved_path = TABLES_DIR / f"phase6_rag_retrieved_{slug}.csv"
        sentences_path = TABLES_DIR / f"phase6_qa_sentences_{slug}.csv"
        result["qa_output"]["retrieved_reviews"].to_csv(retrieved_path, index=False, encoding="utf-8-sig")
        result["qa_output"]["answer_sentences"].to_csv(sentences_path, index=False, encoding="utf-8-sig")

        ranking_table = result["hybrid_answer"].get("ranking_table", pd.DataFrame())
        if isinstance(ranking_table, pd.DataFrame) and not ranking_table.empty:
            ranking_table.to_csv(TABLES_DIR / f"phase6_ranking_{slug}.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(qa_rows)
    summary_path = TABLES_DIR / "phase6_rag_qa_examples_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report_payload = {
        "questions": qa_rows,
        "summary_csv": str(summary_path),
    }
    json_path = REPORTS_DIR / "phase6_rag_qa_examples.json"
    md_path = REPORTS_DIR / "phase6_rag_qa_summary.md"
    json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Phase 6 RAG and QA Examples", ""]
    for row in qa_rows:
        lines.extend(
            [
                f"## {row['question']}",
                "",
                f"- Hybrid mode: `{row['hybrid_mode']}`",
                f"- Hybrid intent: `{row['hybrid_intent']}`",
                f"- Template answer: {row['template_answer']}",
                f"- Hybrid answer: {row['hybrid_answer']}",
                f"- Generative answer: {row['generative_answer']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"summary_csv": str(summary_path), "json_report": str(json_path), "md_report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
