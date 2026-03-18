"""Data loading helpers for the standalone modular project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, ensure_project_dirs


EXPECTED_COLUMNS = [
    "note",
    "auteur",
    "avis",
    "assureur",
    "produit",
    "type",
    "date_publication",
    "date_exp",
    "avis_en",
    "avis_cor",
    "avis_cor_en",
]


def list_review_files(raw_data_dir: Path | None = None) -> list[Path]:
    """Return all source Excel files stored in the modular data folder."""
    ensure_project_dirs()
    data_dir = raw_data_dir or RAW_DATA_DIR
    files = sorted(data_dir.glob("avis_*_traduit.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"No raw Excel files were found in {data_dir}. "
            "Copy the source review files into NLP_ProjetV2_modular/data/raw/."
        )
    return files


def load_single_file(path: Path) -> pd.DataFrame:
    """Load one Excel file and keep only the expected review columns."""
    df = pd.read_excel(path)
    missing_cols = set(EXPECTED_COLUMNS).difference(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise ValueError(f"{path.name} is missing expected columns: {missing_str}")
    return df[EXPECTED_COLUMNS].copy().assign(source_file=path.name)


def load_reviews(raw_data_dir: Path | None = None) -> pd.DataFrame:
    """Load and concatenate all raw review Excel files."""
    files = list_review_files(raw_data_dir=raw_data_dir)
    frames = [load_single_file(path) for path in files]
    merged = pd.concat(frames, ignore_index=True)
    merged["row_id"] = merged.index.astype("int64")
    return merged


def normalize_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight type normalization used across the project."""
    out = df.copy()
    if "note" in out.columns:
        out["note"] = pd.to_numeric(out["note"], errors="coerce")
    for col in ["auteur", "assureur", "produit", "type"]:
        if col in out.columns:
            out[col] = out[col].astype("string")
    for col in ["date_publication", "date_exp"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True)
    return out


def load_raw_reviews_dataset() -> pd.DataFrame:
    """Load the full merged raw dataset from the modular folder."""
    return normalize_column_types(load_reviews())


def _processed_path(phase3: bool = False) -> Path:
    file_name = "clean_reviews_phase3.csv" if phase3 else "clean_reviews.csv"
    return PROCESSED_DATA_DIR / file_name


def load_processed_reviews_dataset(phase3: bool = False) -> pd.DataFrame:
    """Load a processed CSV stored inside the modular project."""
    path = _processed_path(phase3=phase3)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run the preprocessing pipeline first."
        )
    df = pd.read_csv(path, low_memory=False)
    return normalize_column_types(df)


def save_processed_reviews_dataset(df: pd.DataFrame, phase3: bool = False) -> Path:
    """Save a processed dataframe into the modular data folder."""
    ensure_project_dirs()
    path = _processed_path(phase3=phase3)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Fail fast when a dataframe is missing mandatory columns."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact dataset summary for reports or the app."""
    overview: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_values": {col: int(val) for col, val in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "assureur" in df.columns:
        insurer_count = int(df["assureur"].dropna().nunique())
        overview["n_insurers"] = insurer_count
        overview["insurers"] = insurer_count
    if "theme_primary" in df.columns:
        theme_count = int(df["theme_primary"].dropna().nunique())
        overview["n_themes"] = theme_count
        overview["themes"] = theme_count
    if "note" in df.columns:
        note_series = pd.to_numeric(df["note"], errors="coerce")
        overview["note_distribution"] = note_series.value_counts(dropna=False).sort_index().to_dict()
        if note_series.notna().any():
            overview["mean_note"] = float(note_series.mean())
            overview["mean_stars"] = float(note_series.mean())
    return overview
