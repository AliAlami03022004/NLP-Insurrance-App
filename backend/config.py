"""Local path configuration for the standalone modular project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
THEMES_DIR = MODELS_DIR / "themes"
SUPERVISED_DIR = MODELS_DIR / "supervised"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PLOTS_DIR = PROJECT_ROOT / "plots"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_project_dirs() -> None:
    """Create the local project directories used by the pipelines."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EMBEDDINGS_DIR,
        THEMES_DIR,
        SUPERVISED_DIR,
        TABLES_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        PLOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
