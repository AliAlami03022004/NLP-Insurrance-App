## Architecture

```text
NLP_ProjetV2_modular/
|-- backend/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- eda.py
|   |-- themes.py
|   |-- embeddings.py
|   |-- modeling.py
|   |-- evaluation.py
|   |-- search.py
|   |-- rag.py
|   |-- services.py
|-- frontend/
|   |-- __init__.py
|   |-- app.py
|   |-- streamlit_app.py
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|   |-- embeddings/
|   |-- themes/
|   |-- supervised/
|   |-- phase3_embeddings.py
|   |-- phase4_supervised.py
|   |-- phase5_error_analysis.py
|   |-- phase6_rag.py
|-- outputs/
|   |-- figures/
|   |-- reports/
|   |-- tables/
|-- notebooks/
|   |-- 01_eda_matplotlib_plots.ipynb
|   |-- 02_model_workflow.ipynb
|-- plots/
|-- requirements.txt
```

## Design Rules

- `frontend/` contains UI only.
- `backend/services.py` is the main orchestration layer used by the app.
- `backend/` contains the actual NLP / ML / IR / RAG logic.
- `models/phase*.py` are direct runnable entry points, not wrappers.
- `data/raw/` contains the Excel review files.
- `data/processed/` contains locally generated CSV datasets.
- `outputs/` stores local tables, reports, and figures.

## Install

From the `NLP_ProjetV2_modular` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The App

From the repository root:

```bash
streamlit run NLP_ProjetV2_modular/frontend/streamlit_app.py
```

From inside `NLP_ProjetV2_modular`:

```bash
streamlit run frontend/streamlit_app.py
```

## Run The Pipelines

Pre-existing local artifacts are reused by default.  
Use `--force` when you want to rebuild a phase from scratch.

```bash
python models/phase3_embeddings.py
python models/phase4_supervised.py
python models/phase5_error_analysis.py
python models/phase6_rag.py
```

Force rebuild examples:

```bash
python models/phase3_embeddings.py --force
python models/phase4_supervised.py --force
python models/phase5_error_analysis.py --force
```

## What Is Local And Standalone

- raw Excel files are stored in `data/raw/`
- processed CSV files are stored in `data/processed/`
- trained artifacts are stored in `models/embeddings/`, `models/themes/`, and `models/supervised/`
- reports and tables are stored in `outputs/`

The modular app reads only these local modular paths.

## Main Backend Entry Points

Frontend code should only call functions from `backend.services`, for example:

```python
from backend.services import (
    run_preprocessing,
    run_eda,
    run_theme_and_embedding_pipeline,
    run_supervised_pipeline,
    run_error_analysis,
    predict_review,
    search_reviews,
    ask_question,
    get_dashboard_data,
)
```

## Notebooks

- `notebooks/01_eda_matplotlib_plots.ipynb`: Matplotlib / Seaborn plotting walkthrough
- `notebooks/02_model_workflow.ipynb`: model comparison, error analysis, and RAG walkthrough

The notebooks are meant for explanation.
The core reusable logic remains in `.py` files inside `backend/`.
