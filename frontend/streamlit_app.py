"""Main Streamlit frontend entry point for the modularized project."""

from __future__ import annotations

import sys
from pathlib import Path

MODULAR_ROOT = Path(__file__).resolve().parents[1]
if str(MODULAR_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULAR_ROOT))

from frontend.app import main


if __name__ == "__main__":
    main()
