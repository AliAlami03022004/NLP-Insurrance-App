"""Run the standalone Phase 4 supervised benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODULAR_ROOT = Path(__file__).resolve().parents[1]
if str(MODULAR_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULAR_ROOT))

from backend.services import run_supervised_pipeline


def main(force: bool = False) -> None:
    outputs = run_supervised_pipeline(force=force)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4 supervised benchmark.")
    parser.add_argument("--force", action="store_true", help="Retrain the supervised models even if artifacts already exist.")
    args = parser.parse_args()
    main(force=args.force)
