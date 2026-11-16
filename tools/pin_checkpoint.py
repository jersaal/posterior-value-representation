"""Pin a trained checkpoint into studies/<study>/study.json (the only writer).
Usage: pin_checkpoint.py --study <world>/<seed> --weights <path> [--force]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo.subject import load_study, pin_checkpoint  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    pin_checkpoint(load_study(a.study), Path(a.weights), force=a.force)
    print(f"pinned {a.weights} into studies/{a.study}/study.json")
