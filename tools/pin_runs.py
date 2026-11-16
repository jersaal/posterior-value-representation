"""Pin one run-id per experiment into studies/<world>/<seed>/pins.json.
Usage: pin_runs.py --study <world>/<seed> [--experiment 01_behavior=<run-id> ...]
With no --experiment, every experiment's newest run dir is pinned (an explicit
act, recorded here; collectors never look at 'latest' themselves)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pin(study: str, overrides: dict[str, str]) -> dict:
    res = ROOT / "results" / study
    pins_path = ROOT / "studies" / study / "pins.json"
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}
    for exp_dir in sorted(p for p in res.iterdir() if p.is_dir()):
        runs = [p.name for p in sorted(exp_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                if (p / "results.json").exists() and not p.name.startswith("smoke")]
        if not runs:
            continue
        pins[exp_dir.name] = overrides.get(exp_dir.name, runs[-1])
        assert (exp_dir / pins[exp_dir.name] / "results.json").exists(), pins[exp_dir.name]
    pins_path.write_text(json.dumps(pins, indent=2) + "\n")
    return pins


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--experiment", action="append", default=[], help="<experiment>=<run-id>")
    a = ap.parse_args()
    print(json.dumps(pin(a.study, dict(x.split("=", 1) for x in a.experiment)), indent=1))
