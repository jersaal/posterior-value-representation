"""The one argument parser every experiment uses."""

from __future__ import annotations

import argparse
import sys

from jbo import model as M


def parse(description: str, extra=None) -> argparse.Namespace:
    """--study <world>/<seed> --smoke --force --device --label; `extra(parser)`
    adds script-specific flags."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--study", required=True, help="<world>/<seed>, e.g. mess3_m1m2/s0")
    p.add_argument("--smoke", action="store_true", help="tiny run on the smoke seeds")
    p.add_argument("--force", action="store_true", help="overwrite an existing run dir / pin")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--label", default=None, help="run-id prefix")
    if extra is not None:
        extra(p)
    args = p.parse_args()
    assert args.study.count("/") == 1, "--study must be <world>/<seed>"
    args.device = args.device or M.default_device()
    args.argv = sys.argv[1:]
    return args
