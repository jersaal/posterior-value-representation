"""The pretrained transformer's two appendix figures, redrawn from the pinned
03 and 06 payloads with the current figure code: the decoder row
(`Suppl_pretrained_linear.png`, 03's `fig_paper`) and the steering panel
(`Suppl_pretrained_steering.png`, 06's `fig_paper`). Both experiments save
everything their paper figures draw to `payload.npz`, so the exports follow a
relabelling without a rerun of the model, which the steering run needs a GPU
for. Figure text is the paper's terms: posterior, offer value, $Q$.
Usage: fig_pretrained_suppl.py --study mess3_m1m2_qwen3/s0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jbo import plotting as PL   # noqa: E402

# 06's `fig_paper` slope readouts -> the headline entries that carry them (factor A, the panel's factor)
SLOPE_KEYS = {"transfer": "transfer_slope_A", "transfer_ctrl": "control_slope_A",
              "err_nat": "endpoint_error_natural_A", "err_ctrl": "endpoint_error_control_A",
              "ring": "ring_slope", "ring_ctrl": "ring_control_slope",
              "insertion": "insertion_slope_A", "removal": "removal_slope_A"}


def experiment(name: str):
    """Import an `experiments/<name>.py` module (the leading digits keep it off the
    import path). The module goes into `sys.modules` before it executes, which a
    `@dataclass` inside it needs to resolve its own annotations."""
    mod_name = f"jbo_experiment_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def pinned(study: str, exp: str) -> Path:
    pins = json.loads((ROOT / "studies" / study / "pins.json").read_text())
    assert exp in pins, f"{study} has no pinned {exp}"
    return ROOT / "results" / study / exp / pins[exp]


def draw_decoder_row(study: str, out: Path) -> None:
    """03's paper row from its payload: the three portraits at their read sites
    and the two held-out proposition scatters."""
    d = pinned(study, "03_decoder_views")
    z = dict(np.load(d / "payload.npz", allow_pickle=True))
    r = json.loads((d / "results.json").read_text())
    nat, disp = r["natural_r2_by_site"], r["display_site"]
    bsite, dsite = r["portrait_sites"]["posterior"], r["portrait_sites"]["value"]
    dec = {k: z[f"show_dec_{k}"] for k in ("etaA", "etaB", "D")}
    truth = {k: z[f"show_true_{k}"] for k in ("etaA", "etaB", "D")}
    sc = {k: z[k] for k in ("P_true", "P_hat", "D_true", "D_hat")}
    cell_r2 = {"etaA": float(nat[bsite]["etaA"]), "etaB": float(nat[bsite]["etaB"]), "D": float(nat[dsite]["D"])}
    assert set(dec["etaA"].shape) and dec["D"].shape[1] == 2, "payload portraits are not [N, 3] / [N, 2]"
    experiment("03_decoder_views").fig_paper(
        dec, truth, nat[disp], PL.SITE_LABELS[disp], sc, r["prop_median_by_site"][disp],
        len(r["masks_kept"]), (r["read_sites"]["posterior"], r["read_sites"]["value"]), cell_r2, out)


def draw_steering(study: str, out: Path) -> None:
    """06's paper figure from its payload, with the pinned headline slopes."""
    d = pinned(study, "06_steering")
    z = dict(np.load(d / "payload.npz", allow_pickle=True))
    h = json.loads((d / "results.json").read_text())["headline"]
    slopes = {k: float(h[src]["value"]) for k, src in SLOPE_KEYS.items()}
    experiment("06_steering").fig_paper(z, out, slopes=slopes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default="mess3_m1m2_qwen3/s0")
    ap.add_argument("--out-dir", default="figures")
    a = ap.parse_args()
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    draw_decoder_row(a.study, out / "Suppl_pretrained_linear.png")
    draw_steering(a.study, out / "Suppl_pretrained_steering.png")
    print(f"wrote Suppl_pretrained_linear.png and Suppl_pretrained_steering.png to {out}")


if __name__ == "__main__":
    main()
