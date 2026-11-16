"""Read and steering profiles by layer (the two appendix figures): cross-validated
read R² and steering slope at every residual-stream layer, drawn from the pinned
`02_fit_instruments` results of each study (no model, no rerun).
Usage: fig_depth_profiles.py --figure profiles   (toy model and Qwen3-1.7B-Base -> figures/Suppl_depth_profiles.png)
       fig_depth_profiles.py --figure sweep      (two, three, four and eight layers -> figures/Suppl_depth_sweep.png)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jbo import plotting as PL   # noqa: E402

FIGURES = {"profiles": ("Suppl_depth_profiles.png", (("mess3_m1m2/s0", "toy model"), ("mess3_m1m2_qwen3/s0", "Qwen3-1.7B-Base"))),
           "sweep": ("Suppl_depth_sweep.png", (("mess3_m1m2/s0_2l", "2 layers"), ("mess3_m1m2/s0_3l", "3 layers"),
                                               ("mess3_m1m2/s0", "4 layers"), ("mess3_m1m2/s0_8l", "8 layers")))}
SERIES = (("etaA", "posterior M", PL.POSTERIOR_A_COLOR, "o", "-"),
          ("etaB", "posterior N", PL.POSTERIOR_B_COLOR, "o", "-"),
          ("D", "offer values", PL.VALUE_COLOR, "s", "-"))


def pinned_fit(study: str) -> dict:
    pins = json.loads((ROOT / "studies" / study / "pins.json").read_text())
    run = ROOT / "results" / study / "02_fit_instruments" / pins["02_fit_instruments"]
    return json.loads((run / "results.json").read_text())


def layer_of(site: str) -> int:
    """`blocks.<l>.hook_resid_post` -> l; the embedding site is -1."""
    if site.endswith("hook_resid_pre"):
        return -1
    return int(site.split(".")[1])


def profile(r: dict) -> dict:
    """Per-layer arrays, post-block sites only, in layer order."""
    post = sorted((s for s in r["cv"] if s.endswith("resid_post")), key=layer_of)
    L = np.array([layer_of(s) for s in post])                                   # [n_layers]
    read = {k: np.array([r["cv"][s][k] for s in post]) for k in ("etaA", "etaB", "D")}
    steer = {"etaA": np.array([r["posterior_write_scan_slopes"]["A"][s] for s in post]),
             "etaB": np.array([r["posterior_write_scan_slopes"]["B"][s] for s in post]),
             "D": np.array([r["write_scan_slopes"]["own"][s] for s in post])}
    sites = r["sites"]
    marks = {"read": {"etaA": layer_of(sites["posterior"]), "etaB": layer_of(sites["posterior"]),
                      "D": layer_of(sites["value"])},
             "steer": {"etaA": layer_of(sites["posterior_write"]["A"]), "etaB": layer_of(sites["posterior_write"]["B"]),
                       "D": layer_of(sites["value_write"])}}
    assert len(L) >= 2 and all(v.shape == L.shape for v in read.values()), "ragged profile"
    return {"L": L, "read": read, "steer": steer, "marks": marks}


def panel(ax, L: np.ndarray, ys: dict, marks: dict, ylabel: str, legend: bool, xlabel: bool) -> None:
    ax.axhline(0, color="0.75", lw=0.6, zorder=0)
    for key, label, color, mk, ls in SERIES:
        ax.plot(L, ys[key], marker=mk, ms=3, lw=1.1, ls=ls, color=color, label=label, clip_on=False)
        ax.plot([marks[key]], [ys[key][L == marks[key]][0]], marker=mk, ms=6, mfc="none", mec=color,
                mew=0.8, ls="none", clip_on=False, zorder=4)
    PL.frame(ax)
    ax.set_ylim(-0.1, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("layer")
    if len(L) <= 6:
        ax.set_xticks(L)
    else:
        ax.set_xticks(np.arange(0, L.max() + 1, 5))
    ax.set_xlim(L.min() - 0.3, L.max() + 0.3)
    if legend:
        ax.legend(frameon=False, loc="lower right", handlelength=1.6)


def fig_depth(profiles: list, out: Path) -> None:
    n = len(profiles)
    W, H = PL.PAPER_W, round(0.2 + 1.2 * n, 2)                                # 2.6 in for two rows
    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(W, H))
        pw, ph = 2.1, 0.85                                                     # inches; panel box
        x0 = (0.55, 3.25)
        y0 = tuple(0.4 + 1.2 * (n - 1 - r) for r in range(n))                 # top row first
        letters = iter("abcdefghij")
        for row, (p, name) in enumerate(profiles):
            for col, (key, ylabel) in enumerate((("read", "cross-validated $R^2$"), ("steer", "choice-logit slope"))):
                ax = PL.axes_in(fig, x0[col], y0[row], pw, ph)
                panel(ax, p["L"], p[key], p["marks"][key], ylabel, legend=(row == 0 and col == 0),
                      xlabel=(row == len(profiles) - 1))
                PL.text_in(fig, x0[col] - 0.5, y0[row] + ph + 0.04, next(letters), fontsize=PL.PAPER_FS + 1,
                           fontweight="bold", va="bottom")
                if col == 0:
                    PL.text_in(fig, x0[col] - 0.5, y0[row] + ph / 2, name, rotation=90, va="center", ha="center",
                               fontsize=PL.PAPER_FS)
        fig.savefig(out, dpi=300)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=sorted(FIGURES), required=True)
    a = ap.parse_args()
    name, studies = FIGURES[a.figure]
    out = ROOT / "figures" / name
    profiles = [(profile(pinned_fit(s)), label) for s, label in studies]
    for (p, name) in profiles:
        print(f"== {name}: layers {p['L'].min()}..{p['L'].max()}; marks {p['marks']}")
        for key, label, *_ in SERIES:
            print(f"  {label:12s} read  " + " ".join(f"{v:6.3f}" for v in p["read"][key]))
            print(f"  {label:12s} steer " + " ".join(f"{v:6.3f}" for v in p["steer"][key]))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_depth(profiles, out)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
