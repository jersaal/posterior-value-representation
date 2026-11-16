"""One column-width figure for the pretrained transformer, mirroring the toy's Figures 2 and 3:
(a) decoded factor posteriors for both factors at the posterior read site,
(b) decoded offer value $Q(X)$ against exact over the held-out propositions at the display site,
(c) the factor-A posterior steer, observed against exact counterfactual,
with its off-plane control (as Figure 3a), (d) the value direction sweep against theory with the
orthogonal-complement control. Redrawn from the pinned 03 and 06 payloads
with the paper helpers; no titles or panel headings, one text size.
Figure text uses the paper's terms: posterior, offer value, $Q$.
Usage: fig_pretrained.py --study mess3_m1m2_qwen3/s0 --out figures/Fig4_pretrained.png"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jbo import plotting as PL, stats as SS   # noqa: E402

CELL, SQ, RING_W = 0.78, 0.78, 1.10               # simplex cell, scatter side, ring panel width
Y0 = 0.36                                         # panels' floor (tick labels and axis labels below)
HEAD_Y = Y0 + SQ + 0.03                           # inches; baseline of the header line over (b)-(d), and of
H = HEAD_Y + 0.11                                 # the factor titles over (a); one 7 pt line of headroom above
WRITE_COLOR, THEORY_COLOR = "#5b3f99", "#cf6f16"   # choice quantities: purple / orange
IDENTITY = {"color": "0.4", "lw": 0.6, "ls": (0, (3, 2)), "zorder": 0}
ZERO = {"color": "0.4", "lw": 0.6, "ls": (0, (1, 1.5)), "zorder": 0}


def pinned(study: str, experiment: str) -> Path:
    pins = json.loads((ROOT / "studies" / study / "pins.json").read_text())
    return ROOT / "results" / study / experiment / pins[experiment]


def halo_text(ax, x, y, s, color, fs, ha):
    ax.text(x, y, s, color=color, fontsize=fs - 1.5, ha=ha, va="center", transform=ax.transAxes, zorder=7,
            path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="mess3_m1m2_qwen3/s0")
    ap.add_argument("--out", default="figures/Fig4_pretrained.png")
    a = ap.parse_args()
    d04, d07 = pinned(a.study, "03_decoder_views"), pinned(a.study, "06_steering")
    z4 = dict(np.load(d04 / "payload.npz", allow_pickle=True)); r4 = json.loads((d04 / "results.json").read_text())
    z7 = dict(np.load(d07 / "payload.npz", allow_pickle=True)); h7 = json.loads((d07 / "results.json").read_text())["headline"]
    read_sites = r4["read_sites"]                                   # {"posterior": "L0", "value": "L16"}
    nat = r4["natural_r2_by_site"]
    bsite = r4["portrait_sites"]["posterior"]
    cell_r2 = {k: float(nat[bsite][k]) for k in ("etaA", "etaB")} if isinstance(nat.get(bsite), dict) else {}
    fs = PL.PAPER_FS
    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(PL.PAPER_W, H))
        top = H - 0.02
        letter = lambda x_in, s: fig.text(x_in / PL.PAPER_W, top / H, s, fontsize=fs + 1, fontweight="bold", ha="left", va="top")

        # (a) posterior simplices, both factors, at the posterior read site ---------------------------
        x_a = 0.12
        for c, (key, title) in enumerate((("etaA", "factor M"), ("etaB", "factor N"))):
            ax = PL.axes_in(fig, x_a + c * CELL, Y0 - 0.06, CELL, CELL)
            PL.posterior_panel(ax, z4[f"show_dec_{key}"], z4[f"show_true_{key}"], vertex_labels=True, s=1.0, fs=fs)
            if key in cell_r2:
                ax.text(0.5, 0.0, f"$R^2$ = {cell_r2[key]:.3f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=fs - 1)
            PL.text_in(fig, x_a + c * CELL + CELL / 2, HEAD_Y, title, ha="center", va="bottom", fontsize=fs)
        PL.text_in(fig, x_a + CELL, HEAD_Y - 0.09, f"readable from {read_sites['posterior']}",
                   ha="center", va="bottom", fontsize=fs - 1, color="0.45")
        letter(0.02, "a")

        # (b) held-out offer value -------------------------------------------------------------
        x_b = x_a + 2 * CELL + 0.36
        ax = PL.axes_in(fig, x_b, Y0, SQ, SQ)
        t, q = z4["D_true"], z4["D_hat"]
        ax.scatter(t, q, s=0.6, color=PL.VALUE_COLOR, alpha=0.15, linewidths=0, rasterized=True)
        ax.axline((-1, -1), (1, 1), **IDENTITY)
        ax.text(0.97, 0.04, f"$R^2$ = {SS.r2(t, q):.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs - 1)
        ax.set(xticks=[-1, 0, 1], yticks=[-1, 0, 1], xticklabels=["−1", "0", "1"], yticklabels=["−1", "0", "1"])
        ax.set_xlabel("exact $Q(X)$", labelpad=1); ax.set_ylabel("decoded $\\hat{Q}(X)$", labelpad=1)
        PL.text_in(fig, x_b + SQ / 2, HEAD_Y, f"read at {PL.SITE_LABELS[r4['display_site']]}",
                   ha="center", va="bottom", fontsize=fs - 1, color="0.45")
        PL.frame(ax); letter(x_b - 0.30, "b")

        # (c) posterior transplant, factor A, with its off-plane control (as Figure 4a) -------------
        x_c = x_b + SQ + 0.36
        ax = PL.axes_in(fig, x_c, Y0, SQ, SQ)
        dZ, dY = z7["posterior_A_dZ"], z7["posterior_A_dY"]
        lim = float(np.percentile(np.abs(dZ), 99.5))
        ax.scatter(z7["posterior_A_ctrl_dZ"], z7["posterior_A_ctrl_dY"], s=0.6, color="0.6", alpha=0.05, linewidths=0, rasterized=True)
        ax.scatter(dZ, dY, s=0.8, color=PL.POSTERIOR_A_COLOR, alpha=0.2, linewidths=0, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], **IDENTITY)
        tk = float(np.floor(lim)) if lim >= 1.5 else round(lim / 2, 1)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[-tk, 0, tk], yticks=[-tk, 0, tk])
        ax.set_xlabel("predicted logit shift", labelpad=1); ax.set_ylabel("observed shift", labelpad=1)
        halo_text(ax, 0.05, 0.92, f"slope {h7['transfer_slope_A']['value']:.3f}", PL.POSTERIOR_A_COLOR, fs, "left")
        halo_text(ax, 0.05, 0.81, f"control {h7['control_slope_A']['value']:.3f}", "0.45", fs, "left")
        PL.frame(ax); letter(x_c - 0.30, "c")

        # (d) value ring against theory, with the orthogonal-complement control --------------------
        x_d = x_c + SQ + 0.36
        assert x_d + RING_W <= PL.PAPER_W + 1e-6, "row does not fit the column width"
        ax = PL.axes_in(fig, x_d, Y0, RING_W, SQ)
        angles, x, y, yc = z7["ring_angles"], z7["ring_x"], z7["ring_y"], z7["ring_y_control"]
        order = np.argsort(angles)
        ang, pred, obs, ctrl = angles[order], x.mean(1)[order], y.mean(1)[order], yc.mean(1)[order]
        err = (1.96 * y.std(1, ddof=1) / np.sqrt(y.shape[1]))[order]
        tx, ty = np.r_[ang[-1] - 360, ang, ang[0] + 360], np.r_[pred[-1], pred, pred[0]]
        ax.axhline(0, **ZERO)
        ax.plot(tx, ty, color=THEORY_COLOR, lw=2.6, alpha=0.5, solid_capstyle="round", label="predicted", zorder=2)
        ax.errorbar(ang, obs, yerr=err, color=WRITE_COLOR, marker="o", ms=2.6, lw=0.8, capsize=1.2, label="value write", zorder=3)
        ax.plot(ang, ctrl, color="0.55", marker="s", ms=2.4, lw=0.8, label="complement write")
        peak = float(np.abs(np.r_[obs + err, pred]).max())
        ax.set(xlim=(0, 360), xticks=np.arange(0, 361, 90), ylim=(-1.15 * peak, 2.6 * peak))
        ax.set_xlabel("write direction (°)", labelpad=1); ax.set_ylabel("mean logit shift", labelpad=1)
        ax.legend(frameon=False, loc="upper left", handlelength=1.1, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.2)
        halo_text(ax, 0.97, 0.17, f"slope {h7['ring_slope']['value']:.3f}", WRITE_COLOR, fs, "right")
        halo_text(ax, 0.97, 0.06, f"control {h7['ring_control_slope']['value']:.3f}", "0.55", fs, "right")
        PL.frame(ax); letter(x_d - 0.30, "d")

        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300); plt.close(fig)
    print(f"wrote {out} from {d04.name} and {d07.name}")


if __name__ == "__main__":
    main()
