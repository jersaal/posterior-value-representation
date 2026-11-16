"""08 — the paper's decoder-and-representation row, composed from two pinned
runs' saved data: 03's decoder row (portraits read at each quantity's own
site and carrying that site's eval-episode R², held-out proposition scatters
at the display site) beside 07's all-component variance totals per site
(posterior, offer value, card and offers, unexplained). No model, no
instruments: 03's `payload.npz` + `results.json` (`read_sites`) and 07's
`results.json` (`pca_accounting.natural`) carry everything, so a layout
change redraws in seconds (`--run-03` / `--run-07` pick unpinned runs while
iterating). One figure, `paper_decoder_flat_split.png`. The manifest records
which runs were drawn from; the headline repeats the two R² the figure
prints."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, plotting as PL, stats as SS   # noqa: E402
from jbo.subject import load_study                                 # noqa: E402

EXPERIMENT = "08_decoder_representation"
SOURCES = ("03_decoder_views", "07_intrinsic_dim")


def source_run(study: str, experiment: str, run_id: str | None) -> Path:
    """The run dir for `experiment`: `run_id` if given, else the study's pin."""
    pins = json.loads((A.ROOT / "studies" / study / "pins.json").read_text())
    run = A.ROOT / "results" / study / experiment / (run_id or pins[experiment])
    assert (run / "results.json").exists(), run
    return run


def load_decoder(run: Path) -> dict:
    """03's row inputs: portraits (decoded and exact at the display site), held-out scatters, header labels."""
    z = np.load(run / "payload.npz")
    res = json.loads((run / "results.json").read_text())
    assert "read_sites" in res, f"{run} predates the `read_sites` record; re-run 03_decoder_views"
    dec = {k: z[f"show_dec_{k}"] for k in ("etaA", "etaB", "D")}            # etaA/etaB [N, 3], D [N, 2]
    truth = {k: z[f"show_true_{k}"] for k in ("etaA", "etaB", "D")}
    sc = {k: z[k] for k in ("P_true", "P_hat", "D_true", "D_hat")}          # [M] each
    assert "portrait_sites" in res, f"{run} predates the `portrait_sites` record; re-run 03_decoder_views"
    nat, ps = res["natural_r2_by_site"], res["portrait_sites"]              # eval-episode R² of the frozen decoder
    cell_r2 = {"etaA": nat[ps["posterior"]]["etaA"], "etaB": nat[ps["posterior"]]["etaB"], "D": nat[ps["value"]]["D"]}
    return {"dec": dec, "truth": truth, "sc": sc, "display_site": res["display_site"], "cell_r2": cell_r2,
            "read_sites": (res["read_sites"]["posterior"], res["read_sites"]["value"])}


def load_attribution(run: Path) -> dict:
    """07's row inputs: per site, natural-pool variance fractions, attribution by group, totals."""
    res = json.loads((run / "results.json").read_text())
    acc = res["pca_accounting"]["natural"]
    sites = tuple(res["sites"])
    for s in sites:
        att, vf = acc[s]["attribution"], np.array(acc[s]["var_frac"])
        assert np.allclose(sum(np.array(att[g]) for g in PL.GROUPS + ("unexplained",)), vf), s   # partition per PC
    return {"acc": acc, "sites": sites, "n_eval": res["n_eval"]}


# the decoder row narrowed to make room for the all-component totals at its right
ROW_CELL, ROW_SQ, ROW_GAP, ROW_X0 = 0.81, 0.65, 0.27, 0.13   # inches; portrait cell, scatter side, label gap, left edge
TOTALS_GAP, TOTALS_RIGHT = 0.16, 0.40                        # gap after (c), holding the letter; right margin for the totals' tick labels and two-line y label
TOTALS_MIN_W = 0.45                                           # inches; four bars need at least this
Y_SC, LEGEND_ROOM = 0.30, 0.16                                # the scatters' floor (as placed by the drawer); headroom for the one-line group legend


def fig_paper_row(dec: dict, att: dict, path: Path) -> None:
    """One column-width row: (a–c) 03's decoder row without its theory row,
    (d) 07's all-component variance totals per site with context and
    unexplained as separate bands. The group legend is one line in the header
    band, right-aligned above (c, d); the panel letters sit at their axes'
    top-left. Paper typography, no titles."""
    fs = PL.PAPER_FS
    row_h = PL.decoder_row_height(ROW_CELL, False) + LEGEND_ROOM
    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(PL.PAPER_W, row_h))
        x_end = PL.draw_decoder_row(fig, 0.0, dec["dec"], dec["truth"], dec["sc"], dec["read_sites"], letters=("a", "b", "c"),
                                    r2=SS.r2, cell_r2=dec["cell_r2"], x0=ROW_X0, cell=ROW_CELL, sq=ROW_SQ,
                                    gap=ROW_GAP, axis_words=False, theory=False,
                                    row_label_x=ROW_X0 - 0.02)   # clear of the s0 vertex label in the narrower cell
        x_e = x_end + TOTALS_GAP
        w_e = PL.PAPER_W - TOTALS_RIGHT - x_e
        assert w_e >= TOTALS_MIN_W, f"totals panel too narrow ({w_e:.2f} in)"
        ax = PL.axes_in(fig, x_e, Y_SC, w_e, ROW_SQ)
        sites = PL.depth_subset(att["sites"])
        PL.stacked_totals(ax, [att["acc"][s]["totals"] for s in sites], [PL.SITE_LABELS[s] for s in sites], merge_other=False)
        ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "", "1"])   # the middle tick keeps its mark: its "0.5" would cost the bars 0.09 in
        ax.set_xlabel("site", labelpad=1)
        PL.frame(ax)
        ax.yaxis.tick_right(); ax.spines["right"].set_visible(True); ax.spines["left"].set_visible(False)   # last panel: axis outward
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("fraction of\nvariance", labelpad=3)   # beside the outward ticks, reading upward like the row's other y labels
        fig.canvas.draw()
        assert ax.yaxis.label.get_window_extent().x1 / fig.dpi <= PL.PAPER_W - 0.005, "y label leaves the figure"
        groups = PL.plot_groups(False)
        handles = [Patch(color=PL.GROUP_COLOR[g], label=PL.PAPER_GROUP_LABEL[g]) for g in groups]
        leg = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=((PL.PAPER_W - 0.03) / PL.PAPER_W, (row_h - 0.01) / row_h),
                        bbox_transform=fig.transFigure, ncol=len(groups), frameon=False, handlelength=1.1, handleheight=1.0,
                        columnspacing=0.7, handletextpad=0.3, borderaxespad=0.0)   # swatch height kept: the two greys must stay apart
        fig.canvas.draw()
        bb = leg.get_window_extent()
        assert bb.y0 / fig.dpi >= Y_SC + ROW_SQ + 0.05, "legend line sits on the panel letters"   # letters top out 0.04 in above the axes
        assert bb.x0 / fig.dpi >= ROW_X0 + 3 * ROW_CELL + 0.05, "legend line reaches the portrait headers"
        PL.text_in(fig, x_e - 0.14, Y_SC + ROW_SQ + 0.04, "d", fontsize=fs + 1, fontweight="bold", ha="left", va="top")   # in the gap, off (c)
        fig.savefig(path, dpi=300); plt.close(fig)


def main() -> None:
    def extra(p):
        p.add_argument("--run-03", default=None, help="03_decoder_views run-id (default: pinned)")
        p.add_argument("--run-07", default=None, help="07_intrinsic_dim run-id (default: pinned)")
    args = cli.parse("08_decoder_representation: compose the decoder and representation row from pinned runs", extra)
    study = load_study(args.study)
    runs = {"03_decoder_views": source_run(args.study, "03_decoder_views", args.run_03),
            "07_intrinsic_dim": source_run(args.study, "07_intrinsic_dim", args.run_07)}
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dec, att = load_decoder(runs["03_decoder_views"]), load_attribution(runs["07_intrinsic_dim"])
    f1 = run_dir / "figures" / "paper_decoder_flat_split.png"
    fig_paper_row(dec, att, f1)
    sc = dec["sc"]
    results = {"headline": {
        "prop_composed_r2_pooled": A.headline_entry(SS.r2(sc["P_true"], sc["P_hat"]), len(sc["P_true"]), "R²", "pooled R² of composed P̂(X) over the drawn held-out cloud (panel b), from 03"),
        "prop_direct_r2_pooled": A.headline_entry(SS.r2(sc["D_true"], sc["D_hat"]), len(sc["D_true"]), "R²", "pooled R² of direct Q̂(X) over the drawn held-out cloud (panel c), from 03")},
        "sources": {k: v.name for k, v in runs.items()}, "display_site": dec["display_site"],
        "read_sites": {"posterior": dec["read_sites"][0], "value": dec["read_sites"][1]}, "sites": list(att["sites"])}
    A.write_results(run_dir, results)
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv, seeds={},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                     figures=[f1], extra={"sources": {k: str(v.relative_to(A.ROOT)) for k, v in runs.items()},
                                          "source_sha256": {k: A.sha256_file(v / "results.json") for k, v in runs.items()}})
    print(json.dumps(results["headline"], indent=1)); print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
