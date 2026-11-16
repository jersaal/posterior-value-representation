"""04 — representation formation across training: for every
snapshot of the study's 00 run, decode 4,096 eval episodes at each
quantity's read site (posterior at the posterior site, value at the value
site, as in 03's portraits) with a fresh throwaway 5-fold CV probe per
site, read out-of-fold (the per-snapshot fit of Shai et al. 2026). The
step-0 probes, fit to the network at initialisation, are the null the
trained network's reads are compared with. The fresh probes here are the
one sanctioned fit outside 02; they are discarded on the spot. Headline:
the step-0 R² and the first snapshot at which each quantity reaches 0.9.
Figure: `paper_formation.png`, the fresh-probe portrait grid at the
display steps, at column width."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import ROOT, artifacts as A, cli, instruments as I, model as M, plotting as PL, probes as P  # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB   # noqa: E402
from jbo.subject import load_study                          # noqa: E402

EXPERIMENT = "04_formation"
N_EVAL, N_FOLDS, N_SHOW = 4096, 5, 3000
SMOKE_EVAL = 512
DISPLAY_STEPS = (0, 256, 1000, 4000, 8000, 16000, 24000)
ROLE_FOLDS, ROLE_SHOW = 1, 2
THRESH = 0.9


def snapshot_index(study) -> tuple[Path, list[dict]]:
    """The 00 run that produced the pinned checkpoint and its snapshot index."""
    run = (ROOT / study.checkpoint["path"]).parent.parent
    idx = json.loads((run / "snapshots.json").read_text())
    for s in idx["snapshots"]:
        assert A.sha256_file(run / "snapshots" / s["file"]) == s["sha256"], s["file"]
    return run, idx["snapshots"]


def load_snapshot(study, d_vocab: int, path: Path, device: str):
    model = M.build_model(study.arch, d_vocab, seed=0, device="cpu")
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return model.to(device).eval()


def fresh_cv(X: np.ndarray, Y: np.ndarray, valid: np.ndarray, folds: list) -> dict:
    """Throwaway grouped-CV probe, scored per target group on out-of-fold
    predictions; returns the scores and the full OOF decode [E, 8]."""
    oof = P.cv_decode(X[:, None, :], Y[:, None, :], folds, valid[:, None])["oof"][:, 0, :]
    r2 = {g: P.decode_scores(Y[valid][:, lo:hi], oof[valid][:, lo:hi])["r2"] for g, (lo, hi) in GROUPS.items()}
    return {"r2": r2, "oof": oof}


GROUPS = {"etaA": (0, 3), "etaB": (3, 6), "D": (6, 8)}


def fig_paper_formation(steps: list, dec: dict, truth: dict, r2: dict, path: Path) -> None:
    """Column-width formation grid, paper typography, no titles: columns are
    training steps plus the analytic ground truth; rows are the two factor
    posteriors and the value pair, decoded out-of-fold by the per-snapshot
    fresh probes; each panel carries its fresh-probe R² bottom-left; the
    last column keys both colour scales."""
    fs = PL.PAPER_FS
    n = len(steps) + 2
    ML, MR, MT, MB, GX, GY = 0.34, 0.12, 0.16, 0.20, 0.05, 0.13   # inches; row gaps hold the R² line
    cell = (PL.PAPER_W - ML - MR - (n - 1) * GX) / n
    H = MT + 3 * cell + 2 * GY + MB
    gap = truth["D"][:, 0] - truth["D"][:, 1]
    span = float(np.abs(truth["D"]).max()) * 1.15
    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(PL.PAPER_W, H))

        def cell_ax(row: int, col: int):
            x0 = ML + col * (cell + GX)
            y0 = MB + (2 - row) * (cell + GY)
            return fig.add_axes([x0 / PL.PAPER_W, y0 / H, cell / PL.PAPER_W, cell / H])

        def column(col: int, header: str, e: np.ndarray, r2_col: dict | None) -> None:
            axA, axB, axD = cell_ax(0, col), cell_ax(1, col), cell_ax(2, col)
            gt = r2_col is None
            PL.posterior_panel(axA, e[:, 0:3], truth["etaA"], vertex_labels=gt, s=0.7, fs=fs)
            PL.posterior_panel(axB, e[:, 3:6], truth["etaB"], vertex_labels=gt, s=0.7, fs=fs)
            PL.value_panel(axD, e[:, 6:8], gap, s=0.7)
            axD.set_xlim(-span, span); axD.set_ylim(-span, span)
            x_mid = (ML + col * (cell + GX) + cell / 2) / PL.PAPER_W
            fig.text(x_mid, (H - MT + 0.02) / H, header, ha="center", va="bottom", fontsize=fs)
            if r2_col is not None:
                for ax, g in ((axA, "etaA"), (axB, "etaB"), (axD, "D")):
                    ax.text(0.02, -0.04, f"$R^2$ = {r2_col[g]:.2f}", transform=ax.transAxes,
                            ha="left", va="top", fontsize=fs - 1, color="0.35")

        for j, st in enumerate(steps):
            column(j, f"step {st:,}" if j == 0 else f"{st:,}", dec[st], r2[st])
        column(n - 2, "ground truth", np.concatenate([truth["etaA"], truth["etaB"], truth["D"]], 1), None)
        kx0 = ML + (n - 1) * (cell + GX)
        fig.text((ML + (n - 2) * (cell + GX) + cell / 2) / PL.PAPER_W, (MB - 0.10) / H,
                 "Q(A)", ha="center", va="bottom", fontsize=fs)
        fig.text((kx0 - GX / 2) / PL.PAPER_W, (MB + cell / 2) / H,
                 "Q(B)", rotation=90, ha="center", va="center", fontsize=fs)
        # key column: RGB simplex key centred on the posterior rows, gap colourbar on the value row
        y_lo, y_hi = MB + (cell + GY), MB + 2 * (cell + GY) + cell
        ax_key = fig.add_axes([kx0 / PL.PAPER_W, (y_lo + y_hi - cell) / 2 / H, cell / PL.PAPER_W, cell / H])
        PL.posterior_key(ax_key, fs=fs, title=False)
        cax = fig.add_axes([(kx0 + 0.12) / PL.PAPER_W, (MB + 0.06) / H, 0.07 / PL.PAPER_W, (cell - 0.12) / H])
        cb = plt.colorbar(ScalarMappable(norm=PL.GAP_NORM, cmap=PL.GAP_CMAP), cax=cax, extend="both")
        cb.set_ticks([-2.0, 0.0, 2.0]); cb.set_ticklabels(["−2", "0", "+2"]); cb.ax.tick_params(labelsize=fs - 1)
        cb.set_label("gap Q(A) − Q(B)", fontsize=fs - 1)
        for row, name in enumerate(("posterior M", "posterior N", "offer value")):
            y_mid = (MB + (2 - row) * (cell + GY) + cell / 2) / H
            fig.text((ML - 0.22) / PL.PAPER_W, y_mid, name, rotation=90, ha="center", va="center", fontsize=fs)
        fig.savefig(path, dpi=300)
        plt.close(fig)


def main() -> None:
    args = cli.parse("04_formation: representation formation across snapshots")
    study = load_study(args.study)
    world = study.build_world()
    seed = study.seeds["eval"]
    ins = I.load(study, eval_seeds=[seed])
    run00, snaps = snapshot_index(study)
    n_eval = SMOKE_EVAL if args.smoke else N_EVAL
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    tokens = world.sample(n_eval, seed=seed)["tokens"]
    key = world.answer_key(tokens)
    valid = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    Y = np.concatenate([key["etaA"][:, GRADE_POSITION], key["etaB"][:, GRADE_POSITION], key["D"][:, GRADE_READOUT]], 1)
    folds = P.grouped_folds(n_eval, N_FOLDS, np.random.default_rng([seed, ROLE_FOLDS]))
    show = np.random.default_rng([seed, ROLE_SHOW]).choice(np.flatnonzero(valid), size=min(N_SHOW, int(valid.sum())), replace=False)
    b_site, d_site = ins.posterior_site, ins.value_site
    site_of = {"etaA": b_site, "etaB": b_site, "D": d_site}   # each quantity at its read site, as in 03's portraits
    sites_needed = tuple(dict.fromkeys((b_site, d_site)))     # b_site == d_site in some studies
    pos = np.array([GRADE_POSITION])

    schedule = [s["step"] for s in snaps]
    fresh, dec_show = {}, {}
    for s in snaps:
        st = s["step"]
        model = load_snapshot(study, world.d_vocab, run00 / "snapshots" / s["file"], args.device)
        acts = P.collect_site_activations(model, tokens, sites_needed, pos, args.device)
        X = {site: acts[site][:, 0, :].astype(np.float64) for site in sites_needed}
        fresh_by_site = {site: fresh_cv(X[site], Y, valid, folds) for site in sites_needed}
        fresh[st] = {g: fresh_by_site[site_of[g]]["r2"][g] for g in GROUPS}
        e = np.concatenate([fresh_by_site[site_of[g]]["oof"][:, lo:hi] for g, (lo, hi) in GROUPS.items()], 1)
        dec_show[st] = e[show]
        print(f"  step {st:6d}  fresh {fresh[st]}", flush=True)
        del model
    # display columns: the registered steps, or their nearest scheduled neighbours
    steps = sorted({min(schedule, key=lambda s: abs(s - d)) for d in DISPLAY_STEPS})
    truth_show = {"etaA": key["etaA"][show, GRADE_POSITION], "etaB": key["etaB"][show, GRADE_POSITION],
                  "D": key["D"][show, GRADE_READOUT]}
    fig = run_dir / "figures" / "paper_formation.png"
    fig_paper_formation(steps, dec_show, truth_show, fresh, fig)

    def first_above(key_: str):
        hits = [s for s in schedule if fresh[s][key_] >= THRESH]
        return hits[0] if hits else None
    n_valid = int(valid.sum())
    assert schedule[0] == 0, "the first snapshot must be the network at initialisation"
    head = {}
    for g in GROUPS:
        head[f"fresh_r2_step0_{g}"] = A.headline_entry(fresh[0][g], n_valid, "R²",
                                                       f"fresh-probe CV R² ({g}) at step 0, the network at initialisation")
    for g in GROUPS:
        st = first_above(g)
        head[f"fresh_r2_first_above_{THRESH}_{g}"] = A.headline_entry(st if st is not None else float("nan"), n_valid, "step",
                                                                      f"first snapshot where fresh-probe CV R² ({g}) ≥ {THRESH}")
    results = {"headline": head, "schedule": schedule, "fresh": fresh,
               "read_sites": site_of, "display_steps": steps, "n_eval": n_eval, "n_valid": n_valid}
    A.write_results(run_dir, results)
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={"eval": seed, "folds": [seed, ROLE_FOLDS], "show": [seed, ROLE_SHOW]},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                     figures=[fig], extra={"checkpoint": study.checkpoint, "snapshots_run": str(run00.relative_to(ROOT)),
                                              "instruments_sha256": ins.meta["artifact_sha256"]})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
