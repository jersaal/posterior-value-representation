"""05 — dissociation by twins: replace the card, the standing
masks, or round 5's clues from a donor episode; card/mask twins leave the
analytic posterior bit-identical (asserted), the clue twin moves it. Median
decoded posterior and value shift norms per twin, each quantity at its own
read site (posterior at the posterior site, value at the value site), over
the episodes whose offers are both valid in the base and in every twin.
Figure: `paper_dissociation.png`, the shift distributions as paired boxes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, instruments as I, plotting as PL, probes as P, stimuli as ST  # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB   # noqa: E402
from jbo.subject import load_study                          # noqa: E402

EXPERIMENT = "05_dissociation"
N_EVAL, SMOKE_EVAL = 8192, 512
ARMS = ("identity", "card", "masks", "clues5")
PLOT_ARMS = ("card", "masks", "clues5")
ARM_LABEL = {"card": "card", "masks": "mask", "clues5": "clue"}
WHISKERS = (5, 95)


def fig_paper(arrays: dict, path: Path) -> None:
    """Shift distributions per twin as paired boxes (quartiles, median, 5–95
    whiskers), wide and short to sit beside the twin-construction schematic at
    the schematic's height: no title, one text size, legend inside the axes at the upper
    right; the caption carries the units (posterior in simplex tangent units,
    offer value in payoff units)."""
    fs = PL.PAPER_FS
    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(2.35, 1.15))
        ax = fig.add_axes([0.17, 0.30, 0.80, 0.67])
        ys = 1.0 * np.arange(len(PLOT_ARMS))[::-1]
        hi = 0.0
        for y, arm in zip(ys, PLOT_ARMS):
            hi = max(hi, PL.draw_box(ax, y + 0.21, arrays[arm]["posterior"], "#a6c8ec", PL.POSTERIOR_A_COLOR, 0.17, WHISKERS))
            hi = max(hi, PL.draw_box(ax, y - 0.21, arrays[arm]["value"], "#f4a89d", PL.VALUE_COLOR, 0.17, WHISKERS))
        ax.set_yticks(ys); ax.set_yticklabels([ARM_LABEL[a] for a in PLOT_ARMS])
        ax.set_xlim(-0.03, 1.04 * hi); ax.set_ylim(-0.6, ys[0] + 1.35)   # headroom above the top row for a one-line legend
        ax.set_xticks(np.arange(0, hi + 0.01, 0.5))
        ax.set_xlabel("shift in decoded value", labelpad=1)
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="y", length=0, pad=2)
        ax.legend([Patch(facecolor="#a6c8ec", edgecolor=PL.POSTERIOR_A_COLOR), Patch(facecolor="#f4a89d", edgecolor=PL.VALUE_COLOR)],
                  ["posterior", "offer value"], frameon=False, loc="upper right", fontsize=fs, ncol=2,
                  handlelength=1.4, handletextpad=0.5, columnspacing=1.0, borderaxespad=0.0)
        fig.savefig(path, dpi=300); plt.close(fig)


def main() -> None:
    args = cli.parse("05_dissociation: twin stimuli")
    study = load_study(args.study)
    world = study.build_world()
    model, _ = study.load_actor(args.device)
    seed = study.seeds["eval"]
    donor_seed = seed + 1
    ins = I.load(study, eval_seeds=[seed, donor_seed])
    n = SMOKE_EVAL if args.smoke else N_EVAL
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    b_site, d_site, pos = ins.posterior_site, ins.value_site, np.array([GRADE_POSITION])
    read_sites = tuple(dict.fromkeys((b_site, d_site)))     # deduped: the two can coincide on some seeds

    base = world.sample(n, seed=seed)["tokens"]
    donor = world.sample(n, seed=donor_seed)["tokens"]
    key_base = world.answer_key(base)
    twins = {arm: ST.build_twin(world, base, donor, arm) for arm in ARMS}
    keys = {arm: world.answer_key(t) for arm, t in twins.items()}
    for arm in ("identity", "card", "masks"):
        assert np.array_equal(keys[arm]["eta"], key_base["eta"]), f"{arm} twin moved the analytic posterior"
    assert np.abs(keys["clues5"]["eta_ro"][:, GRADE_READOUT] - key_base["eta_ro"][:, GRADE_READOUT]).max() > 1e-3
    keep = key_base["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    for arm in ARMS:
        keep &= keys[arm]["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB

    def decode(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        acts = P.collect_site_activations(model, tokens, read_sites, pos, args.device)
        e_b = ins.decode(b_site, acts[b_site][:, 0, :])
        e_d = ins.decode(d_site, acts[d_site][:, 0, :])
        post = np.concatenate([ins.to_tangent(ins.posterior_A(e_b)), ins.to_tangent(ins.posterior_B(e_b))], 1)   # [N, 4]
        return post, ins.D(e_d)

    post_b, D_b = decode(base)
    medians, arrays = {}, {}
    for arm in ARMS:
        post_t, D_t = decode(twins[arm])
        d_post = np.linalg.norm((post_t - post_b)[keep], axis=1)               # [n_kept] tangent units, both factors
        d_D = np.linalg.norm((D_t - D_b)[keep], axis=1)                     # [n_kept] payoff units
        arrays[arm] = {"posterior": d_post, "value": d_D}
        medians[arm] = {"posterior": float(np.median(d_post)), "value": float(np.median(d_D))}
        print(f"  {arm:9s} posterior {medians[arm]['posterior']:.4f}  value {medians[arm]['value']:.4f}", flush=True)
    for k in ("posterior", "value"):
        assert np.sqrt((arrays["identity"][k] ** 2).mean()) < 1e-9, "identity twin not machine-exact"

    fig = run_dir / "figures" / "paper_dissociation.png"
    fig_paper(arrays, fig)
    np.savez(run_dir / "payload.npz", **{f"{a}_{k}": v for a, d in arrays.items() for k, v in d.items()})   # the boxes' data
    n_kept = int(keep.sum())
    head = {}
    for arm in PLOT_ARMS:
        head[f"posterior_shift_median_{arm}"] = A.headline_entry(medians[arm]["posterior"], n_kept, "tangent", f"median decoded posterior shift, {arm} twin")
        head[f"value_shift_median_{arm}"] = A.headline_entry(medians[arm]["value"], n_kept, "utility", f"median decoded value shift, {arm} twin")
    A.write_results(run_dir, {"headline": head, "n_eval": n, "n_kept": n_kept,
                              "read_sites": {"posterior": b_site, "value": d_site}})
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv, seeds={"eval": seed, "donor": donor_seed},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force, figures=[fig],
                     extra={"checkpoint": study.checkpoint, "instruments_sha256": ins.meta["artifact_sha256"]})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
