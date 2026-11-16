"""07 — what the residual stream represents: principal components of the
residual stream at readout 5 on the evaluation episodes, each component's
variance attributed to an analytic quantity.

At every post-block site the activations are rotated to their principal
components and every component score is regressed (fit on half the episodes,
scored on the other half) on three feature groups — posterior (the joint
posterior, 9), value (D_A, D_B) and context (the payoff card as 27 one-hot
levels and both offers' 18 mask bits) — and on all of them together. The
all-features R² is the explained share of that component's variance; it is
split between the groups in proportion to their own R², and the remainder is
unexplained. Summed over components this gives each quantity's share of the
site's total variance.

Among the posterior site's eight leading components, each factor posterior is
assigned the pair its tangent coordinates explain best (`factor_pairs`).

Figure: `paper_representation_no_totals_split.png` — (a) the first eight
components' variance by group at each site, (b) the posterior site's factor
pairs coloured by the analytic factor posteriors, (c) the display site's first
two components coloured by the analytic gap. `results.json` carries the
per-site accounting (`pca_accounting.natural`, read by 08) and
`factor_pairs`. Uses `instruments` only to name the posterior, value and
display sites."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, instruments as I, plotting as PL, probes as P, stats as SS  # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT      # noqa: E402
from jbo.subject import load_study                             # noqa: E402

EXPERIMENT = "07_intrinsic_dim"
N_EVAL, SMOKE_EVAL, N_SHOW = 8192, 768, 3000
ROLE_SPLIT, ROLE_SHOW = 5, 6                       # default_rng([eval_seed, role])
GROUPS, GROUP_LABEL = PL.GROUPS, PL.GROUP_LABEL    # shared with the paper rows


def feature_groups(key: dict) -> dict:
    """Analytic feature groups at readout 5: posterior [N, 9] (joint posterior,
    which spans the factor marginals and their products), value [N, 2],
    context [N, 45] (card one-hot 27 + both offers' bits 18), all [N, 56]."""
    N = len(key["u"])
    g = {"posterior": key["eta"][:, GRADE_POSITION], "value": key["D"][:, GRADE_READOUT],
         "context": np.column_stack([(key["u"][:, :, None] == np.array([-1, 0, 1])).reshape(N, -1).astype(np.float64),
                                     key["bits_ro"][:, GRADE_READOUT].reshape(N, -1)])}
    g["all"] = np.column_stack([g[k] for k in GROUPS])
    return g


def pc_scores(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal-component scores [N, d] (descending variance) and variance fractions [d]."""
    Xc = X - X.mean(0)
    ev, V = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
    ev, V = ev[::-1], V[:, ::-1]
    return Xc @ V, ev / ev.sum()


def attribute(var_frac: np.ndarray, r2: dict) -> dict:
    """Split each PC's variance fraction between the groups in proportion to
    their R², scaled so the groups sum to the all-features R²; the rest is
    unexplained. {group: [d]} summing to var_frac per PC."""
    parts = np.clip(np.nan_to_num(np.array([r2[g] for g in GROUPS])), 0, 1)     # [3, d]; a constant PC reads as 0
    tot = np.clip(np.nan_to_num(np.asarray(r2["all"])), 0, 1)
    s = parts.sum(0)
    tot = np.where(s > 0, tot, 0.0)     # a PC no single group explains counts as unexplained (deep stacks have such tail PCs)
    share = np.where(s > 0, parts / np.where(s > 0, s, 1.0), 0.0) * tot
    return {**{g: (share[i] * var_frac).tolist() for i, g in enumerate(GROUPS)}, "unexplained": ((1 - tot) * var_frac).tolist()}


def pc_accounting(X: np.ndarray, feats: dict, rng: np.random.Generator) -> dict:
    """Every PC score of X regressed on each feature group (fit half, score the
    other half): {"var_frac": [d], group: [d] R², group + "_all": pooled R² of
    X, "attribution": {group: [d]}, "totals": {group: float}}."""
    scores, var_frac = pc_scores(X)
    perm = rng.permutation(len(X)); fit, test = perm[: len(X) // 2], perm[len(X) // 2:]
    out = {"var_frac": var_frac.tolist()}
    for name in GROUPS + ("all",):
        F = feats[name]
        beta = P.affine_fit(F[fit], scores[fit]); pred = P.affine_predict(F[test], beta)
        out[name] = [SS.r2(scores[test, j], pred[:, j]) for j in range(scores.shape[1])]
        out[name + "_all"] = P.decode_scores(scores[test], pred)["r2"]      # PCA is a rotation: pooled R² is basis-free
    out["attribution"] = attribute(var_frac, out)
    out["totals"] = {g: float(np.sum(v)) for g, v in out["attribution"].items()}
    return out


FACTOR_NAME = {"A": "M", "B": "N"}   # paper names of the two factors (Figure 1: Mess3 processes M and N)
N_LEAD = 8            # leading PCs among which the factor pairs are chosen


def factor_pairs(scores: np.ndarray, tA: np.ndarray, tB: np.ndarray) -> dict:
    """Among the N_LEAD leading PCs, the two each factor's tangent coordinates
    explain best (in-sample R²), assigned greedily so no PC serves both
    factors. {"A": [2], "B": [2], "r2": [N_LEAD, 2]} (PC indices ascending)."""
    r2 = np.array([[SS.r2(scores[:, j], P.affine_predict(t, P.affine_fit(t, scores[:, j]))) for t in (tA, tB)]
                   for j in range(N_LEAD)])                              # [N_LEAD, 2]
    chosen = {0: [], 1: []}
    for j, f in sorted(np.ndindex(r2.shape), key=lambda jf: -r2[jf]):
        if len(chosen[f]) < 2 and j not in chosen[0] + chosen[1]:
            chosen[f].append(int(j))
    assert len(chosen[0]) == 2 and len(chosen[1]) == 2
    return {"A": sorted(chosen[0]), "B": sorted(chosen[1]), "r2": r2.tolist()}


# ----------------------------------------------------------------------------- paper figures
def fig_paper_representation(post: tuple, acc_nat: dict, show: dict, site_labels: dict, path: Path) -> None:
    """Column-width figure of what the residual stream carries at readout 5.
    (a) per site: the first eight PCs' variance split by group (context and
    unexplained as separate bands); (b) posterior site: the PC pair each
    factor's posterior occupies among the leading components, coloured by the
    analytic factor posteriors as RGB; (c) display site: PC1 vs PC2 coloured
    by the analytic gap. Paper typography, no titles."""
    fs = PL.PAPER_FS
    H = 3.45
    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(PL.PAPER_W, H))
        # row 1: (a) the site panels
        y1 = 2.05
        PL.draw_attribution_row(fig, y1, acc_nat, PL.depth_subset(post), site_labels, totals=False, merge_other=False)
        assert y1 + PL.ATTRIBUTION_ROW_H <= H + 1e-6, "attribution row leaves the figure"

        # row 2: (b) posterior simplices from the posterior site's PCs, (c) value plane from the display site's, keys
        y2, h2 = 0.42, 1.0
        pcs, pairs = show["posterior_pc"], show["pairs"]
        for i, (fac, eta) in enumerate((("A", show["etaA"]), ("B", show["etaB"]))):
            ax = PL.axes_in(fig, 0.30 + i * 1.10, y2, 1.0, h2)
            j, k = pairs[fac]
            ax.scatter(pcs[:, j], pcs[:, k], s=0.9, c=np.clip(eta, 0, 1), alpha=0.75, linewidths=0, rasterized=True)
            PL.square_limits(ax, pcs[:, j], pcs[:, k]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"component {j + 1}"); ax.set_ylabel(f"component {k + 1}")
            ax.text(0.5, 1.02, f"factor {FACTOR_NAME[fac]} posterior", transform=ax.transAxes, ha="center", va="bottom", fontsize=fs)
            PL.frame(ax)
        fig.text(0.05 / PL.PAPER_W, (y2 + h2 + 0.14) / H, "b", fontsize=fs + 1, fontweight="bold")
        lead = sorted(pairs["A"] + pairs["B"])
        fig.text(0.28 / PL.PAPER_W, (y2 + h2 + 0.14) / H,
                 f"{site_labels[show['posterior_site']]}: components {', '.join(str(j + 1) for j in lead)}", fontsize=fs, va="bottom")
        x_d = 2.75
        ax = PL.axes_in(fig, x_d, y2, 1.0, h2)
        npc, gap = show["nat_pc"], show["nat_gap"]
        ax.scatter(npc[:, 0], npc[:, 1], s=0.9, c=gap, cmap=PL.GAP_CMAP, norm=PL.GAP_NORM, alpha=0.8, linewidths=0, rasterized=True)
        PL.square_limits(ax, npc[:, 0], npc[:, 1]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
        ax.text(0.5, 1.02, "gap Q(A) − Q(B) as colour", transform=ax.transAxes, ha="center", va="bottom", fontsize=fs)
        PL.frame(ax)
        fig.text((x_d - 0.25) / PL.PAPER_W, (y2 + h2 + 0.14) / H, "c", fontsize=fs + 1, fontweight="bold")
        fig.text((x_d - 0.02) / PL.PAPER_W, (y2 + h2 + 0.14) / H, f"{site_labels[show['nat_site']]}: components 1–2",
                 fontsize=fs, va="bottom")
        PL.posterior_key(PL.axes_in(fig, 3.88, y2 + 0.02, 0.62, h2 - 0.05), fs=fs, title=False)
        PL.value_key(PL.axes_in(fig, 4.42, y2 - 0.02, 0.62, h2 + 0.05), fs=fs, title=False)
        fig.savefig(path, dpi=300); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = cli.parse("07_intrinsic_dim: what the residual stream represents")
    study = load_study(args.study)
    world = study.build_world()
    model, _ = study.load_actor(args.device)
    seed = study.seeds["eval"]
    ins = I.load(study, eval_seeds=[seed])
    post = tuple(s for s in ins.sites if s != P.EMBED_SITE)
    n_eval = SMOKE_EVAL if args.smoke else N_EVAL
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    disp, bsite, vsite = ins.display_site, ins.posterior_site, ins.value_site

    tokens = world.sample(n_eval, seed=seed)["tokens"]
    key = world.answer_key(tokens)
    acts = P.collect_site_activations(model, tokens, post, np.array([GRADE_POSITION]), args.device)

    def at5(s: str) -> np.ndarray:
        """[N, d] float64 activations at site `s`, readout 5."""
        return acts[s][:, 0].astype(np.float64)

    # variance attribution at every post-block site. The split generator's first len(post)
    # permutations are passed over: the reported numbers use the splits that follow them.
    rng_split = np.random.default_rng([seed, ROLE_SPLIT])
    for _ in post:
        rng_split.permutation(n_eval)
    feats = feature_groups(key)
    acc = {s: pc_accounting(at5(s), feats, rng_split) for s in post}

    # the figure's scatters: posterior-site factor pairs and the display site's first two components
    idx = np.random.default_rng([seed, ROLE_SHOW]).choice(n_eval, size=min(N_SHOW, n_eval), replace=False)
    posterior_scores, _ = pc_scores(at5(bsite))
    tA, tB = P.to_tangent(key["etaA"][:, GRADE_POSITION], ins.basis), P.to_tangent(key["etaB"][:, GRADE_POSITION], ins.basis)
    pairs = factor_pairs(posterior_scores[:, :N_LEAD], tA, tB)
    disp_scores, _ = pc_scores(at5(disp))
    show = {"posterior_pc": posterior_scores[idx, :N_LEAD], "etaA": key["etaA"][idx, GRADE_POSITION], "etaB": key["etaB"][idx, GRADE_POSITION],
            "pairs": pairs, "posterior_site": bsite, "nat_pc": disp_scores[idx, :2],
            "nat_gap": key["D"][idx, GRADE_READOUT, 0] - key["D"][idx, GRADE_READOUT, 1], "nat_site": disp}
    f1 = run_dir / "figures" / "paper_representation_no_totals_split.png"
    fig_paper_representation(post, acc, show, PL.SITE_LABELS, f1)

    np.savez(run_dir / "payload.npz",
             show_posterior_pc=show["posterior_pc"], show_etaA=show["etaA"], show_etaB=show["etaB"],
             show_pairA=np.array(pairs["A"]), show_pairB=np.array(pairs["B"]), show_nat_pc=show["nat_pc"], show_nat_gap=show["nat_gap"],
             **{f"att_natural_{s}_{g}": np.array(acc[s]["attribution"][g]) for s in post for g in GROUPS + ("unexplained",)})
    acc_b = acc[bsite]
    fpc = sorted(pairs["A"] + pairs["B"])                                                         # panel (b)'s components
    lead_b = {g: float(np.sum(np.array(acc_b["attribution"][g])[fpc])) for g in GROUPS + ("unexplained",)}
    lead_b_var = float(np.sum(np.array(acc_b["var_frac"])[fpc]))
    assert abs(sum(lead_b.values()) - lead_b_var) < 1e-9, "attribution must partition the variance fraction"
    fpc_label = ", ".join(str(j + 1) for j in fpc)
    read_sites = tuple(dict.fromkeys((bsite, vsite)))                                             # the sites the paper quotes
    results = {"headline": {
        **{f"{g}_share_natural_{PL.SITE_LABELS[s]}": A.headline_entry(
            acc[s]["totals"][g], n_eval, "fraction", f"share of residual variance at {s} attributed to {GROUP_LABEL[g]}")
            for s in read_sites for g in ("posterior", "value")},
        "factor_pcs_var_frac_natural_posterior_site": A.headline_entry(lead_b_var, n_eval, "fraction", f"variance fraction of the factor-pair components ({fpc_label}) at {bsite}"),
        "factor_pcs_posterior_share_natural_posterior_site": A.headline_entry(lead_b["posterior"] / lead_b_var, n_eval, "fraction", f"posterior-attributed share of the factor-pair components' ({fpc_label}) variance at {bsite}")},
        "sites": list(post), "posterior_site": bsite, "value_site": vsite, "display_site": disp,
        "n_eval": n_eval, "pca_accounting": {"natural": acc}, "factor_pairs": pairs}
    A.write_results(run_dir, results)
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={"eval": seed, "split": [seed, ROLE_SPLIT], "show": [seed, ROLE_SHOW]},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                     figures=[f1], extra={"checkpoint": study.checkpoint, "instruments_sha256": ins.meta["artifact_sha256"]})
    print(json.dumps(results["headline"], indent=1)); print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
