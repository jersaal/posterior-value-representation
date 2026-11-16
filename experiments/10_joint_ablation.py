"""10 — joint posterior ablation: is the joint posterior η = η_A ⊗ η_B used for
choice beyond its two factor posteriors?

Encoding subspace. An 8-target affine probe of the joint posterior's tangent
coordinates is fit at the posterior write site on 02's plane rows (the first
half of the fitting episodes), and its rank-8 encoding subspace and aim map are
built exactly as 02 builds the factor planes (JointAim). A commanded change Δη
of the nine probabilities becomes one residual edit that the probe reads back
as Δη. The probe is throwaway: fit, used for the writes, dropped.

Three writes at that site on the evaluation episodes, each scored by the pooled
slope of the observed choice-logit shift on a counterfactual β Δ(Q_A − Q_B):
  ablation        — the joint is displaced to the table with the episode's
                    factor posteriors as margins and nothing else,
                    η_(i,j) = (η_A,i + η_B,j)/3 − 1/9. Both factor posteriors
                    are unchanged; everything beyond an affine function of them
                    is removed. Counterfactual: the offer-value formula applied
                    to the target, over the episodes whose offer probabilities
                    stay at least MIN_OFFER_PROB under it; also on the episodes
                    whose implied offer values both stay inside the payoff range;
  control         — the ablation's edit norms in random directions orthogonal to
                    the encoding subspace, scored on the ablation's
                    counterfactual and rows;
  posterior steer — 06's donor posterior for factor A (same donor stream and
                    pool), applied as η_(i,j) + (η'_A,i − η_A,i)/3: factor A
                    moves to its target, factor B and the rest of the joint stay
                    as they were, so the target leaves the simplex as the
                    ablation's does. Counterfactual: the exact offer values
                    under η'_A ⊗ η_B, as in 06.
A network that reads the joint directions into the offer values follows the
ablation's counterfactual (slope 1); one that composes the offer values from
the factor posteriors is unmoved by it (slope 0) and follows the posterior steer.

Figure labels call the factors M and N."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, instruments as I, plotting as PL, probes as P, stats as S, steering as ST   # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB          # noqa: E402
from jbo.subject import load_study                                                 # noqa: E402

EXPERIMENT = "10_joint_ablation"
SMOKE_FIT = 512                 # full runs take 02's fit and plane populations from the instruments
N_EVAL, SMOKE_EVAL = 16384, 512             # evaluation episodes on the eval seed (06's population)
ROLE_DONOR = 1                              # default_rng([eval_seed, ROLE_DONOR]): 06's factor-A donor stream
ROLE_ABLATE_ORTHO = 5                       # default_rng([eval_seed, ROLE_ABLATE_ORTHO]): the orthogonal control
MAX_COND = 1000.0                           # 8-target aim map Qᵀ W must be invertible in practice
N_BOOT = 300
FIGURE_SEED = 160017                        # scatter subsample of the figure
WRITE_COLOR = "#5b3f99"                     # choice quantities, as in 06
IDENTITY = {"color": "0.4", "lw": 0.6, "ls": (0, (3, 2)), "zorder": 0}
ZERO = {"color": "0.4", "lw": 0.6, "ls": (0, (1, 1.5)), "zorder": 0}


def joint_counterfactual(eta9: np.ndarray, bits: np.ndarray, u: np.ndarray) -> dict:
    """Exact offer probabilities and values under a joint posterior that need not
    be a product. eta9 [N, 9], bits [N, 2, 9], u [N, 9] -> dict(D [N, 2], P [N, 2])."""
    Pm = np.einsum("ns,nos->no", eta9, bits)
    v = np.einsum("ns,nos,ns->no", eta9, bits, u)
    return {"D": v / np.maximum(Pm, 1e-300), "P": Pm}


class JointAim:
    """02's write construction applied to the joint posterior itself: an 8-target
    affine probe of the joint's tangent coordinates y = (η − 1/9) H (H [9, 8] the
    simplex tangent basis), Q = the rank-8 cross-covariance subspace, M = Qᵀ W.
    write(Δη) = (Δη H) M⁻¹ Qᵀ is the residual edit that the probe reads back as
    exactly Δη. No factor structure enters: any 8 coordinates of the joint give
    the same subspace and the same edit."""

    def __init__(self, X: np.ndarray, eta9: np.ndarray, site: str, check_cond: bool):
        self.H = P.simplex_tangent_basis(9)                                              # [9, 8]
        Y = (eta9 - 1.0 / 9) @ self.H                                                    # [N, 8]
        self.Q = P.encoding_subspace(X, Y)                                               # [d, 8]
        self.probe = P.affine_fit(X, Y)                                                  # [d+1, 8]
        self.M = self.Q.T @ self.probe[1:]
        self.cond = float(np.linalg.cond(self.M))
        assert not check_cond or self.cond < MAX_COND, f"degenerate joint aim at {site} (cond {self.cond:.1f})"
        self.Minv, self.site = np.linalg.inv(self.M), site

    def write(self, d_eta: np.ndarray) -> np.ndarray:
        """Commanded change of the joint Δη [N, 9] (rows sum to 0) -> residual edit [N, d]."""
        assert np.allclose(d_eta.sum(1), 0.0, atol=1e-9), "a change of the joint must stay on the simplex plane"
        delta = ((d_eta @ self.H) @ self.Minv) @ self.Q.T
        assert np.allclose(delta @ self.probe[1:], d_eta @ self.H, atol=1e-8), "aim identity"
        return delta


def fit_joint_aim(model, world, ins, n_fit: int, n_plane: int, device: str, check_cond: bool) -> tuple[JointAim, np.ndarray]:
    """The joint's encoding subspace at factor A's posterior write site, fit on
    02's plane rows (valid rows among the first n_plane fitting episodes), and
    06's donor pool: the fitting episodes' factor-A posteriors in tangent
    coordinates [n_fit, 2]."""
    tokens = world.sample(n_fit, seed=ins.fit_seed)["tokens"]
    key = world.answer_key(tokens)
    etaA, etaB = key["etaA"][:, GRADE_POSITION], key["etaB"][:, GRADE_POSITION]         # [N, 3] each
    eta9 = key["eta"][:, GRADE_POSITION]                                                 # [N, 9]
    assert np.allclose(eta9, np.einsum("na,nb->nab", etaA, etaB).reshape(-1, 9), atol=1e-9), \
        "joint posterior is not the product of the factor posteriors"
    valid = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB                         # [N]
    w_site = ins.posterior_write_site["A"]
    acts = P.collect_site_activations(model, tokens, (w_site,), np.array([GRADE_POSITION]), device)
    X = acts[w_site][:, 0, :].astype(np.float64)                                         # [N, d]
    plane_rows = valid & (np.arange(n_fit) < n_plane)
    return JointAim(X[plane_rows], eta9[plane_rows], w_site, check_cond), ins.to_tangent(etaA)


def run_writes(model, world, ins, aim: JointAim, pool_A: np.ndarray, n_eval: int, eval_seed: int, device: str) -> dict:
    """The three writes at the joint write site and their pooled slopes.
    Returns dict(ablation, ablation_in_range, control, posterior_steer: S.slope
    dicts; arrays: the kept rows' counterfactual and observed shifts)."""
    pos, rb, beta = GRADE_POSITION, world.vocab.response_base, world.beta
    tokens = world.sample(n_eval, seed=eval_seed)["tokens"]
    key = world.answer_key(tokens)
    eta9, bits, u = key["eta"][:, pos], key["bits_ro"][:, GRADE_READOUT], key["u"]        # [N, 9], [N, 2, 9], [N, 9]
    etaA, etaB = key["etaA"][:, pos], key["etaB"][:, pos]                                  # [N, 3] each
    D_ana = key["D"][:, GRADE_READOUT]                                                     # [N, 2]
    gap_ana = D_ana[:, 0] - D_ana[:, 1]
    keep_clean = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    clean = ST.run_edited(model, tokens, [], (), pos, device, rb)
    Y_clean = ST.policy_logit(clean["pi"][:, GRADE_READOUT])

    def choice_shift(delta: np.ndarray) -> np.ndarray:
        """Residual edit [N, d] at the write site -> observed choice-logit shift [N]."""
        arm = ST.run_edited(model, tokens, [(aim.site, pos, delta)], (), pos, device, rb)
        assert np.abs(arm["pi"][:, :GRADE_READOUT] - clean["pi"][:, :GRADE_READOUT]).max() < 1e-5, "locality: earlier readouts moved"
        return ST.policy_logit(arm["pi"][:, GRADE_READOUT]) - Y_clean

    def predicted(eta_new: np.ndarray) -> tuple[np.ndarray, dict]:
        """Target joint [N, 9] -> counterfactual choice-logit shift [N] and its offer values / probabilities."""
        cf = joint_counterfactual(eta_new, bits, u)
        return beta * ((cf["D"][:, 0] - cf["D"][:, 1]) - gap_ana), cf

    def margins(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint table [N, 9] -> its factor-A and factor-B sums ([N, 3], [N, 3])."""
        m = table.reshape(-1, 3, 3)
        return m.sum(2), m.sum(1)

    def fit(x: np.ndarray, y: np.ndarray, rows: np.ndarray) -> dict:
        return S.slope(x[rows], y[rows], groups=np.flatnonzero(rows), n_boot=N_BOOT)

    # ablation: the joint replaced by the affine table of its factor posteriors
    target = (etaA[:, :, None] / 3 + etaB[:, None, :] / 3 - 1 / 9).reshape(-1, 9)          # [N, 9] η_(i,j) = (η_A,i + η_B,j)/3 − 1/9
    tA, tB = margins(target)
    assert np.allclose(tA, etaA) and np.allclose(tB, etaB), "target changed a factor posterior"
    d_eta = target - eta9                                                                  # [N, 9], zero row and column sums
    delta = aim.write(d_eta)
    x, cf = predicted(eta9 + d_eta)                                                        # the target as the displacement reaches it (η + Δη): offer values that sit on the payoff bounds keep this rounding in q_ok
    p_ok = keep_clean & (cf["P"].min(-1) >= MIN_OFFER_PROB)                                # the offer-value formula is defined (cells may be negative)
    q_ok = p_ok & (cf["D"] >= u.min()).all(1) & (cf["D"] <= u.max()).all(1)                # both implied offer values inside the payoff range
    y = choice_shift(delta)

    # control: the ablation's edit norms in random directions orthogonal to the encoding subspace
    rng_c = np.random.default_rng([eval_seed, ROLE_ABLATE_ORTHO])
    ortho = rng_c.normal(size=delta.shape); ortho -= (ortho @ aim.Q) @ aim.Q.T
    ortho /= np.linalg.norm(ortho, axis=1, keepdims=True)
    delta_c = np.linalg.norm(delta, axis=1)[:, None] * ortho
    assert np.abs(delta_c @ aim.Q).max() < 1e-8, "control is not orthogonal to the encoding subspace"
    yc = choice_shift(delta_c)

    # posterior steer of factor A through the same subspace: 06's donor target, written as a change of margin only
    yA = ins.to_tangent(etaA)
    donors = ST.draw_natural_donors(yA, pool_A, np.random.default_rng([eval_seed, ROLE_DONOR]))
    etaA_new = ins.from_tangent(yA + donors["dy"])                                         # [N, 3]
    assert etaA_new.min() >= -1e-9, "donor posterior left the simplex"
    d_margin = np.repeat((etaA_new - etaA) / 3, 3, axis=1)                                 # [N, 9] Δη_(i,j) = (η'_A,i − η_A,i)/3
    dA, dB = margins(d_margin)
    assert np.allclose(dA, etaA_new - etaA) and np.allclose(dB, 0.0), "margin-only write"
    xs, cf_s = predicted(np.einsum("na,nb->nab", etaA_new, etaB).reshape(-1, 9))           # exact counterfactual under η'_A ⊗ η_B
    s_ok = keep_clean & (cf_s["P"].min(-1) >= MIN_OFFER_PROB)
    ys = choice_shift(aim.write(d_margin))

    out = {"ablation": fit(x, y, p_ok), "ablation_in_range": fit(x, y, q_ok), "control": fit(x, yc, p_ok),
           "posterior_steer": fit(xs, ys, s_ok), "n_episodes": n_eval,
           "arrays": {"ablation_x": x[p_ok], "ablation_y": y[p_ok], "control_y": yc[p_ok],
                      "posterior_steer_x": xs[s_ok], "posterior_steer_y": ys[s_ok]}}
    for name in ("ablation", "ablation_in_range", "control", "posterior_steer"):
        print(f"  {name:18s} slope {out[name]['slope']:+.4f} ± {out[name]['se']:.4f} (n {out[name]['n']})", flush=True)
    return out


def _symmetric_ticks(lim: float) -> list[float]:
    t = float(np.floor(lim)) if lim >= 1.5 else round(lim / 2, 1)
    return [-t, 0.0, t]


def fig_paper_ablation(arrays: dict, slopes: dict, path: Path) -> None:
    """One-panel paper figure in the form of the steering figure's first panel:
    observed against counterfactual choice-logit shift. Purple: the joint
    ablation. Blue, underneath: the posterior steer of factor M written through
    the same joint subspace. Dashed identity; limits from the blue arm's range,
    so the purple arm's larger predicted shifts clip. 1.25 in square axes on a
    1.75 in wide figure, paper rc, no title."""
    fs, W, H = PL.PAPER_FS, 1.75, 1.72
    xi, yi = arrays["ablation_x"], arrays["ablation_y"]
    xf, yf = arrays["posterior_steer_x"], arrays["posterior_steer_y"]
    rng = np.random.default_rng(FIGURE_SEED)
    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(W, H))
        ax = PL.axes_in(fig, 0.42, 0.40, 1.25, 1.25)
        lim = float(np.percentile(np.abs(xf), 99.5)); ticks = _symmetric_ticks(lim)
        ti = rng.choice(len(xi), min(12000, len(xi)), replace=False); tf = rng.choice(len(xf), min(12000, len(xf)), replace=False)
        ax.plot([-lim, lim], [-lim, lim], **IDENTITY); ax.axhline(0, **ZERO)
        ax.scatter(xf[tf], yf[tf], s=0.6, color=PL.POSTERIOR_A_COLOR, alpha=0.2, linewidths=0, rasterized=True)
        ax.scatter(xi[ti], yi[ti], s=0.6, color=WRITE_COLOR, alpha=0.2, linewidths=0, rasterized=True)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=ticks, yticks=ticks,
               xlabel="predicted logit shift", ylabel="observed shift")
        series = (("posterior steer, factor M", PL.POSTERIOR_A_COLOR, slopes["posterior_steer"]),
                  ("joint content removed", WRITE_COLOR, slopes["ablation"]))
        for k, (name, c, sl) in enumerate(series):                                      # in-panel key and slope readout
            ax.text(0.97, 0.03 + 0.10 * k, name, color=c, fontsize=fs - 1.5, ha="right", va="bottom", transform=ax.transAxes)
            ax.text(0.05, 0.92 - 0.11 * k, f"slope {sl + 0.0:.3f}".replace("-0.000", "0.000"), color=c, fontsize=fs - 1.5, ha="left", va="center", transform=ax.transAxes, zorder=7,
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])
        PL.frame(ax)
        fig.savefig(path, dpi=300); plt.close(fig)


def main() -> None:
    args = cli.parse("10_joint_ablation: write the joint posterior's affine-table target, a factor posterior steer and an orthogonal control through the joint's encoding subspace")
    study = load_study(args.study)
    world = study.build_world()
    model, _ = study.load_actor(args.device)
    eval_seed = study.seeds["eval"]
    ins = I.load(study, eval_seeds=[eval_seed])
    assert not ins.reads_only, "the joint ablation needs the posterior write site (instruments with write planes)"
    n_fit = SMOKE_FIT if args.smoke else int(ins.meta["n_fit"])
    n_plane = SMOKE_FIT if args.smoke else int(ins.meta["n_plane"])                       # 02's plane split: the first half of the fitting episodes
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    aim, pool_A = fit_joint_aim(model, world, ins, n_fit, n_plane, args.device, check_cond=not args.smoke)
    assert aim.Q.shape[1] == 8, "the joint posterior has eight free coordinates"
    res = run_writes(model, world, ins, aim, pool_A, SMOKE_EVAL if args.smoke else N_EVAL, eval_seed, args.device)

    np.savez(run_dir / "payload.npz", **res["arrays"])
    fig_path = run_dir / "figures" / "paper_joint_ablation.png"
    fig_paper_ablation(res["arrays"], {"posterior_steer": res["posterior_steer"]["slope"], "ablation": res["ablation"]["slope"]}, fig_path)

    w = PL.SITE_LABELS[aim.site]
    ab, rg, ct, ps = res["ablation"], res["ablation_in_range"], res["control"], res["posterior_steer"]
    A.write_results(run_dir, {
        "headline": {
            "ablation_slope": A.headline_entry(ab["slope"], ab["n"], "slope",
                f"observed vs exact choice-logit shift when the joint posterior is displaced at {w}, through its 8-dimensional encoding subspace, to η_(i,j) = (η_A,i + η_B,j)/3 − 1/9, "
                f"on the episodes whose offer probabilities stay at least {MIN_OFFER_PROB} under the target (SE {ab['se']:.3f})"),
            "ablation_slope_in_range": A.headline_entry(rg["slope"], rg["n"], "slope",
                f"the same on the episodes whose implied offer values both stay within the payoff range (SE {rg['se']:.3f})"),
            "posterior_steer_slope": A.headline_entry(ps["slope"], ps["n"], "slope",
                f"positive control: the posterior steer of factor A written through the same subspace at {w} as a change of its margin only, observed vs the exact counterfactual under the target factor posterior (SE {ps['se']:.3f})"),
            "control_slope": A.headline_entry(ct["slope"], ct["n"], "slope",
                f"the ablation's edit norms in random directions orthogonal to the encoding subspace, on the ablation's counterfactual and episodes (SE {ct['se']:.3f})")},
        "write_site": w, "subspace_dim": int(aim.Q.shape[1]), "n_episodes": res["n_episodes"], "n_fit": n_fit, "n_plane": n_plane})
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={"fit": ins.fit_seed, "eval": eval_seed, "donor": [eval_seed, ROLE_DONOR],
                            "ablation_orthogonal": [eval_seed, ROLE_ABLATE_ORTHO], "figure_subsample": FIGURE_SEED},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                     figures=[fig_path], extra={"checkpoint": study.checkpoint, "instruments_sha256": ins.meta["artifact_sha256"]})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
