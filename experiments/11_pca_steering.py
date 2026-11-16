"""11 — steering through principal components: the write subspace defined
without ground truth.

At each write site the principal components of the residual stream on the
plane-fitting episodes (no labels) give the write plane, used as they are:
the leading two at the value write site, and at the posterior write site the
pair that 07_intrinsic_dim attributes to each factor among the leading eight
(read from the pinned 07 run's `factor_pairs`). Commands are aimed as in the
paper, dy · M⁻¹ · Qᵀ with M = Qᵀ W, Q the two components and W the fitted
probe's weights.

Value: 06's 12-direction ring at the value write site through the component
plane, with the matched-norm ring in the plane's orthogonal complement as
control. Posterior: 06's natural posterior steer per factor at the posterior
write site through the factor's component pair, with isometric off-plane
copies as controls, slope on the exact counterfactual. Same evaluation
episodes, donors and radii as 06.

Fits PCA bases only; nothing is written to the study."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, instruments as I, plotting as PL, probes as P, stats as S, steering as ST   # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB                                      # noqa: E402
from jbo.subject import load_study                                                                             # noqa: E402

EXPERIMENT = "11_pca_steering"
PAIRS_EXPERIMENT = "07_intrinsic_dim"        # its pinned run assigns a component pair to each factor
N_EVAL, SMOKE_EVAL = 16384, 512
N_CONTROLS, SMOKE_CONTROLS = 12, 3           # isometric off-plane copies per posterior steer, as in 06
ROLE_DONOR, ROLE_POSTERIOR_CTRL, ROLE_ORTHO = 1, 2, 1   # 06's streams: the donor draw reproduces 06's donors exactly
FACTORS = (("A", 0, "posterior_A", "etaA", "etaB"), ("B", 1, "posterior_B", "etaB", "etaA"))
VALUE_PCS = (0, 1)                           # the leading two components at the value write site
MIN_PAIR_R2 = 0.9                            # a factor's pair must carry its posterior on the plane-fitting episodes


# ---------------------------------------------------------------- subspaces

def pca_basis(X: np.ndarray) -> np.ndarray:
    """Principal axes V [d, d] of X [N, d], columns in descending variance."""
    X = np.asarray(X, np.float64)
    assert X.ndim == 2 and len(X) > X.shape[1], "PCA needs more rows than dimensions"
    Xc = X - X.mean(0)
    ev, V = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
    V = V[:, ::-1]
    assert np.allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-8)
    return V


@dataclass(frozen=True)
class Subspace:
    """A rank-2 write plane Q [d, 2] of two principal components, oriented by the fitted probe weights W [d, 2]."""
    Q: np.ndarray
    W: np.ndarray

    @property
    def M(self) -> np.ndarray:                       # [2, 2]: what a unit move along each plane axis does to the probe read
        return self.Q.T @ self.W

    def aim(self, dy: np.ndarray) -> np.ndarray:
        """Commanded probe-read displacement dy [N, 2] -> residual delta [N, d] in the plane with delta · W = dy."""
        delta = (np.asarray(dy, np.float64) @ np.linalg.inv(self.M)) @ self.Q.T
        assert np.allclose(delta @ self.W, dy, atol=1e-6), "aim identity"
        return delta


def component_plane(X: np.ndarray, W: np.ndarray, pcs: list[int]) -> Subspace:
    """The plane of the principal components `pcs` (two indices) of X [N, d]."""
    assert len(pcs) == 2 and len(set(pcs)) == 2, pcs
    return Subspace(pca_basis(X)[:, list(pcs)], W)


def pinned_factor_pairs(study_name: str, posterior_site: str) -> dict:
    """{"A": [2], "B": [2]} component indices from the pinned 07 run, which must have analysed `posterior_site`."""
    pins = json.loads((A.ROOT / "studies" / study_name / "pins.json").read_text())
    assert PAIRS_EXPERIMENT in pins, f"pin a {PAIRS_EXPERIMENT} run for {study_name} first"
    res = json.loads((A.ROOT / "results" / study_name / PAIRS_EXPERIMENT / pins[PAIRS_EXPERIMENT] / "results.json").read_text())
    assert res["posterior_site"] == posterior_site, "the pinned pairs were assigned at a different site than the posterior write site"
    pairs = {name: [int(i) for i in res["factor_pairs"][name]] for name in ("A", "B")}
    assert len(set(pairs["A"]) | set(pairs["B"])) == 4, f"the factor pairs must be disjoint: {pairs}"
    return pairs


def pair_r2(X: np.ndarray, Q: np.ndarray, Y: np.ndarray) -> float:
    """In-sample R² of Y [N, 2] regressed on the scores of X [N, d] along the two components Q [d, 2]."""
    scores = (X - X.mean(0)) @ Q
    return float(P.decode_scores(Y, P.affine_predict(scores, P.affine_fit(scores, Y)))["r2"])


# ---------------------------------------------------------------- shared clean pass

@dataclass(frozen=True)
class Shared:
    device: str
    rb: int
    pos: int
    tokens: np.ndarray        # [N, T]
    key: dict
    bits: np.ndarray          # [N, 2, 9]
    u: np.ndarray             # [N, 9]
    reward: str
    keep_clean: np.ndarray    # [N]
    pi_clean: np.ndarray      # [N, 10, 2]
    Y_clean: np.ndarray       # [N]


def collect_decide(model, tokens: np.ndarray, sites: tuple, pos: int, device: str) -> dict:
    """{site: [N, d]} residual activations at the graded DECIDE position."""
    acts = P.collect_site_activations(model, tokens, sites, np.array([pos]), device)
    return {s: a[:, 0, :].astype(np.float64) for s, a in acts.items()}


# ---------------------------------------------------------------- value ring

def value_ring(sh: Shared, model, ins, sub: Subspace, beta: float, seed: int) -> dict:
    """06's value ring through `sub` at the value write site: pooled slope of the observed
    choice-logit shift on the commanded one, and the same for the matched-norm ring in the
    orthogonal complement of `sub`."""
    d_write, pos = ins.value_write_site, sh.pos
    D_clean = sh.key["D"][:, GRADE_READOUT]
    radius = ST.safe_radii(D_clean)
    keep = sh.keep_clean & (radius >= ST.MIN_RADIUS)
    _, directions = ST.ring_directions()
    controls = ST.orthogonal_directions(sub.Q, ST.N_DIRECTIONS, np.random.default_rng([seed, ROLE_ORTHO]))
    xs, ys, ycs = [], [], []
    loc_err = 0.0
    for k, d in enumerate(directions):
        dy = radius[:, None] * d                                               # [N, 2]
        delta = sub.aim(dy)                                                    # [N, d]
        delta_c = np.linalg.norm(delta, axis=1)[:, None] * controls[:, k][None, :]
        arm = ST.run_edited(model, sh.tokens, [(d_write, pos, delta)], (), pos, sh.device, sh.rb)
        arm_c = ST.run_edited(model, sh.tokens, [(d_write, pos, delta_c)], (), pos, sh.device, sh.rb)
        for a in (arm, arm_c):
            loc_err = max(loc_err, float(np.abs(a["pi"][:, :GRADE_READOUT] - sh.pi_clean[:, :GRADE_READOUT]).max()))
        xs.append(beta * (dy[keep, 0] - dy[keep, 1]))
        ys.append((ST.policy_logit(arm["pi"][:, GRADE_READOUT]) - sh.Y_clean)[keep])
        ycs.append((ST.policy_logit(arm_c["pi"][:, GRADE_READOUT]) - sh.Y_clean)[keep])
    assert loc_err < 1e-5, f"locality failed ({loc_err:.3g})"
    x, y, yc = np.concatenate(xs), np.concatenate(ys), np.concatenate(ycs)
    groups = np.tile(np.flatnonzero(keep), ST.N_DIRECTIONS)
    slope = S.slope(x, y, groups=groups, n_boot=0)
    control = S.slope(x, yc, groups=groups, n_boot=0)
    print(f"  value slope {slope['slope']:.3f}  control {control['slope']:+.3f}", flush=True)
    return {"slope": slope["slope"], "control_slope": control["slope"], "n": slope["n"], "n_kept": int(keep.sum())}


# ---------------------------------------------------------------- posterior steer

def posterior_steer(sh: Shared, model, ins, sub: Subspace, factor: tuple, donors: dict, beta: float,
                    n_ctrl: int, seed: int) -> dict:
    """06's natural posterior steer through `sub` at the factor's posterior write site: slope of
    the observed choice-logit shift on the exact counterfactual, and the pooled slope of
    n_ctrl isometric copies into planes orthogonal to `sub`."""
    name, fi, _, eta_key, other_key = factor
    b_site, pos = ins.posterior_write_site[name], sh.pos
    eta_f, eta_o = sh.key[eta_key][:, pos], sh.key[other_key][:, pos]
    y_clean = ins.to_tangent(eta_f)
    D_clean = sh.key["D"][:, GRADE_READOUT]
    eta_new = ins.from_tangent(y_clean + donors["dy"])
    assert eta_new.min() >= -1e-9, "donor posterior left the simplex"
    cf = ST.counterfactual_D(eta_new, eta_o, fi, sh.bits, sh.u, sh.reward)
    keep = sh.keep_clean & (cf["P"].min(-1) >= MIN_OFFER_PROB)
    dZ = beta * ((cf["D"][:, 0] - cf["D"][:, 1]) - (D_clean[:, 0] - D_clean[:, 1]))
    delta = sub.aim(donors["dy"])                                              # [N, d]
    steered = ST.run_edited(model, sh.tokens, [(b_site, pos, delta)], (), pos, sh.device, sh.rb)
    dY = ST.policy_logit(steered["pi"][:, GRADE_READOUT]) - sh.Y_clean
    transfer = S.slope(dZ[keep], dY[keep], n_boot=0)
    axes = ST.orthogonal_directions(sub.Q, 2 * n_ctrl, np.random.default_rng([seed, ROLE_POSTERIOR_CTRL, fi]))
    ctrl = []
    for c in range(n_ctrl):
        dc = ST.isometric_plane_copy(delta, sub.Q, axes[:, 2 * c: 2 * c + 2])
        arm = ST.run_edited(model, sh.tokens, [(b_site, pos, dc)], (), pos, sh.device, sh.rb)
        ctrl.append((ST.policy_logit(arm["pi"][:, GRADE_READOUT]) - sh.Y_clean)[keep])
    episode_idx = np.flatnonzero(keep)
    control = S.slope(np.tile(dZ[keep], n_ctrl), np.concatenate(ctrl), groups=np.tile(episode_idx, n_ctrl), n_boot=0)
    print(f"  posterior {name} slope {transfer['slope']:.3f}  control {control['slope']:+.3f}", flush=True)
    return {"slope": transfer["slope"], "control_slope": control["slope"], "n": transfer["n"], "n_control": control["n"],
            "n_kept": int(keep.sum())}


# ---------------------------------------------------------------- main

def main() -> None:
    args = cli.parse("11_pca_steering: steer through the principal components attributed to each quantity")
    study = load_study(args.study)
    world = study.build_world()
    assert world.n_factors == 2, "the factor-posterior steer needs a two-factor world"
    model, _ = study.load_actor(args.device)
    seed = study.seeds["eval"]
    ins = I.load(study, eval_seeds=[seed])
    assert not ins.reads_only, f"{study.name} has reads-only instruments — steering needs a trained choice"
    d_write, pos = ins.value_write_site, GRADE_POSITION
    b_sites = dict(ins.posterior_write_site)
    assert len(set(b_sites.values())) == 1, "the pair assignment is one PCA at one posterior site"
    b_write = b_sites["A"]
    pairs = pinned_factor_pairs(args.study, b_write)
    rb, beta = world.vocab.response_base, world.beta
    n, n_ctrl = (SMOKE_EVAL, SMOKE_CONTROLS) if args.smoke else (N_EVAL, N_CONTROLS)
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # PCA populations: 02's plane split of the fit episodes at each write site
    # (value: valid rows, as 02's value plane; posterior: all rows, as 02's posterior planes)
    n_fit, n_plane = int(ins.meta["n_fit"]), int(ins.meta["n_plane"])
    fit_tokens = world.sample(n_fit, seed=ins.fit_seed)["tokens"]
    fit_key = world.answer_key(fit_tokens)
    fit_acts = collect_decide(model, fit_tokens[:n_plane], tuple(dict.fromkeys((d_write, b_write))), pos, args.device)
    fit_valid = fit_key["P"][:n_plane, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    pools = {name: ins.to_tangent(fit_key[eta_key][:, pos]) for name, _, _, eta_key, _ in FACTORS}   # 06's donor pools

    value_sub = component_plane(fit_acts[d_write][fit_valid], ins.planes["value"].probe[1:], list(VALUE_PCS))
    posterior_subs = {}
    for name, _, plane_key, eta_key, _ in FACTORS:
        sub = component_plane(fit_acts[b_write], ins.planes[plane_key].probe[1:], pairs[name])
        carried = pair_r2(fit_acts[b_write], sub.Q, ins.to_tangent(fit_key[eta_key][:n_plane, pos]))
        assert carried > MIN_PAIR_R2, f"components {pairs[name]} do not carry factor {name} on the plane-fitting episodes (R² {carried:.3f})"
        posterior_subs[name] = sub
    print(f"  components {[i + 1 for i in pairs['A']]} for factor A and {[i + 1 for i in pairs['B']]} for factor B at "
          f"{PL.SITE_LABELS[b_write]}; components {[i + 1 for i in VALUE_PCS]} for the offer values at {PL.SITE_LABELS[d_write]}", flush=True)

    # evaluation episodes and the clean pass (06's)
    tokens = world.sample(n, seed=seed)["tokens"]
    key = world.answer_key(tokens)
    keep_clean = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    clean = ST.run_edited(model, tokens, [], (), pos, args.device, rb)
    sh = Shared(device=args.device, rb=rb, pos=pos, tokens=tokens, key=key, bits=key["bits_ro"][:, GRADE_READOUT], u=key["u"],
                reward=world.reward, keep_clean=keep_clean, pi_clean=clean["pi"], Y_clean=ST.policy_logit(clean["pi"][:, GRADE_READOUT]))
    donor_rng = np.random.default_rng([seed, ROLE_DONOR])                      # 06's donor stream, A then B
    donors = {name: ST.draw_natural_donors(ins.to_tangent(key[eta_key][:, pos]), pools[name], donor_rng)
              for name, _, _, eta_key, _ in FACTORS}

    res = {"value": value_ring(sh, model, ins, value_sub, beta, seed),
           "posterior": {factor[0]: posterior_steer(sh, model, ins, posterior_subs[factor[0]], factor, donors[factor[0]], beta, n_ctrl, seed)
                         for factor in FACTORS}}

    H = A.headline_entry
    v_pcs = "–".join(str(i + 1) for i in VALUE_PCS)
    rv = res["value"]
    head = {"value_pcs_slope": H(rv["slope"], rv["n"], "slope", f"value ring at {PL.SITE_LABELS[d_write]} through principal components {v_pcs}"),
            "value_pcs_control_slope": H(rv["control_slope"], rv["n"], "slope",
                                         f"matched-norm ring in the orthogonal complement of components {v_pcs} at {PL.SITE_LABELS[d_write]}")}
    for name in ("A", "B"):
        r, pcs = res["posterior"][name], "–".join(str(i + 1) for i in pairs[name])
        head[f"posterior_{name}_pcs_slope"] = H(r["slope"], r["n"], "slope",
                                                f"natural posterior steer, factor {name}, at {PL.SITE_LABELS[b_write]} through principal components {pcs}")
        head[f"posterior_{name}_pcs_control_slope"] = H(r["control_slope"], r["n_control"], "slope",
                                                        f"pooled slope of {n_ctrl} isometric copies orthogonal to components {pcs}, factor {name}")

    A.write_results(run_dir, {"headline": head, "results": res, "factor_pairs": pairs, "value_pcs": list(VALUE_PCS),
                              "sites": {"value_write": d_write, "posterior_write": b_sites},
                              "n_eval": n, "n_controls": n_ctrl, "n_pca": {"value_rows": int(fit_valid.sum()), "posterior_rows": n_plane}})
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={"eval": seed, "donor": [seed, ROLE_DONOR], "posterior_control": [seed, ROLE_POSTERIOR_CTRL, "factor"],
                            "ring_orthogonal": [seed, ROLE_ORTHO], "fit_pool": ins.fit_seed},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force, figures=[],
                     extra={"checkpoint": study.checkpoint, "instruments_sha256": ins.meta["artifact_sha256"], "smoke": args.smoke})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
