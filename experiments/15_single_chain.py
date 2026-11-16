"""15 — the one-chain world (`mess9`): the toy model's analyses on a world whose
latent state is a single nine-state Mess3-style chain rather than two
three-state factors, so the posterior has no factored form and lives on the
full eight-dimensional simplex. Self-contained: the study has no instruments
artifact; this script fits its own probes and write subspaces for the joint
posterior and the offer values.

Sections, each on the definitions used for the toy model:
  uncertainty — mean probability of the most likely of the nine states at the
                readouts, in this world and in the two-factor world (exact
                posteriors only; no network);
  behavior    — Pearson r of P(bet A) against the target policy;
  reads       — grouped-CV affine reads of the joint posterior (8 tangent
                coordinates), the offer-value pair, their sum and gap, at every
                layer; read sites by the 0.95 formation rule;
  scans       — posterior steer and value sweep at every layer on the fit split;
                write sites by the 0.95-of-maximum rule;
  held-out    — the 64 withheld masks in slot A with a disjoint partner, at the
                final layer: P(X) composed from the decoded joint, Q(X) read directly;
  twins       — card / masks / clue twins; decoded posterior and value shifts;
  steering    — natural posterior steer through the joint subspace with
                isometric off-plane controls and the value endpoint error;
                mediation (insertion / removal of the value-plane part); the
                12-direction value sweep with its orthogonal control; the decoded
                posterior under each.
Posterior shifts are Euclidean distances in the simplex tangent plane (Helmert
basis), in probability units: vertices sqrt(2) apart. Draws no figure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, behavior as B, cli, plotting as PL, probes as P, stats as S, steering as ST, stimuli as STIM  # noqa: E402
from jbo.algebra import N_STATES, mask_bits                                  # noqa: E402
from jbo.world import ACTION_POSITIONS, World                                # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB   # noqa: E402
from jbo.subject import load_study                                          # noqa: E402

EXPERIMENT = "15_single_chain"
FACTORED_WORLD = "mess3_m1m2"   # the two-factor world whose state uncertainty the chain was matched to
N_FOLDS = 5
FORMATION_R2, WRITE_FRACTION = 0.95, 0.95
N_BEHAVIOR, N_FIT, N_PLANE, N_SCAN, N_EVAL, N_CTX = 8192, 4096, 2048, 1024, 8192, 1024
SMOKE = {"behavior": 512, "fit": 512, "plane": 256, "scan": 128, "eval": 512, "ctx": 64}
N_CONTROLS, SMOKE_CONTROLS = 4, 1
MIN_VALID = 64                 # held-out mask kept when at least this many rows have both offers probable
MAX_COND_ETA, MAX_COND_D = 1000.0, 100.0
ROLE_FOLDS, ROLE_SCAN_DONOR, ROLE_DONOR, ROLE_CTRL, ROLE_ORTHO, ROLE_PARTNER = 1, 2, 1, 2, 4, 5
TWIN_ARMS = ("card", "masks", "clues5")


# ------------------------------------------------------------------ joint-posterior helpers

def joint_cf(eta9: np.ndarray, bits: np.ndarray, u: np.ndarray) -> dict:
    """Exact offer values when the joint posterior is eta9 [N, 9]; bits [N, 2, 9], u [N, 9]."""
    Pm = np.einsum("ns,nos->no", eta9, bits)
    v = np.einsum("ns,nos,ns->no", eta9, bits, u)
    return {"D": v / np.maximum(Pm, 1e-300), "P": Pm}


def joint_posterior(key: dict) -> np.ndarray:
    """Exact posterior over the nine joint states [N, L, 9]: the chain's own, or the product of two factors."""
    if "eta" in key:
        return key["eta"]
    eA, eB = key["etaA"], key["etaB"]                                                     # [N, L, 3] each
    return np.einsum("nli,nlj->nlij", eA, eB).reshape(*eA.shape[:2], N_STATES)


def mean_posterior_maximum(world: World, n: int, seed: int) -> float:
    """Mean over episodes and readouts of the probability of the most likely joint state."""
    eta9 = joint_posterior(world.answer_key(world.sample(n, seed=seed)["tokens"]))        # [N, L, 9]
    assert eta9.shape[-1] == N_STATES and np.allclose(eta9.sum(-1), 1.0), "joint posterior is not a distribution"
    return float(eta9[:, list(ACTION_POSITIONS)].max(-1).mean())


def fit_subspace(X: np.ndarray, Y: np.ndarray, max_cond: float) -> dict | None:
    """Rank-k cross-covariance subspace Q [d, k], affine probe [d+1, k], M = Qᵀ W (None if degenerate)."""
    Q = P.encoding_subspace(X, Y)
    probe = P.affine_fit(X, Y)
    M = Q.T @ probe[1:]
    cond = float(np.linalg.cond(M))
    if not np.isfinite(cond) or cond > max_cond:
        return None
    return {"Q": Q, "probe": probe, "M": M, "Minv": np.linalg.inv(M), "cond": cond}


def aim(plane: dict, dy: np.ndarray) -> np.ndarray:
    """Commanded probe-read displacement dy [N, k] -> residual delta [N, d] in span(Q)."""
    delta = (np.asarray(dy, np.float64) @ plane["Minv"]) @ plane["Q"].T
    assert np.allclose(delta @ plane["probe"][1:], dy, atol=1e-6), "aim identity"
    return delta


def read(plane: dict, resid: np.ndarray) -> np.ndarray:
    return P.affine_predict(resid, plane["probe"])


def formation_site(scores: dict[str, float], threshold: float = FORMATION_R2) -> str:
    for site, v in scores.items():
        if v >= threshold:
            return site
    return max(scores, key=scores.get)


def relative_site(scores: dict[str, float], fraction: float = WRITE_FRACTION) -> str:
    finite = {s: v for s, v in scores.items() if np.isfinite(v)}
    top = max(finite.values())
    if top <= 0:                        # an untrained (smoke) actor steers nowhere: fall back to the last site
        print(f"  warning: no site has a positive steering slope ({top:.3f}); using the final site", flush=True)
        return list(finite)[-1]
    return next(s for s, v in finite.items() if v >= fraction * top)


def logit_at_grade(out: dict) -> np.ndarray:
    return ST.policy_logit(out["pi"][:, GRADE_READOUT])


# ------------------------------------------------------------------ sections

def reads_section(model, world, tokens: np.ndarray, key: dict, U: np.ndarray, sites: tuple, n_plane: int,
                  device: str, rng: np.random.Generator) -> dict:
    """CV reads at every site; throwaway subspaces per post-block site on the plane population."""
    pos = GRADE_POSITION
    acts = P.collect_site_activations(model, tokens, sites, np.array([pos]), device)      # {site: [N, 1, d]}
    Y_eta = P.to_tangent(key["eta"][:, pos], U)                                           # [N, 8]
    D = key["D"][:, GRADE_READOUT]                                                        # [N, 2]
    valid = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    targets = {"eta": Y_eta, "D": D, "sum": D.sum(1, keepdims=True), "gap": (D[:, 0] - D[:, 1])[:, None]}
    folds = P.grouped_folds(len(tokens), N_FOLDS, rng)
    cv = {}
    for site in sites:
        X = acts[site]                                                                    # [N, 1, d]
        cv[site] = {}
        for name, Y in targets.items():
            cv[site][name] = float(P.cv_decode(X, Y[:, None, :], folds, valid[:, None])["r2"])
        print(f"  read {PL.SITE_LABELS[site]:5s} " + " ".join(f"{k} {v:.3f}" for k, v in cv[site].items()), flush=True)
    post = tuple(s for s in sites if s != P.EMBED_SITE)
    eta_site = formation_site({s: cv[s]["eta"] for s in post})
    D_site = formation_site({s: cv[s]["D"] for s in post})
    planes = {"eta": {}, "D": {}}
    rows = np.flatnonzero(valid[:n_plane])
    for site in post:
        X = acts[site][rows, 0, :]
        planes["eta"][site] = fit_subspace(X, Y_eta[rows], MAX_COND_ETA)
        planes["D"][site] = fit_subspace(X, D[rows], MAX_COND_D)
    return {"cv": cv, "valid": valid, "Y_eta": Y_eta,
            "eta_site": eta_site, "D_site": D_site, "planes": planes, "post": post}


def scan_section(model, world, tokens: np.ndarray, key: dict, U: np.ndarray, planes: dict, post: tuple,
                 pool_y: np.ndarray, device: str, rng: np.random.Generator) -> dict:
    """Posterior-steer and value-sweep slopes at every post-block site (fit split)."""
    pos, rb, beta = GRADE_POSITION, world.vocab.response_base, world.beta
    Y0 = logit_at_grade(ST.run_edited(model, tokens, [], (), pos, device, rb))
    D_clean, Pc = key["D"][:, GRADE_READOUT], key["P"][:, GRADE_READOUT]
    bits, u = key["bits_ro"][:, GRADE_READOUT], key["u"]
    keep0 = Pc.min(-1) >= MIN_OFFER_PROB
    # posterior steer: one natural donor per episode, the same draw at every site
    y_clean = P.to_tangent(key["eta"][:, pos], U)
    donors = ST.draw_natural_donors(y_clean, pool_y, rng)
    eta_new = P.from_tangent(y_clean + donors["dy"], U)
    assert eta_new.min() >= -1e-9, "donor posterior left the simplex"
    cf = joint_cf(eta_new, bits, u)
    keep = keep0 & (cf["P"].min(-1) >= MIN_OFFER_PROB)
    dZ = beta * ((cf["D"][:, 0] - cf["D"][:, 1]) - (D_clean[:, 0] - D_clean[:, 1]))
    # value sweep
    radius = ST.safe_radii(D_clean)
    keep_r = keep0 & (radius >= ST.MIN_RADIUS)
    _, directions = ST.ring_directions()
    eta_scan, D_scan = {}, {}
    for site in post:
        pe, pd = planes["eta"][site], planes["D"][site]
        if pe is None:
            eta_scan[site] = float("nan")
        else:
            arm = ST.run_edited(model, tokens, [(site, pos, aim(pe, donors["dy"]))], (), pos, device, rb)
            eta_scan[site] = float(S.slope(dZ[keep], (logit_at_grade(arm) - Y0)[keep], n_boot=0)["slope"])
        if pd is None:
            D_scan[site] = float("nan")
        else:
            xs, ys = [], []
            for d in directions:
                dy = radius[:, None] * d
                arm = ST.run_edited(model, tokens, [(site, pos, aim(pd, dy))], (), pos, device, rb)
                xs.append(beta * (dy[keep_r, 0] - dy[keep_r, 1]))
                ys.append((logit_at_grade(arm) - Y0)[keep_r])
            D_scan[site] = float(S.slope(np.concatenate(xs), np.concatenate(ys), n_boot=0)["slope"])
        print(f"  scan {PL.SITE_LABELS[site]:5s} posterior steer {eta_scan[site]:.3f}  value sweep {D_scan[site]:.3f}", flush=True)
    return {"eta": eta_scan, "D": D_scan, "eta_write": relative_site(eta_scan), "D_write": relative_site(D_scan)}


def heldout_section(model, world, tokens: np.ndarray, key: dict, U: np.ndarray, plane_eta: dict, plane_D: dict,
                    site: str, n_ctx: int, device: str, rng: np.random.Generator) -> dict:
    """Withheld masks in slot A with a uniform disjoint training-pool partner; reads at `site`."""
    pos = GRADE_POSITION
    base = tokens[:n_ctx]
    pool = np.arange(1, 511, dtype=np.int64)
    n_masks = 0
    P_true, P_hat, D_true, D_hat = [], [], [], []
    for mask in world.algebra.heldout_masks:
        elig = pool[(pool & int(mask)) == 0]
        partners = elig[rng.integers(0, len(elig), size=n_ctx)]
        toks = STIM.substitute_standing_offers(world, base, np.full(n_ctx, int(mask)), partners)
        k = world.answer_key(toks)
        assert np.allclose(k["eta"], key["eta"][:n_ctx]), "substitution moved the analytic posterior"
        ok = k["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
        if int(ok.sum()) < MIN_VALID:
            continue
        n_masks += 1
        a = P.collect_site_activations(model, toks, (site,), np.array([pos]), device)[site][:, 0, :]
        eta_hat = P.from_tangent(read(plane_eta, a), U)                                   # [n, 9] unclipped
        Ph = (eta_hat * mask_bits(np.full(n_ctx, int(mask)))).sum(-1)
        Dh = read(plane_D, a)[:, 0]
        Pt, Dt = k["P"][:, GRADE_READOUT, 0], k["D"][:, GRADE_READOUT, 0]
        P_true.append(Pt[ok]); P_hat.append(Ph[ok]); D_true.append(Dt[ok]); D_hat.append(Dh[ok])
    P_true, P_hat, D_true, D_hat = map(np.concatenate, (P_true, P_hat, D_true, D_hat))
    out = {"n_masks": n_masks, "n_rows": int(len(P_true)),
           "P_r2_pooled": P.decode_scores(P_true[:, None], P_hat[:, None])["r2"],
           "D_r2_pooled": P.decode_scores(D_true[:, None], D_hat[:, None])["r2"]}
    print(f"  held-out ({n_masks} masks): composed P R² {out['P_r2_pooled']:.3f}, direct Q R² {out['D_r2_pooled']:.3f}", flush=True)
    return out


def twins_section(model, world, tokens: np.ndarray, key: dict, U: np.ndarray, plane_eta: dict, plane_D: dict,
                  eta_site: str, D_site: str, device: str) -> dict:
    """Card / masks / clue twins (donor = next episode): decoded shifts at the read sites."""
    pos = GRADE_POSITION
    donor = np.roll(tokens, 1, axis=0)
    sites = tuple(dict.fromkeys((eta_site, D_site)))
    a0 = P.collect_site_activations(model, tokens, sites, np.array([pos]), device)
    y0, D0 = read(plane_eta, a0[eta_site][:, 0]), read(plane_D, a0[D_site][:, 0])
    keep = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    twins, keys = {}, {}
    for arm in TWIN_ARMS:
        twins[arm] = STIM.build_twin(world, tokens, donor, arm)
        keys[arm] = world.answer_key(twins[arm])
        keep &= keys[arm]["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    for arm in ("card", "masks"):
        assert np.allclose(keys[arm]["eta"], key["eta"]), f"{arm} twin moved the analytic posterior"
    rec = {}
    for arm in TWIN_ARMS:
        a = P.collect_site_activations(model, twins[arm], sites, np.array([pos]), device)
        d_eta = np.linalg.norm(read(plane_eta, a[eta_site][:, 0]) - y0, axis=1)[keep]
        d_D = np.linalg.norm(read(plane_D, a[D_site][:, 0]) - D0, axis=1)[keep]
        rec[arm] = {"eta_shift_median": float(np.median(d_eta)), "D_shift_median": float(np.median(d_D))}
        print(f"  twin {arm:6s} posterior {rec[arm]['eta_shift_median']:.4f}  value {rec[arm]['D_shift_median']:.3f}", flush=True)
    return {"records": rec, "n_kept": int(keep.sum())}


def steering_section(model, world, tokens: np.ndarray, key: dict, U: np.ndarray, planes: dict, sites: dict,
                     pool_y: np.ndarray, n_ctrl: int, device: str, seed: int) -> dict:
    """Natural posterior steer (+ controls, endpoint error, mediation) and the value sweep (+ control).
    Slopes are OLS point estimates (no bootstrap)."""
    pos, rb, beta = GRADE_POSITION, world.vocab.response_base, world.beta
    e_write, d_write, d_read, disp = sites["eta_write"], sites["D_write"], sites["D_site"], sites["display"]
    pe, pd, pe_disp = planes["eta"][e_write], planes["D"][d_write], planes["eta"][disp]
    pd_read = planes["D"][d_read]
    reads = tuple(dict.fromkeys((d_read, d_write, disp)))
    clean = ST.run_edited(model, tokens, [], reads, pos, device, rb)
    Y0 = logit_at_grade(clean)
    D_ana, Pc = key["D"][:, GRADE_READOUT], key["P"][:, GRADE_READOUT]
    bits, u = key["bits_ro"][:, GRADE_READOUT], key["u"]
    keep0 = Pc.min(-1) >= MIN_OFFER_PROB
    D_dec0 = read(pd_read, clean["resid"][d_read])
    y_disp0 = read(pe_disp, clean["resid"][disp])

    # --- natural posterior steer through the joint subspace at the posterior write site
    y_clean = P.to_tangent(key["eta"][:, pos], U)
    donors = ST.draw_natural_donors(y_clean, pool_y, np.random.default_rng([seed, ROLE_DONOR]))
    eta_new = P.from_tangent(y_clean + donors["dy"], U)
    assert eta_new.min() >= -1e-9
    cf = joint_cf(eta_new, bits, u)
    keep = keep0 & (cf["P"].min(-1) >= MIN_OFFER_PROB)
    delta = aim(pe, donors["dy"])
    steered = ST.run_edited(model, tokens, [(e_write, pos, delta)], reads, pos, device, rb)
    dY = logit_at_grade(steered) - Y0
    dZ = beta * ((cf["D"][:, 0] - cf["D"][:, 1]) - (D_ana[:, 0] - D_ana[:, 1]))
    transfer = S.slope(dZ[keep], dY[keep], n_boot=0)
    dD_ana = cf["D"] - D_ana
    err_nat = np.linalg.norm((read(pd_read, steered["resid"][d_read]) - D_dec0 - dD_ana)[keep], axis=1)
    eta_shift_under_eta = np.linalg.norm((read(pe_disp, steered["resid"][disp]) - y_disp0)[keep], axis=1)
    k = pe["Q"].shape[1]
    axes = ST.orthogonal_directions(pe["Q"], k * n_ctrl, np.random.default_rng([seed, ROLE_CTRL]))
    ctrl_dY, ctrl_err = [], []
    for c in range(n_ctrl):
        dc = ST.isometric_plane_copy(delta, pe["Q"], axes[:, k * c: k * (c + 1)])
        arm = ST.run_edited(model, tokens, [(e_write, pos, dc)], (d_read,), pos, device, rb)
        ctrl_dY.append((logit_at_grade(arm) - Y0)[keep])
        ctrl_err.append(np.linalg.norm((read(pd_read, arm["resid"][d_read]) - D_dec0 - dD_ana)[keep], axis=1))
    ctrl_dY, ctrl_err = np.concatenate(ctrl_dY), np.concatenate(ctrl_err)
    control = S.slope(np.tile(dZ[keep], n_ctrl), ctrl_dY, n_boot=0)
    print(f"  posterior steer: transfer {transfer['slope']:.3f}  control {control['slope']:.3f}  "
          f"endpoint err {np.median(err_nat):.3f} vs {np.median(ctrl_err):.3f}", flush=True)

    # --- mediation on the same steer: the value-plane part of the value-write-site displacement
    dh = steered["resid"][d_write] - clean["resid"][d_write]
    parts = ST.split_on_plane(dh, pd["Q"])
    arms = {"insertion": [(d_write, pos, parts["in_plane"])],
            "removal": [(e_write, pos, delta), (d_write, pos, -parts["in_plane"])]}
    med = {}
    for arm, edits in arms.items():
        r = ST.run_edited(model, tokens, edits, (), pos, device, rb)
        med[arm] = float(S.slope(dY[keep], (logit_at_grade(r) - Y0)[keep], n_boot=0)["slope"])
        print(f"  mediation {arm:9s} carried fraction {med[arm]:.3f}", flush=True)

    # --- value sweep at the value write site, with its orthogonal control and the decoded posterior under it
    radius = ST.safe_radii(D_ana)
    keep_r = keep0 & (radius >= ST.MIN_RADIUS)
    _, directions = ST.ring_directions()
    controls = ST.orthogonal_directions(pd["Q"], ST.N_DIRECTIONS, np.random.default_rng([seed, ROLE_ORTHO]))
    xs, ys, ycs, eta_d = [], [], [], []
    for j, d in enumerate(directions):
        dy = radius[:, None] * d
        dv = aim(pd, dy)
        dvc = np.linalg.norm(dv, axis=1)[:, None] * controls[:, j][None, :]
        arm = ST.run_edited(model, tokens, [(d_write, pos, dv)], (disp,), pos, device, rb)
        armc = ST.run_edited(model, tokens, [(d_write, pos, dvc)], (), pos, device, rb)
        xs.append(beta * (dy[keep_r, 0] - dy[keep_r, 1]))
        ys.append((logit_at_grade(arm) - Y0)[keep_r])
        ycs.append((logit_at_grade(armc) - Y0)[keep_r])
        eta_d.append(np.linalg.norm((read(pe_disp, arm["resid"][disp]) - y_disp0)[keep_r], axis=1))
    x, y, yc = np.stack(xs), np.stack(ys), np.stack(ycs)
    ring = S.slope(x.ravel(), y.ravel(), n_boot=0)
    ring_ctrl = S.slope(x.ravel(), yc.ravel(), n_boot=0)
    eta_under_D = np.concatenate(eta_d)
    print(f"  value sweep: slope {ring['slope']:.3f}  control {ring_ctrl['slope']:.3f}  "
          f"posterior RMS shift under value {np.sqrt((eta_under_D ** 2).mean()):.4f} vs under posterior steer "
          f"{np.sqrt((eta_shift_under_eta ** 2).mean()):.3f}", flush=True)
    return {"transfer": float(transfer["slope"]), "control": float(control["slope"]), "n_kept": int(keep.sum()),
            "endpoint_error_median_natural": float(np.median(err_nat)), "endpoint_error_median_control": float(np.median(ctrl_err)),
            "mediation": med, "ring": float(ring["slope"]), "ring_control": float(ring_ctrl["slope"]),
            "n_ring": int(keep_r.sum()),
            "eta_rms_under_value": float(np.sqrt((eta_under_D ** 2).mean())),
            "eta_rms_under_posterior": float(np.sqrt((eta_shift_under_eta ** 2).mean()))}


# ------------------------------------------------------------------ main

def main() -> None:
    args = cli.parse("15_single_chain: the toy model's analyses on the one-chain world")
    study = load_study(args.study)
    world = study.build_world()
    assert world.n_factors == 1, f"{args.study} is a two-factor world; this script is for one-chain worlds"
    model, _ = study.load_actor(args.device)
    n = {k: SMOKE[k] for k in SMOKE} if args.smoke else {"behavior": N_BEHAVIOR, "fit": N_FIT, "plane": N_PLANE,
                                                            "scan": N_SCAN, "eval": N_EVAL, "ctx": N_CTX}
    n_ctrl = SMOKE_CONTROLS if args.smoke else N_CONTROLS
    seeds = study.seeds
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    U = P.simplex_tangent_basis(N_STATES)                                                 # [9, 8]
    sites = (P.EMBED_SITE,) + P.post_block_sites(model.cfg.n_layers)

    print("state uncertainty (exact posteriors)", flush=True)
    top_chain = mean_posterior_maximum(world, n["behavior"], seeds["post_training_eval"])
    top_factored = mean_posterior_maximum(World(FACTORED_WORLD), n["behavior"], seeds["post_training_eval"])
    print(f"  mean probability of the most likely state: one chain {top_chain:.4f}, two factors {top_factored:.4f}", flush=True)

    print("behavior", flush=True)
    tb = world.sample(n["behavior"], seed=seeds["post_training_eval"])["tokens"]
    kb = world.answer_key(tb)
    pi = B.model_policy(model, tb, args.device, world.vocab.response_base)
    r = B.pearson_r(pi[..., 0], kb["policy"][..., 0])
    print(f"  r {r:.4f}", flush=True)

    print("reads (fit seed)", flush=True)
    tf = world.sample(n["fit"], seed=seeds["fit"])["tokens"]
    kf = world.answer_key(tf)
    rd = reads_section(model, world, tf, kf, U, sites, n["plane"], args.device, np.random.default_rng([seeds["fit"], ROLE_FOLDS]))
    pool_y = rd["Y_eta"][rd["valid"]]                                                    # natural donor pool
    print("write-site scans (fit seed)", flush=True)
    sc = scan_section(model, world, tf[:n["scan"]], {k: v[:n["scan"]] for k, v in kf.items()}, U, rd["planes"], rd["post"],
                      pool_y, args.device, np.random.default_rng([seeds["fit"], ROLE_SCAN_DONOR]))
    chosen = {"eta_site": rd["eta_site"], "D_site": rd["D_site"], "eta_write": sc["eta_write"], "D_write": sc["D_write"],
              "display": rd["post"][-1]}
    print("  sites " + ", ".join(f"{k} {PL.SITE_LABELS[v]}" for k, v in chosen.items()), flush=True)

    print("evaluation (eval seed)", flush=True)
    te = world.sample(n["eval"], seed=seeds["eval"])["tokens"]
    ke = world.answer_key(te)
    disp = chosen["display"]
    ho = heldout_section(model, world, te, ke, U, rd["planes"]["eta"][disp], rd["planes"]["D"][disp], disp, n["ctx"],
                         args.device, np.random.default_rng([seeds["eval"], ROLE_PARTNER]))
    tw = twins_section(model, world, te, ke, U, rd["planes"]["eta"][chosen["eta_site"]], rd["planes"]["D"][chosen["D_site"]],
                       chosen["eta_site"], chosen["D_site"], args.device)
    st = steering_section(model, world, te, ke, U, rd["planes"], chosen, pool_y, n_ctrl, args.device, seeds["eval"])

    L = PL.SITE_LABELS
    es, ds = chosen["eta_site"], chosen["D_site"]
    H = A.headline_entry
    nf = int(rd["valid"].sum())
    n_top = n["behavior"] * len(ACTION_POSITIONS)
    head = {
        "posterior_maximum_mean": H(top_chain, n_top, "probability", "mean probability of the most likely of the nine states at the readouts, this world"),
        "posterior_maximum_mean_factored": H(top_factored, n_top, "probability", f"same in the two-factor world {FACTORED_WORLD} (joint = product of the factor posteriors)"),
        "choice_pearson_r": H(r, int(pi[..., 0].size), "r", "Pearson r of P(bet A): model vs target policy"),
        "joint_posterior_cv_r2": H(rd["cv"][es]["eta"], nf, "R²", f"joint posterior (8 tangent coords) CV R² at the posterior site {L[es]}"),
        "offer_values_cv_r2": H(rd["cv"][ds]["D"], nf, "R²", f"offer values (Q_A, Q_B) CV R² at the value site {L[ds]}"),
        "value_sum_cv_r2": H(rd["cv"][ds]["sum"], nf, "R²", f"Q_A + Q_B CV R² at {L[ds]}"),
        "value_gap_cv_r2": H(rd["cv"][ds]["gap"], nf, "R²", f"Q_A - Q_B CV R² at {L[ds]}"),
        "heldout_composed_P_r2": H(ho["P_r2_pooled"], ho["n_rows"], "R²", f"withheld offers: P(X) composed from the decoded joint, pooled, {L[disp]}"),
        "heldout_direct_D_r2": H(ho["D_r2_pooled"], ho["n_rows"], "R²", f"withheld offers: Q(X) read directly, pooled, {L[disp]}"),
    }
    for arm in TWIN_ARMS:
        head[f"twin_{arm}_posterior_shift"] = H(tw["records"][arm]["eta_shift_median"], tw["n_kept"], "tangent", f"median decoded posterior shift, {arm} twin, {L[es]}")
        head[f"twin_{arm}_value_shift"] = H(tw["records"][arm]["D_shift_median"], tw["n_kept"], "payoff", f"median decoded value shift, {arm} twin, {L[ds]}")
    head.update({
        "posterior_steer_slope": H(st["transfer"], st["n_kept"], "slope", f"natural posterior steer through the joint subspace at {L[chosen['eta_write']]}: observed vs exact counterfactual"),
        "posterior_steer_control_slope": H(st["control"], st["n_kept"] * n_ctrl, "slope", "matched isometric off-subspace control"),
        "value_endpoint_error_natural": H(st["endpoint_error_median_natural"], st["n_kept"], "payoff", "median |decoded ΔQ − implied ΔQ| under the posterior steer"),
        "value_endpoint_error_control": H(st["endpoint_error_median_control"], st["n_kept"] * n_ctrl, "payoff", "same under the off-subspace control"),
        "value_sweep_slope": H(st["ring"], st["n_ring"] * ST.N_DIRECTIONS, "slope", f"12-direction value sweep at {L[chosen['D_write']]}"),
        "value_sweep_control_slope": H(st["ring_control"], st["n_ring"] * ST.N_DIRECTIONS, "slope", "matched-norm orthogonal control"),
        "posterior_rms_under_value_sweep": H(st["eta_rms_under_value"], st["n_ring"] * ST.N_DIRECTIONS, "tangent", f"RMS decoded posterior shift at {L[disp]} under the value sweep"),
        "posterior_rms_under_posterior_steer": H(st["eta_rms_under_posterior"], st["n_kept"], "tangent", f"RMS decoded posterior shift at {L[disp]} under the posterior steer"),
        "mediation_insertion": H(st["mediation"]["insertion"], st["n_kept"], "slope", "value-plane part of the posterior steer inserted alone: fraction of the effect carried"),
        "mediation_removal": H(st["mediation"]["removal"], st["n_kept"], "slope", "posterior steer with its value-plane part removed: fraction remaining"),
    })
    for name, k in (("posterior_read", "eta_site"), ("posterior_write", "eta_write"), ("value_read", "D_site"), ("value_write", "D_write")):
        head[f"site_{name}"] = H(int(chosen[k].split(".")[1]), 1, "layer", f"{name.replace('_', ' ')} site: {chosen[k]}")
    results = {"headline": head, "sites": chosen, "cv": rd["cv"], "scan": {"eta": sc["eta"], "D": sc["D"]},
               "n": n, "n_controls": n_ctrl, "n_heldout_masks": ho["n_masks"],
               "checkpoint_sha256": study.checkpoint["sha256"], "tangent_basis": "Helmert, 9 states"}
    A.write_results(run_dir, results)
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={"post_training_eval": seeds["post_training_eval"], "fit": seeds["fit"], "eval": seeds["eval"],
                            "roles": {"folds": ROLE_FOLDS, "scan_donor": ROLE_SCAN_DONOR, "donor": ROLE_DONOR, "ctrl": ROLE_CTRL,
                                      "ortho": ROLE_ORTHO, "partner": ROLE_PARTNER}},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force, figures=[],
                     extra={"checkpoint": study.checkpoint, "sizes": n})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
