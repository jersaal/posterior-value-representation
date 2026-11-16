"""09 — reads of what the offer value is made of, and of what the choice ignores.

The pinned instruments (02) decode the factor posteriors and the two offers'
offer values D(A), D(B). Three questions about that pair are answered here
with throwaway probes, which are fit, scored and discarded.

The sum the choice ignores. The unembedding is linear and the choice tracks
the target policy, so the gap D_A − D_B must be linearly readable near the
output; the two offer values are represented separately only if the sum
D_A + D_B, which adds the same constant to both response logits, is readable
too. On 02's rows and folds a probe reads the pair (D_A, D_B) out of fold at
the value site, reproducing 02's cross-validated R² (asserted); least squares
is linear in its targets, so the out-of-fold sum and gap are the sum and
difference of the pair's reads (asserted against a direct fit of the sum).

Held-out offers. Each never-offered proposition is substituted into slot A
of the evaluation contexts (03's held-out procedure, the same partners) and
every read is scored per proposition at the display site: the pinned factor
posteriors; the nine joint probabilities read directly, beside their best
affine prediction from the decoded factor posteriors (what a linear read
could return if only the factors were represented); the offer probability
P(X) = Σ_{s∈X} η(s) composed from the pinned posterior decoder and read
directly; the offer value read directly (the pinned value read, asserted) and
the pinned decoder's sum and gap of the two offer values; the payoff mass
v(X) = Σ_{s∈X} η(s) u(s) read directly, and v beyond D — the decode of the
residual of v on D, whose regression v ≈ a + b·D is fit on the exact fit-pool
values (the probe is linear in its targets, so that decode is
v̂ − (a + b·D̂)).

Use. Are P and v used by the choice, or only carried? At every post-block
site a rank-4 plane [D_A, D_B, q_A, q_B] is fit (q = v beyond D, or P), so a
command can move the decoded q while holding the decoded D exactly (aim
identity W δ = dy). A 12-direction ring in the q coordinates, at the value
ring's per-episode safe radius, is written and the choice-logit shift is
regressed on β·Δ(q_A − q_B), beside a matched-norm ring orthogonal to the
plane and beside the same ring in the plane's D coordinates with q held
(the positive control: it must reproduce the value-ring slope of 06). The
headline quotes the value write site; `use` in `results.json` lists every
site, with the decoded shift of D and q at the display site under each ring.

Linear-reward worlds (`reward == "mass"`). There the value of an offer is
its payoff mass, D = v, so the decomposition of D into P and v does not
exist: v beyond D is identically zero and its plane is degenerate. On such a
world the script computes the sum and gap of the two values at the value
site and stops; the held-out reads and the use section are skipped and their
headline entries are absent."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, instruments as I, plotting as PL, probes as P, stats as SS, steering as STEER, stimuli as ST  # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB   # noqa: E402
from jbo.subject import load_study                                          # noqa: E402

EXPERIMENT = "09_value_reads"
SMOKE_FIT = 512                              # full runs take 02's fit population from the instruments (ins.meta["n_fit"])
N_EVAL, N_CTX, MIN_VALID = 8192, 512, 32     # 03's held-out procedure
SMOKE_EVAL, SMOKE_CTX = 512, 64
ROLE_PARTNER = 2                             # default_rng([eval_seed, ROLE_PARTNER]) — identical to 03's partners
TARGETS = ("P_A", "P_B", "v_A", "v_B", "D_A", "D_B")
READS = ("etaA_pinned", "etaB_pinned", "joint_direct", "joint_affine", "P_composed", "P_direct", "D_direct", "v_direct", "v_beyond_D")
READ_PLAIN = {"etaA_pinned": "pinned factor-A posterior read (pooled over its three coordinates)",
              "etaB_pinned": "pinned factor-B posterior read (pooled over its three coordinates)",
              "joint_direct": "direct joint-posterior read (pooled over its nine coordinates)",
              "joint_affine": "best affine read of the joint posterior from the pinned factor-posterior reads",
              "P_composed": "offer probability composed from the pinned posterior read",
              "P_direct": "direct offer-probability read", "D_direct": "direct offer-value read",
              "v_direct": "direct payoff-mass read", "v_beyond_D": "read of the payoff-mass residual on D (v − a − bD)"}
QUARTILE_READS = tuple(q for q in READS if q != "joint_affine")   # the paper quotes the affine ceiling's median only
HELDOUT_PAIR = {"sum": "sum D_A + D_B", "gap": "gap D_A − D_B"}  # read by the pinned decoder, both slots
PINNED_TOL = 1e-6                            # direct D̂ vs the pinned value read, same rows -> same probe
N_SCAN, SMOKE_SCAN = 1024, 128               # eval episodes written in the use section
MAX_COND_USE = 1e3                           # a rank-4 plane whose read map is worse conditioned is reported degenerate
ROLE_USE_CTRL = 3                            # default_rng([eval_seed, ROLE_USE_CTRL])
USE_PLANES = {"mass": "v beyond D", "prob": "P"}
USE_KEY = {"mass": "v", "prob": "P"}         # headline prefix of each plane's q ring
N_FOLDS, ROLE_FOLDS = 5, 1                   # 02's folds exactly: default_rng([fit_seed, ROLE_FOLDS])
CV_TOL = 1e-4                                # pair CV R² vs 02's pinned table, same rows and folds -> same numbers
LINEAR_TOL = 1e-8                            # direct fit of the sum vs the sum of the pair's fits: the same least squares


def targets_at_grade(key: dict) -> np.ndarray:
    """Exact (P_A, P_B, v_A, v_B, D_A, D_B) at the grading readout -> [N, 6]."""
    Pk, vk, Dk = (np.asarray(key[k], np.float64)[:, GRADE_READOUT] for k in ("P", "v", "D"))   # [N, 2] each
    assert np.allclose(vk, Pk * Dk), "answer key must satisfy v = P·D"
    return np.column_stack([Pk, vk, Dk])


def mass_on_value(Y: np.ndarray) -> tuple[float, float]:
    """Least squares v ≈ a + b·D on the exact fit-pool values, both slots pooled."""
    v, D = Y[:, 2:4].ravel(), Y[:, 4:6].ravel()
    b, a = np.polyfit(D, v, 1)
    return float(a), float(b)


def fit_read_probes(X: np.ndarray, Y: np.ndarray, eta9: np.ndarray, factors_hat: np.ndarray) -> dict:
    """The display site's throwaway probes on the valid fit rows. X [n, d] activations,
    Y [n, 6] TARGETS, eta9 [n, 9] exact joint posterior, factors_hat [n, 6] the pinned
    decode of both factor posteriors. {"six": [d+1, 6], "joint": [d+1, 9], "joint_affine": [7, 9]}."""
    return {"six": P.affine_fit(X, Y), "joint": P.affine_fit(X, eta9), "joint_affine": P.affine_fit(factors_hat, eta9)}


def reads_for(mask: int, X: np.ndarray, eight: np.ndarray, probes: dict, ins: I.Instruments, ab: tuple) -> dict:
    """Slot-A readings of the READS from the display-site activations X [n, d] and their
    pinned decode `eight` [n, 8]: the factor posteriors are [n, 3], the joint reads [n, 9], the rest [n]."""
    a, b = ab
    six = P.affine_predict(X, probes["six"])                                     # [n, 6]
    return {"etaA_pinned": ins.posterior_A(eight), "etaB_pinned": ins.posterior_B(eight),
            "joint_direct": P.affine_predict(X, probes["joint"]), "joint_affine": P.affine_predict(eight[:, 0:6], probes["joint_affine"]),
            "P_composed": ins.compose_P(eight, np.full(len(X), mask)), "P_direct": six[:, 0],
            "D_direct": six[:, 4], "v_direct": six[:, 2], "v_beyond_D": six[:, 2] - (a + b * six[:, 4])}


def truths_for(k: dict, ab: tuple) -> dict:
    """The exact counterpart of every read in `reads_for`, from the answer key `k`."""
    a, b = ab
    Pt, vt, Dt = (k[q][:, GRADE_READOUT, 0] for q in ("P", "v", "D"))
    eta9 = k["eta"][:, GRADE_POSITION]
    return {"etaA_pinned": k["etaA"][:, GRADE_POSITION], "etaB_pinned": k["etaB"][:, GRADE_POSITION],
            "joint_direct": eta9, "joint_affine": eta9, "P_composed": Pt, "P_direct": Pt,
            "D_direct": Dt, "v_direct": vt, "v_beyond_D": vt - (a + b * Dt)}


def score(truth: np.ndarray, read: np.ndarray) -> float:
    """Per-proposition R²: pooled over coordinates for a [n, k] posterior (02's
    convention), plain for an [n] scalar read."""
    assert truth.shape == read.shape, (truth.shape, read.shape)
    return P.decode_scores(truth, read)["r2"] if truth.ndim == 2 else SS.r2(truth, read)


def sum_and_gap(pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N, 2] -> (sum [N], gap [N])."""
    return pair[:, 0] + pair[:, 1], pair[:, 0] - pair[:, 1]


def sum_gap_cv(X: np.ndarray, D: np.ndarray, valid: np.ndarray, folds: list) -> dict:
    """One grouped-CV probe for (D_A, D_B) at a site, X [N, d], D [N, 2]. R² over the valid
    rows of the pair, and of the sum and the gap formed from the pair's out-of-fold reads;
    `linear_gap` is the largest difference between that sum and a direct out-of-fold fit of it."""
    oof = P.cv_decode(X[:, None, :], D[:, None, :], folds, valid[:, None])["oof"][:, 0, :]             # [N, 2]
    S, G = sum_and_gap(D[valid]); Sh, Gh = sum_and_gap(oof[valid])
    direct = P.cv_decode(X[:, None, :], sum_and_gap(D)[0][:, None, None], folds, valid[:, None])["oof"][:, 0, 0]
    return {"pair": P.decode_scores(D[valid], oof[valid])["r2"], "sum": SS.r2(S, Sh), "gap": SS.r2(G, Gh),
            "linear_gap": float(np.abs(direct - sum_and_gap(oof)[0])[valid].max())}


def fit_use_plane(X: np.ndarray, Y4: np.ndarray) -> dict | None:
    """Rank-4 forward plane for [D_A, D_B, q_A, q_B]: Q = cross-covariance basis,
    probe = affine read, M = Qᵀ W. None when the read map is degenerate."""
    Q = P.encoding_subspace(X, Y4)
    probe = P.affine_fit(X, Y4)
    M = Q.T @ probe[1:]
    cond = float(np.linalg.cond(M))
    if not np.isfinite(cond) or cond > MAX_COND_USE:
        return None
    return {"Q": Q, "probe": probe, "M": M, "cond": cond}


def use_targets(Y: np.ndarray, ab: tuple) -> dict:
    """[N, 4] targets per plane from the six exact quantities: mass = [D, v − a − bD], prob = [D, P]."""
    a, b = ab
    D, v, Pm = Y[:, 4:6], Y[:, 2:4], Y[:, 0:2]
    return {"mass": np.column_stack([D, v - (a + b * D)]), "prob": np.column_stack([D, Pm])}


def use_section(model, world, beta_six: np.ndarray, acts: dict, Y: np.ndarray, valid: np.ndarray, ab: tuple, sites: tuple,
                disp: str, toks: np.ndarray, key: dict, device: str, rng: np.random.Generator) -> dict:
    """Aimed rings in each plane's q coordinates (D held) and in its D coordinates
    (q held), written at every site in `sites` (in that order — it fixes the control
    stream) and read at the display site, whose throwaway probe is `beta_six` [d+1, 6].
    Returns {site: {plane: res}}; res holds stats.slope fits "q", "q_control", "D",
    "D_control", "cond", and "decoded_shift" {"q" | "D": medians of the decoded
    display-site shift of D and of q beside the commanded radius}."""
    rb, pos, beta = world.vocab.response_base, GRADE_POSITION, world.beta
    D_scan, Pk = (key[k][:, GRADE_READOUT] for k in ("D", "P"))                    # [N, 2] each
    radius = STEER.safe_radii(D_scan)
    keep = (Pk.min(-1) >= MIN_OFFER_PROB) & (radius >= STEER.MIN_RADIUS)
    _, directions = STEER.ring_directions()
    clean = STEER.run_edited(model, toks, [], (disp,), pos, device, rb)
    Y0 = STEER.policy_logit(clean["pi"][:, GRADE_READOUT])
    six0 = P.affine_predict(clean["resid"][disp], beta_six)
    b_v = ab[1]
    tg = use_targets(Y[valid], ab)
    groups = np.tile(np.flatnonzero(keep), STEER.N_DIRECTIONS)

    def sweep(site: str, plane: dict, plane_name: str, sub: str, controls: np.ndarray) -> dict:
        """One 12-direction ring in the plane's `sub` coordinates written at `site`; controls [d, 12]."""
        cols = (2, 3) if sub == "q" else (0, 1)
        xs, ys, ycs, dD, dq = [], [], [], [], []
        for k, d in enumerate(directions):
            dy = np.zeros((len(toks), 4)); dy[:, cols[0]] = radius * d[0]; dy[:, cols[1]] = radius * d[1]
            delta = (dy @ np.linalg.inv(plane["M"])) @ plane["Q"].T
            assert np.allclose(delta @ plane["probe"][1:], dy, atol=1e-6), "aim identity"
            delta_c = np.linalg.norm(delta, axis=1)[:, None] * controls[:, k][None, :]
            arm = STEER.run_edited(model, toks, [(site, pos, delta)], (disp,), pos, device, rb)
            arm_c = STEER.run_edited(model, toks, [(site, pos, delta_c)], (), pos, device, rb)
            xs.append(beta * (dy[keep, cols[0]] - dy[keep, cols[1]]))
            ys.append((STEER.policy_logit(arm["pi"][:, GRADE_READOUT]) - Y0)[keep])
            ycs.append((STEER.policy_logit(arm_c["pi"][:, GRADE_READOUT]) - Y0)[keep])
            six = P.affine_predict(arm["resid"][disp], beta_six)
            dD.append(np.linalg.norm((six[:, 4:6] - six0[:, 4:6])[keep], axis=1))
            if plane_name == "mass":
                q1, q0 = six[:, 2:4] - b_v * six[:, 4:6], six0[:, 2:4] - b_v * six0[:, 4:6]
            else:
                q1, q0 = six[:, 0:2], six0[:, 0:2]
            dq.append(np.linalg.norm((q1 - q0)[keep], axis=1))
        x, y, yc = np.concatenate(xs), np.concatenate(ys), np.concatenate(ycs)
        return {sub: SS.slope(x, y, groups=groups, n_boot=300), sub + "_control": SS.slope(x, yc, groups=groups, n_boot=300),
                "decoded_shift": {"D_median": float(np.median(np.concatenate(dD))), "q_median": float(np.median(np.concatenate(dq))),
                                  "commanded_median": float(np.median(radius[keep]))}}

    out = {}
    for s in sites:
        X = acts[s][:, 0, :].astype(np.float64)[valid]
        out[s] = {}
        for plane_name, Y4 in tg.items():
            plane = fit_use_plane(X, Y4)
            if plane is None:
                out[s][plane_name] = {"cond": float("nan"), "degenerate": True}
                print(f"    use {PL.SITE_LABELS[s]} {plane_name}: degenerate plane", flush=True)
                continue
            controls = STEER.orthogonal_directions(plane["Q"], 2 * STEER.N_DIRECTIONS, rng)
            res = {"cond": plane["cond"], "degenerate": False, "decoded_shift": {}}
            for i, sub in enumerate(("q", "D")):
                sw = sweep(s, plane, plane_name, sub, controls[:, i * STEER.N_DIRECTIONS: (i + 1) * STEER.N_DIRECTIONS])
                res["decoded_shift"][sub] = sw.pop("decoded_shift")
                res.update(sw)
                print(f"    use {PL.SITE_LABELS[s]} {plane_name} ring in {sub}: slope {res[sub]['slope']:.3f}±{res[sub]['se']:.3f} "
                      f"control {res[sub + '_control']['slope']:.3f}; decoded shift at {PL.SITE_LABELS[disp]}: "
                      f"D {res['decoded_shift'][sub]['D_median']:.3f}, q {res['decoded_shift'][sub]['q_median']:.3f} "
                      f"(commanded {res['decoded_shift'][sub]['commanded_median']:.3f})", flush=True)
            out[s][plane_name] = res
    return out


def use_headlines(use_w: dict, label: str) -> dict:
    """The value write site's six slopes: each plane's q ring, its off-plane control, and its D ring."""
    def entry(res: dict, fit: str, desc: str) -> dict:
        if res.get("degenerate"):
            return A.headline_entry(float("nan"), 0, "slope", desc + " (degenerate plane)")
        return A.headline_entry(res[fit]["slope"], res[fit]["n"], "slope", desc)
    head = {}
    for pl, qname in USE_PLANES.items():
        head[f"{USE_KEY[pl]}_use_slope_{label}"] = entry(use_w[pl], "q", f"choice-logit slope on β·Δ({qname}) for an aimed ring at {label} with decoded D held")
    for pl in USE_PLANES:
        head[f"{USE_KEY[pl]}_use_control_{label}"] = entry(use_w[pl], "q_control", f"the same commands copied to random directions orthogonal to the {pl} plane at {label}")
    for pl in USE_PLANES:
        head[f"D_in_{pl}_plane_slope_{label}"] = entry(use_w[pl], "D", f"the D ring of the rank-4 {pl} plane at {label} (q held): should match the value ring")
    return head


def main() -> None:
    args = cli.parse("09_value_reads: sum and gap of the offer values, held-out reads of their parts, and whether the parts are used")
    study = load_study(args.study)
    world = study.build_world()
    model, _ = study.load_actor(args.device)
    fit_seed, eval_seed = study.seeds["fit"], study.seeds["eval"]
    ins = I.load(study, eval_seeds=[eval_seed])
    post = tuple(s for s in ins.sites if s != P.EMBED_SITE)
    n_fit, n_eval, n_ctx = (SMOKE_FIT, SMOKE_EVAL, SMOKE_CTX) if args.smoke else (int(ins.meta["n_fit"]), N_EVAL, N_CTX)
    n_scan = SMOKE_SCAN if args.smoke else N_SCAN
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    pos = np.array([GRADE_POSITION])
    disp, vsite = ins.display_site, ins.value_site
    lab = PL.SITE_LABELS
    seeds = {"fit": fit_seed, "eval": eval_seed, "folds": [fit_seed, ROLE_FOLDS]}

    def finish(results: dict) -> None:
        A.write_results(run_dir, results)
        A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv, seeds=seeds,
                         spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                         figures=[], extra={"checkpoint": study.checkpoint, "instruments_sha256": ins.meta["artifact_sha256"]})
        print(json.dumps({k: v["value"] for k, v in results["headline"].items()}, indent=1)); print(f"done: {run_dir}")

    # 02's rows exactly: fit seed, grading readout, valid rows
    tokens = world.sample(n_fit, seed=fit_seed)["tokens"]
    key = world.answer_key(tokens)
    valid = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    n_valid_fit = int(valid.sum())
    acts = P.collect_site_activations(model, tokens, post, pos, args.device)       # {site: [N, 1, d]}

    # the sum the choice ignores: the pair out of fold on 02's rows and folds, at the value site
    D = np.asarray(key["D"], np.float64)[:, GRADE_READOUT]                         # exact (D_A, D_B) [N, 2]
    folds = P.grouped_folds(n_fit, N_FOLDS, np.random.default_rng([fit_seed, ROLE_FOLDS]))
    sg = sum_gap_cv(acts[vsite][:, 0, :].astype(np.float64), D, valid, folds)
    assert args.smoke or abs(sg["pair"] - ins.cv[vsite]["D"]) < CV_TOL, \
        f"pair CV R² differs from 02's pinned table by {abs(sg['pair'] - ins.cv[vsite]['D']):.2e}"
    assert sg["linear_gap"] < LINEAR_TOL, f"direct sum fit differs from the summed pair fit by {sg['linear_gap']:.2e}"
    head = {
        "sum_cv_r2_value_site": A.headline_entry(sg["sum"], n_valid_fit, "R²", f"CV R² of the sum D_A + D_B read from the pair probe at the value site {lab[vsite]} (02's rows and folds)"),
        "gap_cv_r2_value_site": A.headline_entry(sg["gap"], n_valid_fit, "R²", f"CV R² of the gap D_A − D_B read from the pair probe at {lab[vsite]}")}
    common = {"reward": world.reward, "value_site": vsite, "display_site": disp, "n_fit": n_fit, "n_valid_fit": n_valid_fit}
    if world.reward == "mass":                                                     # D = v: no P / v decomposition to read or to write
        finish({"headline": head, **common})
        return

    # throwaway probes at the display site, on the same rows
    Y = targets_at_grade(key)                                                      # [N, 6]
    ab = mass_on_value(Y[valid])
    X_fit = acts[disp][:, 0, :].astype(np.float64)[valid]
    probes = fit_read_probes(X_fit, Y[valid], key["eta"][:, GRADE_POSITION][valid], ins.decode(disp, X_fit)[:, 0:6])

    # read: 03's held-out procedure — each never-offered proposition into slot A of the eval contexts
    base = world.sample(n_eval, seed=eval_seed)["tokens"][:n_ctx]
    rng = np.random.default_rng([eval_seed, ROLE_PARTNER])
    pool = world.algebra.partner_pool()                                            # the family's offerable masks
    kept, skipped = [], []
    r2 = {q: [] for q in READS + tuple(HELDOUT_PAIR)}                              # -> [K] each
    max_pinned_gap = 0.0
    for mask in world.algebra.heldout_masks:
        elig = pool[(pool & int(mask)) == 0]
        partners = elig[rng.integers(0, len(elig), size=n_ctx)]
        toks = ST.substitute_standing_offers(world, base, np.full(n_ctx, int(mask)), partners)
        k = world.answer_key(toks)
        ok = k["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
        if int(ok.sum()) < MIN_VALID:
            skipped.append(int(mask)); continue
        kept.append(int(mask))
        X = P.collect_site_activations(model, toks, (disp,), pos, args.device)[disp][:, 0, :].astype(np.float64)
        eight = ins.decode(disp, X)
        truth, rd = truths_for(k, ab), reads_for(int(mask), X, eight, probes, ins, ab)
        max_pinned_gap = max(max_pinned_gap, float(np.abs(P.affine_predict(X, probes["six"])[:, 4:6] - ins.D(eight)).max()))
        for q in READS:
            r2[q].append(score(truth[q][ok], rd[q][ok]))
        (St, Gt), (Sh, Gh) = sum_and_gap(k["D"][:, GRADE_READOUT][ok]), sum_and_gap(ins.D(eight)[ok])
        r2["sum"].append(SS.r2(St, Sh)); r2["gap"].append(SS.r2(Gt, Gh))
    assert args.smoke or max_pinned_gap < PINNED_TOL, \
        f"direct D̂ differs from the pinned value read by {max_pinned_gap:.2e}"     # smoke fits on fewer rows than 02
    r2 = {q: np.array(v) for q, v in r2.items()}                                   # [K] each
    stat = {"median": lambda v: float(np.nanmedian(v)), "q25": lambda v: float(np.nanpercentile(v, 25)), "q75": lambda v: float(np.nanpercentile(v, 75))}

    def h(val: float, desc: str) -> dict:
        return A.headline_entry(val, len(kept), "R²", desc)
    head.update({f"{q}_median_r2": h(stat["median"](r2[q]), f"median per-held-out-proposition R² of the {READ_PLAIN[q]} at {lab[disp]}") for q in READS})
    head.update({f"{q}_{tag}_r2": h(stat[tag](r2[q]), f"{tag} quartile of the per-held-out-proposition R² of the {READ_PLAIN[q]} at {lab[disp]}")
                 for q in QUARTILE_READS for tag in ("q25", "q75")})
    head.update({f"heldout_{q}_{tag}_r2": h(stat[tag](r2[q]), f"{tag} of the per-held-out-proposition R² of the {desc} read by the pinned decoder at {lab[disp]}")
                 for q, desc in HELDOUT_PAIR.items() for tag in ("median", "q25", "q75")})

    # use: aimed rings with D held, written at every post-block site (the value write site and the
    # display site first: that order fixes the control stream), read at the display site
    assert not ins.reads_only, f"{study.name} has reads-only instruments (no write sites): the use section needs a trained choice"
    wsite = ins.value_write_site
    toks_scan = world.sample(n_eval, seed=eval_seed)["tokens"][:n_scan]
    use = use_section(model, world, probes["six"], acts, Y, valid, ab, tuple(dict.fromkeys((wsite, disp) + post)), disp, toks_scan,
                      world.answer_key(toks_scan), args.device, np.random.default_rng([eval_seed, ROLE_USE_CTRL]))
    head.update(use_headlines(use[wsite], lab[wsite]))

    seeds.update({"partner": [eval_seed, ROLE_PARTNER], "use_control": [eval_seed, ROLE_USE_CTRL]})
    finish({"headline": head, **common,
            "use": {lab[s]: use[s] for s in post}, "value_write_site": wsite, "n_scan": n_scan,
            "per_proposition_r2_display": {q: v.tolist() for q, v in r2.items()}, "kept_masks": kept, "skipped_masks": skipped,
            "v_on_D": {"a": ab[0], "b": ab[1]}, "n_ctx": n_ctx, "targets": list(TARGETS)})
    print(f"kept {len(kept)} propositions, skipped {skipped}; v ≈ {ab[0]:.3f} + {ab[1]:.3f}·D")


if __name__ == "__main__":
    main()
