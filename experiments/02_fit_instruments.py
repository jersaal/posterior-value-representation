"""02 — fit the study's instruments: per-site
read decoders with grouped CV; read sites by the formation threshold (the
earliest post-block site with CV R² >= 0.95); the value and posterior WRITE
sites by steering at every post-block layer on the fit split (the value
direction sweep through each layer's own value plane; the natural posterior
steer through each layer's own posterior plane), each site the earliest
post-block layer whose steering slope reaches 0.95 of the ladder maximum;
posterior/value write planes at the chosen write sites; posterior dose units.
`--reads-only` stops after the read decoders and read sites (for an actor
whose choice is untrained, where no steering slope is defined) and adds the
best value read over the layers to the headline. The per-site CV table and the
two steering-slope ladders in results.json are what the depth-profile figures
draw. No figures here.
The ONLY script that fits anything (the throwaway probes of 04 and 09 excepted)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, probes as P, stats as SS, steering as ST   # noqa: E402
from jbo.instruments import (GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB, ROWS_PER_DIM_FIT,  # noqa: E402
                             ROWS_PER_DIM_PLANE, TARGETS, artifact_paths, fit_sizes)
from jbo.subject import load_study                                      # noqa: E402

EXPERIMENT = "02_fit_instruments"
N_FOLDS = 5
FORMATION_R2 = 0.95        # read site = earliest post-block site with CV R² >= this (fallback: best)
WRITE_FRACTION = 0.95      # write site = earliest post-block site whose steering slope >= this × ladder max
N_SCAN, SMOKE_SCAN = 1024, 128   # fit episodes steered by the write-site scans
MAX_COND = 100.0
ROLE_FOLDS = 1             # default_rng([fit_seed, ROLE_FOLDS])
ROLE_POSTERIOR_SCAN = 2       # default_rng([fit_seed, ROLE_POSTERIOR_SCAN, factor]) — donor draw of the posterior scan
GROUPS = {"etaA": (0, 3), "etaB": (3, 6), "D": (6, 8)}


def fit_plane(X: np.ndarray, Y: np.ndarray) -> dict:
    """Q = rank-2 cross-covariance plane, W = affine probe, M = Qᵀ W (cond < MAX_COND)."""
    Q = P.encoding_subspace(X, Y)
    probe = P.affine_fit(X, Y)
    M = Q.T @ probe[1:]
    cond = float(np.linalg.cond(M))
    assert cond < MAX_COND, f"degenerate plane read map (cond {cond:.1f})"
    return {"Q": Q, "probe": probe, "M": M, "cond": cond}


def formation_site(scores: dict[str, float], threshold: float = FORMATION_R2) -> tuple[str, bool]:
    """Earliest site (ladder order) with score >= threshold; falls back to the
    best site when none reaches it. Returns (site, reached_threshold)."""
    for site, v in scores.items():
        if v >= threshold:
            return site, True
    return max(scores, key=scores.get), False


def relative_formation_site(scores: dict[str, float], fraction: float = WRITE_FRACTION) -> str:
    """Earliest site (ladder order) whose finite score reaches `fraction` of the
    ladder maximum. The write-site analogue of `formation_site`: the write score
    has no fixed ceiling, so the threshold is relative to the ladder."""
    finite = {s: v for s, v in scores.items() if np.isfinite(v)}
    assert finite, "no site has a finite write score"
    top = max(finite.values())
    assert top > 0, f"no site has a positive write score (max {top:.3f})"
    return next(s for s, v in finite.items() if v >= fraction * top)


def clean_policy_logit(model, world, tokens: np.ndarray, device: str) -> np.ndarray:
    """Clean A-minus-B choice logit at the graded readout, [N]."""
    clean = ST.run_edited(model, tokens, [], (), GRADE_POSITION, device, world.vocab.response_base)
    return ST.policy_logit(clean["pi"][:, GRADE_READOUT])


def write_site_scan(model, world, tokens: np.ndarray, key: dict, planes: dict, post: tuple,
                    device: str, Y0: np.ndarray) -> dict[str, float]:
    """Value steering slope at every post-block site, by the direction sweep 06 reports.

    At each site the steering subspace fit at that layer itself, from that
    layer's own activations (plane split, valid rows), is written along the
    12-direction sweep (per-episode safe radii) on the scan episodes, and
    scored by the plain OLS slope of the observed choice-logit shift on the
    commanded one. `planes` maps site -> that layer's value plane (None where
    degenerate, slope NaN). This ladder chooses the value write site. Nothing
    touches the eval seed."""
    D_clean = key["D"][:, GRADE_READOUT]
    radius = ST.safe_radii(D_clean)
    keep = (key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB) & (radius >= ST.MIN_RADIUS)
    _, directions = ST.ring_directions()
    rb = world.vocab.response_base

    def ring_slope(site: str, plane) -> float:
        if plane is None:
            return float("nan")
        xs, ys = [], []
        for d in directions:
            dy = radius[:, None] * d
            delta = (dy @ np.linalg.inv(plane["M"])) @ plane["Q"].T
            arm = ST.run_edited(model, tokens, [(site, GRADE_POSITION, delta)], (), GRADE_POSITION, device, rb)
            xs.append(world.beta * (dy[keep, 0] - dy[keep, 1]))
            ys.append(ST.policy_logit(arm["pi"][:, GRADE_READOUT])[keep] - Y0[keep])
        x, y = np.concatenate(xs), np.concatenate(ys)
        return float(SS.slope(x, y, n_boot=0)["slope"]) if np.ptp(x) > 0 else float("nan")

    slopes = {}
    for site in post:
        slopes[site] = ring_slope(site, planes[site])
        print(f"    value write scan {site}: slope {slopes[site]:.3f}", flush=True)
    return slopes


def posterior_write_site_scan(model, world, tokens: np.ndarray, key: dict, planes: dict, post: tuple,
                           factor: int, pool_y: np.ndarray, basis: np.ndarray, rng: np.random.Generator,
                           device: str, Y0: np.ndarray) -> dict[str, float]:
    """Posterior steering slope at every post-block site, by the natural posterior steer 06 reports.

    Each scan episode gets a natural donor posterior for `factor` from the fit
    pool (the draw 06 makes at evaluation); the commanded tangent displacement
    is aimed through each site's own posterior plane and added there, and the
    observed choice-logit shift is regressed on the exact counterfactual
    beta * delta(D_A - D_B). `planes` maps site -> that factor's plane (None
    where degenerate); `pool_y` [M, 2] is the fit pool in tangent units.
    """
    etaA, etaB = key["etaA"][:, GRADE_POSITION], key["etaB"][:, GRADE_POSITION]
    eta_f, eta_o = (etaA, etaB) if factor == 0 else (etaB, etaA)
    y_clean = P.to_tangent(eta_f, basis)                                   # [N, 2]
    donors = ST.draw_natural_donors(y_clean, pool_y, rng)
    eta_new = P.from_tangent(y_clean + donors["dy"], basis)
    assert eta_new.min() >= -1e-9, "donor posterior left the simplex"
    bits, u = key["bits_ro"][:, GRADE_READOUT], key["u"]
    D_clean = key["D"][:, GRADE_READOUT]
    same = ST.counterfactual_D(eta_f, eta_o, factor, bits, u, world.reward)["D"]
    assert np.allclose(same, D_clean, atol=1e-9), "counterfactual D at the clean posterior must equal the answer key"
    cf = ST.counterfactual_D(eta_new, eta_o, factor, bits, u, world.reward)
    keep = (key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB) & (cf["P"].min(-1) >= MIN_OFFER_PROB)
    x = world.beta * ((cf["D"][:, 0] - cf["D"][:, 1]) - (D_clean[:, 0] - D_clean[:, 1]))
    rb = world.vocab.response_base
    slopes = {}
    for site in post:
        plane = planes[site]
        if plane is None:
            slopes[site] = float("nan")
            continue
        delta = (donors["dy"] @ np.linalg.inv(plane["M"])) @ plane["Q"].T    # [N, d]
        assert np.allclose(delta @ plane["probe"][1:], donors["dy"], atol=1e-6), "aim identity"
        arm = ST.run_edited(model, tokens, [(site, GRADE_POSITION, delta)], (), GRADE_POSITION, device, rb)
        y = ST.policy_logit(arm["pi"][:, GRADE_READOUT]) - Y0
        slopes[site] = float(SS.slope(x[keep], y[keep], n_boot=0)["slope"])
        print(f"    posterior-{'AB'[factor]} write scan {site}: slope {slopes[site]:.3f}", flush=True)
    return slopes


def layer_of(site: str) -> int:
    """`blocks.<l>.hook_resid_post` -> l."""
    assert site.endswith("hook_resid_post"), site
    return int(site.split(".")[1])


def main() -> None:
    args = cli.parse("02_fit_instruments: fit read decoders and write planes",
                     lambda p: p.add_argument("--reads-only", action="store_true",
                                              help="read decoders and read sites only, no write scans or planes: for an actor "
                                                   "whose choice is untrained (a next-token pretrained model), where every "
                                                   "steering slope is zero and the write-site rule is undefined"))
    study = load_study(args.study)
    world = study.build_world()
    model, config = study.load_actor(args.device)
    n_fit, n_plane = fit_sizes(study.arch["d_model"], args.smoke)
    print(f"  fit split {n_fit} episodes, plane split {n_plane} (d_model {study.arch['d_model']})", flush=True)
    fit_seed = study.seeds["fit"]
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None),
                            force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    basis = P.simplex_tangent_basis(3)

    # population
    d = world.sample(n_fit, seed=fit_seed)
    tokens = d["tokens"]
    key = world.answer_key(tokens)
    etaA = key["etaA"][:, GRADE_POSITION]                                   # [N, 3]
    etaB = key["etaB"][:, GRADE_POSITION]
    D = key["D"][:, GRADE_READOUT]                                          # [N, 2]
    valid = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB            # [N]
    Y = np.concatenate([etaA, etaB, D], axis=1)                             # [N, 8]
    sites = (P.EMBED_SITE,) + P.post_block_sites(study.arch["n_layers"])
    acts = P.collect_site_activations(model, tokens, sites, np.array([GRADE_POSITION]), args.device)
    X = {s: acts[s][:, 0, :].astype(np.float64) for s in sites}              # [N, d]
    folds = P.grouped_folds(n_fit, N_FOLDS, np.random.default_rng([fit_seed, ROLE_FOLDS]))

    # read decoders + CV table (fit and scored on valid rows)
    cv, betas = {}, {}
    for s in sites:
        out = P.cv_decode(X[s][:, None, :], Y[:, None, :], folds, valid[:, None])
        oof = out["oof"][:, 0, :][valid]
        cv[s] = {g: P.decode_scores(Y[valid][:, lo:hi], oof[:, lo:hi])["r2"] for g, (lo, hi) in GROUPS.items()}
        betas[s] = P.affine_fit(X[s][valid], Y[valid])
    post = sites[1:]
    posterior_scores = {s: 0.5 * (cv[s]["etaA"] + cv[s]["etaB"]) for s in post}
    value_scores = {s: cv[s]["D"] for s in post}
    (posterior_site, posterior_formed) = formation_site(posterior_scores)
    (value_site, value_formed) = formation_site(value_scores)
    display_site = post[-1]

    # Write planes on the first n_plane episodes. Each write site is chosen by
    # steering the first n_scan episodes at every post-block layer through that
    # layer's own read-fitted plane and applying the relative formation rule.
    sl = slice(0, n_plane)
    n_scan = SMOKE_SCAN if args.smoke else N_SCAN
    yA, yB = P.to_tangent(etaA[sl], basis), P.to_tangent(etaB[sl], basis)
    value_planes, posterior_planes = {}, {"A": {}, "B": {}}
    for site in post:
        try:
            value_planes[site] = fit_plane(X[site][sl][valid[sl]], D[sl][valid[sl]])
        except AssertionError:                      # degenerate read map at this site
            value_planes[site] = None
        for name, y in (("A", yA), ("B", yB)):
            try:
                posterior_planes[name][site] = fit_plane(X[site][sl], y)
            except AssertionError:
                posterior_planes[name][site] = None
    if args.reads_only:
        scan, posterior_scan, value_write_site, posterior_write_site, planes = None, None, None, None, {}
        print("  reads only: no write scans, no planes", flush=True)
    else:
        key_scan = {k: v[:n_scan] for k, v in key.items()}
        tokens_scan = tokens[:n_scan]
        Y0 = clean_policy_logit(model, world, tokens_scan, args.device)
        scan = {"own": write_site_scan(model, world, tokens_scan, key_scan, value_planes, post, args.device, Y0)}
        value_write_site = relative_formation_site(scan["own"])
        pools = {"A": P.to_tangent(etaA, basis), "B": P.to_tangent(etaB, basis)}   # fit-episode donor pools, as in 06
        posterior_scan, posterior_write_site = {}, {}
        for name, fi in (("A", 0), ("B", 1)):
            posterior_scan[name] = posterior_write_site_scan(model, world, tokens_scan, key_scan, posterior_planes[name], post, fi,
                                                       pools[name], basis, np.random.default_rng([fit_seed, ROLE_POSTERIOR_SCAN, fi]),
                                                       args.device, Y0)
            posterior_write_site[name] = relative_formation_site(posterior_scan[name])
        print(f"  value write site (sweep slope, earliest at {WRITE_FRACTION:g} of max): {value_write_site}; "
              f"posterior write sites {posterior_write_site}", flush=True)
        planes = {"posterior_A": posterior_planes["A"][posterior_write_site["A"]], "posterior_B": posterior_planes["B"][posterior_write_site["B"]],
                  "value": value_planes[value_write_site]}
    posterior_unit = {"A": float(0.5 * (yA[:, 0].std() + yA[:, 1].std())),
                   "B": float(0.5 * (yB[:, 0].std() + yB[:, 1].std()))}

    npz, js = artifact_paths(study)
    npz.parent.mkdir(parents=True, exist_ok=True)
    if js.exists() and not args.smoke and not args.force:
        raise FileExistsError(f"{js} exists — refitting invalidates downstream runs; pass --force")
    if args.smoke:
        npz, js = run_dir / "instruments.npz", run_dir / "instruments.json"
    arrays = {f"beta__{s}": betas[s] for s in sites}
    for k, pl in planes.items():
        arrays.update({f"{k}__Q": pl["Q"], f"{k}__probe": pl["probe"], f"{k}__M": pl["M"]})
    np.savez(npz, **arrays)
    meta = {"study": study.name, "checkpoint_sha256": study.checkpoint["sha256"],
            "artifact_sha256": A.sha256_file(npz), "fit_seed": fit_seed, "n_fit": n_fit, "n_plane": n_plane,
            "rows_per_dim": {"fit": ROWS_PER_DIM_FIT, "plane": ROWS_PER_DIM_PLANE},
            "grade_readout": GRADE_READOUT, "grade_position": GRADE_POSITION, "min_offer_prob": MIN_OFFER_PROB,
            "n_valid": int(valid.sum()), "sites": list(sites), "targets": list(TARGETS), "cv": cv,
            "formation_r2": FORMATION_R2, "posterior_site": posterior_site, "value_site": value_site,
            "posterior_site_reached_threshold": posterior_formed, "value_site_reached_threshold": value_formed,
            "value_write_site": value_write_site, "posterior_write_site": posterior_write_site,
            "write_fraction": WRITE_FRACTION,
            "write_site_rule": f"earliest post-block site whose steering slope reaches {WRITE_FRACTION:g} of the ladder maximum",
            "reads_only": bool(args.reads_only), "planes": sorted(planes),
            "write_scan_slopes": scan, "posterior_write_scan_slopes": posterior_scan, "n_scan": n_scan,
            "display_site": display_site, "plane_cond": {k: v["cond"] for k, v in planes.items()},
            "posterior_unit": posterior_unit, "run_dir": str(run_dir.relative_to(A.ROOT)), "smoke": args.smoke}
    js.write_text(json.dumps(meta, indent=2) + "\n")
    if not args.smoke:                                # record the instruments pin in study.json
        raw = json.loads((study.dir / "study.json").read_text())
        raw["instruments"] = {"path": str(js.relative_to(A.ROOT)), "artifact_sha256": meta["artifact_sha256"]}
        (study.dir / "study.json").write_text(json.dumps(raw, indent=2) + "\n")

    n_valid = int(valid.sum())
    headline = {
        "posterior_r2_etaA": A.headline_entry(cv[posterior_site]["etaA"], n_valid, "R²",
                                              f"factor-M posterior CV R² at the posterior site {posterior_site}"),
        "posterior_r2_etaB": A.headline_entry(cv[posterior_site]["etaB"], n_valid, "R²",
                                              f"factor-N posterior CV R² at the posterior site {posterior_site}"),
        "posterior_cv_r2": A.headline_entry(posterior_scores[posterior_site], n_valid, "R²",
                                            f"mean of the two factor-posterior CV R² at the posterior site {posterior_site}"),
        "value_cv_r2": A.headline_entry(value_scores[value_site], n_valid, "R²",
                                        f"offer-value CV R² (the pair pooled) at the value site {value_site}"),
        "posterior_read_layer": A.headline_entry(layer_of(posterior_site), n_valid, "layer",
                                                 f"posterior read site: earliest layer with mean factor CV R² >= {FORMATION_R2:g}"
                                                 + ("" if posterior_formed else " (not reached: best layer)")),
        "value_read_layer": A.headline_entry(layer_of(value_site), n_valid, "layer",
                                             f"value read site: earliest layer with offer-value CV R² >= {FORMATION_R2:g}"
                                             + ("" if value_formed else " (not reached: best layer)")),
    }
    if args.reads_only:
        headline["value_cv_r2_max_over_layers"] = A.headline_entry(
            max(value_scores.values()), n_valid, "R²", "largest offer-value CV R² over the post-block layers")
    else:
        rule = f"earliest layer whose steering slope reaches {WRITE_FRACTION:g} of the maximum over layers"
        for name, factor in (("A", "M"), ("B", "N")):
            headline[f"posterior_write_layer_eta{name}"] = A.headline_entry(
                layer_of(posterior_write_site[name]), n_scan, "layer", f"factor-{factor} posterior write site: {rule}")
        headline["value_write_layer"] = A.headline_entry(layer_of(value_write_site), n_scan, "layer", f"value write site: {rule}")
    results = {"headline": headline,
        "cv": cv, "sites": {"posterior": posterior_site, "value": value_site, "value_write": value_write_site,
                            "posterior_write": posterior_write_site, "display": display_site},
        "formation_r2": FORMATION_R2, "write_fraction": WRITE_FRACTION, "reads_only": bool(args.reads_only),
        "write_scan_slopes": scan, "posterior_write_scan_slopes": posterior_scan,
        "n_fit": n_fit, "n_scan": n_scan, "n_valid": n_valid}
    A.write_results(run_dir, results)
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={"fit": fit_seed, "folds": [fit_seed, ROLE_FOLDS]}, spec_hashes=study.spec_hashes,
                     start_time=t0, device=args.device, force_used=args.force, figures=[],
                     extra={"checkpoint": study.checkpoint, "artifact": str(npz.relative_to(A.ROOT)), "artifact_sha256": meta["artifact_sha256"]})
    print(json.dumps({s: {k: round(v, 4) for k, v in cv[s].items()} for s in sites}, indent=1))
    print(f"posterior site {posterior_site}  value site {value_site}  value write site {value_write_site}  "
          f"posterior write sites {posterior_write_site}\ndone: {run_dir}")


if __name__ == "__main__":
    main()
