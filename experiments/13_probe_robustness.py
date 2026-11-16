"""13 — probe-construction robustness: the paper's affine least-squares probe
refit under (a) ridge regularisation, (b) smaller training sets, and (c) every
readout position, fit and scored at the same readout. Same fit episodes, same
grouped episode folds and same targets (etaA, etaB, D) as 02, each target at
its read site from the fitted instruments (the factor posteriors at the
posterior site, the offer values at the value site). The probes are fit,
scored and discarded; nothing is written back to the study.

The training-size sweep draws its episode subsets from one generator per
target in the order of TRAIN_SIZES; the paper's table reports the sizes in
TABLE_SIZES.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, instruments as I, plotting as PL, probes as P   # noqa: E402
from jbo.instruments import GRADE_READOUT, MIN_OFFER_PROB                           # noqa: E402
from jbo.subject import load_study                                                  # noqa: E402
from jbo.world import DECIDE_POS, N_READOUTS                                        # noqa: E402

EXPERIMENT = "13_probe_robustness"
N_FIT, SMOKE_FIT = 4096, 512
N_FOLDS = 5
ROLE_FOLDS, ROLE_SUBSET = 1, 7           # folds: the draw 02 makes; subset: this script's episode subsampling
LAMBDAS = (0.0, 0.1, 1.0, 10.0)                       # ridge weight on standardised inputs, per-row loss; 0 = least squares
TRAIN_SIZES = (64, 128, 256, 512, 1024, 2048, None)   # episodes per training fold; None = all
TABLE_SIZES = (64, 256, 1024)
TARGETS = ("etaA", "etaB", "D")
TARGET_LABEL = {"etaA": "factor posterior M", "etaB": "factor posterior N", "D": "offer values"}


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> dict:
    """Affine ridge on standardised X [n, d], Y [n, k]: minimise mean ||Y - XW - c||² + lam ||W||²."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)                   # a constant input stays zero
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    n, d = Xs.shape
    if lam == 0.0:                                       # least squares; minimum-norm when rows < d (small training sets)
        W = np.linalg.lstsq(Xs, Y - ym, rcond=None)[0]
    else:
        W = np.linalg.solve(Xs.T @ Xs / n + lam * np.eye(d), Xs.T @ (Y - ym) / n)
    return {"W": W, "mu": mu, "sd": sd, "ym": ym}


def ridge_predict(fit: dict, X: np.ndarray) -> np.ndarray:
    return ((X - fit["mu"]) / fit["sd"]) @ fit["W"] + fit["ym"]


def cv_r2(X: np.ndarray, Y: np.ndarray, valid: np.ndarray, folds: list, lam: float = 0.0,
          n_train: int | None = None, rng: np.random.Generator | None = None) -> float:
    """Pooled out-of-fold R² of the ridge probe; X [N, d], Y [N, k], valid [N] at one readout.
    `n_train` subsamples each fold's training episodes with `rng` before the validity mask."""
    N = len(X)
    oof, scored = np.full(Y.shape, np.nan), np.zeros(N, bool)
    for test in folds:
        train = np.setdiff1d(np.arange(N), test)
        if n_train is not None and n_train < len(train):
            train = np.sort(rng.choice(train, size=n_train, replace=False))
        tr = train[valid[train]]
        fit = ridge_fit(X[tr], Y[tr], lam)
        te = test[valid[test]]
        oof[te] = ridge_predict(fit, X[te])
        scored[te] = True
    return float(P.decode_scores(Y[scored], oof[scored])["r2"])


def main() -> None:
    args = cli.parse("13_probe_robustness: ridge weight, training size and read position")
    study = load_study(args.study)
    world = study.build_world()
    assert world.n_factors == 2, "the factor-posterior targets need a two-factor world"
    model, _ = study.load_actor(args.device)
    seeds = study.seeds
    ins = I.load(study, eval_seeds=[seeds["eval"]])
    n_fit = SMOKE_FIT if args.smoke else N_FIT
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    read_sites = {"etaA": ins.posterior_site, "etaB": ins.posterior_site, "D": ins.value_site}
    sites = tuple(dict.fromkeys(read_sites.values()))

    tokens = world.sample(n_fit, seed=seeds["fit"])["tokens"]
    key = world.answer_key(tokens)
    acts = P.collect_site_activations(model, tokens, sites, DECIDE_POS, args.device)      # {site: [N, 10, d]}
    targets = {"etaA": key["etaA"][:, DECIDE_POS], "etaB": key["etaB"][:, DECIDE_POS], "D": key["D"]}   # [N, 10, k]
    valid = key["P"].min(-1) >= MIN_OFFER_PROB                                             # [N, 10]
    folds = P.grouped_folds(n_fit, N_FOLDS, np.random.default_rng([seeds["fit"], ROLE_FOLDS]))
    g = GRADE_READOUT
    res = {}
    for name in TARGETS:
        X, Y = acts[read_sites[name]].astype(np.float64), targets[name]                    # [N, 10, d], [N, 10, k]
        Xg, Yg, vg = X[:, g], Y[:, g], valid[:, g]
        rng = np.random.default_rng([seeds["fit"], ROLE_SUBSET])
        res[name] = {"ridge": {str(l): cv_r2(Xg, Yg, vg, folds, lam=l) for l in LAMBDAS},
                     "size": {str(n): cv_r2(Xg, Yg, vg, folds, n_train=n, rng=rng) for n in TRAIN_SIZES},
                     "position": [cv_r2(X[:, r], Y[:, r], valid[:, r], folds) for r in range(N_READOUTS)]}
        rr, ss, pp = res[name]["ridge"], res[name]["size"], res[name]["position"]
        print(f"  {PL.SITE_LABELS[read_sites[name]]:5s} {name:4s} ridge " + " ".join(f"{rr[str(l)]:.3f}" for l in LAMBDAS)
              + " | size " + " ".join(f"{ss[str(n)]:.3f}" for n in TRAIN_SIZES)
              + " | readout " + " ".join(f"{v:.3f}" for v in pp), flush=True)

    H = A.headline_entry
    nrows = int(valid[:, g].sum())
    head = {}
    for name in TARGETS:
        L, what = PL.SITE_LABELS[read_sites[name]], TARGET_LABEL[name]
        rr, ss, pp = res[name]["ridge"], res[name]["size"], res[name]["position"]
        head[f"{name}_ols_r2_{L}"] = H(rr["0.0"], nrows, "R²", f"{what}, CV R² at {L}, least squares (the paper's probe)")
        for l in LAMBDAS[1:]:
            head[f"{name}_ridge_{l:g}_r2_{L}"] = H(rr[str(l)], nrows, "R²", f"{what}, CV R² at {L}, ridge weight {l:g}")
        for n in TABLE_SIZES:
            head[f"{name}_r2_{n}_episodes_{L}"] = H(ss[str(n)], nrows, "R²", f"{what}, CV R² at {L}, {n} training episodes per fold")
        head[f"{name}_position_min_same_{L}"] = H(min(pp), nrows, "R²", f"{what} at {L}: lowest CV R² over the {N_READOUTS} readouts, "
                                                  "fit and scored at each readout")
    n_train_all = int(min(len(np.setdiff1d(np.arange(n_fit), f)) for f in folds))
    A.write_results(run_dir, {"headline": head, "results": res, "read_sites": read_sites,
                              "lambdas": list(LAMBDAS), "train_sizes": list(TRAIN_SIZES),
                              "n": {"fit": n_fit, "valid_rows_grade": nrows, "folds": N_FOLDS, "train_episodes_all": n_train_all}})
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv, seeds=seeds,
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                     figures=[], extra={"checkpoint_sha256": study.checkpoint["sha256"], "smoke": args.smoke})
    print("done:", run_dir)


if __name__ == "__main__":
    main()
