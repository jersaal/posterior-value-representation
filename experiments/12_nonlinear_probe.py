"""12 — nonlinear probes beside the linear probe: at every residual-stream
site of the graded DECIDE token, grouped 5-fold CV R² of the factor posteriors
(etaA, etaB) and the offer-value pair (D) under (a) the paper's affine
least-squares probe and (b) multilayer-perceptron probes with one and two
hidden layers, fit on the same episodes and the same episode folds. If the MLP
gains little over the affine probe, the linear probe is not under-reading the
representation. The probes are fit, scored and discarded; the fitted
instruments are loaded only to name the two read sites.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, instruments as I, plotting as PL, probes as P   # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB           # noqa: E402
from jbo.subject import load_study                                                  # noqa: E402

EXPERIMENT = "12_nonlinear_probe"
N_FIT, SMOKE_FIT = 4096, 512
N_FOLDS = 5
ROLE_FOLDS = 1                   # default_rng([fit_seed, ROLE_FOLDS]) — the fold draw 02 makes
HIDDEN = 256
MAX_STEPS, PATIENCE, VAL_FRACTION = 3000, 150, 0.1
LR, WEIGHT_DECAY = 1e-3, 1e-5
TARGETS = ("etaA", "etaB", "D")
PROBES = ("affine", "mlp1", "mlp2")
MLPS = ("mlp1", "mlp2")


class MLP(torch.nn.Module):
    def __init__(self, d: int, k: int, hidden: int, depth: int):
        super().__init__()
        layers, w = [], d
        for _ in range(depth):
            layers += [torch.nn.Linear(w, hidden), torch.nn.GELU()]
            w = hidden
        layers.append(torch.nn.Linear(w, k))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, depth: int, seed: int,
            device: str) -> np.ndarray:
    """Standardise X on the training rows, hold out VAL_FRACTION of them for
    early stopping, full-batch Adam; returns predictions on X_te [n_te, k]."""
    assert X_tr.ndim == 2 and Y_tr.ndim == 2 and len(X_tr) == len(Y_tr)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    n_val = max(int(VAL_FRACTION * len(X_tr)), 16)
    val, tr = perm[:n_val], perm[n_val:]
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)   # noqa: E731
    Xt, Yt, Xv, Yv = t((X_tr[tr] - mu) / sd), t(Y_tr[tr]), t((X_tr[val] - mu) / sd), t(Y_tr[val])
    torch.manual_seed(seed)
    net = MLP(X_tr.shape[1], Y_tr.shape[1], HIDDEN, depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_state, since = float("inf"), None, 0
    for step in range(MAX_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((net(Xt) - Yt) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            v = float(torch.mean((net(Xv) - Yv) ** 2))
        if v < best - 1e-7:
            best, since = v, 0
            best_state = {k: w.detach().clone() for k, w in net.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(t((X_te - mu) / sd)).cpu().numpy().astype(np.float64)


def cv_scores(X: np.ndarray, Y: np.ndarray, valid: np.ndarray, folds: list[np.ndarray], seed: int,
              device: str) -> dict[str, float]:
    """Pooled out-of-fold R² of the three probes; X [N, d], Y [N, k], valid [N]."""
    N, d = X.shape
    oof = {p: np.full(Y.shape, np.nan) for p in PROBES}
    all_ep = np.arange(N)
    for f, test in enumerate(folds):
        train = np.setdiff1d(all_ep, test)
        tr, te = train[valid[train]], test[valid[test]]
        beta = P.affine_fit(X[tr], Y[tr])
        oof["affine"][te] = P.affine_predict(X[te], beta)
        oof["mlp1"][te] = fit_mlp(X[tr], Y[tr], X[te], 1, seed * 10 + f, device)
        oof["mlp2"][te] = fit_mlp(X[tr], Y[tr], X[te], 2, seed * 10 + f, device)
    m = valid
    return {p: float(P.decode_scores(Y[m], oof[p][m])["r2"]) for p in PROBES}


def mlp_gain(score: dict[str, float]) -> float:
    """The better MLP's CV R² minus the affine probe's, for one target at one site."""
    return max(score[p] for p in MLPS) - score["affine"]


def main() -> None:
    args = cli.parse("12_nonlinear_probe: MLP probes beside the affine probe at every site")
    study = load_study(args.study)
    world = study.build_world()
    assert world.n_factors == 2, "the factor-posterior targets need a two-factor world"
    model, _ = study.load_actor(args.device)
    n_fit = SMOKE_FIT if args.smoke else N_FIT
    seeds = study.seeds
    ins = I.load(study, eval_seeds=[seeds["eval"]])
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    sites = (P.EMBED_SITE,) + P.post_block_sites(model.cfg.n_layers)
    read_sites = {"etaA": ins.posterior_site, "etaB": ins.posterior_site, "D": ins.value_site}
    first = P.post_block_sites(model.cfg.n_layers)[0]                    # layer 0: the offer values are not yet formed there
    assert set(read_sites.values()) <= set(sites)

    tokens = world.sample(n_fit, seed=seeds["fit"])["tokens"]
    key = world.answer_key(tokens)
    pos = GRADE_POSITION
    acts = P.collect_site_activations(model, tokens, sites, np.array([pos]), args.device)   # {site: [N, 1, d]}
    targets = {"etaA": key["etaA"][:, pos], "etaB": key["etaB"][:, pos], "D": key["D"][:, GRADE_READOUT]}
    valid = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    folds = P.grouped_folds(n_fit, N_FOLDS, np.random.default_rng([seeds["fit"], ROLE_FOLDS]))
    fit_device = "cpu"                                   # small full-batch fits; deterministic across machines
    scores: dict[str, dict] = {}
    for site in sites:
        X = acts[site][:, 0, :].astype(np.float64)
        scores[site] = {}
        for name, Y in targets.items():
            scores[site][name] = cv_scores(X, Y, valid, folds, seed=seeds["fit"] % 1000, device=fit_device)
            print(f"  {PL.SITE_LABELS[site]:5s} {name:4s} " + "  ".join(f"{p} {v:.3f}" for p, v in scores[site][name].items()),
                  flush=True)

    H = A.headline_entry
    nrows = int(valid.sum())
    gain_read = max(mlp_gain(scores[read_sites[name]][name]) for name in TARGETS)
    gain_any = max(mlp_gain(scores[s][name]) for s in sites for name in TARGETS)
    value_first = max(scores[first]["D"][p] for p in PROBES)
    head = {
        "mlp_gain_read_sites_max": H(gain_read, nrows, "R²", "largest MLP minus affine CV R² (either depth) over the factor posteriors at "
                                     f"{PL.SITE_LABELS[ins.posterior_site]} and the offer values at {PL.SITE_LABELS[ins.value_site]}"),
        "mlp_gain_any_site_max": H(gain_any, nrows, "R²", "largest MLP minus affine CV R² (either depth) over every target and every site"),
        f"D_max_r2_{PL.SITE_LABELS[first]}": H(value_first, nrows, "R²", f"largest CV R² of the offer values at {PL.SITE_LABELS[first]} over the "
                                               "affine and the two MLP probes"),
    }
    A.write_results(run_dir, {"headline": head, "scores": scores, "sites": list(sites), "read_sites": read_sites,
                              "n": {"fit": n_fit, "valid_rows": nrows, "folds": N_FOLDS},
                              "mlp": {"hidden": HIDDEN, "activation": "GELU", "max_steps": MAX_STEPS, "patience": PATIENCE,
                                      "val_fraction": VAL_FRACTION, "lr": LR, "weight_decay": WEIGHT_DECAY}})
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv, seeds=seeds,
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                     figures=[], extra={"checkpoint_sha256": study.checkpoint["sha256"], "smoke": args.smoke})
    print("done:", run_dir)


if __name__ == "__main__":
    main()
