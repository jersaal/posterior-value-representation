"""Linear probes: affine least squares, grouped CV, cross-covariance
subspaces, activation collection, simplex tangent coordinates."""

from __future__ import annotations

import numpy as np
import torch


def simplex_tangent_basis(n_states: int = 3) -> np.ndarray:
    """Orthonormal Helmert basis [S, S-1] of the simplex tangent plane."""
    U = np.zeros((n_states, n_states - 1), dtype=np.float64)
    for k in range(1, n_states):
        U[:k, k - 1] = 1.0
        U[k, k - 1] = -k
        U[:, k - 1] /= np.sqrt(k * (k + 1))
    assert np.allclose(U.T @ U, np.eye(n_states - 1)) and np.allclose(U.sum(0), 0)
    return U


def to_tangent(p: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Posterior [..., S] -> tangent coords [..., S-1]."""
    return (np.asarray(p, np.float64) - 1.0 / U.shape[0]) @ U


def from_tangent(y: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Tangent coords [..., S-1] -> posterior [..., S] (unclipped)."""
    return np.asarray(y, np.float64) @ U.T + 1.0 / U.shape[0]


def affine_fit(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least squares Y ~ [1, X] @ beta. X [n, d], Y [n, k] -> beta [d+1, k]."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Y, rcond=None)
    return beta


def affine_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    return np.column_stack([np.ones(len(X)), X]) @ beta


def decode_scores(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """dict(r2, rmse): multi-output R² = 1 - SS_res / SS_tot, pooled over columns."""
    Y_true, Y_pred = np.asarray(Y_true, np.float64), np.asarray(Y_pred, np.float64)
    assert Y_true.ndim == 2 and Y_true.shape == Y_pred.shape, \
        f"target/prediction row or column mismatch: {Y_true.shape} vs {Y_pred.shape}"
    sq = ((Y_pred - Y_true) ** 2).sum(-1)
    ss_tot = float(((Y_true - Y_true.mean(axis=0)) ** 2).sum())
    return {"r2": 1.0 - float(sq.sum()) / ss_tot if ss_tot > 0 else float("nan"),
            "rmse": float(np.sqrt(sq.mean()))}


def grouped_folds(n_episodes: int, n_splits: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Test-episode index sets per fold (whole episodes held out together)."""
    assert n_episodes >= n_splits
    return [np.sort(f) for f in np.array_split(rng.permutation(n_episodes), n_splits)]


def cv_decode(X: np.ndarray, Y: np.ndarray, folds: list[np.ndarray],
              valid: np.ndarray | None = None) -> dict:
    """Grouped CV affine decode. X [E, P, d], Y [E, P, k], valid [E, P] bool or
    None; probes fit on training episodes' valid rows, scored pooled over the
    held-out episodes' valid rows. Returns dict(r2, rmse, n_rows, oof [E, P, k])."""
    assert X.ndim == Y.ndim == 3, "CV inputs must retain episode and position axes"
    E, P, d = X.shape
    k = Y.shape[-1]
    assert Y.shape[:2] == (E, P), f"target {Y.shape} vs X {X.shape}"
    valid = np.ones((E, P), bool) if valid is None else np.asarray(valid, bool)
    assert valid.shape == (E, P)
    assert len(folds) >= 2, "CV needs at least two episode folds"
    for test in folds:
        assert test.ndim == 1 and np.issubdtype(test.dtype, np.integer), "folds must contain episode indices"
        assert len(test) and ((test >= 0) & (test < E)).all(), "empty or out-of-range episode fold"
    assert np.array_equal(np.sort(np.concatenate(folds)), np.arange(E)), \
        "CV folds must partition the episodes exactly once"
    oof = np.full((E, P, k), np.nan)
    all_ep = np.arange(E)
    for test in folds:
        train = np.setdiff1d(all_ep, test)
        m = valid[train].reshape(-1)
        X_tr = X[train].reshape(-1, d)[m]
        assert len(X_tr) > d + 1, "too few valid training rows"
        beta = affine_fit(X_tr, Y[train].reshape(-1, k)[m])
        oof[test] = affine_predict(X[test].reshape(-1, d), beta).reshape(len(test), P, k)
    m = valid.reshape(-1)
    sc = decode_scores(Y.reshape(-1, k)[m], oof.reshape(-1, k)[m])
    return {"r2": sc["r2"], "rmse": sc["rmse"], "n_rows": int(m.sum()), "oof": oof}


def encoding_subspace(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Orthonormal [d, k] basis of the column space of Cov(X, Y)."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    cov = (X - X.mean(0)).T @ (Y - Y.mean(0)) / len(X)
    Q, _ = np.linalg.qr(cov)
    return Q[:, :Y.shape[1]]


def random_subspace(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return Q[:, :k]


def post_block_sites(n_layers: int) -> tuple[str, ...]:
    return tuple(f"blocks.{l}.hook_resid_post" for l in range(n_layers))


EMBED_SITE = "blocks.0.hook_resid_pre"


@torch.no_grad()
def collect_site_activations(model, tokens: np.ndarray, hooks: tuple[str, ...],
                             keep_pos: np.ndarray, device: str, batch_size: int = 512,
                             fwd_hooks: list | None = None) -> dict:
    """{hook: [N, P, d] float32} residual activations at `keep_pos`."""
    batch_size = min(batch_size, getattr(model, "jbo_batch_size", batch_size))   # LM actors cap the batch
    assert tokens.ndim == 2 and len(tokens) > 0 and batch_size > 0
    assert len(set(hooks)) == len(hooks), "duplicate read sites would duplicate activation rows"
    want = set(hooks)
    out = {h: [] for h in hooks}
    positions = np.atleast_1d(keep_pos)
    assert positions.ndim == 1 and np.issubdtype(positions.dtype, np.integer), "read positions must be integer indices"
    assert ((positions >= 0) & (positions < tokens.shape[1])).all(), "read position outside token stream"
    pos = positions.tolist()
    for i in range(0, len(tokens), batch_size):
        batch = torch.from_numpy(np.asarray(tokens[i: i + batch_size])).to(device)
        if fwd_hooks:
            with model.hooks(fwd_hooks=fwd_hooks):
                _, cache = model.run_with_cache(batch, names_filter=lambda n: n in want)
        else:
            _, cache = model.run_with_cache(batch, names_filter=lambda n: n in want)
        for h in hooks:
            out[h].append(cache[h][:, pos, :].detach().cpu().float().numpy())
        del cache
    acts = {h: np.concatenate(v, axis=0) for h, v in out.items()}
    for h, a in acts.items():
        assert a.shape[:2] == (len(tokens), len(pos)), f"activation shape drifted at {h}"
    return acts
