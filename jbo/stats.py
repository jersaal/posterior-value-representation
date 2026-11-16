"""Slopes with episode-bootstrap standard errors."""

from __future__ import annotations

import numpy as np


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, np.float64), np.asarray(yhat, np.float64)
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss if ss > 0 else float("nan")


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[1]), float(coef[0])


def slope(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, n_boot: int = 1000,
          seed: int = 0) -> dict:
    """OLS slope/intercept/R² of y on x with an episode-bootstrap SE: resample
    whole groups (default: each row is its own episode). dict(slope, se, intercept, r2, n, n_groups)."""
    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    g = np.arange(len(x)) if groups is None else np.asarray(groups).ravel()
    assert len(x) == len(y) == len(g)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    if len(x) < 8 or np.ptp(x) == 0 or len(uniq) < 2:
        return {"slope": float("nan"), "se": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "n": int(len(x)), "n_groups": int(len(uniq))}
    b, a = _ols(x, y)
    rng = np.random.default_rng(seed)
    rows_by_group = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([rows_by_group[k] for k in pick])
        boots[i] = _ols(x[rows], y[rows])[0]
    se = float(boots.std(ddof=1)) if n_boot >= 2 else float("nan")
    return {"slope": b, "se": se, "intercept": a, "r2": r2(y, a + b * x),
            "n": int(len(x)), "n_groups": int(len(uniq))}
