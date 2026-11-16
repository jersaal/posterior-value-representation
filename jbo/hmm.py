"""HMM math. Convention: T[v, i, j] = P(emit v AND move i -> j | state i);
posteriors are row vectors, eta' ∝ eta @ T[v]. Float64, CPU."""

from __future__ import annotations

import numpy as np


def validate_hmm(T: np.ndarray, atol: float = 1e-9) -> None:
    """Assert T:[V,S,S] is non-negative and stochastic over (v, j) per state."""
    assert T.ndim == 3, f"T must be [V,S,S], got {T.shape}"
    assert T.shape[1] == T.shape[2], f"state dims must be square, got {T.shape}"
    assert np.all(T >= -1e-12), "T must be non-negative"
    mass = T.sum(axis=(0, 2))                                   # [S]
    assert np.allclose(mass, 1.0, atol=atol), f"T not stochastic over (v,j): {mass}"


def mess3(x: float, a: float) -> np.ndarray:
    """Mess3 (Marzen & Crutchfield 2017) as [3, 3, 3]: stay prob y = 1 - 2x,
    hop x to each other state; emitted token names the destination with
    prob a, each other token with prob b = (1 - a)/2. Entry-identical to
    Shai et al. 2026's `mess3(x, a)`."""
    b = (1.0 - a) / 2.0
    y = 1.0 - 2.0 * x
    ay, ax, by, bx = a * y, a * x, b * y, b * x
    T = np.array([
        [[ay, bx, bx], [ax, by, bx], [ax, bx, by]],
        [[by, ax, bx], [bx, ay, bx], [bx, ax, by]],
        [[by, bx, ax], [bx, by, ax], [bx, bx, ay]],
    ], dtype=np.float64)                                         # [V=3, S=3, S=3]
    validate_hmm(T)
    return T


def mess_n(n: int, x: float, a: float) -> np.ndarray:
    """The Mess3 construction on n states as [n, n, n]: stay with prob
    1 - (n - 1) x, hop x to each other state; the emitted token names the
    destination with prob a, each other token with prob (1 - a) / (n - 1).
    mess_n(3, x, a) is entry-identical to mess3(x, a)."""
    assert n >= 2 and 0.0 < x < 1.0 / (n - 1) and 0.0 < a < 1.0, (n, x, a)
    stay, b = 1.0 - (n - 1) * x, (1.0 - a) / (n - 1)
    move = np.full((n, n), x) + np.eye(n) * (stay - x)                   # [S, S] P(j | i)
    emit = np.full((n, n), b) + np.eye(n) * (a - b)                      # [V, S] P(v | destination j)
    T = move[None, :, :] * emit[:, None, :]                              # [V, S, S]
    validate_hmm(T)
    return T


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """pi with pi @ M = pi for M = sum_v T[v]. Shape [S]."""
    M = T.sum(axis=0)
    evals, evecs = np.linalg.eig(M.T)
    idx = int(np.argmin(np.abs(evals - 1.0)))
    assert abs(evals[idx] - 1.0) < 1e-8, f"no eigenvalue 1 found: {evals}"
    pi = np.real(evecs[:, idx])
    pi = np.clip(pi / pi.sum(), 0.0, None)
    return pi / pi.sum()


def emission_probs(T: np.ndarray) -> np.ndarray:
    """E[v, i] = P(emit v | state i). Shape [V, S]."""
    return T.sum(axis=2)


def nkron(Ts: list[np.ndarray]) -> np.ndarray:
    """Iterated Kronecker product of independent factors -> one joint HMM.
    Token v = a * VB + b, state s = iA * SB + iB (factor 0 most significant)."""
    J = Ts[0]
    for T in Ts[1:]:
        VA, SA, _ = J.shape
        VB, SB, _ = T.shape
        out = np.zeros((VA * VB, SA * SB, SA * SB))
        for a in range(VA):
            for b in range(VB):
                out[a * VB + b] = np.kron(J[a], T[b])
        J = out
    validate_hmm(J)
    return J


def factor_marginals(posterior: np.ndarray, shapes: tuple[int, ...]) -> list[np.ndarray]:
    """Joint posterior [..., prod(shapes)] -> per-factor marginals [..., S_i]."""
    lead = posterior.shape[:-1]
    nl = len(lead)
    B = np.asarray(posterior).reshape(*lead, *shapes)
    out = []
    for i in range(len(shapes)):
        axes = tuple(nl + j for j in range(len(shapes)) if j != i)
        out.append(B.sum(axis=axes))
    return out

