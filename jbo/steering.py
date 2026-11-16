"""Residual-stream edits and the geometry around write planes."""

from __future__ import annotations

import numpy as np
import torch

from jbo import probes as P
from jbo.train import policy_at_actions

DONOR_TOP_FRACTION = 0.5
RADIUS_CAP, RADIUS_FRACTION, MIN_RADIUS = 0.25, 0.8, 0.05
N_DIRECTIONS = 12


@torch.no_grad()
def run_edited(model, tokens: np.ndarray, edits: list, read_sites: tuple, read_position: int,
               device: str, response_base: int | None = None, batch_size: int = 512) -> dict:
    """One forward pass with residual edits: `edits` = [(hook, position, delta [N, d])],
    each ADDED at its site/position. Returns dict(pi [N, 10, 2], resid {site: [N, d]})."""
    batch_size = min(batch_size, getattr(model, "jbo_batch_size", batch_size))   # LM actors cap the batch
    assert tokens.ndim == 2 and len(tokens) > 0 and batch_size > 0
    assert len(set(read_sites)) == len(read_sites), "duplicate read sites would duplicate activation rows"
    assert isinstance(read_position, (int, np.integer)) and 0 <= read_position < tokens.shape[1], \
        "read position outside token stream"
    for _, position, delta in edits:
        assert isinstance(position, (int, np.integer)) and 0 <= position < tokens.shape[1], \
            "write position outside token stream"
        assert delta.ndim == 2 and len(delta) == len(tokens), "each edit needs one residual vector per episode"
    want = set(read_sites)
    pis, out = [], {h: [] for h in read_sites}
    for i in range(0, len(tokens), batch_size):
        bt = torch.from_numpy(np.asarray(tokens[i: i + batch_size])).to(device)
        hooks = []
        for hook_name, position, delta in edits:
            d = torch.as_tensor(np.ascontiguousarray(delta[i: i + batch_size]), dtype=torch.float32, device=device)

            def fn(resid, hook, d=d, position=position):
                assert d.shape == resid[:, position, :].shape, "edit must match episode and residual axes without broadcasting"
                resid[:, position, :] = resid[:, position, :] + d
                return resid
            hooks.append((hook_name, fn))
        with model.hooks(fwd_hooks=hooks):
            logits, cache = model.run_with_cache(bt, names_filter=lambda n: n in want)
        _, pi, _ = policy_at_actions(logits, response_base)
        pis.append(pi.detach().cpu().double().numpy())
        for h in read_sites:
            out[h].append(cache[h][:, read_position, :].detach().cpu().float().numpy())
        del cache
    return {"pi": np.concatenate(pis, 0),
            "resid": {h: np.concatenate(v, 0).astype(np.float64) for h, v in out.items()}}


def policy_logit(pi: np.ndarray, clip: float = 1e-12) -> np.ndarray:
    pi = np.clip(np.asarray(pi, np.float64), clip, 1 - clip)
    return np.log(pi[..., 0]) - np.log(pi[..., 1])


def counterfactual_D(eta_new: np.ndarray, eta_other: np.ndarray, factor: int, bits: np.ndarray,
                     u: np.ndarray, reward: str = "conditional") -> dict:
    """Exact offer values when factor `factor` has posterior eta_new and the other
    factor is unchanged. eta_* [N, 3], bits [N, 2, 9], u [N, 9] -> dict(D [N, 2], P [N, 2]).
    `reward` is the world's rule: "conditional" gives m(X)/P(X); "mass" (the linear
    variant) gives m(X) itself."""
    assert reward in ("conditional", "mass"), reward
    joint = ((eta_new[:, :, None] * eta_other[:, None, :]) if factor == 0
             else (eta_other[:, :, None] * eta_new[:, None, :])).reshape(-1, 9)
    Pm = np.einsum("ns,nos->no", joint, bits)
    v = np.einsum("ns,nos,ns->no", joint, bits, u)
    return {"D": v / np.maximum(Pm, 1e-300) if reward == "conditional" else v, "P": Pm}


def draw_natural_donors(y_clean: np.ndarray, y_pool: np.ndarray, rng: np.random.Generator,
                        top_fraction: float = DONOR_TOP_FRACTION, chunk: int = 2048) -> dict:
    """One pool posterior per episode, uniform among the farthest `top_fraction`
    of the pool (tangent distance). dict(donor_idx [N], dy [N, 2], dose [N])."""
    n, m = len(y_clean), len(y_pool)
    k = int(np.ceil(m * top_fraction))
    donor_idx = np.empty(n, dtype=np.int64)
    for i in range(0, n, chunk):
        d = np.linalg.norm(y_clean[i: i + chunk, None, :] - y_pool[None, :, :], axis=-1)
        far = np.argpartition(d, m - k, axis=1)[:, m - k:]
        pick = rng.integers(0, k, size=len(far))
        donor_idx[i: i + chunk] = far[np.arange(len(far)), pick]
    dy = y_pool[donor_idx] - y_clean
    return {"donor_idx": donor_idx, "dy": dy, "dose": np.linalg.norm(dy, axis=-1)}


def orthogonal_directions(Q: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """[d, n] orthonormal directions orthogonal to the plane Q [d, k]."""
    d = Q.shape[0]
    assert n <= d - Q.shape[1]
    r = rng.normal(size=(d, n))
    r -= Q @ (Q.T @ r)
    basis, _ = np.linalg.qr(r)
    basis = basis[:, :n]
    assert np.allclose(Q.T @ basis, 0, atol=1e-10)
    return basis


def isometric_plane_copy(delta: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Copy in-plane commands delta [N, d] (in span(source)) into span(target), norm-preserving."""
    coords = delta @ source
    assert np.allclose(delta, coords @ source.T, atol=1e-8), "delta must lie in the source plane"
    copied = coords @ target.T
    assert np.allclose(np.linalg.norm(copied, axis=1), np.linalg.norm(delta, axis=1), atol=1e-10)
    return copied


def split_on_plane(dh: np.ndarray, Q: np.ndarray) -> dict:
    """Orthogonal split of dh [N, d] into in-plane (dh Q)Qᵀ and complement."""
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)
    in_plane = (dh @ Q) @ Q.T
    return {"in_plane": in_plane, "complement": dh - in_plane}


def matched_random_part(dh: np.ndarray, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """dh projected on a random rank-2 plane orthogonal to Q, rescaled per row
    to the norm of dh's in-plane part."""
    d = dh.shape[1]
    R = P.random_subspace(d, 2, rng)
    R = R - Q @ (Q.T @ R)
    R, _ = np.linalg.qr(R)
    R = R[:, :2]
    part = (dh @ R) @ R.T
    want = np.linalg.norm(split_on_plane(dh, Q)["in_plane"], axis=-1)
    have = np.linalg.norm(part, axis=-1)
    return part * np.divide(want, have, out=np.zeros_like(want), where=have > 0)[:, None]


def safe_radii(D: np.ndarray) -> np.ndarray:
    """Per-episode value-ring radius: 80 % of the nearest square-edge margin, capped at 0.25."""
    margin = np.minimum(1.0 - np.abs(np.asarray(D, np.float64)), 1.0).min(axis=1)
    return np.minimum(RADIUS_CAP, RADIUS_FRACTION * np.maximum(margin, 0.0))


def ring_directions(n: int = N_DIRECTIONS) -> tuple[np.ndarray, np.ndarray]:
    """(angles [n] degrees, unit directions [n, 2]) at equal increments."""
    angles = np.arange(0.0, 360.0, 360.0 / n)
    rad = np.deg2rad(angles)
    return angles, np.column_stack([np.cos(rad), np.sin(rad)])
