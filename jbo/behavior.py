"""Behavioural grading primitives shared by 01_behavior."""

from __future__ import annotations

import numpy as np
import torch

from jbo.train import policy_at_actions

PERM_BATCH = 2000


@torch.no_grad()
def model_policy(model, tokens: np.ndarray, device: str, response_base: int,
                 batch_size: int = 2048) -> np.ndarray:
    """[N, 91] tokens -> [N, 10, 2] float64 policy, via the trainer's readout.
    `response_base` comes from the world's own alphabet (world.vocab.response_base)."""
    batch_size = min(batch_size, getattr(model, "jbo_batch_size", batch_size))   # LM actors cap the batch
    out = []
    for i in range(0, len(tokens), batch_size):
        _, pi, _ = policy_at_actions(model(torch.from_numpy(tokens[i: i + batch_size]).to(device)),
                                     response_base)
        out.append(pi.detach().cpu().double().numpy())
    return np.concatenate(out, axis=0)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])


def permutation_null_r(pi_model: np.ndarray, pi_star: np.ndarray, n_perm: int,
                       rng: np.random.Generator) -> dict:
    """Null of Pearson r(P_model(A), P*(A)) under whole-episode permutations of
    the organism's rows. Moment identity: r = (mean(xy) - mx my)/(sx sy); the
    marginal moments are permutation-invariant, so only mean(xy) is recomputed.
    p = (1 + #{null >= observed}) / (n_perm + 1). Returns {observed, null [P], p}."""
    x = np.asarray(pi_model, float)[..., 0]                    # [N, 10]
    y = np.asarray(pi_star, float)[..., 0]
    N = len(x)
    mx, my, sx, sy = x.mean(), y.mean(), x.std(), y.std()

    def r_of(perms: np.ndarray) -> np.ndarray:                 # [P, N] -> [P]
        return ((x[perms] * y).mean(axis=(1, 2)) - mx * my) / (sx * sy)

    observed = float(r_of(np.arange(N)[None, :])[0])
    assert abs(observed - pearson_r(x, y)) < 1e-9
    null = np.empty(n_perm)
    done = 0
    while done < n_perm:
        b = min(PERM_BATCH, n_perm - done)
        null[done: done + b] = r_of(rng.permuted(np.tile(np.arange(N), (b, 1)), axis=1))
        done += b
    p = (1 + int((null >= observed).sum())) / (n_perm + 1)
    return {"observed": observed, "null": null, "p": p}
