"""Evaluate-only loader for a study's instrument artifact (fit by 02).
Contains no fitting code."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from jbo import algebra
from jbo.artifacts import sha256_file
from jbo.probes import affine_predict, from_tangent, simplex_tangent_basis, to_tangent
from jbo.world import ACTION_POSITIONS

TARGETS = ("etaA0", "etaA1", "etaA2", "etaB0", "etaB1", "etaB2", "D_A", "D_B")
GRADE_READOUT = 5
GRADE_POSITION = ACTION_POSITIONS[GRADE_READOUT]
MIN_OFFER_PROB = 0.02      # valid rows need min(P_A, P_B) >= this at the grading readout
                           # (D = v/P is ill-conditioned at small P); the ONE definition —
                           # every experiment imports it from here
N_FIT, N_PLANE = 4096, 2048              # 02's fit and plane populations for a narrow residual (the toy)
SMOKE_FIT, SMOKE_PLANE = 512, 256
ROWS_PER_DIM_FIT, ROWS_PER_DIM_PLANE = 8, 4   # least-squares reads/planes need rows >> d_model: the populations
                                              # grow with the residual width (toy: 64 -> the defaults hold)


def fit_sizes(d_model: int, smoke: bool) -> tuple[int, int]:
    """(n_fit, n_plane): the defaults, or ROWS_PER_DIM_* x d_model when the
    residual is wide enough that the defaults would under-determine the probes.
    The ONE rule: 02 fits with it and the throwaway probes of 09 size with it."""
    n_fit = max(SMOKE_FIT if smoke else N_FIT, ROWS_PER_DIM_FIT * d_model)
    n_plane = max(SMOKE_PLANE if smoke else N_PLANE, ROWS_PER_DIM_PLANE * d_model)
    assert n_plane <= n_fit
    return n_fit, n_plane


@dataclass(frozen=True)
class Plane:
    """A rank-2 write plane: Q [d, 2] orthonormal, probe [d+1, 2], M = Qᵀ W [2, 2]."""
    Q: np.ndarray
    probe: np.ndarray
    M: np.ndarray

    def aim(self, dy: np.ndarray) -> np.ndarray:
        """Commanded probe-read displacement dy [N, 2] -> residual delta [N, d]."""
        return (np.asarray(dy, np.float64) @ np.linalg.inv(self.M)) @ self.Q.T

    def read(self, resid: np.ndarray) -> np.ndarray:
        return affine_predict(resid, self.probe)


@dataclass(frozen=True)
class Instruments:
    sites: tuple[str, ...]           # embed + post-block sites, ladder order
    posterior_site: str                 # read: earliest post-block site with posterior CV R² >= FORMATION_R2
    value_site: str                 # read: same rule on (D_A, D_B)
    value_write_site: str | None    # write: earliest post-block site at WRITE_FRACTION of the value-sweep slope maximum
    posterior_write_site: dict | None   # write: {"A": site, "B": site}, same rule on the posterior-steer slope ladder
    display_site: str                # final post-block residual
    cv: dict                         # {site: {target: r2}}
    betas: dict                      # {site: [d+1, 8]}
    planes: dict                     # {"posterior_A", "posterior_B", "value": Plane}; empty for a reads-only fit
                                     # (02 --reads-only: an actor whose choice is untrained has no write sites)
    posterior_unit: dict                # {"A": float, "B": float}
    fit_seed: int
    meta: dict
    basis: np.ndarray = field(default_factory=lambda: simplex_tangent_basis(3))

    @property
    def reads_only(self) -> bool:
        return not self.planes

    def decode(self, site: str, resid: np.ndarray) -> np.ndarray:
        """[N, d] -> [N, 8] decoded (etaA, etaB, D_A, D_B)."""
        assert site in self.betas, f"{site!r} is not an instrument site"
        return affine_predict(resid, self.betas[site])

    def posterior_A(self, eight: np.ndarray) -> np.ndarray:
        return np.asarray(eight, np.float64)[:, 0:3]

    def posterior_B(self, eight: np.ndarray) -> np.ndarray:
        return np.asarray(eight, np.float64)[:, 3:6]

    def D(self, eight: np.ndarray) -> np.ndarray:
        return np.asarray(eight, np.float64)[:, 6:8]

    def gap(self, eight: np.ndarray) -> np.ndarray:
        e = np.asarray(eight, np.float64)
        return e[:, 6] - e[:, 7]

    def compose_P(self, eight: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Decoded P(X) [N] = outer(etaA, etaB) summed over mask bits (unclipped)."""
        joint = np.einsum("na,nb->nab", self.posterior_A(eight), self.posterior_B(eight)).reshape(-1, 9)
        return (joint * algebra.mask_bits(masks)).sum(-1)

    def to_tangent(self, p: np.ndarray) -> np.ndarray:
        return to_tangent(p, self.basis)

    def from_tangent(self, y: np.ndarray) -> np.ndarray:
        return from_tangent(y, self.basis)


def artifact_paths(study) -> tuple[Path, Path]:
    d = study.dir / "instruments"
    return d / "instruments.npz", d / "instruments.json"


def load(study, eval_seeds: Sequence[int]) -> Instruments:
    """Load and verify: artifact sha, checkpoint sha, eval ∩ fit = ∅."""
    npz, js = artifact_paths(study)
    assert js.exists() and npz.exists(), f"no instruments for {study.name} — run 02_fit_instruments"
    meta = json.loads(js.read_text())
    assert (meta["grade_readout"], meta["grade_position"]) == (GRADE_READOUT, GRADE_POSITION), \
        "instrument grading position differs from the evaluation readout"
    assert sha256_file(npz) == meta["artifact_sha256"], "instrument artifact drifted"
    assert meta["checkpoint_sha256"] == study.checkpoint["sha256"], \
        "instruments were fit on a different checkpoint — refit with 02"
    declared = {int(s) for s in eval_seeds}
    assert not declared & {int(meta["fit_seed"])}, "evaluation seed equals the fit seed"
    plane_keys = tuple(meta.get("planes", ("posterior_A", "posterior_B", "value")))   # older artifacts: all three
    assert plane_keys == () or set(plane_keys) == {"posterior_A", "posterior_B", "value"}, plane_keys
    with np.load(npz) as z:
        betas = {s: z[f"beta__{s}"] for s in meta["sites"]}
        planes = {k: Plane(z[f"{k}__Q"], z[f"{k}__probe"], z[f"{k}__M"]) for k in plane_keys}
    bws = meta["posterior_write_site"]
    return Instruments(tuple(meta["sites"]), meta["posterior_site"], meta["value_site"],
                       meta["value_write_site"], None if bws is None else dict(bws), meta["display_site"],
                       meta["cv"], betas, planes, meta["posterior_unit"], int(meta["fit_seed"]), meta)
