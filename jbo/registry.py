"""Named JSON specs: hmms/<name>.json (parametric mess3, messn) and worlds/<name>.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jbo import ROOT, hmm

HMM_DIR = ROOT / "hmms"
WORLD_DIR = ROOT / "worlds"


def _load_spec(directory: Path, name: str) -> dict:
    path = directory / f"{name}.json"
    if not path.exists():
        known = sorted(p.stem for p in directory.glob("*.json"))
        raise FileNotFoundError(f"no spec {name!r} in {directory.name}/; known: {known}")
    return json.loads(path.read_text())


def load_hmm(name: str) -> np.ndarray:
    """[V, S, S] tensor for a named HMM (parametric 'mess3' or 'messn')."""
    spec = _load_spec(HMM_DIR, name)
    kind = spec.get("type")
    if kind == "mess3":
        T = hmm.mess3(float(spec["x"]), float(spec["a"]))
    elif kind == "messn":
        T = hmm.mess_n(int(spec["n"]), float(spec["x"]), float(spec["a"]))
    else:
        raise ValueError(f"HMM {name!r}: unknown spec type {kind!r} (expected 'mess3' or 'messn')")
    hmm.validate_hmm(T)
    return T


def load_world(name: str) -> dict:
    """World spec: {name, factors [2] (or [1]: one nine-state chain), beta, levels [3]}."""
    spec = _load_spec(WORLD_DIR, name)
    for key in ("factors", "beta", "levels"):
        if key not in spec:
            raise KeyError(f"world spec {name!r} is missing {key!r}")
    assert len(spec["factors"]) in (1, 2), "a world has two factors, or one nine-state chain"
    return spec
