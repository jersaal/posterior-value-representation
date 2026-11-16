"""Immutable run outputs: results/<world>/<seed>/<experiment>/<run-id>/
{manifest.json, results.json, figures/, ...}."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from jbo import ROOT

RESULTS_DIR = ROOT / "results"
TRACKED_PACKAGES = ("torch", "transformer-lens", "transformers", "numpy", "scipy", "matplotlib", "scikit-learn")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions() -> dict:
    v = {}
    for p in TRACKED_PACKAGES:
        try:
            v[p] = pkg_version(p)
        except PackageNotFoundError:                    # transformers is only needed for the pretrained actor
            v[p] = None
    v["python"] = platform.python_version()
    return v


def new_run_dir(study: str, experiment: str, label: str | None = None,
                force: bool = False) -> Path:
    """results/<study>/<experiment>/<label->YYYYMMDD-HHMMSS/; refuses to
    overwrite unless `force` (caller records it in the manifest). Runs
    labelled `smoke*` are throwaway and never pinned."""
    run_id = (f"{label}-" if label else "") + time.strftime("%Y%m%d-%H%M%S")
    d = RESULTS_DIR / study / experiment / run_id
    if d.exists() and not force:
        raise FileExistsError(f"{d} exists — results are immutable; pass --force to overwrite")
    d.mkdir(parents=True, exist_ok=force)
    (d / "figures").mkdir(exist_ok=True)
    return d


def write_manifest(run_dir: Path, *, experiment: str, study: str, argv: list[str],
                   seeds: dict, spec_hashes: dict, start_time: str, device: str,
                   force_used: bool, figures: list[Path], extra: dict | None = None) -> Path:
    """manifest.json: spec hashes, seeds, argv, runtime, device, timings,
    figure paths + hashes, plus experiment-specific `extra` (no key collisions)."""
    manifest = {
        "experiment": experiment, "study": study,
        "spec_hashes": spec_hashes, "argv": argv, "seeds": seeds,
        "runtime_versions": runtime_versions(), "device": device,
        "start_time": start_time, "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "force_used": force_used,
        "figures": [{"path": str(Path(f).relative_to(ROOT)), "sha256": sha256_file(f)}
                    for f in figures],
    }
    for k, v in (extra or {}).items():
        assert k not in manifest, f"extra field {k!r} collides with a required field"
        manifest[k] = v
    p = Path(run_dir) / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2))
    return p


def headline_entry(value: float, n: int, unit: str, description: str) -> dict:
    return {"value": value, "n": int(n), "unit": unit, "description": description}


def write_results(run_dir: Path, results: dict) -> Path:
    """results.json; requires a `headline` block of {name: headline_entry}."""
    assert "headline" in results and isinstance(results["headline"], dict), \
        "results.json needs a `headline` block"
    for k, h in results["headline"].items():
        assert set(h) >= {"value", "n", "unit", "description"}, f"headline {k!r} malformed"
    p = Path(run_dir) / "results.json"
    p.write_text(json.dumps(results, indent=2))
    return p
