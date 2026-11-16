"""01 — does the actor's choice follow the target policy softmax(beta Q)?
8,192 episodes on the post_training_eval seed, all ten readouts: Pearson r of
P(choose A) between the actor and the target policy, a 100k whole-episode
permutation null of r, and the mean entropy of the target policy itself (a
property of the world, used to match beta across reward variants).
`--snapshot K` scores the 00 run's snapshot after K updates (K=0: the network
at initialisation) instead of the pinned checkpoint (toy actors only), reports
r alone, and writes under 01_behavior_stepK so it pins beside the trained run.
No figures."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, behavior as B, cli, model as M   # noqa: E402
from jbo.subject import ROOT, load_study                    # noqa: E402

EXPERIMENT = "01_behavior"
N_EPISODES, N_PERM = 8192, 100_000
SMOKE_EPISODES, SMOKE_PERM = 512, 2000
ROLE_PERM = 1                                               # default_rng([eval_seed, ROLE_PERM])


def policy_entropy(pi: np.ndarray) -> np.ndarray:
    """[..., 2] choice probabilities -> [...] entropy in nats (0 log 0 = 0)."""
    assert pi.shape[-1] == 2 and np.allclose(pi.sum(-1), 1.0), "rows must be two-way distributions"
    return -(pi * np.log(np.clip(pi, 1e-300, None))).sum(-1)


def load_snapshot(study, world, step: int, device: str) -> tuple[torch.nn.Module, dict]:
    """The 00 run that produced the pinned checkpoint holds snapshots/; load the
    one after exactly `step` updates (sha-verified) into a fresh toy model."""
    assert study.actor["kind"] == "toy", "snapshots are a toy-actor feature"
    run00 = (ROOT / study.checkpoint["path"]).parent.parent
    idx = json.loads((run00 / "snapshots.json").read_text())
    match = [s for s in idx["snapshots"] if s["step"] == step]
    assert len(match) == 1, f"no snapshot at step {step}; schedule {idx['schedule']}"
    snap = match[0]
    path = run00 / "snapshots" / snap["file"]
    assert A.sha256_file(path) == snap["sha256"], f"snapshot sha mismatch: {path}"
    model = M.build_model(study.arch, world.d_vocab, seed=0, device="cpu")
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval(), {"step": step, "file": str(path.relative_to(ROOT)), "sha256": snap["sha256"]}


def main() -> None:
    args = cli.parse("01_behavior: choice against the target policy",
                     lambda p: p.add_argument("--snapshot", type=int, default=None,
                                              help="score the 00 run's snapshot after K updates (K=0: at initialisation)"))
    study = load_study(args.study)
    world = study.build_world()
    if args.snapshot is None:
        model, _ = study.load_actor(args.device)
        snapshot = None
    else:
        model, snapshot = load_snapshot(study, world, args.snapshot, args.device)
    n, n_perm = (SMOKE_EPISODES, SMOKE_PERM) if args.smoke else (N_EPISODES, N_PERM)
    seed = study.seeds["post_training_eval"]
    experiment = EXPERIMENT if snapshot is None else f"{EXPERIMENT}_step{args.snapshot}"   # its own pin
    run_dir = A.new_run_dir(args.study, experiment, label=args.label or ("smoke" if args.smoke else None),
                            force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    tokens = world.sample(n, seed=seed)["tokens"]                       # [n, 91]
    pi_star = world.answer_key(tokens)["policy"]                        # [n, 10, 2] target policy
    pi = B.model_policy(model, tokens, args.device, world.vocab.response_base)   # [n, 10, 2]
    assert pi.shape == pi_star.shape
    n_dec = int(pi[..., 0].size)
    r = B.pearson_r(pi[..., 0], pi_star[..., 0])
    headline = {"pearson_r": A.headline_entry(r, n_dec, "r", "Pearson r of P(choose A): actor vs target policy")}
    seeds = {"post_training_eval": seed}
    if snapshot is None:
        perm = B.permutation_null_r(pi, pi_star, n_perm, np.random.default_rng([seed, ROLE_PERM]))
        assert abs(perm["observed"] - r) < 1e-9
        target_entropy = float(policy_entropy(pi_star).mean())         # how sharp the world's target policy is
        headline["perm_p"] = A.headline_entry(perm["p"], n_perm, "p", "one-sided p of r under whole-episode permutations")
        headline["target_policy_entropy"] = A.headline_entry(
            target_entropy, n_dec, "nats", "mean entropy of the target policy softmax(beta Q) per decision (model-free)")
        seeds["perm"] = [seed, ROLE_PERM]
    results = {
        "headline": headline,
        "n_episodes": n, "n_decisions": n_dec,
        "checkpoint_sha256": study.checkpoint["sha256"],
        "snapshot": snapshot,                       # None: the pinned checkpoint was scored
    }
    A.write_results(run_dir, results)
    A.write_manifest(run_dir, experiment=experiment, study=args.study, argv=args.argv,
                     seeds=seeds, spec_hashes=study.spec_hashes, start_time=t0, device=args.device,
                     force_used=args.force, figures=[],
                     extra={"checkpoint": study.checkpoint, "snapshot": snapshot,
                            "n_episodes": n, "n_perm": n_perm if snapshot is None else 0})
    print(" ".join(f"{k}={v['value']:.6g}" for k, v in headline.items()) + f"\ndone: {run_dir}")


if __name__ == "__main__":
    main()
