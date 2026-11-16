"""00 — train the actor (a fresh seeded toy actor, a toy actor initialised
from another study's pinned checkpoint via `init_from`, or a pretrained one
restricted to the world's symbols; study.json says which). The training
config's `loss` picks REINFORCE on the sampled reward (headline: the final
logged reward and policy entropy) or next-token prediction (the prediction
stage of the two-stage model; headline: the excess next-clue cross-entropy
over the exact Bayes predictor, measured after training on fresh episodes).
`--smoke`: 31 steps on the smoke seeds. `--snapshots`: capture the 23-point
weight schedule that 04 reads (toy actors only). Prints the pin command on a
full run."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import ROOT, artifacts as A, cli, hmm, model as M, train as T   # noqa: E402
from jbo.subject import load_study                                         # noqa: E402
from jbo.world import ACTION_POSITIONS, L                                  # noqa: E402

EXPERIMENT = "00_train"
SMOKE_STEPS = 31
SMOKE_BATCH = 64
N_NTP_EVAL = 4096


def save_checkpoint(study, world, out: dict, run_dir: Path, seeds: dict, num_steps: int,
                    symbol_map: dict | None, init_from: dict | None) -> Path:
    """checkpoint/{weights.pt, config.json, metrics.json}; exact-reload check
    through the study's own actor builder (architecture only, no init)."""
    ck = run_dir / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    model = out["model"]
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ck / "weights.pt")
    config = {"study": study.name, "world": world.name, "factors": list(world.factors),
              "d_vocab": world.d_vocab, "n_ctx": L, "action_positions": list(ACTION_POSITIONS),
              "arch": study.arch, "training": study.training, "actor": study.actor, "symbol_map": symbol_map,
              "init_from": init_from,
              "model_seed": seeds["model_init"], "data_stream_base": seeds["data_stream_base"],
              "num_steps": num_steps, "world_spec_sha256": study.spec_hashes["world"]}
    (ck / "config.json").write_text(json.dumps(config, indent=2))
    (ck / "metrics.json").write_text(json.dumps(
        {"log": out["log"], "wall_s": out["wall_s"], "final_lr": out["final_lr"],
         "n_params": int(sum(p.numel() for p in model.parameters()))}, indent=2))
    clone, _ = study.build_actor(world, "cpu", init=False)
    clone.load_state_dict(torch.load(ck / "weights.pt", map_location="cpu"), strict=True)
    src = {k: v.cpu() for k, v in model.state_dict().items()}
    for k, v in clone.state_dict().items():
        assert torch.equal(src[k], v), f"checkpoint reload drift at {k}"
    print(f"  checkpoint reload verified exactly ({len(src)} tensors)")
    return ck / "weights.pt"


def save_snapshots(out: dict, run_dir: Path, init_state: dict, schedule) -> list[dict]:
    """snapshots/step_XXXXX.pt + snapshots.json; verify snapshot 0 == the
    starting weights and the last snapshot == the trained weights."""
    snaps = out["snapshots"]
    assert sorted(snaps) == list(schedule)
    for k, v in init_state.items():
        assert torch.equal(v, snaps[0][k]), f"snapshot 0 != starting weights at {k}"
    for k, v in out["model"].state_dict().items():
        assert torch.equal(v.cpu(), snaps[schedule[-1]][k]), f"final snapshot != weights at {k}"
    d = run_dir / "snapshots"
    d.mkdir(exist_ok=True)
    index = []
    for step in schedule:
        p = d / f"step_{step:05d}.pt"
        torch.save(snaps[step], p)
        index.append({"step": step, "file": p.name, "sha256": A.sha256_file(p)})
    (run_dir / "snapshots.json").write_text(json.dumps(
        {"schedule": list(schedule), "snapshots": index,
         "semantics": "snapshot k = parameters after exactly k updates; k=0 is the starting weights "
                      "(the seeded init, or the init_from checkpoint)"},
        indent=2))
    print(f"  {len(index)} snapshots saved under {d}")
    return index


def next_clue_evaluation(world, model, seed: int, n: int, device: str) -> dict:
    """Post-training measurement on `n` fresh episodes: the model's mean
    next-clue cross-entropy and the exact Bayes predictor's (posterior after
    the context, pushed through the emission kernel). The trainer never sees
    this; it is the answer key used after training."""
    tokens = world.sample(n, seed=seed)["tokens"]                              # [n, 91]
    eta = world.answer_key(tokens)["eta"]                                      # [n, 91, 9] posterior after each position
    E = hmm.emission_probs(world.T)                                            # [V, 9] P(clue v | state)
    ctx = np.asarray(T.CLUE_CONTEXT_POS)
    q = eta[:, ctx] @ E.T                                                      # [n, 36, V] Bayes next-clue law
    target = tokens[:, ctx + 1]                                                # [n, 36]
    assert (target < world.V).all() and np.allclose(q.sum(-1), 1.0)
    bayes_ce = -np.log(np.take_along_axis(q, target[..., None], axis=2)[..., 0])   # [n, 36]
    ce_parts = []
    bs = int(getattr(model, "jbo_batch_size", 512))
    with torch.no_grad():
        for i in range(0, n, bs):
            t = torch.from_numpy(tokens[i:i + bs]).to(device)
            ce, _ = T.next_token_loss(model(t), t, vocab=world.vocab)          # [b, 36] at the clue targets
            ce_parts.append(ce.cpu().numpy())
    ce = np.concatenate(ce_parts)
    assert ce.shape == bayes_ce.shape == (n, len(ctx))
    return {"n_episodes": n, "n_targets": int(ce.size), "seed": seed,
            "model_ce": float(ce.mean()), "bayes_ce": float(bayes_ce.mean()),
            "excess_ce": float(ce.mean() - bayes_ce.mean())}


def main() -> None:
    args = cli.parse("00_train: train the actor",
                     lambda p: p.add_argument("--snapshots", action="store_true"))
    study = load_study(args.study)
    world = study.build_world()
    s = study.seeds
    if args.smoke:
        seeds = {"model_init": s["smoke_model_init"], "data_stream_base": s["smoke_data_stream_base"]}
        num_steps, training = SMOKE_STEPS, {**study.training, "batch_size": SMOKE_BATCH, "log_every": 5}
    else:
        seeds = {"model_init": s["model_init"], "data_stream_base": s["data_stream_base"]}
        num_steps, training = study.training["num_steps"], study.training
    schedule = T.snapshot_schedule(num_steps) if args.snapshots else None
    assert not (schedule and study.actor["kind"] != "toy"), "snapshots are a toy-actor feature (each LM snapshot is the full model)"
    label = args.label or ("smoke" if args.smoke else "full")
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=label, force=args.force)
    t_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"run dir: {run_dir}")

    lines: list[str] = []
    init_from = None
    if study.actor["kind"] == "toy":
        model, symbol_map = M.build_model(study.arch, world.d_vocab, seed=seeds["model_init"], device=args.device), None
        init_state = study.initial_state()
        if init_state is not None:
            model.load_state_dict(init_state, strict=True)
            src = load_study(study.init_from)
            init_from = {"study": study.init_from, **src.checkpoint}
            print(f"  toy actor initialised from {study.init_from} ({src.checkpoint['path']})", flush=True)
    else:
        model, symbol_map = study.build_actor(world, args.device, init=True)
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        print(f"  pretrained actor {study.actor['tl_config_name']} from {study.actor['hf_init']}; symbols {symbol_map['chars']}", flush=True)
    start_state = T.capture_snapshot(model)
    out = T.train(world, study.arch, training, model_seed=seeds["model_init"],
                  data_seed=seeds["data_stream_base"], num_steps=num_steps, device=args.device,
                  snapshot_steps=set(schedule) if schedule else None, log_lines=lines, model=model)
    out["model"].eval()
    (run_dir / "run.log").write_text("\n".join(lines) + "\n")
    weights = save_checkpoint(study, world, out, run_dir, seeds, num_steps, symbol_map, init_from)
    snap_index = save_snapshots(out, run_dir, start_state, schedule) if schedule else []
    ntp = training.get("loss") == T.NEXT_TOKEN_LOSS
    ntp_eval = next_clue_evaluation(world, out["model"], study.seeds["post_training_eval"],
                                    SMOKE_BATCH if args.smoke else N_NTP_EVAL, args.device) if ntp else None

    last = out["log"][-1]
    if ntp:
        headline = {
            "excess_ce": A.headline_entry(ntp_eval["excess_ce"], ntp_eval["n_targets"], "nats",
                                          "model next-clue cross-entropy minus the exact Bayes predictor's, fresh eval episodes"),
        }
    else:
        headline = {
            "final_reward": A.headline_entry(last["r_mean"], training["batch_size"] * 10, "reward",
                                             "mean sampled reward at the final logged step"),
            "final_entropy": A.headline_entry(last["entropy"], training["batch_size"] * 10, "nats",
                                              "mean policy entropy at the final logged step"),
        }
    results = {
        "headline": headline,
        "loss": training.get("loss", T.DEFAULT_LOSS), "init_from": init_from, "next_clue_eval": ntp_eval,
        "num_steps": num_steps, "batch_size": training["batch_size"], "smoke": args.smoke,
        "wall_s": out["wall_s"], "weights_sha256": A.sha256_file(weights),
        "n_snapshots": len(snap_index), "log": out["log"],
    }
    A.write_results(run_dir, results)
    if not args.smoke:
        print(f"  to pin: python tools/pin_checkpoint.py --study {args.study} --weights {weights}")
    backend = {"gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None}
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={**seeds, "substream_rule": "episodes seed=data+step; actions "
                            "default_rng([data, step, 0]); rewards default_rng([data, step, 1])"},
                     spec_hashes=study.spec_hashes, start_time=t_start, device=args.device,
                     force_used=args.force, figures=[],
                     extra={"backend": backend, "checkpoint": {"path": str(weights.relative_to(ROOT)),
                            "sha256": results["weights_sha256"]}, "init_from": init_from,
                            "loss": results["loss"],
                            "snapshots": snap_index, "wall_s": out["wall_s"], "num_steps": num_steps})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
