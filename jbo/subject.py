"""Study = studies/<world>/<seed>/study.json: world name, arch, training,
seeds, checkpoint pin, instruments pin, and optionally `init_from`: another
study of the same world and arch whose pinned checkpoint is this study's
starting weights (a second training stage). Loads and hash-verifies everything."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from jbo import ROOT, model as M
from jbo.artifacts import sha256_file
from jbo.world import L, ACTION_POSITIONS, World

STUDIES_DIR = ROOT / "studies"


@dataclass
class Study:
    name: str                 # "<world>/<seed>"
    study: dict
    arch: dict
    training: dict
    spec_hashes: dict

    @property
    def dir(self) -> Path:
        return STUDIES_DIR / self.name

    @property
    def world_name(self) -> str:
        return self.study["world"]

    @property
    def seeds(self) -> dict:
        return self.study["seeds"]

    def build_world(self) -> World:
        return World(self.world_name)

    @property
    def checkpoint(self) -> dict:
        ck = self.study.get("checkpoint")
        assert ck is not None, f"study {self.name} has no checkpoint pin — run 00_train"
        return ck

    @property
    def init_from(self) -> str | None:
        """Name of the study whose pinned checkpoint initialises this one's
        training (toy actors only), or None for the seeded init."""
        name = self.study.get("init_from")
        assert name is None or (isinstance(name, str) and name.count("/") == 1), f"bad init_from {name!r}"
        return name

    def initial_state(self) -> dict | None:
        """The init_from study's pinned weights (sha-verified, same world and
        arch, CPU state_dict), or None when this study trains from the seeded init."""
        if self.init_from is None:
            return None
        assert self.actor["kind"] == "toy", "init_from is a toy-actor feature"
        src = load_study(self.init_from)
        assert src.world_name == self.world_name and src.arch == self.arch, \
            f"init_from {self.init_from} is a different world or arch"
        assert src.seeds["data_stream_base"] != self.seeds["data_stream_base"] or \
            src.training != self.training, "init_from would replay the source study's own training"
        weights = ROOT / src.checkpoint["path"]
        assert sha256_file(weights) == src.checkpoint["sha256"], f"init_from checkpoint sha mismatch: {weights}"
        return torch.load(weights, map_location="cpu")

    @property
    def actor(self) -> dict:
        """{"kind": "toy"} (built fresh by 00 from `arch`) or
        {"kind": "pretrained", "tl_config_name", "hf_init", "batch_size"} (initialised
        from a Hugging Face checkpoint, restricted to the world's symbols)."""
        return self.study.get("actor", {"kind": "toy"})

    def build_actor(self, world: World, device: str, init: bool):
        """A fresh actor of this study's kind. `init`: seeded toy weights, or the
        pretrained weights (a download); False gives the architecture only, for
        loading a state_dict into. Returns (model, symbol map or None)."""
        a = self.actor
        if a["kind"] == "toy":
            seed = self.seeds["model_init"] if init else 0
            return M.build_model(self.arch, world.d_vocab, seed=seed, device=device), None
        assert a["kind"] == "pretrained", f"unknown actor kind {a['kind']!r}"
        model, smap = M.build_pretrained_model(a["tl_config_name"], a["hf_init"] if init else None,
                                              world.vocab.n_clue_tokens, world.d_vocab, device)
        shape = {"n_layers": model.cfg.n_layers, "d_model": model.cfg.d_model, "n_heads": model.cfg.n_heads,
                 "d_head": model.cfg.d_head, "d_mlp": model.cfg.d_mlp}
        for k, v in shape.items():
            assert self.arch[k] == v, f"arch spec {k}={self.arch[k]} but the pretrained architecture has {v}"
        assert model.cfg.n_ctx >= L
        model.jbo_batch_size = int(a["batch_size"])         # caps the batched passes in probes/steering/behavior
        return model, smap

    def load_actor(self, device: str = "cpu"):
        """Load the pinned checkpoint strict, verify sha and config geometry,
        and probe a finite forward. Returns (model.eval(), config)."""
        ck = self.checkpoint
        weights = ROOT / ck["path"]
        assert weights.exists(), f"checkpoint missing: {weights}"
        actual = sha256_file(weights)
        assert actual == ck["sha256"], f"checkpoint sha mismatch: pin {ck['sha256']} != {actual}"
        config = json.loads((weights.parent / "config.json").read_text())
        assert config["world"] == self.world_name and config["study"] == self.name
        assert config["n_ctx"] == L and tuple(config["action_positions"]) == ACTION_POSITIONS
        assert config["world_spec_sha256"] == self.spec_hashes["world"], "world spec drifted"
        assert config["arch"] == self.arch, "arch drifted"
        world = self.build_world()
        model, _ = self.build_actor(world, device, init=self.actor["kind"] == "toy")
        if self.actor["kind"] == "toy":
            assert config["model_seed"] == self.seeds["model_init"]
        sd = torch.load(weights, map_location=device)
        model.load_state_dict(sd, strict=True)
        for name, p in model.named_parameters():
            assert torch.equal(p.detach(), sd[name].to(p.device)), f"{name} drifted during load"
            p.requires_grad_(False)
        model.eval()
        probe = world.sample(2, seed=self.seeds["smoke_data_stream_base"])["tokens"]
        with torch.no_grad():
            logits = model(torch.from_numpy(probe).to(device))
        assert tuple(logits.shape) == (2, L, config["d_vocab"]) and bool(torch.isfinite(logits).all())
        return model, config


def load_study(name: str) -> Study:
    """`name` = "<world>/<seed>"."""
    path = STUDIES_DIR / name / "study.json"
    assert path.exists(), f"no study at {path}"
    study = json.loads(path.read_text())
    arch = json.loads((ROOT / study["arch"]).read_text())
    training = json.loads((ROOT / study["training"]).read_text())
    world_spec = ROOT / "worlds" / f"{study['world']}.json"
    assert world_spec.exists(), f"missing world spec {world_spec}"
    hashes = {"study": sha256_file(path), "arch": sha256_file(ROOT / study["arch"]),
              "training": sha256_file(ROOT / study["training"]), "world": sha256_file(world_spec)}
    for f in json.loads(world_spec.read_text())["factors"]:
        hashes[f"hmm:{f}"] = sha256_file(ROOT / "hmms" / f"{f}.json")
    assert study["seed_tag"] == name.split("/")[1]
    return Study(name, study, arch, training, hashes)


def pin_checkpoint(study: Study, weights: Path, force: bool = False) -> None:
    """Record the checkpoint into study.json (refuses to replace unless force)."""
    raw = json.loads((study.dir / "study.json").read_text())
    assert raw.get("checkpoint") is None or force, \
        f"{study.name} already pins a checkpoint — pass --force to replace"
    raw["checkpoint"] = {"path": str(Path(weights).resolve().relative_to(ROOT)), "sha256": sha256_file(weights)}
    (study.dir / "study.json").write_text(json.dumps(raw, indent=2) + "\n")
