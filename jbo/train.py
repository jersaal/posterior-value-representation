"""Zero-baseline REINFORCE trainer, and the next-token pretraining stage.
Evaluator-free: never calls world.answer_key; logs sampled quantities only.
`training["loss"]` selects the per-decision loss (see zero_baseline_loss);
default is the paper's REINFORCE + entropy bonus. `"next_token"` instead
trains next-clue prediction on the same episodes (see next_token_loss): the
Shai et al. objective, no choices sampled, no rewards drawn. One trainer for
every actor: a step may be sliced
into `training["micro_batch"]` episodes (gradient accumulation; the batch-mean
gradient either way) and the blocks may run under bfloat16 autocast
(`training["autocast"]`, CUDA only; master weights and the response head stay
float32). Neither is set for the toy studies."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from jbo import model as M
from jbo.world import ACTION_POSITIONS, CLUE_POS, L, StreamVocab, World

N_READOUTS = len(ACTION_POSITIONS)
CLUE_CONTEXT_POS = tuple(int(p) - 1 for p in CLUE_POS)       # the 36 positions whose next token is a clue
POW2_MAX = 512
COARSE_STEPS = (1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000, 20000)


def snapshot_schedule(num_steps: int) -> tuple[int, ...]:
    """{0} ∪ {2^k ≤ min(512, num_steps)} ∪ {coarse ≤ num_steps} ∪ {num_steps}."""
    assert num_steps >= 1
    pow2 = {1 << k for k in range(POW2_MAX.bit_length()) if 1 << k <= min(POW2_MAX, num_steps)}
    coarse = {s for s in COARSE_STEPS if s <= num_steps}
    return tuple(sorted({0} | pow2 | coarse | {num_steps}))


def policy_at_actions(logits: torch.Tensor, response_base: int | None = None):
    """logits [N, 91, d_vocab] -> (logpi [N, 10, 2], pi [N, 10, 2], entropy [N, 10]):
    softmax over the two response channels at each DECIDE position."""
    assert logits.shape[1] == L, f"expected all {L} positions, got {logits.shape}"
    if response_base is None:
        response_base = StreamVocab.from_d_vocab(int(logits.shape[-1])).response_base
    assert response_base + 2 <= logits.shape[-1]
    resp = logits[:, list(ACTION_POSITIONS), response_base: response_base + 2].float()
    logpi = F.log_softmax(resp, dim=-1)
    pi = logpi.exp()
    entropy = -(pi * logpi).sum(-1)
    return logpi, pi, entropy


def sample_choices(logpi: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
    """[N, 10, 2] log-policy -> [N, 10] choices by inverse CDF on float64 CPU probs."""
    pi = logpi.detach().cpu().double().exp().numpy()
    pi /= pi.sum(axis=-1, keepdims=True)
    u = rng.random(pi.shape[:2])
    a = (np.cumsum(pi, axis=-1) < u[..., None]).sum(axis=-1)
    return np.minimum(a, pi.shape[-1] - 1)


LOSSES = ("reinforce_entropy", "reinforce")
NEXT_TOKEN_LOSS = "next_token"
DEFAULT_LOSS = "reinforce_entropy"


NTP_TARGETS = ("clues", "all")


def next_token_loss(logits: torch.Tensor, tokens: torch.Tensor,
                    targets: str = "clues", vocab: StreamVocab | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token prediction on the episode. logits [N, 91, d_vocab], tokens
    [N, 91] -> (cross-entropy [N, T], correct [N, T]).
      "clues": T = 36, the positions whose next token is a clue; softmax over
               the clue alphabet (the process's emissions, Shai et al.'s
               vocabulary). Card, offer and DECIDE tokens are never targets.
      "all":   T = 90, every position but the last; softmax over the whole
               stream vocabulary except the two response channels, so the card,
               offer bits and DECIDE tokens are predicted too (the language-model
               analogue: everything in context is text).
    Either way the two response channels are outside the softmax, so no gradient
    reaches the response head: the RL stage inherits it as seeded up to AdamW's
    uniform weight decay (a factor (1 - lr * 0.01) per step)."""
    assert logits.shape[1] == L and tokens.shape == logits.shape[:2]
    assert targets in NTP_TARGETS, f"unknown next-token targets {targets!r}; one of {NTP_TARGETS}"
    if vocab is None:                                                    # "states" worlds only; pass world.vocab otherwise
        vocab = StreamVocab.from_d_vocab(int(logits.shape[-1]))
    assert vocab.d_vocab == logits.shape[-1], "vocabulary does not match the logits"
    if targets == "clues":
        ctx = list(CLUE_CONTEXT_POS)
        pred = logits[:, ctx, :vocab.n_clue_tokens].float()              # [N, 36, V] clue channels only
        target = tokens[:, [p + 1 for p in ctx]]                         # [N, 36] clue ids
        assert int(target.max()) < vocab.n_clue_tokens, "next-clue target outside the clue alphabet"
    else:
        pred = logits[:, :-1].float().clone()                            # [N, 90, d_vocab]
        pred[..., vocab.response_base:vocab.response_base + 2] = float("-inf")   # response channels never compete
        target = tokens[:, 1:]                                           # [N, 90]
        assert not ((target == vocab.response_base) | (target == vocab.response_base + 1)).any()
    ce = F.cross_entropy(pred.transpose(1, 2), target, reduction="none")  # [N, T]
    correct = (pred.argmax(-1) == target).float()
    return ce, correct


def zero_baseline_loss(logpi: torch.Tensor, entropy: torch.Tensor, beta: float,
                       choices: torch.Tensor, rewards: torch.Tensor,
                       loss: str = DEFAULT_LOSS) -> torch.Tensor:
    """L[n, k], returns [N, 10]:
      reinforce_entropy  -(log pi(choice) * r + H(pi) / beta)   (the paper's trainer)
      reinforce          -(log pi(choice) * r)                  (no entropy bonus, beta unused)"""
    assert loss in LOSSES, f"unknown loss {loss!r}; one of {LOSSES}"
    assert not rewards.requires_grad and not choices.requires_grad
    lp_choice = torch.gather(logpi, 2, choices[..., None])[..., 0]
    if loss == "reinforce_entropy":
        return -(lp_choice * rewards + entropy / beta)
    return -(lp_choice * rewards)


def lr_at(step: int, spec: dict) -> float:
    """Constant lr to decay_start, linear to lr_final at decay_end, then held."""
    lr0, s0, s1 = spec["lr"], spec["decay_start"], spec["decay_end"]
    if step <= s0:
        return lr0
    if step >= s1:
        return spec["lr_final"]
    return lr0 + (step - s0) / (s1 - s0) * (spec["lr_final"] - lr0)


def grad_norm(params) -> float:
    return sum(float((p.grad.detach() ** 2).sum()) for p in params if p.grad is not None) ** 0.5


def capture_snapshot(model) -> dict:
    """Detached CPU copy of state_dict; reads state only, consumes no RNG."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def response_logits(model, toks: torch.Tensor, autocast: bool) -> torch.Tensor:
    """[N, 91] tokens -> [N, 91, d_vocab] float32 logits. Without autocast this
    is the plain forward. With it the blocks run under bfloat16 autocast and the
    final norm and unembedding run in float32 on the float32 residual, so the
    policy the loss differentiates is the float32 head's."""
    if not autocast:
        return model(toks)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        resid = model(toks, stop_at_layer=model.cfg.n_layers)                 # [N, 91, d] after the last block
    return model.unembed(model.ln_final(resid.float()))


def train(world: World, arch: dict, training: dict, model_seed: int | None, data_seed: int,
          num_steps: int, device: str, snapshot_steps: set[int] | None = None,
          log_lines: list[str] | None = None, model=None) -> dict:
    """Train one actor: a fresh seeded toy actor built from `arch`, or `model`
    when given (a pretrained actor). Returns {model, log, snapshots {k: state_dict},
    final_lr, wall_s}. Snapshot k = parameters after exactly k updates."""
    bs, beta, log_every = training["batch_size"], world.beta, training["log_every"]
    micro = int(training.get("micro_batch") or bs)
    autocast = bool(training.get("autocast", False)) and device.startswith("cuda")
    assert 0 < micro <= bs and bs % micro == 0, f"micro_batch {micro} must divide batch_size {bs}"
    loss_name = training.get("loss", DEFAULT_LOSS)
    ntp = loss_name == NEXT_TOKEN_LOSS
    assert ntp or loss_name in LOSSES, f"unknown training loss {loss_name!r}"
    ntp_targets = training.get("ntp_targets", "clues")
    assert ntp_targets in NTP_TARGETS
    if model is None:
        model = M.build_model(arch, world.d_vocab, seed=model_seed, device=device)
    assert model.cfg.d_vocab == world.d_vocab, "actor vocabulary must be the world's"
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "no trainable parameters"
    opt = torch.optim.AdamW(params, lr=training["lr"])
    dev = next(model.parameters()).device
    snap_set = frozenset(int(s) for s in (snapshot_steps or ()))
    assert all(0 <= s <= num_steps for s in snap_set)
    snapshots: dict[int, dict] = {}
    if 0 in snap_set:
        snapshots[0] = capture_snapshot(model)

    def emit(line: str):
        print(line, flush=True)
        if log_lines is not None:
            log_lines.append(line)

    emit(f"train: world={world.name} vocab={world.d_vocab} arch={arch['name']} beta={beta:g} "
         f"loss={loss_name}{'/' + ntp_targets if ntp else ''} steps={num_steps} bs={bs} micro={micro} autocast={autocast} "
         f"model_seed={model_seed} data_seed={data_seed} device={dev} "
         f"trainable={sum(p.numel() for p in params) / 1e6:.2f}M")
    log, t0 = [], time.time()
    for step in range(num_steps):
        toks = world.sample(bs, seed=data_seed + step)["tokens"]              # [N, 91]
        rng_choice = np.random.default_rng([data_seed, step, 0])              # consumed in slice order
        rng_reward = np.random.default_rng([data_seed, step, 1])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, training)
        opt.zero_grad()
        r_parts, h_parts = [], []                                            # reward / entropy, or CE / accuracy
        for i in range(0, bs, micro):
            sl = slice(i, i + micro)
            toks_t = torch.from_numpy(toks[sl]).to(dev)
            logits = response_logits(model, toks_t, autocast)                 # [n, 91, d_vocab]
            if ntp:
                ce, correct = next_token_loss(logits, toks_t, ntp_targets, world.vocab)   # [n, T] each
                loss = ce.mean()
                r_parts.append(ce.detach())
                h_parts.append(correct)
            else:
                logpi, pi, entropy = policy_at_actions(logits, world.vocab.response_base)
                choices = sample_choices(logpi, rng_choice)
                r_np = world.sample_rewards(toks[sl], choices, rng_reward)
                choices_t = torch.from_numpy(choices).to(dev)
                r_t = torch.from_numpy(r_np).float().to(dev)
                loss = zero_baseline_loss(logpi, entropy, beta, choices_t, r_t, loss=loss_name).mean()
                r_parts.append(r_t.detach())
                h_parts.append(entropy.detach())
            (loss * (len(toks_t) / bs)).backward()                            # accumulates the batch mean
        r_all, h_all = torch.cat(r_parts), torch.cat(h_parts)
        log_this = step % log_every == 0 or step == num_steps - 1
        gn = grad_norm(params) if log_this else None
        opt.step()
        if (step + 1) in snap_set:
            snapshots[step + 1] = capture_snapshot(model)
        if log_this:
            entry = {"step": step, "grad_norm": round(gn, 5), "lr": lr_at(step, training),
                     "wall_s": round(time.time() - t0, 1)}
            if ntp:
                entry.update(ce=round(float(r_all.mean()), 5), acc=round(float(h_all.mean()), 5))
                head = f"ce {entry['ce']:.4f} acc {entry['acc']:.3f}"
            else:
                entry.update(r_mean=round(float(r_all.mean()), 5), r_std=round(float(r_all.std()), 5),
                             entropy=round(float(h_all.mean()), 5))
                head = f"r {entry['r_mean']:+.3f}±{entry['r_std']:.3f} H {entry['entropy']:.3f}"
            log.append(entry)
            emit(f"  step {step:5d}/{num_steps} {head} |g| {entry['grad_norm']:.3f} "
                 f"lr {entry['lr']:.2e} ({entry['wall_s']:.0f}s)")
    return {"model": model, "log": log, "snapshots": snapshots,
            "final_lr": lr_at(num_steps - 1, training), "wall_s": round(time.time() - t0, 1)}
