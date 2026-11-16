"""The actor: a TransformerLens HookedTransformer. Toy studies build one
fresh with a seeded init; language-model studies start from a pretrained
Hugging Face checkpoint, restricted to the world's symbols. Either way the
trained weights are a plain state_dict the same loader reads back."""

from __future__ import annotations

import os

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402
from transformer_lens import loading_from_pretrained as loading           # noqa: E402

SHAPE_KEYS = ("n_layers", "d_model", "n_heads", "d_head", "d_mlp", "n_ctx",
              "act_fn", "normalization_type")
# One single-token string per world symbol, in the world's symbol order:
# clue tokens, then response A/B, utility levels -1/0/+1, membership bits 0/1, DECIDE.
CLUE_CHARS = "abcdefghijklmnop"
FIXED_CHARS = ("A", "B", "-", "0", "+", "x", "o", "?")


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(arch: dict, d_vocab: int, seed: int, device: str) -> HookedTransformer:
    """Fresh HookedTransformer with the named shape and a seeded init."""
    cfg = HookedTransformerConfig(d_vocab=d_vocab, seed=seed, device=device,
                                  **{k: arch[k] for k in SHAPE_KEYS})
    return HookedTransformer(cfg)


def symbol_map(n_clue_tokens: int, tokenizer) -> dict:
    """{"chars": [d_vocab str], "table": [d_vocab int]}: world symbol i -> the
    pretrained tokenizer's single token for its character (asserted single
    and distinct), the mapping the pretrained study trains with."""
    chars = [CLUE_CHARS[i] for i in range(n_clue_tokens)] + list(FIXED_CHARS)
    table = []
    for ch in chars:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        assert len(ids) == 1, f"{ch!r} is not a single token: {ids}"
        table.append(int(ids[0]))
    assert len(set(table)) == len(table) == len(chars), "symbol map must be a bijection"
    return {"chars": chars, "table": table}


def pretrained_config(tl_config_name: str, d_vocab: int, device: str) -> HookedTransformerConfig:
    """The pretrained architecture's HookedTransformerConfig with the vocabulary
    set to the world's `d_vocab` (no folding, no centring, float32)."""
    cfg = loading.get_pretrained_model_config(tl_config_name, fold_ln=False, device=device, dtype=torch.float32)
    cfg.d_vocab = cfg.d_vocab_out = int(d_vocab)
    cfg.tokenizer_name = None                    # the actor speaks the world's symbols, never text
    return cfg


def build_pretrained_model(tl_config_name: str, hf_init: str | None, n_clue_tokens: int, d_vocab: int,
                           device: str) -> tuple[HookedTransformer, dict | None]:
    """`hf_init` given: the Hugging Face weights loaded without TransformerLens'
    processing (every residual site equals the HF hidden state), then the
    embedding rows and unembedding columns restricted to the world's symbols in
    world order, so the actor takes world tokens and emits [N, T, d_vocab]
    logits exactly as a toy actor does; the discarded rows were never fed and
    never read. `hf_init` None: the same architecture with random weights, for
    loading a trained state_dict into. Returns (model, symbol map or None)."""
    if hf_init is None:
        return HookedTransformer(pretrained_config(tl_config_name, d_vocab, device)), None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_init)
    smap = symbol_map(n_clue_tokens, tok)
    assert len(smap["table"]) == d_vocab
    hf = AutoModelForCausalLM.from_pretrained(hf_init, dtype=torch.float32)
    model = HookedTransformer.from_pretrained_no_processing(tl_config_name, hf_model=hf, tokenizer=tok,
                                                            device=device, dtype=torch.float32)
    del hf
    idx = torch.as_tensor(np.asarray(smap["table"], dtype=np.int64), device=model.cfg.device)
    with torch.no_grad():
        W_E = model.embed.W_E.data[idx].clone()                   # [d_vocab, d]
        W_U = model.unembed.W_U.data[:, idx].clone()              # [d, d_vocab]
        b_U = model.unembed.b_U.data[idx].clone()
    model.embed.W_E = torch.nn.Parameter(W_E)
    model.unembed.W_U = torch.nn.Parameter(W_U)
    model.unembed.b_U = torch.nn.Parameter(b_U)
    model.cfg.d_vocab = model.cfg.d_vocab_out = int(d_vocab)
    model.cfg.tokenizer_name = None
    model.tokenizer = None
    return model, smap
