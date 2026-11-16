"""The 91-token news-value game: joint HMM over 9 states, utility card,
two standing offers, nine 4-clue rounds each followed by DECIDE, a terminal
surprise pair and its DECIDE. Sampler, exact answer key, reward lottery.

Layout (L = 91): card 0-8 | standing A bits 9-17 | B bits 18-26 |
round r: clues 27+5r..30+5r, DECIDE 31+5r | terminal A 72-80 | B 81-89 |
DECIDE 90. Vocab: clues (VA*VB) | responses 2 | levels 3 | bits 2 | DECIDE 1
| (offers "factors" only) 14 condition symbols | 1 pad.
Offer encodings (spec key "offers"): "states" (default) writes an offer's nine
membership bits, slot k = state k; "factors" writes one condition symbol per
chain (slot 0: M's subset, slot 1: N's subset) and pads the other seven slots,
so no slot stands for a state and the offer family is the 48 product sets.
Response index 0 = slot A, 1 = slot B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jbo import algebra, hmm, registry
from jbo.algebra import N_BITS, N_STATES

L = 91
N_ROUNDS = 9
CLUES_PER_ROUND = 4
N_CLUES = N_ROUNDS * CLUES_PER_ROUND                                   # 36
N_READOUTS = N_ROUNDS + 1                                              # 10
CARD_POS = np.arange(0, 9)
STAND_A_POS = np.arange(9, 18)
STAND_B_POS = np.arange(18, 27)
CLUE_POS = (27 + 5 * np.arange(N_ROUNDS)[:, None] + np.arange(CLUES_PER_ROUND)).ravel()  # [36]
ACTION_POSITIONS = (31, 36, 41, 46, 51, 56, 61, 66, 71, 90)
DECIDE_POS = np.array(ACTION_POSITIONS)                                # [10]
TERM_A_POS = np.arange(72, 81)
TERM_B_POS = np.arange(81, 90)
assert len(np.unique(np.concatenate([CARD_POS, STAND_A_POS, STAND_B_POS, CLUE_POS,
                                     DECIDE_POS, TERM_A_POS, TERM_B_POS]))) == L


N_CONDITIONS = 7                                                       # nonempty subsets of a factor's 3 states
N_CONDITION_TOKENS = 2 * N_CONDITIONS                                  # one symbol set per chain
OFFER_ENCODINGS = ("states", "factors")


@dataclass(frozen=True)
class StreamVocab:
    """Token ids for a world with `n_clue_tokens` composite clue tokens; the
    condition block (and its pad) exists only under the "factors" offer encoding."""
    n_clue_tokens: int
    n_condition_tokens: int = 0

    @property
    def response_base(self) -> int:
        return self.n_clue_tokens

    @property
    def level_base(self) -> int:
        return self.n_clue_tokens + 2

    @property
    def bit_base(self) -> int:
        return self.n_clue_tokens + 5

    @property
    def decide_token(self) -> int:
        return self.n_clue_tokens + 7

    @property
    def cond_base(self) -> int:
        """First condition symbol: M's seven subsets, then N's seven ("factors" only)."""
        assert self.n_condition_tokens == N_CONDITION_TOKENS, "no condition block in this vocabulary"
        return self.n_clue_tokens + 8

    @property
    def pad_token(self) -> int:
        return self.cond_base + self.n_condition_tokens

    @property
    def d_vocab(self) -> int:
        return self.n_clue_tokens + 8 + (self.n_condition_tokens + 1 if self.n_condition_tokens else 0)

    @classmethod
    def from_d_vocab(cls, d_vocab: int) -> "StreamVocab":
        """The "states" vocabulary of a given size (the fallback when no world is
        at hand; a "factors" world must pass its own vocab)."""
        assert d_vocab >= 9, f"d_vocab {d_vocab} cannot hold the 8 non-clue tokens"
        return cls(int(d_vocab) - 8)


class World:
    """One world from its spec name (worlds/<name>.json)."""

    def __init__(self, name: str):
        self.spec = registry.load_world(name)
        self.name = name
        self.factors = tuple(self.spec["factors"])
        self.n_factors = len(self.factors)
        Ts = [registry.load_hmm(f) for f in self.factors]
        self.factor_shapes = tuple(T.shape[1] for T in Ts)
        if self.n_factors == 2:
            self.TA, self.TB = Ts
            self.SA, self.SB = self.factor_shapes
            assert self.SA == 3 and self.SB == 3, "the algebra is built on 9 joint states"
            self.VA, self.VB = self.TA.shape[0], self.TB.shape[0]
            self.piA = hmm.stationary_distribution(self.TA)
            self.piB = hmm.stationary_distribution(self.TB)
        else:
            assert self.factor_shapes == (N_STATES,), "a one-chain world needs exactly nine states"
        self.T = hmm.nkron(Ts)                                       # [V, 9, 9]; two factors: v = VB*a + b
        self.V, self.S = self.T.shape[0], N_STATES
        self.pi0 = hmm.stationary_distribution(self.T)               # [9]
        self.levels = np.asarray(self.spec["levels"], dtype=np.float64)
        assert list(self.levels) == [-1.0, 0.0, 1.0], "levels are frozen at {-1, 0, +1}"
        self.beta = float(self.spec["beta"])
        assert self.beta > 0
        self.offers = str(self.spec.get("offers", "states"))
        assert self.offers in OFFER_ENCODINGS, f"unknown offer encoding {self.offers!r}"
        assert self.offers == "states" or self.n_factors == 2, "factor conditions need a two-factor world"
        self.vocab = StreamVocab(self.V, N_CONDITION_TOKENS if self.offers == "factors" else 0)
        self.d_vocab = self.vocab.d_vocab
        self.algebra = algebra.build_algebra(tuple(self.spec.get("heldout_strata", ())),
                                             family=self.offers)
        self.reward = str(self.spec.get("reward", "conditional"))
        assert self.reward in ("conditional", "mass"), f"unknown reward rule {self.reward!r}"
        self.n_readouts = N_READOUTS
        self.action_positions = DECIDE_POS

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def sample(self, n: int, seed: int) -> dict:
        """[n, 91] episodes. RNG order: cards, standing pairs, terminal pairs,
        initial states, then one uniform per clue position. Returns
        dict(tokens [n, 91], card [n, 9], stand [n, 2], term [n, 2])."""
        rng = np.random.default_rng(seed)
        alg, v = self.algebra, self.vocab
        card = alg.train_cards[rng.integers(0, len(alg.train_cards), n)]   # [n, 9]
        stand = alg.draw_pairs(rng, n)                                      # [n, 2]
        term = alg.draw_pairs(rng, n)
        tokens = np.empty((n, L), dtype=np.int64)
        tokens[:, CARD_POS] = v.level_base + card
        tokens[:, STAND_A_POS] = self.offer_tokens(stand[:, 0])
        tokens[:, STAND_B_POS] = self.offer_tokens(stand[:, 1])
        tokens[:, TERM_A_POS] = self.offer_tokens(term[:, 0])
        tokens[:, TERM_B_POS] = self.offer_tokens(term[:, 1])
        tokens[:, DECIDE_POS] = v.decide_token
        # hidden chain: one inverse-CDF draw of (token, next state) per clue
        K = self.T.transpose(1, 0, 2).reshape(self.S, self.V * self.S)     # [9, V*9]
        Kcum = np.cumsum(K, axis=1)
        Kcum /= Kcum[:, -1:]
        cur = rng.choice(self.S, size=n, p=self.pi0)
        for p in CLUE_POS:
            u = rng.random(n)
            flat = np.empty(n, dtype=np.int64)
            for s in np.unique(cur):
                m = cur == s
                flat[m] = np.searchsorted(Kcum[s], u[m], side="right")
            tokens[:, p] = flat // self.S
            cur = flat % self.S
        return {"tokens": self.check_stream(tokens), "card": card, "stand": stand, "term": term}

    # ------------------------------------------------------------------ #
    # offer blocks: mask <-> the nine tokens of an offer block
    # ------------------------------------------------------------------ #
    def offer_tokens(self, masks: np.ndarray) -> np.ndarray:
        """Masks [N] -> the nine tokens of an offer block [N, 9] under this
        world's offer encoding ("states": membership bits; "factors": M's and
        N's condition symbols then seven pads)."""
        masks = np.asarray(masks, dtype=np.int64)
        v = self.vocab
        if self.offers == "states":
            return v.bit_base + algebra.mask_bits(masks)
        cond_a, cond_b = algebra.factor_conditions(masks)                # [N] each, 1..7
        block = np.full((len(masks), N_BITS), v.pad_token, dtype=np.int64)
        block[:, 0] = v.cond_base + cond_a - 1
        block[:, 1] = v.cond_base + N_CONDITIONS + cond_b - 1
        return block

    def offer_bits(self, block: np.ndarray) -> np.ndarray:
        """Offer block tokens [N, 9] -> membership bits [N, 9] (bit k = state k)."""
        v = self.vocab
        block = np.asarray(block, dtype=np.int64)
        assert block.shape[1:] == (N_BITS,), block.shape
        if self.offers == "states":
            bits = block - v.bit_base
            assert bits.min() >= 0 and bits.max() <= 1, "bad offer bits"
            return bits
        cond_a, cond_b = block[:, 0] - v.cond_base + 1, block[:, 1] - v.cond_base - N_CONDITIONS + 1
        assert cond_a.min() >= 1 and cond_a.max() <= N_CONDITIONS, "slot 0 is not an M condition"
        assert cond_b.min() >= 1 and cond_b.max() <= N_CONDITIONS, "slot 1 is not an N condition"
        assert (block[:, 2:] == v.pad_token).all(), "offer slots 2-8 must be pads"
        return algebra.mask_bits(algebra.product_mask(cond_a, cond_b))

    def check_stream(self, tokens: np.ndarray) -> np.ndarray:
        """Assert an [N, 91] stream obeys the layout and this world's alphabet."""
        v = self.vocab
        tokens = np.asarray(tokens, dtype=np.int64)
        assert tokens.ndim == 2 and tokens.shape[1] == L, f"expected [N, {L}], got {tokens.shape}"
        assert tokens.min() >= 0 and tokens.max() < v.d_vocab, "token outside the vocabulary"
        assert (tokens[:, DECIDE_POS] == v.decide_token).all(), "DECIDE missing at an action position"
        non_decide = np.setdiff1d(np.arange(L), DECIDE_POS)
        assert (tokens[:, non_decide] != v.decide_token).all(), "DECIDE outside an action position"
        card = tokens[:, CARD_POS]
        assert card.min() >= v.level_base and card.max() <= v.level_base + 2, "bad card block"
        for cols in (STAND_A_POS, STAND_B_POS, TERM_A_POS, TERM_B_POS):
            bits = self.offer_bits(tokens[:, cols])
            assert bits.any(axis=1).all(), "empty proposition offered"
            if self.offers == "factors":
                assert not (bits.all(axis=1)).any(), "the full set is never an offer"
        clues = tokens[:, CLUE_POS]
        assert clues.min() >= 0 and clues.max() < self.V, "bad clue block"
        return tokens

    # ------------------------------------------------------------------ #
    # parsing and the forward algorithm (shared by answer key and lottery)
    # ------------------------------------------------------------------ #
    def _parse(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(card [N, 9] levels, u [N, 9], bits_ro [N, 10, 2, 9])."""
        v = self.vocab
        N = len(tokens)
        card = tokens[:, CARD_POS] - v.level_base
        u = self.levels[card]
        bits_ro = np.empty((N, N_READOUTS, 2, N_BITS))
        bits_ro[:, :N_ROUNDS, 0] = self.offer_bits(tokens[:, STAND_A_POS])[:, None, :]
        bits_ro[:, :N_ROUNDS, 1] = self.offer_bits(tokens[:, STAND_B_POS])[:, None, :]
        bits_ro[:, N_ROUNDS, 0] = self.offer_bits(tokens[:, TERM_A_POS])
        bits_ro[:, N_ROUNDS, 1] = self.offer_bits(tokens[:, TERM_B_POS])
        return card, u, bits_ro

    def _posteriors(self, tokens: np.ndarray) -> np.ndarray:
        """eta [N, 91, 9]: joint posterior after each position; starts at the
        stationary distribution and moves only at the 36 clue positions."""
        N = len(tokens)
        eta = np.empty((N, L, self.S))
        cur = np.broadcast_to(self.pi0, (N, self.S)).copy()
        clue_set = set(int(p) for p in CLUE_POS)
        for p in range(L):
            if p in clue_set:
                cur = np.einsum("ns,nsj->nj", cur, self.T[tokens[:, p]])
                cur /= cur.sum(axis=1, keepdims=True)
            eta[:, p] = cur
        return eta

    # ------------------------------------------------------------------ #
    # the answer key (post-training measurement only)
    # ------------------------------------------------------------------ #
    def answer_key(self, tokens: np.ndarray) -> dict:
        """Exact float64 ground truth. NEVER called by the trainer.
        eta [N,91,9], etaA/etaB [N,91,3] (two-factor worlds only), eta_ro [N,10,9], u [N,9], card [N,9],
        bits_ro [N,10,2,9], P/v/D/policy [N,10,2] (index 0 = slot A)."""
        tokens = self.check_stream(tokens)
        card, u, bits_ro = self._parse(tokens)
        eta = self._posteriors(tokens)
        eta_ro = eta[:, DECIDE_POS]                                        # [N, 10, 9]
        P = np.einsum("nrs,nros->nro", eta_ro, bits_ro)                    # [N, 10, 2]
        assert P.min() > 0.0, "P(offer) = 0 in float64 — full support violated"
        val = np.einsum("nrs,nros,ns->nro", eta_ro, bits_ro, u)
        D = val / P if self.reward == "conditional" else val             # "mass": the linear variant's value is m(X)
        out = {"eta": eta, "eta_ro": eta_ro, "u": u, "card": card, "bits_ro": bits_ro,
               "P": P, "v": val, "D": D, "policy": self.policy_from_D(D)}
        if self.n_factors == 2:                                            # one-chain worlds carry no factor marginals
            out["etaA"], out["etaB"] = hmm.factor_marginals(eta, (self.SA, self.SB))
        return out

    def policy_from_D(self, D: np.ndarray) -> np.ndarray:
        """softmax(beta * D) over the last axis."""
        z = self.beta * np.asarray(D, dtype=np.float64)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    # ------------------------------------------------------------------ #
    # the reward lottery (the trainer's only window)
    # ------------------------------------------------------------------ #
    def sample_rewards(self, tokens: np.ndarray, choices: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
        """r [N, 10] in {-1, 0, +1}: sample s~ from eta_ro restricted to the
        chosen offer, pay u[s~]. E[r | choice] = D(chosen). Never touches
        answer_key / policy. Under reward "mass" (the linear variant) s~ is
        drawn from the unrestricted eta_ro and pays u[s~] only if s~ lies in
        the chosen offer, so E[r | choice] = m(chosen)."""
        tokens = self.check_stream(tokens)
        choices = np.asarray(choices, dtype=np.int64)
        N = len(tokens)
        assert choices.shape == (N, N_READOUTS), "one choice per readout"
        assert choices.min() >= 0 and choices.max() <= 1, "2AFC choices are 0/1"
        card, u, bits_ro = self._parse(tokens)
        eta_ro = self._posteriors(tokens)[:, DECIDE_POS]                   # [N, 10, 9]
        chosen = np.take_along_axis(bits_ro, choices[..., None, None], axis=2)[:, :, 0, :]
        w = eta_ro * chosen if self.reward == "conditional" else eta_ro    # [N, 10, 9]
        mass = w.sum(-1)
        assert mass.min() > 0.0, "reward sampler: P(X) = 0 in float64"
        cdf = np.cumsum(w, axis=-1) / mass[..., None]
        draw = rng.random(mass.shape)
        s = np.minimum((cdf < draw[..., None]).sum(-1), self.S - 1)        # [N, 10]
        r = np.take_along_axis(u, s, axis=1)
        if self.reward == "mass":
            r = r * np.take_along_axis(chosen, s[..., None], axis=2)[..., 0]   # zero unless s~ in the chosen offer
        return r
