"""The frozen event algebra over 9 joint states (3 x 3): utility cards,
proposition masks with their 8 strata and held-out splits, and the disjoint
v2 pair law. Every RNG call below reproduces the source repo's call order
exactly. Factor-independent: nothing here reads an HMM.

Two mask families. "states" (default): all 511 nonempty subsets of the nine
states, 8 strata, 64 held out. "factors": the 48 product sets U x W of a
nonempty subset of each factor's three states (not both full), the four
strata atom / factorA / factorB / rect, 6 held out; an offer in this family
is a pair of conditions, one per chain."""

from __future__ import annotations

import itertools
import zlib
from dataclasses import dataclass

import numpy as np

N_STATES = 9
N_BITS = 9
N_MASKS = 512                       # masks 1..511 are the nonempty propositions
N_CARD_HELDOUT = 1000
N_PROP_HELDOUT = 64
N_PAIR_HELDOUT = 2000
CARD_SEED = zlib.crc32(b"news_world:cards") % (2 ** 31)     # 1957601876
PROP_SEED = zlib.crc32(b"news_world:props") % (2 ** 31)     # 722620417
PAIR_SEED = zlib.crc32(b"news_world:pairs") % (2 ** 31)     # 1594209856

STRATA = ("atom", "coatom", "factorA", "factorB", "xor", "rect", "union", "generic")
STRATUM_SIZES = {"atom": 9, "coatom": 9, "factorA": 6, "factorB": 6,
                 "xor": 18, "rect": 27, "union": 27, "generic": 409}
HELDOUT_PER_STRATUM = {"atom": 1, "coatom": 1, "factorA": 1, "factorB": 1,
                       "xor": 2, "rect": 3, "union": 3, "generic": 52}

FAMILIES = ("states", "factors")
FACTOR_STRATA = ("atom", "factorA", "factorB", "rect")                 # the 48 product sets U x W
N_FACTOR_MASKS = 48
N_FACTOR_PROP_HELDOUT = 6
N_FACTOR_PAIR_HELDOUT = 20                                             # of C(42, 2) = 861 train pairs
HELDOUT_PER_FACTOR_STRATUM = {"atom": 1, "factorA": 1, "factorB": 1, "rect": 3}

MASK_BITS = (np.arange(N_MASKS)[:, None] >> np.arange(N_BITS)) & 1   # [512, 9]


def mask_bits(masks: np.ndarray) -> np.ndarray:
    """Masks [...] (ints in 0..511) -> membership bits [..., 9]; bit k = state k."""
    return MASK_BITS[np.asarray(masks, dtype=np.int64)]


def bits_to_mask(bits: np.ndarray) -> np.ndarray:
    """Membership bits [..., 9] -> mask ints [...]."""
    return np.asarray(bits, dtype=np.int64) @ (1 << np.arange(N_BITS))


# --------------------------------------------------------------------------- #
# product sets: a mask U x W as a pair of 3-bit factor conditions
# --------------------------------------------------------------------------- #
def product_mask(cond_a: np.ndarray, cond_b: np.ndarray) -> np.ndarray:
    """Factor conditions cond_a, cond_b [...] (3-bit ints in 1..7 over factor
    states 0..2) -> joint mask [...]: state s = 3 a + b is in iff a in U and b in W."""
    ua = (np.asarray(cond_a, dtype=np.int64)[..., None] >> np.arange(3)) & 1   # [..., 3] over a
    wb = (np.asarray(cond_b, dtype=np.int64)[..., None] >> np.arange(3)) & 1   # [..., 3] over b
    bits = (ua[..., :, None] * wb[..., None, :]).reshape(*ua.shape[:-1], N_BITS)   # [..., 9], s = 3a + b
    return bits_to_mask(bits)


def factor_conditions(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Product masks [...] -> (cond_a, cond_b) [...] 3-bit ints; asserts every
    mask is a nonempty product set U x W (its bits form a rank-one 3 x 3 grid)."""
    bits = mask_bits(masks).reshape(*np.shape(masks), 3, 3)                 # [..., a, b]
    ua, wb = bits.any(-1).astype(np.int64), bits.any(-2).astype(np.int64)   # [..., 3] each
    cond_a, cond_b = ua @ (1 << np.arange(3)), wb @ (1 << np.arange(3))
    assert np.array_equal(product_mask(cond_a, cond_b), np.asarray(masks, dtype=np.int64)), \
        "not a product set"
    return cond_a, cond_b


# --------------------------------------------------------------------------- #
# utility cards
# --------------------------------------------------------------------------- #
def utility_cards() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eligible [18660, 9], train [17660, 9], heldout [1000, 9]) level
    vectors in {0, 1, 2}; eligible = contains a 0 (u = -1) and a 2 (u = +1)."""
    cards = np.array(list(itertools.product(range(3), repeat=N_STATES)), dtype=np.int64)
    eligible = cards[(cards == 0).any(1) & (cards == 2).any(1)]
    rng = np.random.default_rng(CARD_SEED)
    idx = rng.choice(len(eligible), size=N_CARD_HELDOUT, replace=False)
    held = np.zeros(len(eligible), dtype=bool)
    held[idx] = True
    return eligible, eligible[~held], eligible[held]


# --------------------------------------------------------------------------- #
# propositions: strata and the frozen splits
# --------------------------------------------------------------------------- #
def _proper_subsets() -> list[frozenset]:
    """The 6 proper nonempty subsets of {0, 1, 2}."""
    return [frozenset(c) for r in (1, 2) for c in itertools.combinations(range(3), r)]


def classify_propositions() -> tuple[np.ndarray, dict[str, list[int]]]:
    """Stratum of every mask 1..511 by priority order (first match wins):
    atom |X|=1, coatom |X|=8, factorA U x Omega, factorB Omega x W,
    xor (sA in U) != (sB in W), rect U x W, union (U x Omega) u (Omega x W),
    generic. Returns (strata_of [512] index into STRATA, {name: sorted masks})."""
    subsets = _proper_subsets()

    def mask_of(pred) -> int:
        m = 0
        for s in range(N_STATES):
            if pred(s // 3, s % 3):
                m |= 1 << s
        return m

    factorA = {mask_of(lambda a, b, U=U: a in U) for U in subsets}
    factorB = {mask_of(lambda a, b, W=W: b in W) for W in subsets}
    xor = {mask_of(lambda a, b, U=U, W=W: (a in U) != (b in W)) for U in subsets for W in subsets}
    rect = {mask_of(lambda a, b, U=U, W=W: a in U and b in W) for U in subsets for W in subsets}
    union = {mask_of(lambda a, b, U=U, W=W: a in U or b in W) for U in subsets for W in subsets}

    strata_of = np.zeros(N_MASKS, dtype=np.int64)
    stratum_masks: dict[str, list[int]] = {name: [] for name in STRATA}
    for m in range(1, N_MASKS):
        n = int(bin(m).count("1"))
        if n == 1:
            name = "atom"
        elif n == 8:
            name = "coatom"
        elif m in factorA:
            name = "factorA"
        elif m in factorB:
            name = "factorB"
        elif m in xor:
            name = "xor"
        elif m in rect:
            name = "rect"
        elif m in union:
            name = "union"
        else:
            name = "generic"
        strata_of[m] = STRATA.index(name)
        stratum_masks[name].append(m)
    sizes = {k: len(v) for k, v in stratum_masks.items()}
    assert sizes == STRATUM_SIZES, sizes
    return strata_of, stratum_masks


def family_strata(family: str) -> tuple[tuple[str, ...], dict[str, int], int]:
    """(strata, held-out count per stratum, held-out pair count) of a mask family."""
    assert family in FAMILIES, f"unknown mask family {family!r}; one of {FAMILIES}"
    if family == "states":
        return STRATA, HELDOUT_PER_STRATUM, N_PAIR_HELDOUT
    return FACTOR_STRATA, HELDOUT_PER_FACTOR_STRATUM, N_FACTOR_PAIR_HELDOUT


def proposition_split(heldout_strata: tuple[str, ...] = (), family: str = "states"
                      ) -> tuple[np.ndarray, dict[str, list[int]], np.ndarray, np.ndarray]:
    """(strata_of, stratum_masks, train_masks [447], heldout_masks [64]):
    HELDOUT_PER_STRATUM masks drawn with default_rng(PROP_SEED) in stratum order.
    `heldout_strata` withholds every mask of the named strata as well (the
    standard 64 are drawn first, so the RNG stream is unchanged); the held-out
    set then has 64 + the strata's masks not already among them.
    Family "factors": stratum_masks holds only the four product strata (48
    masks), 6 are held out, train_masks has 42."""
    strata, per_stratum, _ = family_strata(family)
    strata_of, stratum_masks = classify_propositions()
    stratum_masks = {name: stratum_masks[name] for name in strata}
    universe = sorted(m for masks in stratum_masks.values() for m in masks)
    assert len(universe) == (N_MASKS - 1 if family == "states" else N_FACTOR_MASKS)
    rng = np.random.default_rng(PROP_SEED)
    held: list[int] = []
    for name in strata:
        pool = stratum_masks[name]
        pick = rng.choice(len(pool), size=per_stratum[name], replace=False)
        held += [pool[i] for i in pick]
    held = sorted(held)
    assert len(held) == sum(per_stratum.values())
    assert all(name in strata for name in heldout_strata), heldout_strata
    extra = sorted(set(m for name in heldout_strata for m in stratum_masks[name]) - set(held))
    held = sorted(held + extra)
    assert len(held) == sum(per_stratum.values()) + len(extra)
    train = sorted(set(universe) - set(held))
    return strata_of, stratum_masks, np.array(train), np.array(held)


def pair_split(train_masks: np.ndarray, n_pairs: int = N_PAIR_HELDOUT) -> list[tuple[int, int]]:
    """The n_pairs unordered train-mask pairs never offered together, from
    default_rng(PAIR_SEED).integers in the source's literal loop (a pair that
    is not disjoint is held out vacuously)."""
    assert 0 < n_pairs < len(train_masks) * (len(train_masks) - 1) // 2
    rng = np.random.default_rng(PAIR_SEED)
    held: set[tuple[int, int]] = set()
    while len(held) < n_pairs:
        i, j = rng.integers(0, len(train_masks), size=2)
        if i == j:
            continue
        a, b = int(train_masks[i]), int(train_masks[j])
        held.add((min(a, b), max(a, b)))
    return sorted(held)


# --------------------------------------------------------------------------- #
# the disjoint v2 pair law
# --------------------------------------------------------------------------- #
def pair_law(stratum_masks: dict[str, list[int]], heldout_masks: np.ndarray,
             heldout_pairs: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    """Q [512, 512]: A[x, y] = 1[x & y == 0] minus held-out pairs, restricted to
    train masks; uniform over admissible ordered (stratum, stratum) cells and
    uniform within a cell. Symmetric, sums to one. Returns (Q, n_cells)."""
    m = np.arange(N_MASKS)
    A = (m[:, None] & m[None, :]) == 0                             # [512, 512]
    for a, b in heldout_pairs:
        A[a, b] = A[b, a] = False
    held = set(int(x) for x in heldout_masks)
    pools = [np.array([x for x in masks if x not in held], dtype=np.int64) for masks in stratum_masks.values()]
    pools = [p for p in pools if len(p)]                            # a wholly withheld stratum has no cell
    cells = [(pa, pb) for pa in pools for pb in pools if A[np.ix_(pa, pb)].any()]
    Q = np.zeros((N_MASKS, N_MASKS))
    for pa, pb in cells:
        ix = np.ix_(pa, pb)
        Q[ix] += A[ix] / (len(cells) * A[ix].sum())
    assert abs(Q.sum() - 1.0) < 1e-12 and np.allclose(Q, Q.T)
    return Q, len(cells)


@dataclass(frozen=True)
class Algebra:
    """Everything frozen about cards and propositions, built once."""
    cards_all: np.ndarray            # [18660, 9]
    train_cards: np.ndarray          # [17660, 9]
    heldout_cards: np.ndarray        # [1000, 9]
    strata_of: np.ndarray            # [512] index into STRATA (every mask, whatever the family)
    stratum_masks: dict              # the family's strata only
    train_masks: np.ndarray          # [447] ("factors": [42])
    heldout_masks: np.ndarray        # [64], or 64 + whole strata (`heldout_strata`); "factors": [6]
    heldout_pairs: list              # 2000 (lo, hi); "factors": 20
    pair_law: np.ndarray             # [512, 512]
    pair_cum: np.ndarray             # [512*512] normalised cumsum of pair_law.ravel()
    n_cells: int
    slot_marginal: np.ndarray        # [512] row sums of pair_law
    unofferable_masks: np.ndarray    # train masks with zero slot mass
    family: str = "states"           # "states" (511 masks) or "factors" (48 product sets)

    def partner_pool(self) -> np.ndarray:
        """[K] every offerable mask of the family, sorted: the pool a disjoint partner is drawn from
        ("states": 1..511, so filtering by disjointness leaves 1..510; "factors": the 48 product sets)."""
        pool = np.sort(np.concatenate([self.train_masks, self.heldout_masks])).astype(np.int64)
        assert len(np.unique(pool)) == len(pool), "train and held-out masks overlap"
        return pool

    def draw_pairs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """[n, 2] offer pairs: one categorical draw per pair from Q."""
        idx = np.searchsorted(self.pair_cum, rng.random(n), side="right")
        return np.stack([idx // N_MASKS, idx % N_MASKS], axis=1).astype(np.int64)


def build_algebra(heldout_strata: tuple[str, ...] = (), family: str = "states") -> Algebra:
    """The frozen algebra; `heldout_strata` withholds whole strata from training
    as well; `family` picks the mask universe (module docstring)."""
    cards_all, train_cards, heldout_cards = utility_cards()
    strata_of, stratum_masks, train_masks, heldout_masks = proposition_split(tuple(heldout_strata), family)
    heldout_pairs = pair_split(train_masks, family_strata(family)[2])
    Q, n_cells = pair_law(stratum_masks, heldout_masks, heldout_pairs)
    cum = np.cumsum(Q.ravel())
    cum /= cum[-1]
    slot = Q.sum(axis=1)
    unofferable = np.array([x for x in train_masks if slot[x] == 0.0])
    return Algebra(cards_all, train_cards, heldout_cards, strata_of, stratum_masks,
                   train_masks, heldout_masks, heldout_pairs, Q, cum, n_cells,
                   slot, unofferable, family)
