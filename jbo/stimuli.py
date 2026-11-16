"""Constructed stimuli: offer substitution and twins."""

from __future__ import annotations

import numpy as np

from jbo.world import CARD_POS, CLUE_POS, STAND_A_POS, STAND_B_POS, World


def substitute_standing_offers(world: World, tokens: np.ndarray, maskA: np.ndarray,
                               maskB: np.ndarray) -> np.ndarray:
    """Replace the standing blocks (9–26) by the commanded disjoint mask pair;
    card, clues, terminal blocks untouched (so the analytic posterior is too)."""
    maskA, maskB = np.asarray(maskA, np.int64), np.asarray(maskB, np.int64)
    assert (maskA >= 1).all() and (maskB >= 1).all(), "offers must be nonempty"
    assert (maskA & maskB == 0).all(), "offers must be disjoint"
    out = tokens.copy()
    out[:, STAND_A_POS] = world.offer_tokens(maskA)                      # the world's offer encoding
    out[:, STAND_B_POS] = world.offer_tokens(maskB)
    return world.check_stream(out)


def build_twin(world: World, base: np.ndarray, donor: np.ndarray, arm: str) -> np.ndarray:
    """Twin of `base` taking one block from `donor`: 'card' (0–8), 'masks'
    (9–26), 'clues5' (round 5's four clues, 52–55), or 'identity'."""
    out = base.copy()
    if arm == "card":
        out[:, CARD_POS] = donor[:, CARD_POS]
    elif arm == "masks":
        out[:, STAND_A_POS] = donor[:, STAND_A_POS]
        out[:, STAND_B_POS] = donor[:, STAND_B_POS]
    elif arm == "clues5":
        cols = CLUE_POS[20:24]
        assert list(cols) == [52, 53, 54, 55]
        out[:, cols] = donor[:, cols]
    else:
        assert arm == "identity", f"unknown twin arm {arm!r}"
    return world.check_stream(out)
