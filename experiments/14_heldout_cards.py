"""14 — unseen payoff cards: the 1,000 eligible payoff cards that the algebra
withholds from training (`heldout_cards`) are substituted into the evaluation
episodes, and the fitted probes and the policy are scored on them exactly as
on training cards: choice against the target policy, and the factor posteriors
and offer values read at their read sites. The same episodes with their
original (training) cards give the reference reads. Nothing is fit; the
fitted instruments are loaded.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, behavior as B, cli, instruments as I, plotting as PL, probes as P   # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB                         # noqa: E402
from jbo.subject import load_study                                                                # noqa: E402
from jbo.world import CARD_POS                                                                    # noqa: E402

EXPERIMENT = "14_heldout_cards"
N_EVAL, SMOKE_EVAL = 8192, 512
ROLE_CARDS = 11                   # default_rng([eval_seed, ROLE_CARDS]) draws the held-out card per episode
ARMS = ("training_cards", "heldout_cards")


def with_heldout_cards(world, tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Copy of `tokens` [N, T] with each episode's card replaced by a random held-out card."""
    alg, v = world.algebra, world.vocab
    cards = alg.heldout_cards[rng.integers(0, len(alg.heldout_cards), len(tokens))]      # [N, 9] levels 0..2
    out = tokens.copy()
    out[:, CARD_POS] = v.level_base + cards
    train = set(map(tuple, alg.train_cards.tolist()))
    assert not any(tuple(c) in train for c in cards.tolist()), "a held-out card is also a training card"
    return world.check_stream(out)


def score(model, world, ins, tokens: np.ndarray, device: str) -> dict:
    """Choice correlation with the target policy over all readouts, and the R² of the
    fitted probes' reads at the graded readout on the valid rows; tokens [N, T]."""
    key = world.answer_key(tokens)
    pos, rb = GRADE_POSITION, world.vocab.response_base
    pi = B.model_policy(model, tokens, device, rb)                                        # [N, 10, 2]
    valid = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB                           # [N]
    sites = tuple(dict.fromkeys((ins.posterior_site, ins.value_site)))
    acts = P.collect_site_activations(model, tokens, sites, np.array([pos]), device)      # {site: [N, 1, d]}
    eb = ins.decode(ins.posterior_site, acts[ins.posterior_site][:, 0, :])
    ed = ins.decode(ins.value_site, acts[ins.value_site][:, 0, :])
    D_true, D_hat = key["D"][:, GRADE_READOUT], ins.D(ed)
    return {"pearson_r": B.pearson_r(pi[..., 0], key["policy"][..., 0]),
            "etaA_r2": P.decode_scores(key["etaA"][valid, pos], ins.posterior_A(eb)[valid])["r2"],
            "etaB_r2": P.decode_scores(key["etaB"][valid, pos], ins.posterior_B(eb)[valid])["r2"],
            "D_r2": P.decode_scores(D_true[valid], D_hat[valid])["r2"],
            "n_valid": int(valid.sum()), "n": int(len(tokens))}


def main() -> None:
    args = cli.parse("14_heldout_cards: policy and reads on the withheld payoff cards")
    study = load_study(args.study)
    world = study.build_world()
    assert world.n_factors == 2, "the factor-posterior reads need a two-factor world"
    model, _ = study.load_actor(args.device)
    seed = study.seeds["eval"]
    ins = I.load(study, eval_seeds=[seed])
    n = SMOKE_EVAL if args.smoke else N_EVAL
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    base = world.sample(n, seed=seed)["tokens"]
    swapped = with_heldout_cards(world, base, np.random.default_rng([seed, ROLE_CARDS]))
    res = {"training_cards": score(model, world, ins, base, args.device),
           "heldout_cards": score(model, world, ins, swapped, args.device)}
    for arm in ARMS:
        r = res[arm]
        print(f"  {arm:15s} r {r['pearson_r']:.4f}  etaA {r['etaA_r2']:.3f}  etaB {r['etaB_r2']:.3f}  D {r['D_r2']:.3f}  "
              f"(n valid {r['n_valid']})", flush=True)

    H = A.headline_entry
    b_lab, d_lab = PL.SITE_LABELS[ins.posterior_site], PL.SITE_LABELS[ins.value_site]
    head = {}
    for arm in ARMS:
        r = res[arm]
        head[f"etaA_r2_{arm}"] = H(r["etaA_r2"], r["n_valid"], "R²", f"factor posterior M read at {b_lab} with the fitted probe, {arm}")
        head[f"etaB_r2_{arm}"] = H(r["etaB_r2"], r["n_valid"], "R²", f"factor posterior N read at {b_lab} with the fitted probe, {arm}")
        head[f"D_r2_{arm}"] = H(r["D_r2"], r["n_valid"], "R²", f"offer values read at {d_lab} with the fitted probe, {arm}")
    r = res["heldout_cards"]
    head["choice_pearson_r_heldout_cards"] = H(r["pearson_r"], r["n"] * world.n_readouts, "r",
                                               "Pearson r of P(choose A), model vs target policy, heldout_cards")
    n_held, n_train = int(len(world.algebra.heldout_cards)), int(len(world.algebra.train_cards))
    A.write_results(run_dir, {"headline": head, "n": {"eval": n, "heldout_cards": n_held, "train_cards": n_train,
                                                      "eligible_cards": n_held + n_train},
                              "sites": {"posterior": ins.posterior_site, "value": ins.value_site}})
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv, seeds=study.seeds,
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                     figures=[], extra={"checkpoint_sha256": study.checkpoint["sha256"], "smoke": args.smoke})
    print("done:", run_dir)


if __name__ == "__main__":
    main()
