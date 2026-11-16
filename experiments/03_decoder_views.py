"""03 — the instruments' read decoder on held-out data:
(a) natural pass: 8,192 eval episodes decoded at the post-block sites, each
quantity scored at its own read site (posterior at the posterior site, value
at the value site), the R² the portraits print; (b) withheld offers: 512 base
contexts × the world's held-out masks substituted into standing slot A with a
uniform disjoint partner, composed P̂(X) vs direct Q̂(X) at the display site,
pooled R² over all (offer, context) pairs and, for a world that withholds a
structural class of offers wholesale (`heldout_strata`), the per-offer median
R² over that class. Writes no figure of its own: `payload.npz` holds the
portraits and the withheld-offer scatters, which 08 composes into the
decoder-and-representation figure and `tools/fig_pretrained_suppl.py` draws
for the pretrained study (through `fig_paper` below). Unclipped decode; the
scatters quote the pooled R² of the cloud they draw."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import algebra as ALG, artifacts as A, cli, instruments as I, plotting as PL, probes as P, stats as SS, stimuli as ST  # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB   # noqa: E402
from jbo.subject import load_study                          # noqa: E402

EXPERIMENT = "03_decoder_views"
N_EVAL, N_CTX, N_SHOW, MIN_VALID = 8192, 512, 3000, 32
SMOKE_EVAL, SMOKE_CTX = 512, 64
ROLE_SHOW, ROLE_PARTNER = 1, 2


READABLE_R2 = 0.9     # figure header only: "readable from" = earliest post-block site with CV R² >= this


def readable_from(cv: dict, post: tuple, threshold: float = READABLE_R2) -> tuple[str, str]:
    """Site labels for the figure header: earliest post-block site whose CV R²
    reaches `threshold` for posterior (mean of the two factors) and for value —
    02's formation rule at a looser threshold. Falls back to the best site."""
    def earliest(scores: dict) -> str:
        for site, v in scores.items():
            if v >= threshold:
                return PL.SITE_LABELS[site]
        return PL.SITE_LABELS[max(scores, key=scores.get)]
    posterior = {s: 0.5 * (cv[s]["etaA"] + cv[s]["etaB"]) for s in post}
    value = {s: cv[s]["D"] for s in post}
    return earliest(posterior), earliest(value)


def fig_paper(dec: dict, truth: dict, r2: dict, site_label: str, sc: dict, med: dict, n_kept: int,
              read_sites: tuple[str, str], cell_r2: dict, path: Path) -> None:
    """The decoder views as one column-width figure, drawn by
    `PL.draw_decoder_row`: (a) portraits, each quantity read at its own
    site (posterior at the posterior site, value at the value site), activations
    above a half-height theory row in the same coordinates, each column headed
    by its identifier and by the earliest site the quantity is readable from
    (`read_sites` = posterior, value labels from `readable_from`), each cell
    carrying its own R² at that site (`cell_r2`); (b, c)
    held-out proposition scatters, each with the pooled R² of the cloud they
    draw. No titles: the caption carries the display site, the counts, the
    colour mappings, that P̂ is composed from the decoded factors while Q̂ is
    read directly, the natural-pass R² and the per-proposition medians (`r2`,
    `site_label`, `med`, `n_kept` stay in the signature for the caption
    writer)."""
    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(PL.PAPER_W, PL.DECODER_ROW_H))
        PL.draw_decoder_row(fig, 0.0, dec, truth, sc, read_sites, r2=SS.r2, cell_r2=cell_r2)
        fig.savefig(path, dpi=300); plt.close(fig)


def main() -> None:
    args = cli.parse("03_decoder_views: read decoder on held-out data")
    study = load_study(args.study)
    world = study.build_world()
    model, _ = study.load_actor(args.device)
    seed = study.seeds["eval"]
    ins = I.load(study, eval_seeds=[seed])
    post = tuple(s for s in ins.sites if s != P.EMBED_SITE)
    n_eval, n_ctx = (SMOKE_EVAL, SMOKE_CTX) if args.smoke else (N_EVAL, N_CTX)
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    pos = np.array([GRADE_POSITION])

    # (a) natural pass
    tokens = world.sample(n_eval, seed=seed)["tokens"]
    key = world.answer_key(tokens)
    valid = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    acts = P.collect_site_activations(model, tokens, post, pos, args.device)
    truth = {"etaA": key["etaA"][:, GRADE_POSITION], "etaB": key["etaB"][:, GRADE_POSITION], "D": key["D"][:, GRADE_READOUT]}
    eight = {s: ins.decode(s, acts[s][:, 0, :]) for s in post}
    r2_nat = {s: {"etaA": P.decode_scores(truth["etaA"][valid], ins.posterior_A(eight[s])[valid])["r2"],
                  "etaB": P.decode_scores(truth["etaB"][valid], ins.posterior_B(eight[s])[valid])["r2"],
                  "D": P.decode_scores(truth["D"][valid], ins.D(eight[s])[valid])["r2"]} for s in post}
    disp = ins.display_site
    show = np.random.default_rng([seed, ROLE_SHOW]).choice(np.flatnonzero(valid), size=min(N_SHOW, int(valid.sum())), replace=False)
    dec_show = {"etaA": ins.posterior_A(eight[ins.posterior_site])[show], "etaB": ins.posterior_B(eight[ins.posterior_site])[show],
                "D": ins.D(eight[ins.value_site])[show]}    # paper portraits: each quantity at its read site
    truth_show = {k: v[show] for k, v in truth.items()}

    # (b) held-out propositions in slot A with a uniform disjoint partner
    base = tokens[:n_ctx]
    rng = np.random.default_rng([seed, ROLE_PARTNER])
    pool = world.algebra.partner_pool()    # the family's offerable masks
    kept, skipped, r2c, r2d = [], [], [], []
    scat = {k: [] for k in ("P_true", "P_hat", "D_true", "D_hat")}
    for mask in world.algebra.heldout_masks:
        elig = pool[(pool & int(mask)) == 0]
        partners = elig[rng.integers(0, len(elig), size=n_ctx)]
        toks = ST.substitute_standing_offers(world, base, np.full(n_ctx, int(mask)), partners)
        k = world.answer_key(toks)
        assert np.allclose(k["eta"], key["eta"][:n_ctx]), "substitution moved the analytic posterior"
        ok = k["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
        if int(ok.sum()) < MIN_VALID:
            skipped.append(int(mask)); continue
        kept.append(int(mask))
        a = P.collect_site_activations(model, toks, post, pos, args.device)
        P_true, D_true = k["P"][:, GRADE_READOUT, 0], k["D"][:, GRADE_READOUT, 0]
        rc, rd = [], []
        for s in post:
            e = ins.decode(s, a[s][:, 0, :])
            P_hat, D_hat = ins.compose_P(e, np.full(n_ctx, int(mask))), ins.D(e)[:, 0]
            rc.append(SS.r2(P_true[ok], P_hat[ok])); rd.append(SS.r2(D_true[ok], D_hat[ok]))
            if s == disp:
                for kk, v in (("P_true", P_true), ("P_hat", P_hat), ("D_true", D_true), ("D_hat", D_hat)):
                    scat[kk].append(v[ok])
        r2c.append(rc); r2d.append(rd)
    r2c, r2d = np.array(r2c), np.array(r2d)                                  # [K, 4]
    scat = {k: np.concatenate(v) for k, v in scat.items()}
    i_disp = post.index(disp)
    withheld_classes = tuple(world.spec.get("heldout_strata", ()))      # classes withheld wholesale (e.g. "rect")
    by_class = {}
    if withheld_classes:
        in_class = np.isin(world.algebra.strata_of[np.array(kept)], [ALG.STRATA.index(c) for c in withheld_classes])
        assert in_class.any() and (~in_class).any(), "need held-out masks inside and outside the withheld classes"
        by_class["withheld_class"] = {"n": int(in_class.sum()), "P": float(np.median(r2c[in_class, i_disp])),
                                      "D": float(np.median(r2d[in_class, i_disp]))}
    read_sites = readable_from(ins.cv, post)
    cell_r2 = {"etaA": r2_nat[ins.posterior_site]["etaA"], "etaB": r2_nat[ins.posterior_site]["etaB"],
               "D": r2_nat[ins.value_site]["D"]}            # each portrait's fit at the site it is drawn from
    np.savez(run_dir / "payload.npz", r2_composed=r2c, r2_direct=r2d, masks_kept=np.array(kept), **scat,
             **{f"show_dec_{k}": v for k, v in dec_show.items()}, **{f"show_true_{k}": v for k, v in truth_show.items()})

    n_valid = int(valid.sum())
    results = {"headline": {
        "portrait_r2_etaA": A.headline_entry(cell_r2["etaA"], n_valid, "R²", f"factor-M posterior R² on eval episodes at the posterior site {ins.posterior_site}"),
        "portrait_r2_etaB": A.headline_entry(cell_r2["etaB"], n_valid, "R²", f"factor-N posterior R² on eval episodes at the posterior site {ins.posterior_site}"),
        "portrait_r2_D": A.headline_entry(cell_r2["D"], n_valid, "R²", f"offer-value R² on eval episodes at the value site {ins.value_site}"),
        "prop_composed_r2_pooled": A.headline_entry(SS.r2(scat["P_true"], scat["P_hat"]), len(scat["P_true"]), "R²", "pooled R² of composed P̂(X) over all withheld (offer, context) pairs at the display site"),
        "prop_direct_r2_pooled": A.headline_entry(SS.r2(scat["D_true"], scat["D_hat"]), len(scat["D_true"]), "R²", "pooled R² of direct Q̂(X) over all withheld (offer, context) pairs at the display site"),
        **{f"prop_{name}_r2_median_{label}": A.headline_entry(
               v[q], v["n"], "R²", f"median per-offer R² of {desc} at the display site over the held-out masks "
               f"of the classes withheld wholesale ({', '.join(withheld_classes)})")
           for label, v in by_class.items() for name, q, desc in (("composed", "P", "composed P̂(X)"), ("direct", "D", "direct Q̂(X)"))}},
        "withheld_classes": list(withheld_classes),
        "natural_r2_by_site": r2_nat, "n_eval": n_eval, "n_valid": n_valid, "n_ctx": n_ctx,
        "display_site": disp, "read_sites": {"posterior": read_sites[0], "value": read_sites[1]},   # paper-row header
        "portrait_sites": {"posterior": ins.posterior_site, "value": ins.value_site},   # sites the paper portraits decode at
        "masks_kept": kept, "masks_skipped": skipped,
        "prop_median_by_site": {s: {"P": float(np.median(r2c[:, i])), "D": float(np.median(r2d[:, i]))} for i, s in enumerate(post)}}
    A.write_results(run_dir, results)
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={"eval": seed, "show": [seed, ROLE_SHOW], "partner": [seed, ROLE_PARTNER]},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force,
                     figures=[], extra={"checkpoint": study.checkpoint, "instruments_sha256": ins.meta["artifact_sha256"]})
    print(json.dumps(results["headline"], indent=1)); print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
