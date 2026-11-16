"""06 — steering: every causal write to the two subspaces, one clean pass.

Posterior steering. Per factor: one target posterior per episode, drawn from
the fit pool → commanded tangent displacement → write Q·M⁻¹·dy at that
factor's scanned posterior write site → observed choice-logit shift vs the exact
counterfactual (slope); 12 isometric off-plane copies → pooled control slope;
decoded (D_A, D_B) shift at the value read site vs the commanded analytic
shift → endpoint error, posterior plane vs off-plane. Fan example: one episode
with five commands.

Value ring. A 12-direction ring in the value plane written at the
scanned value write site, per-episode safe radius; per-direction mean observed
vs commanded choice-logit shift; pooled bootstrap slope; the same ring in the
orthogonal complement (matched norm) as control; posterior read at the display
site under the value ring and under a posterior-plane ring per factor at one
spread unit (the positive control of that read). Asserts aim identity and
locality.

Mediation. The same posterior steer; split the value-write-site
displacement into its value-plane part and the complement; insertion arm
(clean + in-plane part), removal arm (steered − in-plane part), and the same
two arms with a matched-norm random part orthogonal to the value plane; slope
of each arm's choice-logit shift on the full steer's.

One figure, `paper_steering.png` (the paper's steering figure), drawn from the
arrays saved in `payload.npz`; `tools/fig_pretrained_suppl.py` redraws it for
the pretrained study from that study's payload."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jbo import artifacts as A, cli, instruments as I, plotting as PL, stats as S, steering as ST  # noqa: E402
from jbo.instruments import GRADE_POSITION, GRADE_READOUT, MIN_OFFER_PROB   # noqa: E402
from jbo.subject import load_study                          # noqa: E402

EXPERIMENT = "06_steering"
N_EVAL, SMOKE_EVAL = 16384, 512
N_CONTROLS, SMOKE_CONTROLS = 12, 3
ROLE_DONOR, ROLE_POSTERIOR_CTRL, ROLE_MEDIATION_CTRL, ROLE_ORTHO = 1, 2, 3, 1   # donor and ring streams share a seed, separate generators
FAN_COMMANDS, FAN_DOSE_BAND, FAN_CANDIDATES = 5, (0.5, 1.3), 200
FACTORS = (("A", 0, "posterior_A", "etaA", "etaB"), ("B", 1, "posterior_B", "etaB", "etaA"))
FAN_COLORS = ("#1b9e77", "#d95f02", "#e7298a", "#7570b3", "#66a61e")
WRITE_COLOR, THEORY_COLOR = "#5b3f99", "#cf6f16"       # choice quantities: purple ↔ orange
SUFF, NEC = "#4F7563", "#9B7448"                       # insertion / removal arm identities


@dataclass(frozen=True)
class Shared:
    """The clean pass and everything every section reads from it."""
    device: str
    rb: int
    pos: int
    tokens: np.ndarray        # [N, T]
    key: dict                 # world.answer_key(tokens)
    bits: np.ndarray          # [N, 2, 9] offer masks at the graded readout
    u: np.ndarray             # [N, 9]
    reward: str               # world.reward: "conditional" (D = m/P) or "mass" (D = m)
    keep_clean: np.ndarray    # [N] both offers above MIN_OFFER_PROB
    clean: dict               # run_edited output: pi [N, 10, 2], resid {site: [N, d]}
    Y_clean: np.ndarray       # [N] clean choice logit


# ---------------------------------------------------------------- posterior transplants + mediation

def pick_fan(keep: np.ndarray, y_clean: np.ndarray, y_pool: np.ndarray, eta_o: np.ndarray, bits: np.ndarray,
             u: np.ndarray, D_ana: np.ndarray, ins, reward: str = "conditional") -> dict | None:
    """Among the most interior retained episodes, the one whose five banded donors
    give the best-spread value targets (largest minimum pairwise distance)."""
    cands = np.flatnonzero(keep)[np.argsort(np.linalg.norm(y_clean[keep], axis=1))][:FAN_CANDIDATES]
    best, best_spread = None, -1.0
    for ep in cands:
        dy = y_pool - y_clean[ep][None]
        eta_new = ins.from_tangent(y_pool)
        cf = ST.counterfactual_D(eta_new, np.tile(eta_o[ep], (len(y_pool), 1)), 0, np.tile(bits[ep], (len(y_pool), 1, 1)), np.tile(u[ep], (len(y_pool), 1)), reward)
        tgt = cf["D"]
        norms = np.linalg.norm(tgt - D_ana[ep][None], axis=1)
        ok = (eta_new.min(-1) >= -1e-9) & (cf["P"].min(-1) >= MIN_OFFER_PROB) & (norms >= FAN_DOSE_BAND[0]) & (norms <= FAN_DOSE_BAND[1])
        if ok.sum() < FAN_COMMANDS:
            continue
        chosen: list[int] = []
        while len(chosen) < FAN_COMMANDS:
            ref = np.vstack([D_ana[ep][None]] + [tgt[i][None] for i in chosen])
            dmin = np.linalg.norm(tgt[:, None, :] - ref[None], axis=-1).min(1)
            dmin[~ok] = -1.0
            for i in chosen:
                dmin[i] = -1.0
            chosen.append(int(np.argmax(dmin)))
        pts = np.vstack([D_ana[ep][None], tgt[chosen]])                                   # [6, 2] start + targets
        spread = float(np.linalg.norm(pts[:, None] - pts[None], axis=-1)[np.triu_indices(len(pts), 1)].min())
        if spread > best_spread:
            best, best_spread = {"episode": int(ep), "y0": y_clean[ep], "donors": y_pool[chosen], "dy": dy[chosen], "D_cf": tgt[chosen]}, spread
    return best


def posterior_section(sh: Shared, model, ins, beta: float, pools: dict, n_ctrl: int, seed: int) -> tuple[dict, dict, dict]:
    """Per factor: the natural posterior steer, its off-plane controls, the decoded-value
    endpoint error, and the four mediation arms on the same steer.
    Returns (posterior results, mediation results, fan example)."""
    d_read, d_write = ins.value_site, ins.value_write_site
    reads = tuple(dict.fromkeys((d_read, d_write)))
    key, pos = sh.key, sh.pos
    D_ana = key["D"][:, GRADE_READOUT]
    D_dec_clean = ins.D(ins.decode(d_read, sh.clean["resid"][d_read]))
    h_clean = sh.clean["resid"][d_write]
    gap_clean = D_ana[:, 0] - D_ana[:, 1]
    Q = ins.planes["value"].Q
    donor_rng = np.random.default_rng([seed, ROLE_DONOR])
    med_rng = np.random.default_rng([seed, ROLE_MEDIATION_CTRL])

    bel, med, fan_data = {}, {}, None
    for name, fi, plane_key, eta_key, other_key in FACTORS:
        b_site = ins.posterior_write_site[name]
        eta_f, eta_o = key[eta_key][:, pos], key[other_key][:, pos]
        y_clean = ins.to_tangent(eta_f)
        plane = ins.planes[plane_key]
        donors = ST.draw_natural_donors(y_clean, pools[name], donor_rng)
        delta = plane.aim(donors["dy"])                                          # [N, d]
        assert np.allclose(delta @ plane.probe[1:], donors["dy"], atol=1e-6), "aim identity"
        eta_new = ins.from_tangent(y_clean + donors["dy"])
        assert eta_new.min() >= -1e-9
        cf = ST.counterfactual_D(eta_new, eta_o, fi, sh.bits, sh.u, sh.reward)
        keep = sh.keep_clean & (cf["P"].min(-1) >= MIN_OFFER_PROB)
        episode_idx = np.flatnonzero(keep)
        ctrl_episode_idx = np.tile(episode_idx, n_ctrl)
        steered = ST.run_edited(model, sh.tokens, [(b_site, pos, delta)], reads, pos, sh.device, sh.rb)
        dY = ST.policy_logit(steered["pi"][:, GRADE_READOUT]) - sh.Y_clean
        dZ = beta * ((cf["D"][:, 0] - cf["D"][:, 1]) - gap_clean)
        transfer = S.slope(dZ[keep], dY[keep])
        dD_dec = ins.D(ins.decode(d_read, steered["resid"][d_read])) - D_dec_clean
        dD_ana = cf["D"] - D_ana
        err_nat = np.linalg.norm((dD_dec - dD_ana)[keep], axis=1)

        # off-plane controls: the same commands copied isometrically into planes orthogonal to the posterior plane
        axes = ST.orthogonal_directions(plane.Q, 2 * n_ctrl, np.random.default_rng([seed, ROLE_POSTERIOR_CTRL, fi]))
        ctrl_dY, ctrl_err = [], []
        for c in range(n_ctrl):
            dc = ST.isometric_plane_copy(delta, plane.Q, axes[:, 2 * c: 2 * c + 2])
            arm = ST.run_edited(model, sh.tokens, [(b_site, pos, dc)], (d_read,), pos, sh.device, sh.rb)
            ctrl_dY.append((ST.policy_logit(arm["pi"][:, GRADE_READOUT]) - sh.Y_clean)[keep])
            dDc = ins.D(ins.decode(d_read, arm["resid"][d_read])) - D_dec_clean
            ctrl_err.append(np.linalg.norm((dDc - dD_ana)[keep], axis=1))
        ctrl_dY, ctrl_err = np.concatenate(ctrl_dY), np.concatenate(ctrl_err)
        control = S.slope(np.tile(dZ[keep], n_ctrl), ctrl_dY, groups=ctrl_episode_idx, n_boot=300)
        bel[name] = {"n_kept": int(keep.sum()), "transfer": transfer, "control": control,
                     "episode_idx": episode_idx, "ctrl_episode_idx": ctrl_episode_idx, "keep": keep,
                     # Donor records retain all input episodes; result rows follow episode_idx.
                     "donor_idx": donors["donor_idx"], "donor_dy": donors["dy"],
                     "endpoint_error_median_natural": float(np.median(err_nat)),
                     "endpoint_error_median_control": float(np.median(ctrl_err)),
                     "dZ": dZ[keep], "dY": dY[keep], "ctrl_dZ": np.tile(dZ[keep], n_ctrl), "ctrl_dY": ctrl_dY,
                     "err_nat": err_nat, "err_ctrl": ctrl_err}
        print(f"  posterior {name}: transfer {transfer['slope']:.3f}±{transfer['se']:.3f} (R² {transfer['r2']:.3f}, n {transfer['n']}); "
              f"control {control['slope']:.3f}±{control['se']:.3f}; endpoint err {np.median(err_nat):.3f} vs {np.median(ctrl_err):.3f}", flush=True)

        # mediation: split the value-write-site displacement of the same steer on the value plane
        dh = steered["resid"][d_write] - h_clean
        parts = ST.split_on_plane(dh, Q)
        ctrl = ST.matched_random_part(dh, Q, med_rng)
        arms = {"insertion": [(d_write, pos, parts["in_plane"])],
                "removal": [(b_site, pos, delta), (d_write, pos, -parts["in_plane"])],
                "insertion_control": [(d_write, pos, ctrl)],
                "removal_control": [(b_site, pos, delta), (d_write, pos, -ctrl)]}
        fits, dYs = {}, {}
        for arm, edits in arms.items():
            r = ST.run_edited(model, sh.tokens, edits, (), pos, sh.device, sh.rb)
            dYs[arm] = ST.policy_logit(r["pi"][:, GRADE_READOUT]) - sh.Y_clean
            fits[arm] = S.slope(dY[keep], dYs[arm][keep])
            print(f"  mediation {name} {arm:18s} carried fraction {fits[arm]['slope']:.3f} ± {fits[arm]['se']:.3f} (R² {fits[arm]['r2']:.3f})", flush=True)
        med[name] = {"fits": fits, "n_kept": int(keep.sum()),
                     "episode_idx": episode_idx,
                     "full": dY[keep], "insertion": dYs["insertion"][keep], "removal": dYs["removal"][keep]}

        if fi == 0:
            fan = pick_fan(keep, y_clean, pools[name], eta_o, sh.bits, sh.u, D_ana, ins, sh.reward)
            assert fan is not None, "no fan candidate satisfies the dose band"
            ep = fan["episode"]
            st = ST.run_edited(model, np.tile(sh.tokens[ep], (FAN_COMMANDS, 1)), [(b_site, pos, plane.aim(fan["dy"]))], (d_read,), pos, sh.device, sh.rb)
            fan["D0"] = D_dec_clean[ep]
            fan["D1"] = ins.D(ins.decode(d_read, st["resid"][d_read]))
            fan["D_cloud"] = D_dec_clean[sh.keep_clean]
            fan_data = fan
    return bel, med, fan_data


# ---------------------------------------------------------------- value ring

def ring_section(sh: Shared, model, ins, beta: float, seed: int) -> dict:
    """The value-plane ring at the write site with its orthogonal-complement control,
    the decoded posterior under it, and the posterior-plane ring that calibrates that read."""
    d_write, disp, pos = ins.value_write_site, ins.display_site, sh.pos
    key, n = sh.key, len(sh.tokens)
    D_clean = key["D"][:, GRADE_READOUT]
    radius = ST.safe_radii(D_clean)
    keep = sh.keep_clean & (radius >= ST.MIN_RADIUS)
    angles, directions = ST.ring_directions()                                  # raw (D_A, D_B) directions; 0° = raise D_A
    plane = ins.planes["value"]
    controls = ST.orthogonal_directions(plane.Q, ST.N_DIRECTIONS, np.random.default_rng([seed, ROLE_ORTHO]))
    e0 = ins.decode(disp, sh.clean["resid"][disp])
    bel0 = np.concatenate([ins.to_tangent(ins.posterior_A(e0)), ins.to_tangent(ins.posterior_B(e0))], 1)    # [N, 4]

    def posterior_shift(resid: np.ndarray) -> np.ndarray:
        e = ins.decode(disp, resid)
        return np.concatenate([ins.to_tangent(ins.posterior_A(e)), ins.to_tangent(ins.posterior_B(e))], 1) - bel0

    xs, dys, dys_c, bel_d = [], [], [], []
    aim_err, loc_err = 0.0, 0.0
    for k, d in enumerate(directions):
        dy = radius[:, None] * d
        delta = plane.aim(dy)
        aim_err = max(aim_err, float(np.abs(delta @ plane.probe[1:] - dy).max()))
        delta_c = np.linalg.norm(delta, axis=1)[:, None] * controls[:, k][None, :]
        arm = ST.run_edited(model, sh.tokens, [(d_write, pos, delta)], (disp,), pos, sh.device, sh.rb)
        arm_c = ST.run_edited(model, sh.tokens, [(d_write, pos, delta_c)], (), pos, sh.device, sh.rb)
        for a in (arm, arm_c):
            loc_err = max(loc_err, float(np.abs(a["pi"][:, :GRADE_READOUT] - sh.clean["pi"][:, :GRADE_READOUT]).max()))
        xs.append(beta * (dy[keep, 0] - dy[keep, 1]))
        dys.append((ST.policy_logit(arm["pi"][:, GRADE_READOUT]) - sh.Y_clean)[keep])
        dys_c.append((ST.policy_logit(arm_c["pi"][:, GRADE_READOUT]) - sh.Y_clean)[keep])
        bel_d.append(posterior_shift(arm["resid"][disp])[keep])
        print(f"  ring {angles[k]:3.0f}°: commanded {xs[-1].mean():+.3f}  observed {dys[-1].mean():+.3f}  control {dys_c[-1].mean():+.3f}", flush=True)
    assert aim_err < 1e-8, f"aim identity failed ({aim_err:.3g})"
    assert loc_err < 1e-5, f"locality failed ({loc_err:.3g})"
    x, y, yc = np.stack(xs), np.stack(dys), np.stack(dys_c)                    # [12, k]
    groups = np.tile(np.flatnonzero(keep), ST.N_DIRECTIONS)
    slope_in = S.slope(x.ravel(), y.ravel(), groups=groups, n_boot=300)
    slope_ctrl = S.slope(x.ravel(), yc.ravel(), groups=groups, n_boot=300)
    post_under_value = np.linalg.norm(np.concatenate(bel_d), axis=1)           # joint 4-d tangent shift

    # decoder-sensitivity rows: a posterior-plane ring per factor at one spread unit, written at the posterior site
    bel_own, bel_implied = [], []
    for name, fi, plane_key, eta_key, other_key in FACTORS:
        bplane, unit = ins.planes[plane_key], ins.posterior_unit[name]
        eta_f, eta_o = key[eta_key][:, pos], key[other_key][:, pos]
        y_clean = ins.to_tangent(eta_f)
        for d in directions:
            dy = np.broadcast_to(unit * d, (n, 2)).copy()
            eta_new = ins.from_tangent(y_clean + dy)
            cf = ST.counterfactual_D(eta_new, eta_o, fi, sh.bits, sh.u, sh.reward)
            ok = sh.keep_clean & (eta_new.min(-1) >= -1e-12) & (cf["P"].min(-1) >= MIN_OFFER_PROB)
            arm = ST.run_edited(model, sh.tokens, [(ins.posterior_write_site[name], pos, bplane.aim(dy))], (disp,), pos, sh.device, sh.rb)
            shift = posterior_shift(arm["resid"][disp])[ok]
            bel_own.append(np.linalg.norm(shift[:, 2 * fi: 2 * fi + 2], axis=1))
            bel_implied.append(np.full(int(ok.sum()), unit))
    bel_own, bel_implied = np.concatenate(bel_own), np.concatenate(bel_implied)
    rows = {"posterior": {"obs": bel_own, "rms": float(np.sqrt((bel_own ** 2).mean())), "implied": float(bel_implied.mean())},
            "value": {"obs": post_under_value, "rms": float(np.sqrt((post_under_value ** 2).mean())), "implied": 0.0}}
    print(f"  ring slope {slope_in['slope']:.3f}±{slope_in['se']:.3f}  control {slope_ctrl['slope']:.3f}  posterior RMS value-ring {rows['value']['rms']:.4f} "
          f"posterior-ring {rows['posterior']['rms']:.3f} (ref {rows['posterior']['implied']:.3f})", flush=True)
    return {"angles": angles, "directions": directions, "x": x, "y": y, "y_control": yc, "slope": slope_in, "control": slope_ctrl,
            "episode_idx": np.flatnonzero(keep),
            "rows": rows, "n_kept": int(keep.sum()),
            "D_cloud": D_clean[keep], "aim_error": aim_err, "locality_error": loc_err}


# ---------------------------------------------------------------- paper figure

PAPER_H = 2.82        # inches; width, text size and rc come from jbo.plotting
IDENTITY = {"color": "0.4", "lw": 0.6, "ls": (0, (3, 2)), "zorder": 0}
ZERO = {"color": "0.4", "lw": 0.6, "ls": (0, (1, 1.5)), "zorder": 0}
ARROW = {"arrowstyle": "-|>", "lw": 0.9, "mutation_scale": 6, "shrinkA": 0, "shrinkB": 0}


def _symmetric_ticks(lim: float) -> list[float]:
    t = float(np.floor(lim)) if lim >= 1.5 else round(lim / 2, 1)
    return [-t, 0.0, t]


def _key(ax, fs: float, rows: list[tuple[str, str]], x: float = 0.03, y: float = 0.93, dy: float = 0.10) -> None:
    """Upper-left glyph key in axes fraction: each row is ("arrow" | "x" | "dot", label), drawn in dark grey with a white halo."""
    kc = "0.3"
    for k, (glyph, name) in enumerate(rows):
        yk = y - dy * k
        if glyph == "arrow":
            ax.annotate("", xy=(x + 0.12, yk), xytext=(x, yk), xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops={**ARROW, "color": kc}, zorder=7)
        elif glyph == "x":
            ax.scatter([x + 0.06], [yk], marker="x", s=12, color=kc, linewidths=0.8, transform=ax.transAxes, zorder=7, clip_on=False)
        else:
            assert glyph == "dot", glyph
            ax.scatter([x + 0.06], [yk], s=9, color="#1e2430", transform=ax.transAxes, zorder=7, clip_on=False)
        ax.text(x + 0.16, yk, name, color=kc, fontsize=fs - 1.5, ha="left", va="center", transform=ax.transAxes, zorder=7,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])


def fig_paper(z: dict, path: Path, slopes: dict[str, float] | None = None) -> None:
    """Two rows at column width, redrawn from `payload.npz`. Top, posterior
    steering (factor A): (a) observed vs exact choice-logit shift over the
    pooled off-plane control; (b) one episode's five posterior commands on the
    factor simplex and, in the paired panel, the value they imply (arrows)
    against the value decoded after the steer (crosses); (c) the endpoint error over the
    population, natural steer vs off-plane control. Bottom, value steering:
    (d) the twelve-direction write compass on the decoded value map and,
    under the same letter, the per-direction mean choice-logit shift ± 95 % CI
    against theory and the orthogonal-complement control; (e) insertion and
    removal arms on the full posterior steer. When `slopes` is given (keys
    transfer/transfer_ctrl, ring/ring_ctrl, insertion/removal, and the panel-c
    medians err_nat/err_ctrl), panels a, c, d, and e print the treatment and
    control numbers beside the data, coloured to match their series."""
    fs = PL.PAPER_FS
    h = 0.95
    y1, y2 = PAPER_H - 0.14 - h, 0.34            # row baselines

    def slope_column(ax, rows: list[tuple[str, str]], x: float, y: float, ha: str, dy: float = 0.11, label: str = "slope") -> None:
        """Per-panel numeric readout: each row is (slopes key, series colour), printed
        as `<label> <value>` in the series colour with a white halo, axes fraction."""
        if slopes is None:
            return
        for k, (key, c) in enumerate(rows):
            ax.text(x, y - dy * k, f"{label} {slopes[key]:.3f}", color=c, fontsize=fs - 1.5, ha=ha,
                    va="center", transform=ax.transAxes, zorder=7,
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])

    with plt.rc_context(PL.PAPER_RC):
        fig = plt.figure(figsize=(PL.PAPER_W, PAPER_H))
        letter = lambda x_in, y_in, s: fig.text(x_in / PL.PAPER_W, y_in / PAPER_H, s, fontsize=fs + 1, fontweight="bold", ha="left", va="top")
        top1, top2 = PAPER_H - 0.02, y2 + h + 0.12

        # (a) posterior transplant → choice -------------------------------------------
        ax = PL.axes_in(fig, 0.40, y1, h, h)
        dZ, dY = z["posterior_A_dZ"], z["posterior_A_dY"]
        lim = float(np.percentile(np.abs(dZ), 99.5))
        ax.scatter(z["posterior_A_ctrl_dZ"], z["posterior_A_ctrl_dY"], s=0.6, color="0.6", alpha=0.05, linewidths=0, rasterized=True)
        ax.scatter(dZ, dY, s=0.8, color=PL.POSTERIOR_A_COLOR, alpha=0.2, linewidths=0, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], **IDENTITY)
        ticks = _symmetric_ticks(lim)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=ticks, yticks=ticks, xlabel="predicted logit shift", ylabel="observed shift")
        for k, (name, c) in enumerate((("off-plane control", "0.45"), ("posterior-plane steer", PL.POSTERIOR_A_COLOR))):
            ax.text(0.97, 0.03 + 0.10 * k, name, color=c, fontsize=fs - 1.5, ha="right", va="bottom", transform=ax.transAxes)
        slope_column(ax, [("transfer", PL.POSTERIOR_A_COLOR), ("transfer_ctrl", "0.45")], x=0.05, y=0.92, ha="left")
        PL.frame(ax); letter(0.02, top1, "a")

        # (b) fan: five posterior commands on the simplex ----------------------------------
        ax = PL.axes_in(fig, 1.55, y1 - 0.08, h, h + 0.08)
        pool = z["posterior_fan_pool_xy"]
        ax.scatter(pool[:, 0], pool[:, 1], s=0.5, color="0.8", linewidths=0, rasterized=True, zorder=1)
        edge = np.vstack([PL.TRI, PL.TRI[:1]]); ax.plot(edge[:, 0], edge[:, 1], color="0.55", lw=0.9, zorder=2)
        p0 = z["posterior_fan_p0"]
        for c, p1 in zip(FAN_COLORS, z["posterior_fan_p1"]):
            ax.annotate("", xy=p1, xytext=p0, arrowprops={**ARROW, "color": c}, zorder=4)
        ax.scatter([p0[0]], [p0[1]], s=9, color="#1e2430", zorder=5)
        for (x, y), name, dx, dy, ha in ((PL.TRI[0], "s0", -0.06, -0.05, "center"), (PL.TRI[1], "s1", 0.06, -0.05, "center"), (PL.TRI[2], "s2", 0.05, 0.0, "left")):
            ax.text(x + dx, y + dy, name, ha=ha, va="center", fontsize=fs, fontweight="bold", color="0.25")
        ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 1.25); ax.set_aspect("equal"); ax.axis("off")   # headroom for the two-row key above the apex
        _key(ax, fs, [("arrow", "posterior target"), ("dot", "unsteered")], y=0.905)                    # same figure height as the paired panel's key
        letter(1.45, top1, "b")

        # (b, right) fan: the decoded value follows --------------------------------------
        sq = 0.80
        ax = PL.axes_in(fig, 2.90 + (h - sq) / 2, y1 + (h - sq) / 2, sq, sq)
        cloud = z["posterior_fan_D_cloud"]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=0.5, color="0.8", linewidths=0, rasterized=True, zorder=1)
        D0 = z["posterior_fan_D0"]
        for c, d1, dcf in zip(FAN_COLORS, z["posterior_fan_D1"], z["posterior_fan_D_cf"]):
            ax.annotate("", xy=dcf, xytext=D0, arrowprops={**ARROW, "color": c}, zorder=4)     # the value the command implies
            ax.scatter([d1[0]], [d1[1]], marker="x", s=12, color=c, linewidths=0.8, zorder=5)   # decoded after the steer
        ax.scatter([D0[0]], [D0[1]], s=9, color="#1e2430", zorder=6)
        _key(ax, fs, [("arrow", "implied value"), ("x", "decoded"), ("dot", "unsteered")])
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[-1, 0, 1], yticks=[-1, 0, 1], xlabel="decoded $Q_A$", ylabel="decoded $Q_B$")
        PL.frame(ax)

        # (c) endpoint error over the population -----------------------------------------
        ax = PL.axes_in(fig, 4.32, y1, 1.10, h)
        err_nat, err_ctrl = z["posterior_A_err_nat"], z["posterior_A_err_ctrl"]
        bins = np.linspace(0, float(np.percentile(err_ctrl, 99)), 40)
        ax.hist(err_nat, bins=bins, density=True, color=PL.POSTERIOR_A_COLOR, alpha=0.75, label="posterior-plane steer")
        ax.hist(err_ctrl, bins=bins, density=True, color="0.6", alpha=0.6, label="off-plane control")
        ax.set(xlim=(0, bins[-1]), xlabel="value endpoint error", ylabel="density")
        ax.set_ylim(0, 1.45 * ax.get_ylim()[1])
        ax.legend(frameon=False, loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
        if slopes is not None and "err_nat" in slopes:
            slope_column(ax, [("err_nat", PL.POSTERIOR_A_COLOR), ("err_ctrl", "0.45")], x=0.97, y=0.60, ha="right", label="median error")
        PL.frame(ax); letter(3.80, top1, "c")

        # (d) write compass on the value map ---------------------------------------------
        sq = 0.80
        ax = PL.axes_in(fig, 0.40, y2 + (h - sq) / 2, sq, sq)
        Dc = z["ring_D_cloud"]
        gap = Dc[:, 0] - Dc[:, 1]
        scale = float(np.quantile(np.abs(gap), 0.98))
        colors = plt.get_cmap(PL.GAP_CMAP)(0.5 * (np.clip(gap / scale, -1, 1) + 1))
        ax.scatter(Dc[:, 0], Dc[:, 1], c=colors, s=0.5, alpha=0.55, linewidths=0, rasterized=True)
        for d in z["ring_directions"]:
            ax.annotate("", xy=tuple(0.5 * d), xytext=(0, 0), arrowprops={**ARROW, "color": WRITE_COLOR,
                        "path_effects": [patheffects.withStroke(linewidth=1.8, foreground="white")]})
        ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xticks=[-1, 0, 1], yticks=[-1, 0, 1], xlabel="$Q_A$", ylabel="$Q_B$")
        ax.set_aspect("equal"); PL.frame(ax); letter(0.02, top2, "d")

        # (d, right) value ring → choice ------------------------------------------------------
        ax = PL.axes_in(fig, 2.00, y2, 1.50, h)               # centred between the compass and the mediation panel
        angles, x, y, yc = z["ring_angles"], z["ring_x"], z["ring_y"], z["ring_y_control"]
        order = np.argsort(angles)
        a, pred, obs, ctrl = angles[order], x.mean(1)[order], y.mean(1)[order], yc.mean(1)[order]
        err = (1.96 * y.std(1, ddof=1) / np.sqrt(y.shape[1]))[order]
        tx = np.r_[a[-1] - 360, a, a[0] + 360]; ty = np.r_[pred[-1], pred, pred[0]]
        ax.axhline(0, **ZERO)
        # theory as a translucent band under the data: the two curves coincide, so a dashed line would be hidden
        ax.plot(tx, ty, color=THEORY_COLOR, lw=2.6, alpha=0.5, solid_capstyle="round", label=r"predicted $\beta\,\Delta(Q_A - Q_B)$", zorder=2)
        ax.errorbar(a, obs, yerr=err, color=WRITE_COLOR, marker="o", ms=2.6, lw=0.8, capsize=1.2, label="value-plane write", zorder=3)
        ax.plot(a, ctrl, color="0.55", marker="s", ms=2.4, lw=0.8, label="orthogonal-complement write")
        top = float(np.abs(np.r_[obs + err, pred]).max())
        ax.set(xlim=(0, 360), xticks=np.arange(0, 361, 90), ylim=(-1.15 * top, 2.45 * top),
               xlabel="write direction (°)", ylabel="mean logit shift")
        ax.legend(frameon=False, loc="upper left", handlelength=1.3, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.25)
        slope_column(ax, [("ring", WRITE_COLOR), ("ring_ctrl", "0.55")], x=0.97, y=0.17, ha="right")
        PL.frame(ax)                                         # shares letter (d) with the compass

        # (e) mediation -------------------------------------------------------------------
        ax = PL.axes_in(fig, 4.32, y2, h, h)
        full, ins_, rem = z["mediation_A_full"], z["mediation_A_insertion"], z["mediation_A_removal"]
        lim = float(np.percentile(np.abs(np.concatenate([full, ins_, rem])), 99.5))
        take = np.random.default_rng(290071).choice(len(full), min(3000, len(full)), replace=False)
        ax.scatter(full[take], ins_[take], s=0.8, color=SUFF, alpha=0.3, linewidths=0, rasterized=True)
        ax.scatter(full[take], rem[take], s=0.8, color=NEC, alpha=0.3, linewidths=0, rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], **IDENTITY); ax.axhline(0, **ZERO)
        ticks = _symmetric_ticks(lim)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=ticks, yticks=ticks, xlabel="posterior-steer logit shift", ylabel="edited-pass logit shift")
        for txt, c, ytxt in (("clean pass\n+ value part", SUFF, 0.89), ("steered pass\n− value part", NEC, 0.67)):
            ax.text(0.05, ytxt, txt, color=c, fontsize=fs - 1.5, ha="left", va="center", transform=ax.transAxes, zorder=7,
                    linespacing=1.15, path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])
        slope_column(ax, [("insertion", SUFF), ("removal", NEC)], x=0.97, y=0.15, ha="right")
        PL.frame(ax); letter(3.80, top2, "e")

        fig.savefig(path, dpi=300); plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    args = cli.parse("06_steering: posterior transplants, value ring, mediation")
    study = load_study(args.study)
    world = study.build_world()
    model, _ = study.load_actor(args.device)
    seed = study.seeds["eval"]
    ins = I.load(study, eval_seeds=[seed])
    d_read, d_write, disp, pos = ins.value_site, ins.value_write_site, ins.display_site, GRADE_POSITION
    assert not ins.reads_only, f"{study.name} has reads-only instruments (no write planes): steering needs a trained choice"
    b_sites = dict(ins.posterior_write_site)
    for name, b_site in b_sites.items():
        assert int(b_site.split(".")[1]) < int(d_read.split(".")[1]), \
            f"posterior {name} write site {b_site} must be upstream of the value read site {d_read}"
        assert int(b_site.split(".")[1]) < int(d_write.split(".")[1]), \
            f"posterior {name} steer at {b_site} must be strictly upstream of the value write site {d_write}"
    rb = world.vocab.response_base
    n, n_ctrl = (SMOKE_EVAL, SMOKE_CONTROLS) if args.smoke else (N_EVAL, N_CONTROLS)
    run_dir = A.new_run_dir(args.study, EXPERIMENT, label=args.label or ("smoke" if args.smoke else None), force=args.force)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    beta = world.beta

    # donor pool: the fit episodes' analytic posteriors at readout 5 (regenerated from the fit seed)
    fit_key = world.answer_key(world.sample(int(ins.meta["n_fit"]), seed=ins.fit_seed)["tokens"])
    pools = {name: ins.to_tangent(fit_key[eta_key][:, pos]) for name, _, _, eta_key, _ in FACTORS}

    tokens = world.sample(n, seed=seed)["tokens"]
    key = world.answer_key(tokens)
    keep_clean = key["P"][:, GRADE_READOUT].min(-1) >= MIN_OFFER_PROB
    clean = ST.run_edited(model, tokens, [], tuple(dict.fromkeys((d_read, d_write, disp))), pos, args.device, rb)
    sh = Shared(device=args.device, rb=rb, pos=pos, tokens=tokens, key=key, bits=key["bits_ro"][:, GRADE_READOUT], u=key["u"], reward=world.reward,
                keep_clean=keep_clean, clean=clean, Y_clean=ST.policy_logit(clean["pi"][:, GRADE_READOUT]))

    bel, med, fan = posterior_section(sh, model, ins, beta, pools, n_ctrl, seed)
    ring = ring_section(sh, model, ins, beta, seed)

    arrays = {f"posterior_{k}_{kk}": v for k, r in bel.items() for kk, v in r.items() if isinstance(v, np.ndarray)}
    arrays.update({f"mediation_{k}_{kk}": r[kk] for k, r in med.items() for kk in ("full", "insertion", "removal", "episode_idx")})
    arrays.update({f"ring_{kk}": ring[kk] for kk in ("angles", "directions", "x", "y", "y_control", "D_cloud", "episode_idx")})
    arrays.update({"eval_tokens": tokens, "fit_pool_A": pools["A"], "fit_pool_B": pools["B"]})
    arrays.update({"ring_posterior_under_value": ring["rows"]["value"]["obs"], "ring_posterior_own": ring["rows"]["posterior"]["obs"]})
    arrays.update({"posterior_fan_pool_xy": PL.bary(ins.from_tangent(pools["A"])), "posterior_fan_p0": PL.bary(ins.from_tangent(fan["y0"][None]))[0],
                   "posterior_fan_p1": PL.bary(ins.from_tangent(fan["donors"])),
                   **{f"posterior_fan_{k}": fan[k] for k in ("D0", "D1", "D_cf", "D_cloud")}})
    np.savez(run_dir / "payload.npz", **arrays)

    figs = run_dir / "figures"
    fig_paper(arrays, figs / "paper_steering.png",
              slopes={"transfer": bel["A"]["transfer"]["slope"], "transfer_ctrl": bel["A"]["control"]["slope"],
                      "err_nat": bel["A"]["endpoint_error_median_natural"], "err_ctrl": bel["A"]["endpoint_error_median_control"],
                      "ring": ring["slope"]["slope"], "ring_ctrl": ring["control"]["slope"],
                      "insertion": med["A"]["fits"]["insertion"]["slope"], "removal": med["A"]["fits"]["removal"]["slope"]})
    figure_paths = [figs / "paper_steering.png"]

    head = {}
    for name in ("A", "B"):
        r = bel[name]
        head[f"transfer_slope_{name}"] = A.headline_entry(r["transfer"]["slope"], r["transfer"]["n"], "slope", f"observed vs exact choice-logit shift, factor {name} (SE {r['transfer']['se']:.3f})")
        head[f"control_slope_{name}"] = A.headline_entry(r["control"]["slope"], r["control"]["n"], "slope", f"pooled slope of {n_ctrl} isometric off-plane copies, factor {name} (SE {r['control']['se']:.3f})")
        head[f"endpoint_error_natural_{name}"] = A.headline_entry(r["endpoint_error_median_natural"], r["n_kept"], "utility", f"median decoded-vs-commanded value-shift error, factor {name}")
        head[f"endpoint_error_control_{name}"] = A.headline_entry(r["endpoint_error_median_control"], r["n_kept"] * n_ctrl, "utility", f"same, off-plane control, factor {name}")
    rows = ring["rows"]
    head["ring_slope"] = A.headline_entry(ring["slope"]["slope"], ring["slope"]["n"], "slope", f"pooled slope, observed on commanded choice-logit shift, 12 value-plane directions (SE {ring['slope']['se']:.3f})")
    head["ring_control_slope"] = A.headline_entry(ring["control"]["slope"], ring["control"]["n"], "slope", f"same for the matched-norm orthogonal-complement ring (SE {ring['control']['se']:.3f})")
    head["posterior_rms_shift_value_ring"] = A.headline_entry(rows["value"]["rms"], len(rows["value"]["obs"]), "tangent", "RMS decoded posterior shift under the value ring (reference 0)")
    head["posterior_rms_shift_posterior_ring"] = A.headline_entry(rows["posterior"]["rms"], len(rows["posterior"]["obs"]), "tangent", f"RMS decoded own-factor posterior shift under a posterior-plane ring (reference {rows['posterior']['implied']:.3f})")
    for name in ("A", "B"):
        for arm, desc in (("insertion", "clean + value-plane part"), ("removal", "steered − value-plane part")):
            f = med[name]["fits"][arm]
            head[f"{arm}_slope_{name}"] = A.headline_entry(f["slope"], f["n"], "slope", f"factor {name}: {desc}, slope on the full steer (SE {f['se']:.3f})")

    per_dir = [{"angle": float(a), "commanded": float(ring["x"][k].mean()), "observed": float(ring["y"][k].mean()),
                "ci95": float(1.96 * ring["y"][k].std(ddof=1) / np.sqrt(ring["y"].shape[1])), "control": float(ring["y_control"][k].mean())}
               for k, a in enumerate(ring["angles"])]
    scalar = lambda d: {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
    A.write_results(run_dir, {
        "headline": head,
        "posterior": {"factors": {k: scalar(r) for k, r in bel.items()},
                   "fan": {"episode": fan["episode"], "D_cf": fan["D_cf"].tolist(), "D1": fan["D1"].tolist()}},
        "ring": {"per_direction": per_dir, "n_kept": ring["n_kept"], "aim_error": ring["aim_error"], "locality_error": ring["locality_error"],
                 "caveat": ("value write site equals the display site: the posterior read is taken at the write site itself" if d_write == disp else None)},
        "mediation": {"factors": {k: {"fits": r["fits"], "n_kept": r["n_kept"]} for k, r in med.items()}},
        "sites": {"posterior_read": ins.posterior_site, "posterior_write": b_sites, "value_read": d_read, "value_write": d_write, "display": disp},
        "n_eval": n, "n_controls": n_ctrl})
    A.write_manifest(run_dir, experiment=EXPERIMENT, study=args.study, argv=args.argv,
                     seeds={"eval": seed, "donor": [seed, ROLE_DONOR], "posterior_control": [seed, ROLE_POSTERIOR_CTRL, "factor"],
                            "mediation_control": [seed, ROLE_MEDIATION_CTRL], "ring_orthogonal": [seed, ROLE_ORTHO], "fit_pool": ins.fit_seed},
                     spec_hashes=study.spec_hashes, start_time=t0, device=args.device, force_used=args.force, figures=figure_paths,
                     extra={"checkpoint": study.checkpoint, "instruments_sha256": ins.meta["artifact_sha256"]})
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
