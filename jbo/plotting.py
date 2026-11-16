"""One figure style: posterior = RGB simplex / blue family, value = red family."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])   # s0, s1, s2 vertices
POSTERIOR_COLOR, POSTERIOR_A_COLOR, POSTERIOR_B_COLOR = "#1f4e8c", "#1f5fbf", "#7fb2e5"
VALUE_COLOR = "#c0392b"
GAP_CMAP, GAP_NORM = "RdGy_r", Normalize(vmin=-2.0, vmax=2.0)
SITE_LABELS = {"blocks.0.hook_resid_pre": "embed", **{f"blocks.{l}.hook_resid_post": f"L{l}" for l in range(64)}}


def bary(p: np.ndarray) -> np.ndarray:
    """Posterior [..., 3] -> 2-D simplex coordinates (unclipped)."""
    return np.asarray(p, np.float64) @ TRI


def posterior_panel(ax, p_plot: np.ndarray, p_colour: np.ndarray, vertex_labels: bool = False,
                 s: float = 2.5, fs: float = 9.0) -> None:
    """Simplex cloud: points at bary(p_plot), coloured by p_colour as RGB; `fs` sizes the vertex labels."""
    edge = np.vstack([TRI, TRI[:1]])
    ax.plot(edge[:, 0], edge[:, 1], color="0.55", lw=0.9, zorder=1)
    xy = bary(p_plot)
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=np.clip(p_colour, 0, 1), alpha=0.75, linewidths=0, zorder=2)
    if vertex_labels:
        for (x, y), name, dx, dy in ((TRI[0], "s0", -0.06, -0.05), (TRI[1], "s1", 0.06, -0.05),
                                     (TRI[2], "s2", 0.0, 0.07)):
            ax.text(x + dx, y + dy, name, ha="center", va="center", fontsize=fs, fontweight="bold", color="0.25")
    ax.set_xlim(-0.25, 1.25); ax.set_ylim(-0.31, 1.19); ax.set_aspect("equal"); ax.axis("off")


def value_panel(ax, D: np.ndarray, gap: np.ndarray, labeled: bool = False, s: float = 2.5,
                 fs: float = 9.0) -> None:
    """(D_A, D_B) cloud coloured by the analytic gap on the fixed ±2 red/grey scale; `fs` sizes the axis labels."""
    ax.scatter(D[:, 0], D[:, 1], s=s, c=gap, cmap=GAP_CMAP, norm=GAP_NORM, alpha=0.8, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
    if labeled:
        ax.set_xlabel("Q(A)", fontsize=fs); ax.set_ylabel("Q(B)", fontsize=fs)


def posterior_key(ax, fs: float = 9.0, title: bool = True) -> None:
    """RGB simplex legend; `fs` sizes all text, `title=False` drops the two-line heading."""
    # exact fill: an RGBA raster whose pixel colour is its own barycentric weight, transparent outside
    n = 400
    xs, ys = np.linspace(-0.2, 1.2, n), np.linspace(-0.18, 1.05, n)
    X, Y = np.meshgrid(xs, ys)                                   # [n, n] simplex-plane coordinates
    w2 = Y / TRI[2, 1]                                           # weight on s2 (apex)
    w1 = X - 0.5 * w2                                            # weight on s1
    w0 = 1.0 - w1 - w2                                           # weight on s0
    W = np.stack([w0, w1, w2], axis=-1)                          # [n, n, 3]
    inside = (W >= -1e-9).all(axis=-1)
    img = np.concatenate([np.clip(W, 0, 1), inside[..., None].astype(np.float64)], axis=-1)   # [n, n, 4] RGBA
    ax.imshow(img, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), interpolation="bilinear", zorder=1)
    edge = np.vstack([TRI, TRI[:1]])
    ax.plot(edge[:, 0], edge[:, 1], color="0.55", lw=0.9, zorder=2)
    for (x, y), name, dx, dy in ((TRI[0], "s0", -0.08, -0.06), (TRI[1], "s1", 0.08, -0.06), (TRI[2], "s2", 0.0, 0.08)):
        ax.text(x + dx, y + dy, name, ha="center", va="center", fontsize=fs, fontweight="bold", color="0.25")
    if title:
        ax.set_title("posterior key\n(s0, s1, s2) as RGB", fontsize=fs)
    ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.18, 1.05); ax.set_aspect("equal"); ax.axis("off")


def value_key(ax, fs: float = 9.0, title: bool = True) -> None:
    """Gap colourbar legend; `fs` sizes all text, `title=False` drops the two-line heading."""
    ax.axis("off")
    cax = ax.inset_axes([0.30, 0.04, 0.16, 0.80])
    cb = plt.colorbar(ScalarMappable(norm=GAP_NORM, cmap=GAP_CMAP), cax=cax, extend="both")
    cb.set_ticks([-2.0, 0.0, 2.0]); cb.set_ticklabels(["−2", "0", "+2"]); cb.ax.tick_params(labelsize=fs - 1)
    cb.ax.yaxis.set_ticks_position("left")
    if title:
        ax.set_title("value key\ngap Q(A) − Q(B)", fontsize=fs)
    ax.text(0.52, 0.80, "offer A\nvalued higher", transform=ax.transAxes, fontsize=fs - 1, va="center", color="#67001f")
    ax.text(0.52, 0.08, "offer B\nvalued higher", transform=ax.transAxes, fontsize=fs - 1, va="center", color="0.25")


def draw_box(ax, y: float, v: np.ndarray, fill: str, edge: str, half: float,
             whiskers: tuple = (5, 95)) -> float:
    """One horizontal quartile box with median line and whisker caps; returns the high whisker."""
    from matplotlib.patches import Rectangle
    lo, q25, q50, q75, hi = np.percentile(v, [whiskers[0], 25, 50, 75, whiskers[1]])
    ax.plot([lo, hi], [y, y], color=edge, lw=0.8, zorder=2)
    for w in (lo, hi):
        ax.plot([w, w], [y - 0.4 * half, y + 0.4 * half], color=edge, lw=0.8, zorder=2)
    ax.add_patch(Rectangle((q25, y - half), q75 - q25, 2 * half, facecolor=fill, edgecolor=edge, lw=0.7, zorder=3))
    ax.plot([q50, q50], [y - half, y + half], color=edge, lw=1.1, zorder=4)
    return float(hi)


# ----------------------------------------------------------------------------- paper figures
PAPER_FS = 7.0        # one text size for the paper figures; ticks/legend one point smaller
PAPER_RC = {"font.size": PAPER_FS, "axes.titlesize": PAPER_FS, "axes.labelsize": PAPER_FS,
            "xtick.labelsize": PAPER_FS - 1, "ytick.labelsize": PAPER_FS - 1, "legend.fontsize": PAPER_FS - 1,
            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.major.size": 2.5, "ytick.major.size": 2.5}
PAPER_W = 5.5         # inches; the paper's text width


def axes_in(fig, x0: float, y0: float, w: float, h: float):
    """Add an axes at an inch-specified rectangle (origin bottom-left) so panels are designed at print size."""
    W, H = fig.get_size_inches()
    assert x0 + w <= W + 1e-6 and y0 + h <= H + 1e-6, f"panel ({x0}, {y0}, {w}, {h}) leaves the {W} x {H} in figure"
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def text_in(fig, x: float, y: float, s: str, **kw):
    """`fig.text` at an inch position (origin bottom-left)."""
    W, H = fig.get_size_inches()
    return fig.text(x / W, y / H, s, **kw)


def frame(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def square_limits(ax, x: np.ndarray, y: np.ndarray, pad: float = 1.06) -> None:
    """Equal data scale on both axes without letting matplotlib shrink the box."""
    span = pad * max(np.ptp(x), np.ptp(y))
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2); ax.set_ylim(cy - span / 2, cy + span / 2)


# ----------------------------------------------------------------------------- variance attribution (12)
GROUPS = ("posterior", "value", "context")                       # regression groups of 07's accounting
GROUP_COLOR = {"posterior": POSTERIOR_COLOR, "value": VALUE_COLOR, "context": "0.55", "unexplained": "0.88", "other": "0.72"}
PAPER_GROUP_LABEL = {"posterior": "posterior", "value": "offer value", "context": "card and offers",
                     "unexplained": "unexplained", "other": "other"}   # the paper's paper terms for the group keys
GROUP_LABEL = {"posterior": "posterior (joint posterior)", "value": "offer value (Q(A), Q(B))",
               "context": "context (card, offers)", "unexplained": "unexplained", "other": "other (context, unexplained)"}


def plot_groups(merge_other: bool) -> tuple[str, ...]:
    """The drawn groups, bottom to top: merged, context and the unexplained
    remainder become one "other" band; split, they stack separately."""
    return ("posterior", "value", "other") if merge_other else GROUPS + ("unexplained",)


def plot_parts(att: dict, merge_other: bool) -> dict:
    """The accounting's {posterior, value, context, unexplained} parts (each a
    float or [d]) keyed by drawn group; merged, other = context + unexplained."""
    parts = {g: np.asarray(att[g], dtype=np.float64) for g in GROUPS + ("unexplained",)}
    if merge_other:
        parts = {"posterior": parts["posterior"], "value": parts["value"], "other": parts["context"] + parts["unexplained"]}
    return parts


def stacked_pcs(ax, att: dict, k: int, width: float = 0.8, merge_other: bool = True) -> None:
    """Stacked bars of the first k PCs' variance fractions split by drawn group ({group: [d]} accounting parts)."""
    x = np.arange(1, k + 1)
    base = np.zeros(k)
    drawn = plot_parts(att, merge_other)
    for g in plot_groups(merge_other):
        v = drawn[g][:k]
        ax.bar(x, v, bottom=base, width=width, color=GROUP_COLOR[g], edgecolor="none", label=GROUP_LABEL[g])
        base += v


def stacked_totals(ax, totals: list, labels: list, width: float = 0.7, merge_other: bool = True) -> None:
    """One stacked bar per site of the total variance by drawn group ([{group: float}] accounting parts)."""
    x = np.arange(len(totals))
    base = np.zeros(len(totals))
    drawn = [plot_parts(t, merge_other) for t in totals]
    for g in plot_groups(merge_other):
        v = np.array([d[g] for d in drawn])
        ax.bar(x, v, bottom=base, width=width, color=GROUP_COLOR[g], edgecolor="none", label=GROUP_LABEL[g])
        base += v
    ax.set_xticks(x); ax.set_xticklabels(labels)


# ----------------------------------------------------------------------------- paper rows
# Each drawer paints one column-width row of a paper figure at a vertical offset in inches, inside the
# caller's `plt.rc_context(PAPER_RC)`; a script draws its own paper figure with the row at the bottom, and a
# composer stacks rows from several experiments' saved data (experiments/08_decoder_representation.py).

DECODER_CELL = 0.94                                           # inches; the portrait cell the row is designed around


def decoder_row_height(cell: float = DECODER_CELL, theory: bool = True) -> float:
    """Row height in inches for a portrait cell of `cell`: theory row and activations row scale with the
    cell, the one-line header above keeps its text height. Without the theory row the activations cell
    sits 0.08 in above the floor (its vertex labels stay inside the cell)."""
    return (0.50 * cell / DECODER_CELL if theory else 0.08) + cell + 0.12


DECODER_ROW_H = decoder_row_height()                          # 1.56 in at the design cell


def draw_decoder_row(fig, y0: float, dec: dict, truth: dict, sc: dict, read_sites: tuple[str, str],
                     letters: tuple[str, str, str] = ("a", "b", "c"), r2=None, x0: float = 0.10,
                     cell: float = DECODER_CELL, sq: float = 0.95, gap: float = 0.32, axis_words: bool = True,
                     theory: bool = True, row_label_x: float | None = None,
                     cell_r2: dict | None = None) -> float:
    """03's paper row: (a) portraits at the display site, activations above a
    half-height theory row in the same coordinates, each column headed by its
    identifier and by the earliest site the quantity is readable from
    (`read_sites` = posterior, value labels); (b, c) held-out proposition
    scatters, each with the pooled R² of the cloud it draws (`r2` = a function
    (true, hat) -> float). `dec` / `truth`: {"etaA", "etaB": [N, 3], "D": [N, 2]};
    `sc`: {"P_true", "P_hat", "D_true", "D_hat": [M]}. Geometry in inches:
    `x0` the block's left edge, `cell` the portrait cell (the vertical layout
    scales with it; see `decoder_row_height`), `sq` the scatter side, `gap`
    the label gap before each scatter; `cell_r2` = {"etaA", "etaB", "D": float},
    the R² of each portrait cloud at the site it is read from, printed in that
    cell (simplices below the triangle base, the value square inside its
    lower-left corner over a white patch), so (a) carries its fit the way
    (b, c) do; `axis_words=False` drops "exact" /
    "decoded" from the scatter axis labels (the hat carries the identity) so
    the panels can be narrower; `theory=False` drops the theory row (the
    caption then points to the analytic portraits elsewhere); `row_label_x`
    moves the rotated row labels (default 0.06 cell inside the block; a
    narrower cell needs them further left to clear the s0 vertex label).
    Returns the right edge of panel (c)."""
    from jbo.stats import r2 as _r2
    r2 = r2 or _r2
    fs = PAPER_FS
    W = fig.get_size_inches()[0]
    k = cell / DECODER_CELL                                  # vertical scale of the portrait block
    CELL, THEORY, SQ, GAP = cell, 0.55 * k, sq, gap          # portrait cell, theory-row height, scatter side, label gap
    row_h = decoder_row_height(cell, theory)
    if theory:
        x_a, y_t, y_a = x0, y0 + 0.04 * k, y0 + 0.50 * k     # portrait block origin; theory row, activations row
        y_sc = y0 + 0.36 * k + (0.95 * k - SQ) / 2           # scatters centred where the design places them
    else:
        x_a, y_t, y_a = x0, None, y0 + 0.08                  # activations row only
        y_sc = y0 + 0.30                                     # tick labels and axis label below the scatters
    # The rows overlap on purpose: a simplex cell is empty below its base and above its apex, so the theory
    # row tucks under the activations row and the two header lines sit inside the activations cell's top.
    assert x_a + 3 * CELL + 2 * (GAP + SQ) + 0.03 <= W, "row does not fit the column width"
    assert (not theory or y_t + 0.76 * THEORY <= y_a) and y_a + CELL + 0.09 <= y0 + row_h, "portrait rows and header do not fit"
    assert y_sc >= y0 and y_sc + SQ + 0.06 <= y0 + row_h, "scatters (and their letters) leave the row"

    # (a) portraits: activations over theory, same coordinates --------------------
    gap = truth["D"][:, 0] - truth["D"][:, 1]
    cols = (("factor M", "etaA"), ("factor N", "etaB"), ("offer value", "D"))   # as the pretrained figure heads its portraits; "posterior (factor M)" overruns a 0.81 in cell
    for c, (title, key) in enumerate(cols):
        x0 = x_a + c * CELL
        ax_dec = axes_in(fig, x0, y_a, CELL, CELL)
        base = 0.31 / 1.5 * THEORY                  # simplex base sits this far above its cell floor
        ax_tru = (axes_in(fig, x0 + (CELL - THEORY) / 2, y_t - (0 if key == "D" else base), THEORY, THEORY)
                  if theory else None)
        # square, centred across the cell; simplex cells drop so their base meets the value square's floor
        if key == "D":                              # framed square ≈ the simplex footprint
            for ax, D, lab in ((ax_dec, dec["D"], True), (ax_tru, truth["D"], False)):
                if ax is None:
                    continue
                p, k = ax.get_position(), 0.76
                yy = p.y0 + (1 - k) / 2 * p.height if lab else p.y0      # theory square sits at its cell's floor
                ax.set_position([p.x0 + (1 - k) / 2 * p.width, yy, k * p.width, k * p.height])
                value_panel(ax, D, gap, labeled=lab, s=1.0 if lab else 0.4, fs=fs)
            ax_dec.xaxis.labelpad = 1.5; ax_dec.yaxis.labelpad = 1.5
        else:
            posterior_panel(ax_dec, dec[key], truth[key], vertex_labels=True, s=1.0, fs=fs)
            if theory:
                posterior_panel(ax_tru, truth[key], truth[key], s=0.3, fs=fs)
        if cell_r2 is not None and key in cell_r2:
            # the cloud's own fit at the site it is read from, in the cell's empty corner: under the
            # triangle base (clear of the s0/s1 vertex labels), inside the square's light lower-left
            xy, ha, va, box = ((0.03, 0.03), "left", "bottom",
                               dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.6)) if key == "D" \
                else ((0.5, 0.0), "center", "bottom", None)
            ax_dec.text(*xy, f"$R^2$ = {cell_r2[key]:.3f}", transform=ax_dec.transAxes,
                        ha=ha, va=va, fontsize=fs - 1, bbox=box, zorder=5)
        text_in(fig, x0 + CELL / 2, y_a + CELL - 0.01, title, ha="center", va="bottom", fontsize=fs)
        if c == 0 and theory:                       # row labels hug the first simplex; without a theory row, "activations" contrasts with nothing
            rows = ((y_a, CELL, "activations"), (y_t, 0.76 * THEORY, "theory"))
            x_lab = x_a + 0.06 * CELL if row_label_x is None else row_label_x
            for yy, h, label in rows:
                text_in(fig, x_lab, yy + h / 2, label, ha="center", va="center", fontsize=fs, rotation=90)
    for x_mid, site in ((x_a + CELL, read_sites[0]), (x_a + 2.5 * CELL, read_sites[1])):   # under the two posterior
        text_in(fig, x_mid, y_a + CELL - 0.11, f"readable from {site}",                     # columns; under value
                ha="center", va="bottom", fontsize=fs - 1, color="0.45")
    text_in(fig, 0.066, y0 + row_h - 0.023, letters[0], fontsize=fs + 1, fontweight="bold", ha="left", va="top")

    # (b, c) held-out proposition scatters --------------------------------------
    ex, de = ("exact ", "decoded ") if axis_words else ("", "")
    panels = (("P", POSTERIOR_COLOR, f"{ex}$P(X)$", f"{de}$\\hat{{P}}(X)$", [0.0, 0.5, 1.0], ["0", "0.5", "1"]),
              ("D", VALUE_COLOR, f"{ex}$Q(X)$", f"{de}$\\hat{{Q}}(X)$", [-1.0, 0.0, 1.0], ["−1", "0", "1"]))
    for i, (key, color, xlabel, ylabel, ticks, labels) in enumerate(panels):
        ax = axes_in(fig, x_a + 3 * CELL + GAP + i * (GAP + SQ), y_sc, SQ, SQ)
        t, q = sc[f"{key}_true"], sc[f"{key}_hat"]
        ax.scatter(t, q, s=0.6, color=color, alpha=0.15, linewidths=0, rasterized=True)
        lo, hi = min(t.min(), q.min()), max(t.max(), q.max())
        ax.axline((lo, lo), (hi, hi), color="0.4", lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.text(0.97, 0.04, f"$R^2$ = {r2(t, q):.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=fs - 1)           # pooled over the drawn cloud; below the identity line
        ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.set_yticks(ticks); ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel, labelpad=1); ax.set_ylabel(ylabel, labelpad=1)   # the reading's identity is the axis
        frame(ax)
        ax.text(-0.32, 1.06, letters[1 + i], transform=ax.transAxes, fontsize=fs + 1, fontweight="bold", ha="left", va="top")
    return x_a + 3 * CELL + 2 * (GAP + SQ)


ATTRIBUTION_AXES_H = 1.05                                     # inches; bar panels' height (header sits 0.08 above)
ATTRIBUTION_ROW_H = 1.40                                      # inches, from the bar panels' floor to the header's top


N_DEPTH_PANELS = 4    # a paper depth row shows this many sites: every post-block site of the toy, an even sample of a deep stack


def depth_subset(sites: tuple, k: int = N_DEPTH_PANELS) -> tuple:
    """`sites` itself when it has at most `k` entries; otherwise `k` sites evenly
    spaced through the stack, first and last included."""
    if len(sites) <= k:
        return tuple(sites)
    idx = np.unique(np.round(np.linspace(0, len(sites) - 1, k)).astype(int))
    return tuple(sites[i] for i in idx)


def draw_attribution_row(fig, y0: float, acc: dict, sites: tuple, site_labels: dict, k_pc: int = 8,
                         letters: tuple[str, ...] = ("a", "b"), totals: bool = True, merge_other: bool = True) -> None:
    """07's paper row, natural pool: (a) per site, the first `k_pc` PCs'
    variance fractions split by analytic group; (b) the same summed over all
    components. `acc`: {site: {"var_frac": [d], "attribution": {group: [d]},
    "totals": {group: float}}}. `y0` is the bar panels' floor in inches.
    `totals=False` drops (b) and spreads the site panels over the column;
    `merge_other=False` stacks context and unexplained as separate bands."""
    fs = PAPER_FS
    h1, gap_a, x_a = ATTRIBUTION_AXES_H, 0.16, 0.42
    # with totals, the totals panel ends 0.32 in short of the column: room for its right-hand tick labels;
    # without, the site panels share the column up to the same right margin
    w_a = 0.83 if totals else (PAPER_W - x_a - 0.32 - (len(sites) - 1) * gap_a) / len(sites)
    top_a = 1.12 * max(max(acc[s]["var_frac"][:k_pc]) for s in sites)
    for i, s in enumerate(sites):
        ax = axes_in(fig, x_a + i * (w_a + gap_a), y0, w_a, h1)
        stacked_pcs(ax, acc[s]["attribution"], k_pc, merge_other=merge_other)
        ax.set_xticks([1, 4, 8]); ax.set_ylim(0, top_a); ax.set_xlim(0.4, k_pc + 0.6)
        ax.text(0.97, 0.97, site_labels[s], transform=ax.transAxes, ha="right", va="top", fontsize=fs)
        frame(ax)
        if i == 0:
            ax.set_ylabel("fraction of total variance")
            ax.legend(handles=[Patch(color=GROUP_COLOR[g], label=PAPER_GROUP_LABEL[g]) for g in plot_groups(merge_other)],
                      loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, handlelength=1.0, handleheight=0.8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("component")
    text_in(fig, 0.05, y0 + h1 + 0.08, letters[0], fontsize=fs + 1, fontweight="bold")
    if not totals:
        return
    x_b = x_a + len(sites) * (w_a + gap_a) + 0.02
    ax = axes_in(fig, x_b, y0, 0.78, h1)
    stacked_totals(ax, [acc[s]["totals"] for s in sites], [site_labels[s] for s in sites], merge_other=merge_other)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1]); ax.set_xlabel("site"); frame(ax)
    ax.yaxis.tick_right(); ax.spines["right"].set_visible(True); ax.spines["left"].set_visible(False)
    text_in(fig, x_b - 0.26, y0 + h1 + 0.08, letters[1], fontsize=fs + 1, fontweight="bold")
    text_in(fig, x_b + 0.39, y0 + h1 + 0.1, f"all {len(acc[sites[0]]['var_frac'])} components", ha="center", fontsize=fs)
