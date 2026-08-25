"""Draw a publication-grade ASL principle and challenge schematic.

The figure is intentionally data-free: all image-like elements are vector
schematics, so the SVG contains no embedded clinical raster images.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "asl_principle_challenges_nature"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.4,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


COL = {
    "ink": "#1E2933",
    "muted": "#5A6573",
    "line": "#B8C2CC",
    "panel": "#F8FAFC",
    "asl": "#2F6B9A",
    "asl_light": "#E9F2F8",
    "label": "#C95B59",
    "label_light": "#FBEDEC",
    "t1": "#4F8358",
    "t1_light": "#EDF5EE",
    "time": "#D08A22",
    "time_light": "#FCF3E3",
    "n2n": "#76569C",
    "n2n_light": "#F2EDF8",
    "warn": "#B94A48",
    "warn_light": "#FDF0EF",
    "dark_img": "#242B31",
    "mid_img": "#AEB8BF",
    "light_img": "#E3E7EA",
}


def add_text(ax, x, y, s, *, size=6.4, color=None, weight="normal", ha="left", va="center", **kwargs):
    kwargs.setdefault("zorder", 10)
    return ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        fontsize=size,
        color=color or COL["ink"],
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=1.15,
        **kwargs,
    )


def rounded(ax, x, y, w, h, *, fc="white", ec=None, lw=0.7, radius=0.007, zorder=1, ls="-"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=ax.transAxes,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or COL["line"],
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color=None, lw=1.0, ms=7, ls="-", zorder=5, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color or COL["ink"],
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel(ax, x, y, w, h, label, title, color):
    rounded(ax, x, y, w, h, fc="white", ec=color, lw=0.9, radius=0.006, zorder=0)
    add_text(ax, x + 0.008, y + h - 0.018, label, size=8.2, color=color, weight="bold")
    add_text(ax, x + 0.031, y + h - 0.018, title, size=8.2, color=color, weight="bold")


def bezier(ax, verts, *, color, lw=1.0, zorder=3, alpha=1.0):
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * (len(verts) - 1)
    path = MplPath(verts, codes)
    patch = PathPatch(path, transform=ax.transAxes, fill=False, color=color, lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    return patch


def draw_axial_brain(ax, x, y, w, h, *, kind="magnitude", seed=0, border=None, ghost=False):
    """Draw a stylized axial brain as vector patches inside a thumbnail box."""
    rounded(ax, x, y, w, h, fc="#11171C", ec=border or COL["line"], lw=0.7, radius=0.003, zorder=2)
    cx, cy = x + w / 2, y + h / 2
    outer = Ellipse(
        (cx, cy),
        w * 0.79,
        h * 0.86,
        transform=ax.transAxes,
        facecolor=COL["mid_img"] if kind in {"magnitude", "t1", "biased"} else "#4C5962",
        edgecolor="#F1F3F4",
        lw=0.45,
        zorder=3,
    )
    ax.add_patch(outer)
    if ghost:
        ax.add_patch(
            Ellipse(
                (cx + w * 0.045, cy),
                w * 0.79,
                h * 0.86,
                transform=ax.transAxes,
                fill=False,
                edgecolor=COL["label"],
                lw=0.55,
                alpha=0.55,
                zorder=3,
            )
        )

    # Hemisphere separation and ventricles.
    ax.plot([cx, cx], [cy - h * 0.30, cy + h * 0.31], transform=ax.transAxes, color="#65717A", lw=0.35, zorder=4)
    for side in (-1, 1):
        vent = Ellipse(
            (cx + side * w * 0.082, cy + h * 0.015),
            w * 0.075,
            h * 0.16,
            angle=-side * 12,
            transform=ax.transAxes,
            facecolor="#303941",
            edgecolor="none",
            zorder=4,
        )
        ax.add_patch(vent)

    rng = np.random.default_rng(seed)
    if kind in {"magnitude", "t1", "biased"}:
        # Cortical arcs and tissue bands.
        for frac, alpha in [(0.64, 0.85), (0.49, 0.55), (0.34, 0.40)]:
            ax.add_patch(
                Ellipse(
                    (cx, cy),
                    w * frac,
                    h * (frac + 0.09),
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor=COL["light_img"],
                    lw=0.38 if kind != "biased" else 0.65,
                    alpha=alpha,
                    zorder=4,
                )
            )
        if kind == "biased":
            for a in (-0.23, 0.23):
                bezier(
                    ax,
                    [(cx + a * w, cy - 0.25 * h), (cx + a * 0.8 * w, cy), (cx + a * w, cy + 0.27 * h), (cx + a * 0.5 * w, cy + 0.34 * h)],
                    color="#F7F9FA",
                    lw=0.55,
                    zorder=5,
                )
    else:
        # Broad perfusion fields plus controllable vector speckle.
        for px, py, rr, aa in [(-0.18, 0.13, 0.18, 0.28), (0.20, 0.09, 0.16, 0.24), (0.0, -0.17, 0.20, 0.21)]:
            c = Circle(
                (cx + px * w, cy + py * h),
                min(w, h) * rr,
                transform=ax.transAxes,
                facecolor="#E8EDF0",
                edgecolor="none",
                alpha=aa,
                zorder=4,
            )
            c.set_clip_path(outer)
            ax.add_patch(c)
        n = 24 if kind == "delta" else 8
        for _ in range(n):
            px = cx + rng.normal(0, w * 0.20)
            py = cy + rng.normal(0, h * 0.23)
            rr = rng.uniform(0.004, 0.010)
            speck = Circle(
                (px, py),
                rr,
                transform=ax.transAxes,
                facecolor="#F7F9FA" if rng.random() > 0.30 else "#171C20",
                edgecolor="none",
                alpha=0.62 if kind == "delta" else 0.35,
                zorder=5,
            )
            speck.set_clip_path(outer)
            ax.add_patch(speck)
    return outer


def draw_spin_lane(ax, x, y, w, h, *, labeled=False):
    color = COL["label"] if labeled else COL["asl"]
    # Simplified feeding artery.
    ax.plot([x, x + w * 0.18, x + w * 0.28], [y + h * 0.45, y + h * 0.45, y + h * 0.69], transform=ax.transAxes, color=color, lw=2.1, solid_capstyle="round", zorder=3)
    ax.plot([x + w * 0.18, x + w * 0.28], [y + h * 0.45, y + h * 0.22], transform=ax.transAxes, color=color, lw=1.4, zorder=3)
    for j in range(4):
        sx = x + w * (0.06 + 0.055 * j)
        sy = y + h * 0.45
        ax.add_patch(Circle((sx, sy), 0.0032, transform=ax.transAxes, fc=color, ec="white", lw=0.2, zorder=4))
    # Spin arrows: upward for control, downward for label.
    for j in range(4):
        sx = x + w * (0.39 + 0.07 * j)
        if labeled:
            arrow(ax, sx, y + h * 0.72, sx, y + h * 0.31, color=color, lw=0.65, ms=4)
        else:
            arrow(ax, sx, y + h * 0.29, sx, y + h * 0.70, color=color, lw=0.65, ms=4)
    # Brain destination.
    brain = Ellipse((x + w * 0.83, y + h * 0.50), w * 0.25, h * 0.78, transform=ax.transAxes, fc="#F4F6F7", ec=COL["muted"], lw=0.45, zorder=2)
    ax.add_patch(brain)
    for j in range(3):
        sx = x + w * (0.76 + 0.06 * j)
        if labeled:
            arrow(ax, sx, y + h * 0.68, sx, y + h * 0.38, color=color, lw=0.55, ms=3.5)
        else:
            arrow(ax, sx, y + h * 0.36, sx, y + h * 0.66, color=color, lw=0.55, ms=3.5)


def draw_head_and_label_plane(ax):
    # Head, neck, brain and carotids.
    head = Ellipse((0.080, 0.787), 0.092, 0.175, transform=ax.transAxes, fc="#F7F8F9", ec=COL["muted"], lw=0.65, zorder=2)
    ax.add_patch(head)
    ax.add_patch(Rectangle((0.069, 0.665), 0.024, 0.070, transform=ax.transAxes, fc="#F7F8F9", ec=COL["muted"], lw=0.55, zorder=1))
    brain = Ellipse((0.078, 0.807), 0.064, 0.085, transform=ax.transAxes, fc="#E4E9EC", ec=COL["muted"], lw=0.4, zorder=3)
    ax.add_patch(brain)
    # Carotid trunks and branches.
    for dx in (-0.009, 0.009):
        bezier(ax, [(0.081 + dx, 0.665), (0.081 + dx, 0.705), (0.080 + dx, 0.748), (0.078 + dx, 0.785)], color=COL["label"], lw=1.25, zorder=4)
    for xoff in (0.069, 0.087):
        bezier(ax, [(xoff, 0.767), (xoff - 0.010, 0.785), (xoff - 0.006, 0.806), (xoff - 0.019, 0.826)], color=COL["label"], lw=0.75, zorder=4)
    ax.plot([0.030, 0.130], [0.710, 0.710], transform=ax.transAxes, color=COL["asl"], lw=0.75, ls=(0, (3, 2)), zorder=5)
    add_text(ax, 0.031, 0.700, "labeling plane", size=5.7, color=COL["asl"], va="top")


def draw_frame_stack(ax, x, y, *, color, n=5, scale=1.0):
    for i in range(n):
        dx = i * 0.0045 * scale
        dy = i * 0.0038 * scale
        ax.add_patch(Rectangle((x + dx, y + dy), 0.028 * scale, 0.043 * scale, transform=ax.transAxes, fc="#3F4A52", ec="white", lw=0.25, zorder=3 + i))
        ax.add_patch(Ellipse((x + dx + 0.014 * scale, y + dy + 0.022 * scale), 0.021 * scale, 0.033 * scale, transform=ax.transAxes, fc="#AEB8BF", ec="none", alpha=0.85, zorder=4 + i))
    ax.add_patch(Rectangle((x - 0.003, y - 0.003), 0.028 * scale + n * 0.0045 * scale + 0.006, 0.043 * scale + n * 0.0038 * scale + 0.006, transform=ax.transAxes, fill=False, ec=color, lw=0.55, zorder=2))


def build_figure():
    width_in = 183.0 / 25.4
    height_in = 122.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_text(
        ax,
        0.5,
        0.978,
        "From arterial spin labeling to PWI: signal formation and project challenges",
        size=12.2,
        weight="bold",
        ha="center",
    )

    # ------------------------------------------------------------------ panel a
    panel(ax, 0.012, 0.642, 0.976, 0.310, "a", "ASL acquisition: endogenous arterial-water tracer", COL["asl"])
    rounded(ax, 0.842, 0.920, 0.128, 0.021, fc=COL["time_light"], ec=COL["time"], lw=0.5, radius=0.002)
    add_text(ax, 0.906, 0.9305, "7 T  •  single PLD  •  12 NEX", size=5.2, color=COL["time"], weight="bold", ha="center")
    draw_head_and_label_plane(ax)

    rounded(ax, 0.150, 0.795, 0.455, 0.104, fc=COL["asl_light"], ec=COL["asl"], lw=0.75, radius=0.004)
    add_text(ax, 0.160, 0.883, "CONTROL", size=7.2, color=COL["asl"], weight="bold")
    add_text(ax, 0.226, 0.883, "(no inversion)", size=5.4, color=COL["muted"])
    add_text(ax, 0.160, 0.842, "arterial spins pass\nwithout net inversion", size=5.8)
    draw_spin_lane(ax, 0.260, 0.808, 0.285, 0.076, labeled=False)

    rounded(ax, 0.150, 0.674, 0.455, 0.104, fc=COL["label_light"], ec=COL["label"], lw=0.75, radius=0.004)
    add_text(ax, 0.160, 0.762, "LABEL", size=7.2, color=COL["label"], weight="bold")
    add_text(ax, 0.211, 0.762, "(inversion)", size=5.4, color=COL["muted"])
    add_text(ax, 0.160, 0.721, "RF plane inverts\narterial blood-water spins", size=5.8)
    draw_spin_lane(ax, 0.260, 0.687, 0.285, 0.076, labeled=True)

    # PLD and acquired magnitude images.
    arrow(ax, 0.613, 0.846, 0.682, 0.846, color=COL["asl"], lw=1.0, ms=6)
    arrow(ax, 0.613, 0.725, 0.682, 0.725, color=COL["label"], lw=1.0, ms=6)
    add_text(ax, 0.647, 0.865, "post-labeling delay", size=5.6, color=COL["asl"], ha="center")
    add_text(ax, 0.647, 0.744, "post-labeling delay", size=5.6, color=COL["label"], ha="center")
    draw_axial_brain(ax, 0.690, 0.807, 0.060, 0.080, kind="magnitude", seed=1, border=COL["asl"])
    draw_axial_brain(ax, 0.690, 0.686, 0.060, 0.080, kind="magnitude", seed=1, border=COL["label"])
    add_text(ax, 0.757, 0.851, r"$M_{\mathrm{control},t}$", size=7.2, color=COL["asl"], weight="bold")
    add_text(ax, 0.757, 0.826, "ASL control image", size=5.7)
    add_text(ax, 0.757, 0.730, r"$M_{\mathrm{label},t}$", size=7.2, color=COL["label"], weight="bold")
    add_text(ax, 0.757, 0.705, "ASL label image", size=5.7)
    rounded(ax, 0.852, 0.739, 0.113, 0.095, fc="#F7F8F9", ec=COL["muted"], lw=0.6, radius=0.003)
    add_text(ax, 0.908, 0.787, "Nearly identical\nmagnitude images;\nperfusion is a small residual", size=5.8, ha="center")

    # Alternating timeline.
    add_text(ax, 0.153, 0.657, "alternating acquisition", size=5.4, color=COL["muted"])
    x0 = 0.260
    labels = [("Control 1", COL["asl"]), ("Label 1", COL["label"]), ("Control 2", COL["asl"]), ("Label 2", COL["label"]), ("...", COL["muted"]), ("Control T", COL["asl"]), ("Label T", COL["label"])]
    widths = [0.068, 0.060, 0.068, 0.060, 0.040, 0.068, 0.060]
    for idx, ((lab, color), ww) in enumerate(zip(labels, widths)):
        if lab == "...":
            add_text(ax, x0 + ww / 2, 0.657, lab, size=7.0, ha="center")
        else:
            rounded(ax, x0, 0.646, ww, 0.026, fc="white", ec=color, lw=0.55, radius=0.002)
            add_text(ax, x0 + ww / 2, 0.659, lab, size=5.3, color=color, ha="center")
        if idx < len(labels) - 1:
            arrow(ax, x0 + ww + 0.004, 0.659, x0 + ww + 0.016, 0.659, color=COL["muted"], lw=0.45, ms=3.5)
        x0 += ww + 0.020
    add_text(ax, 0.950, 0.659, "time", size=5.4, color=COL["muted"], ha="right")

    # ------------------------------------------------------------------ panel b
    panel(ax, 0.012, 0.373, 0.976, 0.250, "b", "Pairwise subtraction and repeated averaging", COL["asl"])
    draw_axial_brain(ax, 0.030, 0.493, 0.047, 0.064, kind="magnitude", seed=2, border=COL["asl"])
    draw_axial_brain(ax, 0.030, 0.408, 0.047, 0.064, kind="magnitude", seed=2, border=COL["label"])
    add_text(ax, 0.082, 0.525, r"$M_{\mathrm{control},t}$", size=6.2, color=COL["asl"])
    add_text(ax, 0.082, 0.440, r"$M_{\mathrm{label},t}$", size=6.2, color=COL["label"])
    arrow(ax, 0.135, 0.508, 0.172, 0.477, color=COL["asl"], lw=0.8, ms=5)
    arrow(ax, 0.135, 0.427, 0.172, 0.465, color=COL["label"], lw=0.8, ms=5)
    add_text(ax, 0.177, 0.471, "−", size=11, weight="bold", ha="center")
    rounded(ax, 0.196, 0.444, 0.170, 0.052, fc="#F7F9FA", ec=COL["asl"], lw=0.65, radius=0.003)
    add_text(ax, 0.281, 0.470, r"$\Delta M_t=M_{\mathrm{control},t}-M_{\mathrm{label},t}$", size=6.6, ha="center")
    arrow(ax, 0.368, 0.470, 0.392, 0.470, color=COL["asl"], lw=0.8, ms=5)
    draw_axial_brain(ax, 0.395, 0.432, 0.052, 0.075, kind="delta", seed=5, border=COL["asl"])
    add_text(ax, 0.421, 0.419, "single-NEX ΔM", size=5.5, color=COL["asl"], ha="center")

    # Repetition stack, 2 x 6.
    stack_x, stack_y = 0.477, 0.428
    rounded(ax, stack_x - 0.010, stack_y - 0.012, 0.257, 0.139, fc="#F7F9FA", ec=COL["line"], lw=0.6, radius=0.003)
    add_text(ax, stack_x + 0.118, stack_y + 0.116, "12 heterogeneous single-NEX ΔM frames", size=5.8, color=COL["asl"], weight="bold", ha="center")
    for j in range(12):
        col, row = j % 6, 1 - j // 6
        bx = stack_x + col * 0.039
        by = stack_y + row * 0.052
        draw_axial_brain(ax, bx, by, 0.032, 0.040, kind="delta", seed=20 + j, border=COL["warn"] if j == 3 else (COL["time"] if j == 9 else "#D0D7DD"), ghost=(j == 3))
    add_text(ax, stack_x + 0.133, 0.407, "motion / labeling drift / physiological fluctuation", size=4.5, color=COL["muted"], ha="center")

    # Few-frame and full-average outcomes.
    arrow(ax, 0.725, 0.505, 0.770, 0.542, color=COL["time"], lw=0.9, ms=5)
    arrow(ax, 0.725, 0.447, 0.770, 0.414, color=COL["asl"], lw=0.9, ms=5)
    rounded(ax, 0.772, 0.520, 0.100, 0.045, fc=COL["time_light"], ec=COL["time"], lw=0.65, radius=0.003)
    add_text(ax, 0.822, 0.543, "few-frame n = 2–8", size=6.0, color=COL["time"], weight="bold", ha="center")
    draw_axial_brain(ax, 0.875, 0.510, 0.045, 0.068, kind="delta", seed=52, border=COL["time"])
    add_text(ax, 0.925, 0.544, "low SNR\nlow redundancy", size=4.4, color=COL["time"], ha="left")

    rounded(ax, 0.772, 0.391, 0.100, 0.045, fc=COL["asl_light"], ec=COL["asl"], lw=0.65, radius=0.003)
    add_text(ax, 0.822, 0.414, r"$\mathrm{PWI}=T^{-1}\sum_t\Delta M_t$", size=6.0, color=COL["asl"], weight="bold", ha="center")
    draw_axial_brain(ax, 0.875, 0.386, 0.045, 0.068, kind="pwi", seed=61, border=COL["asl"])
    add_text(ax, 0.925, 0.420, "higher SNR\nstill not clean", size=4.4, color=COL["asl"], ha="left")
    rounded(ax, 0.350, 0.378, 0.300, 0.018, fc=COL["n2n_light"], ec=COL["n2n"], lw=0.5, radius=0.002)
    add_text(ax, 0.500, 0.387, "Normalized PWI / ΔM — not quantitative CBF", size=5.4, color=COL["n2n"], weight="bold", ha="center")

    # ------------------------------------------------------------------ panel c
    panel(ax, 0.012, 0.050, 0.976, 0.302, "c", "Challenges addressed in this project", COL["time"])
    card_y, card_h, card_w = 0.077, 0.238, 0.230
    card_xs = [0.024, 0.266, 0.508, 0.750]
    card_colors = [COL["warn"], COL["time"], COL["n2n"], COL["t1"]]
    card_fills = [COL["warn_light"], COL["time_light"], COL["n2n_light"], COL["t1_light"]]
    titles = ["Weak subtraction signal", "Scan time vs reliability", "No clean ground truth", "Auxiliary-anatomy bias"]
    for idx, (cx, cc, ff, title) in enumerate(zip(card_xs, card_colors, card_fills, titles), start=1):
        rounded(ax, cx, card_y, card_w, card_h, fc=ff, ec=cc, lw=0.7, radius=0.004)
        ax.add_patch(Circle((cx + 0.017, card_y + card_h - 0.023), 0.011, transform=ax.transAxes, fc=cc, ec="none", zorder=4))
        add_text(ax, cx + 0.017, card_y + card_h - 0.023, str(idx), size=7.0, color="white", weight="bold", ha="center", zorder=12)
        add_text(ax, cx + 0.034, card_y + card_h - 0.023, title, size=6.7, color=cc, weight="bold")

    # Card 1: nearly equal magnitude bars and tiny residual.
    add_text(ax, 0.139, 0.270, r"$\Delta M \ll M_0$; subtraction exposes" + "\nthermal and physiological noise.", size=5.5, ha="center")
    base_y = 0.108
    ax.plot([0.050, 0.225], [base_y, base_y], transform=ax.transAxes, color=COL["muted"], lw=0.55)
    ax.add_patch(Rectangle((0.065, base_y), 0.035, 0.103, transform=ax.transAxes, fc=COL["asl"], ec="none", alpha=0.85))
    ax.add_patch(Rectangle((0.178, base_y), 0.035, 0.099, transform=ax.transAxes, fc=COL["label"], ec="none", alpha=0.85))
    arrow(ax, 0.129, 0.205, 0.129, 0.112, color=COL["warn"], lw=0.65, ms=4)
    add_text(ax, 0.083, 0.096, "control", size=5.0, color=COL["asl"], ha="center")
    add_text(ax, 0.195, 0.096, "label", size=5.0, color=COL["label"], ha="center")
    add_text(ax, 0.129, 0.101, "tiny ΔM", size=5.0, color=COL["warn"], ha="center")

    # Card 2: clock, 12-frame stack and few-frame stack.
    add_text(ax, 0.381, 0.270, "More NEX improves averaging but lengthens\nthe 7 T examination; one bad frame matters more.", size=5.4, ha="center")
    ax.add_patch(Circle((0.295, 0.177), 0.021, transform=ax.transAxes, fc="white", ec=COL["time"], lw=0.8))
    ax.plot([0.295, 0.295], [0.177, 0.191], transform=ax.transAxes, color=COL["time"], lw=0.8)
    ax.plot([0.295, 0.306], [0.177, 0.169], transform=ax.transAxes, color=COL["time"], lw=0.8)
    draw_frame_stack(ax, 0.326, 0.145, color=COL["time"], n=6, scale=0.92)
    add_text(ax, 0.367, 0.128, "12 NEX", size=5.1, color=COL["time"], ha="center")
    arrow(ax, 0.411, 0.171, 0.436, 0.171, color=COL["muted"], lw=0.55, ms=4)
    draw_frame_stack(ax, 0.446, 0.145, color=COL["warn"], n=3, scale=0.92)
    add_text(ax, 0.474, 0.128, "2–8 NEX", size=5.1, color=COL["warn"], ha="center")
    rounded(ax, 0.287, 0.086, 0.188, 0.026, fc="white", ec=COL["time"], lw=0.5, radius=0.002)
    add_text(ax, 0.381, 0.099, r"$B_0/B_1$ sensitivity  •  motion  •  labeling instability", size=4.9, color=COL["muted"], ha="center")

    # Card 3: noisy reference and Noise2Noise subsets.
    add_text(ax, 0.623, 0.270, "Structured motion, physiology and labeling drift\npersist across repetitions and their mean.", size=5.4, ha="center")
    draw_axial_brain(ax, 0.522, 0.145, 0.045, 0.064, kind="delta", seed=70, border=COL["n2n"], ghost=True)
    add_text(ax, 0.545, 0.133, "12-NEX mean\nreference ≠ truth", size=4.4, color=COL["n2n"], ha="center", va="top")
    draw_frame_stack(ax, 0.586, 0.159, color=COL["n2n"], n=3, scale=0.58)
    add_text(ax, 0.602, 0.139, "Set A\nnoisy input", size=4.3, color=COL["n2n"], ha="center", va="top")
    arrow(ax, 0.612, 0.179, 0.631, 0.179, color=COL["n2n"], lw=0.65, ms=3.5)
    rounded(ax, 0.634, 0.164, 0.038, 0.030, fc="white", ec=COL["n2n"], lw=0.55, radius=0.002)
    add_text(ax, 0.653, 0.179, "model", size=4.5, color=COL["n2n"], ha="center")
    arrow(ax, 0.675, 0.179, 0.691, 0.179, color=COL["n2n"], lw=0.65, ms=3.5)
    draw_frame_stack(ax, 0.694, 0.159, color=COL["n2n"], n=3, scale=0.58)
    add_text(ax, 0.710, 0.139, "mean(Set B)\nnoisy target", size=4.3, color=COL["n2n"], ha="center", va="top")
    add_text(ax, 0.623, 0.093, "disjoint subsets; neither side is clean", size=4.7, color=COL["n2n"], ha="center")

    # Card 4: sharp T1, noisy PWI and risky unrestricted fusion.
    add_text(ax, 0.865, 0.270, "T1w is sharper but is not perfusion; unrestricted\nfusion can transfer unsupported anatomical edges.", size=5.3, ha="center")
    draw_axial_brain(ax, 0.764, 0.147, 0.040, 0.058, kind="t1", seed=1, border=COL["t1"])
    add_text(ax, 0.784, 0.137, "T1w", size=4.5, color=COL["t1"], ha="center", va="top")
    add_text(ax, 0.815, 0.176, "+", size=7.5, color=COL["muted"], weight="bold", ha="center")
    draw_axial_brain(ax, 0.828, 0.147, 0.040, 0.058, kind="delta", seed=91, border=COL["asl"])
    add_text(ax, 0.848, 0.137, "PWI", size=4.5, color=COL["asl"], ha="center", va="top")
    arrow(ax, 0.871, 0.176, 0.888, 0.176, color=COL["muted"], lw=0.65, ms=3.5)
    rounded(ax, 0.891, 0.158, 0.043, 0.036, fc="white", ec=COL["muted"], lw=0.55, radius=0.002)
    add_text(ax, 0.9125, 0.176, "naive\nfusion", size=4.4, ha="center")
    arrow(ax, 0.937, 0.176, 0.945, 0.176, color=COL["warn"], lw=0.65, ms=3.5)
    draw_axial_brain(ax, 0.947, 0.147, 0.029, 0.046, kind="biased", seed=1, border=COL["warn"])
    add_text(ax, 0.961, 0.136, "biased", size=4.2, color=COL["warn"], ha="center", va="top")
    add_text(ax, 0.865, 0.096, "correlated prior ≠ output content", size=4.8, color=COL["warn"], weight="bold", ha="center")

    # Motivation banner.
    rounded(ax, 0.090, 0.012, 0.820, 0.027, fc="#F7F8FA", ec=COL["muted"], lw=0.55, radius=0.002)
    add_text(
        ax,
        0.500,
        0.0255,
        "Design objective: robust few-frame reconstruction + self-supervision without clean targets + anatomical guidance without direct T1 content transfer",
        size=5.9,
        color=COL["ink"],
        weight="bold",
        ha="center",
    )

    return fig


def export_and_check(fig):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg_path = OUT.with_suffix(".svg")
    pdf_path = OUT.with_suffix(".pdf")
    png_path = OUT.with_suffix(".png")
    fig.savefig(svg_path, format="svg", facecolor="white", bbox_inches=None)
    fig.savefig(pdf_path, format="pdf", facecolor="white", bbox_inches=None)
    fig.savefig(png_path, format="png", dpi=600, facecolor="white", bbox_inches=None)

    root = ElementTree.parse(svg_path).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    text_nodes = root.findall(".//svg:text", ns)
    image_nodes = root.findall(".//svg:image", ns)
    if len(text_nodes) < 25:
        raise RuntimeError(f"SVG text is not sufficiently editable: only {len(text_nodes)} text nodes")
    if image_nodes:
        raise RuntimeError(f"SVG unexpectedly embeds {len(image_nodes)} raster image(s)")
    print(f"SVG: {svg_path}")
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"QA: {len(text_nodes)} editable text nodes; 0 embedded raster images")


if __name__ == "__main__":
    figure = build_figure()
    export_and_check(figure)
    plt.close(figure)
