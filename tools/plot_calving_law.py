"""
Illustrate the gap-blended calving criterion analytically (no model).

glide's calving flag is psi = sigmoid(c r (H - H_calve)) with the critical
thickness blended between the grounded (height-above-buoyancy) and shelf
(minimum-thickness) laws by the gap between ice base and bed:

    H_g     = (depth / r + h0) / (1 - q)      grounded: calve if HAB < q H + h0
    H_s     = min(H_c, H_g)                   shelf: H < H_c, capped by the grounded
                                               threshold (H_c acts only in deep water)
    gap     = max(depth - r H, 0)              ice base above the bed
    G       = r (H_g - H_s)
    w       = max(1 - gap / G, 0)              w(0) = 1, w -> 0 once detached
    H_calve = H_s + w (H_g - H_s)

Panels: (a) the ramp w against the gap, next to an exponential of the same
scale for contrast; (b) F = H - H_calve against H at three water depths for
a negative and a positive margin — non-decreasing everywhere, with the flat
floating regime F = -h0 down to H_c - h0; (c) the resulting flag psi(H)
against the earlier phi-blended pair of criteria, which is non-monotone
at the root of a tongue (the sink switches off as the cell thins).

    python tools/plot_calving_law.py [--out calving_law.png]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = 0.917            # rho_i / rho_w
C = 1.0              # flag sharpness (1/m of flotation excess), glide's default

# validated categorical palette (slots 1-3) + text/neutral tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#9a9891", "#fcfcfb"


def sigmoid(z, c=C):
    return 1.0 / (1.0 + np.exp(-np.clip(c * z, -20, 20)))


def calving_F(H, depth, q, h0, H_c):
    """F = H - H_calve; psi = sigmoid(c r F). Mirrors common.cu calving_F."""
    H = np.asarray(H, dtype=float)
    Hg = (depth / R + h0) / (1.0 - q)
    Hs = min(H_c, Hg)
    gap = np.maximum(depth - R * H, 0.0)
    G = max(R * (Hg - Hs), 1e-3)
    w = np.maximum(1.0 - gap / G, 0.0)
    return H - (Hs + w * (Hg - Hs))


def psi_blended(H, depth, q, h0, H_c):
    """The earlier phi-blended pair of criteria (non-monotone at tongue roots)."""
    H = np.asarray(H, dtype=float)
    z = R * H - depth
    phi = sigmoid(z)
    g = sigmoid(z - R * (q * H + h0))
    f = sigmoid(R * (H - H_c))
    return phi * g + (1.0 - phi) * f


def style(ax, title):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)


def main(out: str):
    H_c, q = 200.0, 0.0
    depths = [400.0, 800.0, 1200.0]
    H = np.linspace(1.0, 1500.0, 3000)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.patch.set_facecolor(SURFACE)

    # (a) the ramp against the gap, for a deep fjord and a negative margin
    ax = axes[0]
    depth, h0 = 800.0, -50.0
    Hg = (depth / R + h0) / (1.0 - q)
    G = R * abs(Hg - H_c)
    gap = np.linspace(0.0, 1.6 * G, 400)
    ax.plot(gap, np.maximum(1.0 - gap / G, 0.0), color=SERIES[0], linewidth=2.0)
    ax.plot(gap, np.exp(-gap / G), color=SERIES[0], linewidth=1.4, linestyle=":")
    ax.axvline(G, color=MUTED, linewidth=1.0, linestyle="--")
    ax.text(G, 1.02, f"G = r |H_g − H_s| = {G:.0f} m", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.text(0.30 * G, 0.86, "linear ramp (used)", color=SERIES[0], fontsize=9)
    ax.text(0.95 * G, 0.42, "exponential, same scale:\nstill 0.37 at the ramp's end,\nkeeps the grounded law in play", color=INK2, fontsize=8.5)
    ax.set_xlabel("gap between ice base and bed  (m)", color=INK2)
    ax.set_ylabel("w", color=INK2)
    ax.set_ylim(-0.03, 1.12)
    style(ax, f"(a) blend weight w(gap)\ndepth {depth:.0f} m, h0 = {h0:+.0f} m")

    # (b) F(H) at three depths, negative and positive margin
    ax = axes[1]
    for k, depth in enumerate(depths):
        for h0, ls, lab in ((-50.0, "-", "h0 = −50 m"), (50.0, "--", "h0 = +50 m")):
            F = calving_F(H, depth, q, h0, H_c)
            ax.plot(H, F, color=SERIES[k], linewidth=2.0 if ls == "-" else 1.5, linestyle=ls)
        Hf = depth / R
        ax.text(30, 520 - 45 * k, f"depth {depth:.0f} m, H_f = {Hf:.0f} m", color=SERIES[k], fontsize=8.5, ha="left", va="center")
    ax.axhline(0.0, color=MUTED, linewidth=1.0)
    ax.axvline(H_c, color=MUTED, linewidth=1.0, linestyle="--")
    ax.text(H_c + 12, 250, "H_c", color=INK2, fontsize=8.5)
    ax.text(30, -100, "F < 0: calving", color=INK2, fontsize=8.5)
    ax.text(30, 30, "F > 0: no calving", color=INK2, fontsize=8.5)
    ax.text(300, -110, "floating regime: F = −h0, flat in H\nsolid h0 = −50 m, dashed h0 = +50 m", color=INK2, fontsize=8.5, va="top", ha="left")
    ax.set_xlabel("ice thickness H  (m)", color=INK2)
    ax.set_ylabel("F = H − H_calve  (m)", color=INK2)
    ax.set_xlim(0, 1500)
    ax.set_ylim(-260, 560)
    style(ax, f"(b) F(H) is non-decreasing: thinning never reduces calving\nH_c = {H_c:.0f} m, three water depths")

    # (c) psi(H): monotone law vs the phi-blended pair, at a tongue root
    ax = axes[2]
    depth, h0 = 800.0, 50.0
    psi_new = sigmoid(R * calving_F(H, depth, q, h0, H_c))
    psi_old = psi_blended(H, depth, q, h0, H_c)
    ax.plot(H, psi_new, color=SERIES[1], linewidth=2.4)
    ax.plot(H, psi_old, color=INK2, linewidth=1.6, linestyle=(0, (5, 4)))
    Hf = depth / R
    ax.axvline(Hf, color=MUTED, linewidth=1.0, linestyle=":")
    ax.text(Hf + 12, 0.55, "flotation", color=INK2, fontsize=8.5)
    ax.text(230, 0.86, "phi-blended pair (dashed, dropped):\nthick floating ice protected,\ngrounded root below the margin calves,\nso psi RISES as the cell thins across flotation", color=INK2, fontsize=8.5, va="top")
    ax.text(Hf + 90, 0.30, "gap-blended law (solid):\npsi = 0 up to H_f + h0,\nthen 1; never decreasing", color=SERIES[1], fontsize=8.5)
    ax.set_xlabel("ice thickness H  (m)", color=INK2)
    ax.set_ylabel("calving flag psi  (1 = no calving)", color=INK2)
    ax.set_xlim(0, 1500)
    ax.set_ylim(-0.03, 1.08)
    style(ax, f"(c) flag psi(H) at a tongue root\ndepth {depth:.0f} m, h0 = {h0:+.0f} m, H_c = {H_c:.0f} m")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="calving_law.png")
    main(ap.parse_args().out)
