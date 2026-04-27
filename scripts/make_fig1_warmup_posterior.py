#!/usr/bin/env python3
"""Figure 1: 2-row visualisation of how each method navigates the strong x-x
posterior dependency on LV (N=25, seed 42).

Top row: the entire warmup phase (5000 iterations), plotted from initial point
  to converged region. Shows how fast each chain moves from its start to the
  posterior ridge (= the x-x dependency region).
Bottom row: 200 consecutive post-warmup samples within the converged region.
  Shows the local step behaviour once each chain has settled.

Inputs:  experiments/pgm2026/results/exp1_scatter/exp1s_{method}_seed42_full.csv
Output:  paper/figures/mcmc_trajectory.png
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAT = os.path.join(BASE, "experiments/pgm2026/results/exp1_scatter")
OUT = os.path.join(BASE, "paper/figures/mcmc_trajectory.png")

METHODS = ["RFFGM", "GPGM"]
LABELS = {"RFFGM": "(a) RFFGM (ours)", "GPGM": "(b) GPGM"}
COLOR = {"RFFGM": "#d73027", "GPGM": "#4575b4"}
WARMUP_LEN = 5000

POST_TRAIL = 200  # number of post-warmup samples to show in bottom row


def load_full(path):
    its, iw, xi, xj = [], [], [], []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            its.append(int(row["iter"]))
            iw.append(int(row["is_warmup"]))
            xi.append(float(row["x_i"]))
            xj.append(float(row["x_j"]))
    return (
        np.array(its),
        np.array(iw),
        np.array(xi),
        np.array(xj),
    )


def iterations_to_reach(xi, xj, target_xi, target_xj, tol=0.3):
    """Return first iteration index where chain enters tol-ball around target."""
    d = np.sqrt((xi - target_xi) ** 2 + (xj - target_xj) ** 2)
    hit = np.where(d < tol)[0]
    return int(hit[0]) if len(hit) > 0 else len(xi)


def main():
    data = {}
    for m in METHODS:
        d = load_full(os.path.join(DAT, f"exp1s_{m}_seed42_full.csv"))
        if d is None:
            print(f"Missing: {m}", file=sys.stderr)
            continue
        data[m] = d

    # Shared target region from the post-warmup posterior mean of RFFGM
    # (all methods' targets should be near the same ridge in a well-posed problem).
    _, iw, xi, xj = data["RFFGM"]
    post_mask = iw == 0
    target_xi = float(np.mean(xi[post_mask]))
    target_xj = float(np.mean(xj[post_mask]))

    # Global axis limits (use WARMUP values so the full journey is visible).
    all_xi_w = np.concatenate([d[2][d[1] == 1] for d in data.values()])
    all_xj_w = np.concatenate([d[3][d[1] == 1] for d in data.values()])
    xlim = (min(all_xi_w.min(), 0) - 0.5, all_xi_w.max() + 0.5)
    ylim = (min(all_xj_w.min(), 0) - 0.5, all_xj_w.max() + 0.5)

    # Zoomed-in limits for post-warmup row.
    all_xi_p = np.concatenate([d[2][d[1] == 0] for d in data.values()])
    all_xj_p = np.concatenate([d[3][d[1] == 0] for d in data.values()])
    xlim_p = (all_xi_p.min() - 0.1, all_xi_p.max() + 0.1)
    ylim_p = (all_xj_p.min() - 0.1, all_xj_p.max() + 0.1)

    fig, axes = plt.subplots(
        2, 2, figsize=(9, 7.2), constrained_layout=True,
    )

    for col, m in enumerate(METHODS):
        _, iw, xi, xj = data[m]
        warmup_mask = iw == 1
        xi_w = xi[warmup_mask]
        xj_w = xj[warmup_mask]
        xi_p = xi[iw == 0]
        xj_p = xj[iw == 0]

        # ---------- Top row: warmup trajectory ----------
        ax_top = axes[0, col]
        cmap = matplotlib.colormaps.get_cmap("viridis")
        n_w = len(xi_w)
        # subsample for rendering speed (still shows evolution)
        step = max(1, n_w // 500)
        idx_sub = np.arange(0, n_w, step)
        # connect consecutive sub-sampled points with coloured segments
        for k in range(len(idx_sub) - 1):
            i0 = idx_sub[k]; i1 = idx_sub[k + 1]
            ax_top.plot(xi_w[i0:i1+1], xj_w[i0:i1+1],
                        color=cmap(i0 / max(1, n_w - 1)),
                        lw=0.6, alpha=0.85)
        # Start marker
        ax_top.scatter([xi_w[0]], [xj_w[0]], marker="s", s=80, c="black",
                       edgecolor="white", lw=1.2, zorder=5, label="start")
        # End of warmup marker
        ax_top.scatter([xi_w[-1]], [xj_w[-1]], marker="D", s=80, c=COLOR[m],
                       edgecolor="black", lw=1.0, zorder=5,
                       label="end of warmup")
        # Target region (post-warmup mean of RFFGM, reference)
        ax_top.scatter([target_xi], [target_xj], marker="*", s=220,
                       c="gold", edgecolor="black", linewidth=1.0, zorder=6,
                       label="posterior mean (ref.)")

        # Two useful metrics:
        #   (1) maximum excursion from the target during warmup
        #   (2) distance from the target at end of warmup
        d_warmup = np.sqrt((xi_w - target_xi) ** 2 + (xj_w - target_xj) ** 2)
        max_excursion = float(d_warmup.max())
        end_distance = float(d_warmup[-1])
        txt = (
            f"max dist.\\ from target = {max_excursion:.2f}\n"
            f"end-of-warmup dist.     = {end_distance:.2f}"
        )
        ax_top.text(
            0.03, 0.97, txt,
            transform=ax_top.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="gray", alpha=0.9),
        )
        ax_top.set_xlim(*xlim)
        ax_top.set_ylim(*ylim)
        ax_top.set_title(LABELS[m], fontsize=11, loc="left")
        if col == 0:
            ax_top.set_ylabel(r"$x_1(t_{25})$  (warmup)", fontsize=10)
        ax_top.tick_params(labelsize=9)
        ax_top.grid(True, alpha=0.2)
        if col == 0:
            ax_top.legend(loc="lower right", fontsize=8, framealpha=0.9)

        # ---------- Bottom row: post-warmup local behaviour ----------
        ax_bot = axes[1, col]
        # Full post-warmup cloud (light grey)
        ax_bot.scatter(xi_p, xj_p, s=3, c="lightgray",
                       alpha=0.35, edgecolor="none", zorder=1)

        # Last POST_TRAIL consecutive samples with plasma colour
        tr_i = xi_p[:POST_TRAIL]
        tr_j = xj_p[:POST_TRAIL]
        cmap2 = matplotlib.colormaps.get_cmap("plasma")
        for k in range(len(tr_i) - 1):
            ax_bot.plot(tr_i[k:k+2], tr_j[k:k+2],
                        color=cmap2(k / max(1, POST_TRAIL - 2)),
                        lw=1.0, alpha=0.85, zorder=2)
        ax_bot.scatter(tr_i, tr_j, s=10, c=np.arange(len(tr_i)),
                       cmap="plasma", edgecolor="white", lw=0.3, zorder=3)

        # Mean step annotation
        steps = np.hypot(np.diff(tr_i), np.diff(tr_j))
        ax_bot.text(
            0.03, 0.97,
            f"mean step = {steps.mean():.3f}\n(over {POST_TRAIL} samples)",
            transform=ax_bot.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="gray", alpha=0.9),
        )
        ax_bot.set_xlim(*xlim_p)
        ax_bot.set_ylim(*ylim_p)
        ax_bot.set_xlabel(r"$x_1(t_{24})$", fontsize=10)
        if col == 0:
            ax_bot.set_ylabel(r"$x_1(t_{25})$  (post-warmup)", fontsize=10)
        ax_bot.tick_params(labelsize=9)
        ax_bot.grid(True, alpha=0.2)

    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
