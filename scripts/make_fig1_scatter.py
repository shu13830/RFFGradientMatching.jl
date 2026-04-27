#!/usr/bin/env python3
"""Figure 1: MCMC exploration in the (x(t_i), x(t_{i+1})) plane for two
temporally adjacent latent states, where the X--X dependency induced by the GP
prior is most visible. 3 columns: RFFGM | GPGM | MAGI (matches Figure 2).

Inputs:  experiments/pgm2026/results/exp1_scatter/exp1s_{method}_seed42_{post,traj}.csv
         experiments/pgm2026/results/exp1_scatter/exp1s_meta.csv
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

METHODS = ["RFFGM", "GPGM", "MAGI"]
LABELS = {"RFFGM": "(a) RFFGM (ours)", "GPGM": "(b) GPGM", "MAGI": "(c) MAGI"}
COLOR = {"RFFGM": "#d73027", "GPGM": "#4575b4", "MAGI": "#1a9850"}


def load(path):
    xi, xj = [], []
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        for row in csv.DictReader(f):
            xi.append(float(row["x_i"]))
            xj.append(float(row["x_j"]))
    return np.array(xi), np.array(xj)


def main():
    # Metadata (for axis labels)
    meta = {}
    meta_path = os.path.join(DAT, "exp1s_meta.csv")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for row in csv.DictReader(f):
                meta = row
    def _f(key):
        try:
            return float(meta[key]) if meta and key in meta else None
        except (ValueError, KeyError, TypeError):
            return None
    def _i(key):
        try:
            return int(float(meta[key])) if meta and key in meta else None
        except (ValueError, KeyError, TypeError):
            return None
    ti = _f("t_i")
    tj = _f("t_j")
    pair_i = _i("pair_i_obs") if (meta and "pair_i_obs" in meta) else _i("pair_i")
    pair_j = _i("pair_j_obs") if (meta and "pair_j_obs" in meta) else _i("pair_j")

    data = {}
    for m in METHODS:
        post_xi, post_xj = load(os.path.join(DAT, f"exp1s_{m}_seed42_post.csv"))
        traj_xi, traj_xj = load(os.path.join(DAT, f"exp1s_{m}_seed42_traj.csv"))
        if post_xi is None:
            print(f"Missing: {m}", file=sys.stderr)
            continue
        data[m] = dict(post_xi=post_xi, post_xj=post_xj,
                       traj_xi=traj_xi, traj_xj=traj_xj)

    # Shared axis range from union of posteriors
    all_xi = np.concatenate([d["post_xi"] for d in data.values()])
    all_xj = np.concatenate([d["post_xj"] for d in data.values()])
    xlo, xhi = all_xi.min(), all_xi.max()
    ylo, yhi = all_xj.min(), all_xj.max()
    # 5% padding
    px = 0.05 * (xhi - xlo)
    py = 0.05 * (yhi - ylo)
    xlim = (xlo - px, xhi + px)
    ylim = (ylo - py, yhi + py)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0),
                             constrained_layout=True, sharex=True, sharey=True)

    for col, m in enumerate(METHODS):
        ax = axes[col]
        if m not in data:
            ax.text(0.5, 0.5, f"{m}: no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        d = data[m]

        # Posterior cloud
        ax.scatter(d["post_xi"], d["post_xj"], s=3, c="lightgray",
                   alpha=0.35, edgecolor="none", zorder=1)

        # Consecutive MCMC samples: line + markers, coloured by step index
        tr_xi, tr_xj = d["traj_xi"], d["traj_xj"]
        n_tr = len(tr_xi)
        cmap = matplotlib.colormaps.get_cmap("plasma")
        for i in range(n_tr - 1):
            ax.plot(tr_xi[i:i+2], tr_xj[i:i+2],
                    color=cmap(i / max(1, n_tr - 2)),
                    lw=1.2, alpha=0.85, zorder=2)
        ax.scatter(tr_xi, tr_xj, s=14, c=np.arange(n_tr), cmap="plasma",
                   edgecolor="white", lw=0.4, zorder=3)

        # Identity line y = x for reference (X-X dependence is tight if
        # the posterior cloud + trajectory lie close to this line)
        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.4, zorder=0)

        # Correlation annotation (MAGI can be stuck -> NaN)
        if d["post_xi"].std() == 0 or d["post_xj"].std() == 0:
            r_text = "stuck"
        else:
            r = float(np.corrcoef(d["post_xi"], d["post_xj"])[0, 1])
            r_text = f"{r:.3f}"
        # Mean Euclidean step of consecutive trajectory samples
        steps = np.hypot(np.diff(tr_xi), np.diff(tr_xj))
        txt = (
            f"post.\\ corr = {r_text}\n"
            f"mean step = {steps.mean():.4f}"
        )
        ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", alpha=0.9))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        xlabel = (f"$x_1(t_{{{pair_i}}}) \\approx x_1({ti:.2f})$"
                  if ti is not None else "$x_1(t_i)$")
        ylabel = (f"$x_1(t_{{{pair_j}}}) \\approx x_1({tj:.2f})$"
                  if tj is not None else "$x_1(t_{i+1})$")
        ax.set_xlabel(xlabel, fontsize=10)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(LABELS[m], fontsize=11, loc="left")
        ax.tick_params(labelsize=9)

    # Colorbar showing MCMC step order
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=1, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.04, pad=0.12, aspect=40)
    cbar.set_label("Step index in the 100-step trace (colour = post-warmup iteration)",
                   fontsize=9)

    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
