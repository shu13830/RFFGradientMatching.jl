#!/usr/bin/env python3
"""Figure 1: per-method (x_i, x_{i+1}) posterior scatter + 100 consecutive MCMC
steps on Lotka--Volterra (N=25, seed 42). All three methods face the same
narrow X-X posterior ridge (corr approx 1) induced by the ODE smoothness, so the
scatter directly visualises the variable-to-variable dependency that makes MCMC
hard. Annotations: integrated autocorrelation time tau and effective sample
size (ESS) of the x_i chain. RFFGM reaches ESS approx 56 vs GPGM 30 vs MAGI 3
on the same 5000-sample budget, even though all three chains move along the
same ridge.

Inputs:  experiments/pgm2026/results/exp1_scatter/exp1s_{method}_seed42_{post,traj}.csv
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
LABELS = {"RFFGM": "(a) RFFGM (ours)", "GPGM": "(b) GPGM", "MAGI": "(c) Julia MAGI"}
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


def integrated_act(x, c=5):
    x = x - x.mean()
    var = x.var()
    if var == 0 or len(x) < 2:
        return float("inf")
    max_lag = min(len(x) // 3, 2000)
    tau = 1.0
    sum_r = 0.0
    for k in range(1, max_lag + 1):
        r_k = np.mean(x[:-k] * x[k:]) / var
        sum_r += r_k
        tau = 1 + 2 * sum_r
        if k >= c * tau:
            break
    return max(tau, 1.0)


def main():
    meta = {}
    with open(os.path.join(DAT, "exp1s_meta.csv")) as f:
        for row in csv.DictReader(f):
            meta = row
    pair_i = meta.get("pair_i_obs", "?")
    pair_j = meta.get("pair_j_obs", "?")
    ti = float(meta.get("t_i", "0"))
    tj = float(meta.get("t_j", "0"))

    data = {}
    for m in METHODS:
        post_xi, post_xj = load(os.path.join(DAT, f"exp1s_{m}_seed42_post.csv"))
        traj_xi, traj_xj = load(os.path.join(DAT, f"exp1s_{m}_seed42_traj.csv"))
        if post_xi is None:
            print(f"Missing: {m}", file=sys.stderr)
            continue
        data[m] = dict(post_xi=post_xi, post_xj=post_xj,
                       traj_xi=traj_xi, traj_xj=traj_xj)

    all_xi = np.concatenate([d["post_xi"] for d in data.values()])
    all_xj = np.concatenate([d["post_xj"] for d in data.values()])
    px = 0.05 * (all_xi.max() - all_xi.min())
    py = 0.05 * (all_xj.max() - all_xj.min())
    xlim = (all_xi.min() - px, all_xi.max() + px)
    ylim = (all_xj.min() - py, all_xj.max() + py)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0),
                             constrained_layout=True, sharex=True, sharey=True)

    for col, m in enumerate(METHODS):
        ax = axes[col]
        d = data[m]

        # Posterior cloud
        ax.scatter(d["post_xi"], d["post_xj"], s=3, c="lightgray",
                   alpha=0.35, edgecolor="none", zorder=1)

        # 100 consecutive samples with plasma colouring
        tr_xi, tr_xj = d["traj_xi"], d["traj_xj"]
        n_tr = len(tr_xi)
        cmap = matplotlib.colormaps.get_cmap("plasma")
        for i in range(n_tr - 1):
            ax.plot(tr_xi[i:i+2], tr_xj[i:i+2],
                    color=cmap(i / max(1, n_tr - 2)),
                    lw=1.2, alpha=0.85, zorder=2)
        ax.scatter(tr_xi, tr_xj, s=14, c=np.arange(n_tr), cmap="plasma",
                   edgecolor="white", lw=0.4, zorder=3)

        # Diagonal y=x reference
        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.4, zorder=0)

        # ESS + tau annotation
        N = len(d["post_xi"])
        tau = integrated_act(d["post_xi"])
        ess = N / tau if tau > 0 else 0
        # Also posterior correlation (the "ridge")
        r = float(np.corrcoef(d["post_xi"], d["post_xj"])[0, 1]) if d["post_xi"].std() > 0 else 1.0
        txt = (
            f"post.\\ corr = {r:.3f}\n"
            f"$\\tau$ = {tau:.0f}    ESS = {ess:.0f}"
        )
        ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", alpha=0.9))

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        xlabel = f"$x_1(t_{{{pair_i}}}) \\approx x_1({ti:.2f})$"
        ylabel = f"$x_1(t_{{{pair_j}}}) \\approx x_1({tj:.2f})$"
        ax.set_xlabel(xlabel, fontsize=10)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(LABELS[m], fontsize=11, loc="left")
        ax.tick_params(labelsize=9)

    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=1, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.04, pad=0.12, aspect=40)
    cbar.set_label("Step index in the 100-step trace (colour $=$ post-warmup iteration)",
                   fontsize=9)

    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
