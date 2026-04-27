#!/usr/bin/env python3
"""Figure 1: per-seed posterior mean scatter on protein signaling transduction (PST).

Each method's 10 seeds are plotted as points in the (theta_4, theta_6) plane
(true theta = (0.07, 0.6, 0.05, 0.3, 0.017, 0.3)). This single view shows three
things at once:

  * RFFGM (with ExpPower kernel) clusters tightly near truth on both axes
    (RMSD 0.18 at N=50) --- a visually and numerically clear accuracy win.
  * GPGM (with Matern-5/2) hits truth on most seeds but has visible seed-level
    failures in theta_6 (two seeds > 1, one seed > 3), matching the RMSD 0.32
    with high variance reported in Table 2.
  * MAGI's chain DOES mix (it is not stuck); however, it converges to a
    consistent biased posterior around (theta_4, theta_6) approx (0.48, 3.9),
    which is what the paper's "implementation-sensitive" story means.

Inputs:  scripts/results/main/{rffgm,gpgm,magi}/results.csv and
         scripts/results/kernel/rffgm_kernels.csv
Output:  paper/figures/mcmc_trajectory.png (replaces the older scatter fig)
"""
from __future__ import annotations
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "paper/figures/mcmc_trajectory.png")

# (true theta values; paper convention)
TRUE_T4 = 0.3
TRUE_T6 = 0.3

# colours matching Figure 2
COLOUR = {"RFFGM": "#d73027", "GPGM": "#4575b4", "MAGI": "#1a9850"}
LABEL = {
    "RFFGM": "(a) RFFGM (ours)\nExpPower, $\\gamma{=}10^{-3}$",
    "GPGM": "(b) GPGM\nMat\\'ern-5/2, $\\gamma{=}10^{-3}$",
    "MAGI": "(c) Julia MAGI\nRBF, $\\gamma{=}0.5$",
}


def load(path, method, kernel, gamma, ode="PST", N="50", theta="1"):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("method") != method:
                continue
            if row["ode"] != ode or row["N"] != N or row.get("theta_id", "1") != theta:
                continue
            if row.get("kernel") != kernel:
                continue
            try:
                g = float(row.get("gamma", 0))
            except ValueError:
                continue
            if abs(g - gamma) > 1e-5:
                continue
            tm = [float(x) for x in row["theta_mean"].split(";")]
            out.append((int(row["seed"]), tm))
    return out


def main():
    rffgm = load(
        os.path.join(BASE, "scripts/results/kernel/rffgm_kernels.csv"),
        "RFFGM", "ExpPower", 0.001,
    )
    gpgm = load(
        os.path.join(BASE, "scripts/results/main/gpgm/results.csv"),
        "GPGM", "Matern52", 0.001,
    )
    magi = load(
        os.path.join(BASE, "scripts/results/main/magi/results.csv"),
        "MAGI", "RBF", 0.5,
    )

    data = {"RFFGM": rffgm, "GPGM": gpgm, "MAGI": magi}

    fig, axes = plt.subplots(
        1, 3,
        figsize=(12, 3.8),
        constrained_layout=True,
        sharex=True, sharey=True,
    )

    # use log-y to accommodate MAGI's large theta_6 biases
    for ax, method in zip(axes, ["RFFGM", "GPGM", "MAGI"]):
        pts = data[method]
        t4 = np.array([p[1][3] for p in pts])
        t6 = np.array([p[1][5] for p in pts])
        c = COLOUR[method]

        # Posterior-mean points
        ax.scatter(t4, t6 + 1e-4, s=55, c=c, edgecolor="white",
                   linewidth=0.7, zorder=3, alpha=0.95)
        # Truth marker
        ax.scatter([TRUE_T4], [TRUE_T6], marker="*", s=240,
                   c="gold", edgecolor="black", linewidth=1.0, zorder=5)
        # Dashed lines at truth for reference
        ax.axvline(TRUE_T4, color="k", lw=0.4, ls="--", alpha=0.4)
        ax.axhline(TRUE_T6, color="k", lw=0.4, ls="--", alpha=0.4)

        # Aggregate RMSD annotation (posterior-mean vs. truth)
        true_theta = np.array([0.07, 0.6, 0.05, 0.3, 0.017, 0.3])
        rmsds = []
        for _, tm in pts:
            tm_arr = np.array(tm[:6])
            rmsds.append(np.sqrt(np.mean((tm_arr - true_theta) ** 2)))
        mean_rmsd = float(np.mean(rmsds)) if rmsds else float("nan")
        std_rmsd = float(np.std(rmsds, ddof=1)) if len(rmsds) >= 2 else 0.0
        txt = f"RMSD = {mean_rmsd:.2f} $\\pm$ {std_rmsd:.2f}\n(10 seeds, $n{{=}}{len(pts)}$)"
        ax.text(
            0.04, 0.96, txt,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="gray", alpha=0.92),
        )

        ax.set_yscale("log")
        ax.set_xlim(-0.02, 0.65)
        ax.set_ylim(0.007, 10)
        ax.set_xlabel(r"$\theta_4$ (true $= 0.3$)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(r"$\theta_6$ (true $= 0.3$, log scale)", fontsize=10)
        ax.set_title(LABEL[method], fontsize=10, loc="left")
        ax.tick_params(labelsize=9)
        ax.grid(True, which="both", alpha=0.2)

    # Single legend for truth marker
    fig.text(
        0.5, -0.02,
        r"Posterior-mean $(\theta_4, \theta_6)$ over 10 seeds. "
        r"Gold star: true value $(0.3, 0.3)$. "
        "Tighter clusters closer to the star indicate more accurate and more reliable estimates.",
        ha="center", va="top", fontsize=9,
    )

    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
