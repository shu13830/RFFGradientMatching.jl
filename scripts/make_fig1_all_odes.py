#!/usr/bin/env python3
"""Generate per-seed posterior-mean scatter figures for all four ODE systems.

Produces four PNGs (paper/figures/posterior_scatter_{lv,sir,pst,fn}.png) so we
can choose the single most compelling one as main Figure 1 and move the others
to the appendix. Each PNG is a 1 x 3 scatter (RFFGM | GPGM | Julia MAGI).
"""
from __future__ import annotations
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "paper/figures")

COLOUR = {"RFFGM": "#d73027", "GPGM": "#4575b4", "MAGI": "#1a9850"}

# (kernel, gamma, N, theta_id) for each (ODE, method) — oracle settings at N=50 θ_id=1.
ORACLE = {
    "LV":  {"RFFGM": ("RBF",      0.5,   "50"),
            "GPGM":  ("RBF",      0.5,   "50"),
            "MAGI":  ("RBF",      0.5,   "50")},
    "SIR": {"RFFGM": ("Matern52", 0.05,  "50"),
            "GPGM":  ("Matern52", 0.01,  "50"),
            "MAGI":  ("Matern52", 0.1,   "50")},
    "PST": {"RFFGM": ("ExpPower", 0.001, "50"),
            "GPGM":  ("Matern52", 0.001, "50"),
            "MAGI":  ("RBF",      0.5,   "50")},
    "FN":  {"RFFGM": ("RBF",      0.001, "50"),
            "GPGM":  ("RBF",      0.01,  "50"),
            "MAGI":  ("RBF",      0.05,  "50")},
}

# theta components to plot + true value + axis config.
PLOT_SPEC = {
    # ODE : (idx_x, idx_y, true_x, true_y, xlabel, ylabel, xlim, ylim, yscale)
    "LV":  (0, 1, 2.0, 1.0,
            r"$\theta_1$ (true $= 2.0$)", r"$\theta_2$ (true $= 1.0$)",
            (-0.05, 2.5), (-0.05, 2.5), "linear"),
    "SIR": (0, 1, 0.5, 0.25,
            r"$\theta_1$ (true $= 0.5$)", r"$\theta_2$ (true $= 0.25$)",
            (0.35, 0.65), (0.18, 0.32), "linear"),
    "PST": (3, 5, 0.3, 0.3,
            r"$\theta_4$ (true $= 0.3$)", r"$\theta_6$ (true $= 0.3$)",
            (-0.02, 0.65), (0.007, 10), "log"),
    "FN":  (0, 1, 3.0, 0.2,
            r"$\theta_1$ (true $= 3.0$)", r"$\theta_2$ (true $= 0.2$)",
            (-0.2, 5.8), (-0.2, 3.0), "linear"),
}

TRUE_THETA = {
    "LV":  np.array([2.0, 1.0, 4.0, 1.0]),
    "SIR": np.array([0.5, 0.25]),
    "PST": np.array([0.07, 0.6, 0.05, 0.3, 0.017, 0.3]),
    "FN":  np.array([3.0, 0.2, 0.2]),
}


def load_rows(path, method, kernel, gamma, ode, N, theta="1"):
    out = []
    if not os.path.exists(path):
        return out
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


def collect(method, ode):
    kernel, gamma, N = ORACLE[ode][method]
    paths = [
        os.path.join(BASE, "scripts/results/main/rffgm/results.csv"),
        os.path.join(BASE, "scripts/results/kernel/rffgm_kernels.csv"),
        os.path.join(BASE, "scripts/results/main/gpgm/results.csv"),
        os.path.join(BASE, "scripts/results/main/magi/results.csv"),
    ]
    rows = []
    for path in paths:
        rows.extend(load_rows(path, method, kernel, gamma, ode, N))
    return rows, (kernel, gamma, N)


def make_figure(ode):
    idx_x, idx_y, tx, ty, xl, yl, xlim, ylim, yscale = PLOT_SPEC[ode]
    true_theta = TRUE_THETA[ode]

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.8), constrained_layout=True,
        sharex=True, sharey=True,
    )

    for ax, method in zip(axes, ["RFFGM", "GPGM", "MAGI"]):
        pts, (kernel, gamma, N) = collect(method, ode)
        if not pts:
            ax.text(0.5, 0.5, f"No data for {method}",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        tx_vals = np.array([p[1][idx_x] for p in pts])
        ty_vals = np.array([p[1][idx_y] for p in pts])

        c = COLOUR[method]
        # offset log-scale tiny values for visibility
        ty_plot = ty_vals + (1e-4 if yscale == "log" else 0.0)
        ax.scatter(tx_vals, ty_plot, s=55, c=c, edgecolor="white",
                   linewidth=0.7, zorder=3, alpha=0.95)

        # Truth star
        ty_star = ty + (1e-4 if yscale == "log" else 0.0)
        ax.scatter([tx], [ty_star], marker="*", s=240,
                   c="gold", edgecolor="black", linewidth=1.0, zorder=5)
        ax.axvline(tx, color="k", lw=0.4, ls="--", alpha=0.4)
        ax.axhline(ty, color="k", lw=0.4, ls="--", alpha=0.4)

        # per-seed RMSD annotation
        rmsds = [float(np.sqrt(np.mean((np.array(p[1][:len(true_theta)]) - true_theta) ** 2)))
                 for p in pts]
        m_r = float(np.mean(rmsds)) if rmsds else float("nan")
        s_r = float(np.std(rmsds, ddof=1)) if len(rmsds) >= 2 else 0.0

        label_name = {
            "RFFGM": f"(a) RFFGM (ours)\n{kernel}, $\\gamma{{=}}{gamma:g}$",
            "GPGM":  f"(b) GPGM\n{kernel}, $\\gamma{{=}}{gamma:g}$",
            "MAGI":  f"(c) Julia MAGI\n{kernel}, $\\gamma{{=}}{gamma:g}$",
        }[method]
        ax.set_title(label_name, fontsize=10, loc="left")

        ax.text(
            0.04, 0.96,
            f"RMSD = {m_r:.2f} $\\pm$ {s_r:.2f}\n(10 seeds, $n{{=}}{len(pts)}$)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="gray", alpha=0.92),
        )

        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xl, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(yl, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, which="both", alpha=0.2)

    fig.text(
        0.5, -0.02,
        f"{ode}: posterior-mean per seed at $N{{=}}50$ ($\\theta_\\mathrm{{id}}=1$, 10 seeds). "
        "Gold star: true $\\theta$.",
        ha="center", va="top", fontsize=9,
    )

    out_path = os.path.join(OUT_DIR, f"posterior_scatter_{ode.lower()}.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    for ode in ["LV", "SIR", "PST", "FN"]:
        make_figure(ode)
