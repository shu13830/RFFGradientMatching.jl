#!/usr/bin/env python3
"""Generate Figure 1: 3-method MCMC mixing comparison (RFFGM | GPGM | MAGI),
matching Figure 2's layout. Single row of θ_a trace plots.

Proximity-to-truth indicators (star, start/end markers) are intentionally
removed; the message is about exploration / x--x dependency reduction.

Inputs:  experiments/pgm2026/results/exp1/exp1_background_{RFFGM,GPGM,MAGI}_seed42.csv
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
EXP1 = os.path.join(BASE, "experiments/pgm2026/results/exp1")
OUT = os.path.join(BASE, "paper/figures/mcmc_trajectory.png")

METHODS = ["RFFGM", "GPGM", "MAGI"]
LABELS = {"RFFGM": "(a) RFFGM (ours)", "GPGM": "(b) GPGM", "MAGI": "(c) MAGI"}
COLOR = {"RFFGM": "#d73027", "GPGM": "#4575b4", "MAGI": "#1a9850"}


def load(path):
    theta = []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            theta.append(float(row["theta_a"]))
    return np.array(theta)


def integrated_act(x, c=5):
    """Sokal's windowed integrated autocorrelation time."""
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
    data = {m: load(os.path.join(EXP1, f"exp1_background_{m}_seed42.csv")) for m in METHODS}
    if any(v is None for v in data.values()):
        missing = [m for m, v in data.items() if v is None]
        print(f"Missing data for: {missing}", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True, sharey=True)

    # Shared y-range so chain amplitudes are visually comparable
    y_lo = min(data[m].min() for m in METHODS)
    y_hi = max(data[m].max() for m in METHODS)
    pad = 0.08 * (y_hi - y_lo)
    y_range = (y_lo - pad, y_hi + pad)

    for col, m in enumerate(METHODS):
        ax = axes[col]
        trace = data[m]
        it = np.arange(1, len(trace) + 1)
        ax.plot(it, trace, color=COLOR[m], lw=0.5, alpha=0.9)
        ax.set_ylim(*y_range)
        ax.set_xlim(0, len(trace))
        if col == 0:
            ax.set_ylabel(r"$\theta_a$", fontsize=11)
        ax.set_xlabel("Post-warmup MCMC iteration", fontsize=10)
        ax.set_title(LABELS[m], fontsize=11, loc="left")
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2, axis="y")

        # Annotations
        amp = trace.max() - trace.min()
        tau = integrated_act(trace)
        ess = len(trace) / tau if tau > 0 else 0
        if amp < 1e-3:
            txt = f"trace amplitude\n$\\approx 0$\n(chain stuck)"
        else:
            txt = (
                f"trace amplitude = {amp:.2f}\n"
                f"$\\tau$ = {tau:.0f}    $N/\\tau$ = {ess:.0f}"
            )
        ax.text(
            0.98, 0.03, txt,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.92),
        )

    plt.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
