#!/usr/bin/env python3
"""Compute the numerical content of the paper's main tables from the raw CSVs.

Reads:
  scripts/results/main/{rffgm,gpgm,magi}/results.csv
  scripts/results/kernel/rffgm_kernels.csv
  scripts/results/scaling/results.csv
  scripts/results/rmagi/rmagi_{lv,sir,pst,fn,lvc}.csv
  scripts/results/auto_gamma/results.csv
Prints:
  Table 1 (main_rmsd, N=25 oracle)
  Table 2 (rmsd, N=50 oracle + N=25 per-theta)
  Table 5 (kernels)
  Table 6 (scaling)
  Table 9 (auto_gamma)
  Table 8 (theta robustness) — Appendix
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev

try:
    from scipy.stats import t as tdist
except ImportError:
    print("Please install scipy (pip install scipy) for Welch t-test p-values.", file=sys.stderr)
    tdist = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "scripts", "results")

ODES = ["LV", "SIR", "PST", "FN"]


def welch_t(a, b):
    """Two-sided Welch t-test. Returns (t, p). Falls back to p=1 if scipy missing."""
    if len(a) < 2 or len(b) < 2 or tdist is None:
        return 0.0, 1.0
    n1, n2 = len(a), len(b)
    m1, m2 = mean(a), mean(b)
    v1, v2 = stdev(a) ** 2, stdev(b) ** 2
    if v1 == 0 and v2 == 0:
        return 0.0, 1.0
    se = sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    p = 2 * (1 - tdist.cdf(abs(t), df))
    return t, p


def load_csv(path, N=None, theta_id=None):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            if N is not None and str(row.get("N")) != str(N):
                continue
            if theta_id is not None and str(row.get("theta_id", "1")) != str(theta_id):
                continue
            try:
                row["rmsd"] = float(row["rmsd"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(row)
    return rows


def oracle(rows):
    """For a list of rows, find (kernel, gamma) with lowest mean rmsd; return its rmsd list."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("kernel", "?"), r.get("gamma", "?"))].append(r["rmsd"])
    if not groups:
        return [], None
    best = min(groups, key=lambda k: mean(groups[k]))
    return groups[best], best


def method_rows(method, ode, N=25, theta_id=1):
    out = []
    for d in ["main/rffgm", "main/gpgm", "main/magi", "kernel"]:
        path = os.path.join(RES, d, "results.csv" if d != "kernel" else "rffgm_kernels.csv")
        for r in load_csv(path, N=N, theta_id=theta_id):
            if r.get("method") == method and r.get("ode") == ode:
                out.append(r)
    return out


def print_table1_2(N, theta_id=1):
    print(f"\n=== N={N}, θ_id={theta_id} oracle ===")
    for ode in ODES:
        r_rows = method_rows("RFFGM", ode, N=N, theta_id=theta_id)
        g_rows = method_rows("GPGM", ode, N=N, theta_id=theta_id)
        m_rows = method_rows("MAGI", ode, N=N, theta_id=theta_id)
        r_vals, r_key = oracle(r_rows)
        g_vals, g_key = oracle(g_rows)
        m_vals, m_key = oracle(m_rows)
        if not r_vals:
            continue
        line = f"  {ode}: RFFGM {mean(r_vals):.4f} ± {stdev(r_vals):.4f} [k={r_key}]"
        if g_vals:
            t, p = welch_t(r_vals, g_vals)
            line += f" | GPGM {mean(g_vals):.4f} ± {stdev(g_vals):.4f} (p={p:.3f})"
        if m_vals:
            t, p = welch_t(r_vals, m_vals)
            line += f" | MAGI {mean(m_vals):.4f} ± {stdev(m_vals):.4f} (p={p:.3f})"
        print(line)


def print_table5():
    print("\n=== Table 5: RFFGM kernel flexibility (N=25, θ_id=1) ===")
    kernel_path = os.path.join(RES, "kernel", "rffgm_kernels.csv")
    for ode in ODES:
        rows = [r for r in load_csv(kernel_path, N=25, theta_id=1) if r["ode"] == ode]
        also = [r for r in load_csv(os.path.join(RES, "main", "rffgm", "results.csv"), N=25, theta_id=1) if r["ode"] == ode]
        all_rows = rows + also
        per_kernel = defaultdict(list)
        for r in all_rows:
            per_kernel[r["kernel"]].append(r["rmsd"])
        cells = []
        for k in ["RBF", "Matern52", "Laplace", "GenCauchy", "ExpPower"]:
            if k in per_kernel:
                per_gamma = defaultdict(list)
                for r in all_rows:
                    if r["kernel"] == k:
                        per_gamma[r["gamma"]].append(r["rmsd"])
                best = min(per_gamma, key=lambda g: mean(per_gamma[g]))
                cells.append(f"{k}={mean(per_gamma[best]):.3f} (γ={best})")
            else:
                cells.append(f"{k}=--")
        print(f"  {ode}: " + " | ".join(cells))


def print_table6():
    print("\n=== Table 6: LVC scaling ===")
    path = os.path.join(RES, "scaling", "results.csv")
    if not os.path.exists(path):
        print("  (scaling/results.csv not found)")
        return
    by_method_K = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                by_method_K[(row["method"], int(row["K"]))].append((
                    float(row["rmsd"]),
                    float(row["ess_mean"]),
                    float(row["rhat_max"]),
                ))
            except (KeyError, ValueError):
                continue
    for K in [2, 5, 10, 15]:
        for meth in ["RFFGM", "GPGM"]:
            vals = by_method_K.get((meth, K), [])
            if not vals:
                continue
            rmsds = [v[0] for v in vals]
            ess = [v[1] for v in vals]
            rhat = [v[2] for v in vals]
            print(
                f"  K={K:2d} {meth}: RMSD {mean(rmsds):.3f}±{stdev(rmsds):.3f} "
                f"ESS {mean(ess):.0f} Rhat_max {max(rhat):.2f} (n={len(vals)})"
            )


def print_table8():
    print("\n=== Table 8 (App H): θ-pattern robustness (N=25, CV) ===")
    for ode in ODES:
        for meth in ["RFFGM", "GPGM", "MAGI"]:
            per_theta = defaultdict(list)
            for tid in [1, 2, 3]:
                rows = method_rows(meth, ode, N=25, theta_id=tid)
                vals, key = oracle(rows)
                if vals:
                    per_theta[tid] = vals
            if len(per_theta) < 3:
                continue
            means = [mean(per_theta[t]) for t in [1, 2, 3]]
            cv = (stdev(means) / mean(means) * 100) if mean(means) > 0 else float("inf")
            print(f"  {ode} {meth}: θ1={means[0]:.3f} θ2={means[1]:.3f} θ3={means[2]:.3f} CV={cv:.1f}%")


def print_table9():
    print("\n=== Table 9 (App K): auto-γ selection ===")
    path = os.path.join(RES, "auto_gamma", "results.csv")
    if not os.path.exists(path):
        print("  (auto_gamma/results.csv not found)")
        return
    groups = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["rmsd"] = float(row["rmsd"])
                row["resid_mean"] = float(row["resid_mean"])
                row["rhat_max"] = float(row["rhat_max"])
            except (KeyError, ValueError):
                continue
            groups[(row["ode"], row["N"], row["theta_id"], row["seed"])].append(row)
    for label, filt in [
        ("Rhat<1.1", lambda r: r["rhat_max"] < 1.1),
        ("no filter", lambda r: True),
    ]:
        e = w = t = 0
        ratios = []
        for g in groups.values():
            conv = [r for r in g if filt(r)]
            if len(conv) < 2:
                continue
            orc = min(conv, key=lambda r: r["rmsd"])
            auto = min(conv, key=lambda r: r["resid_mean"])
            t += 1
            if auto["gamma"] == orc["gamma"]:
                e += 1
            if auto["rmsd"] <= 1.5 * orc["rmsd"]:
                w += 1
            if orc["rmsd"] > 0:
                ratios.append(auto["rmsd"] / orc["rmsd"])
        ratios.sort()
        med = ratios[len(ratios) // 2] if ratios else float("nan")
        if t > 0:
            print(
                f"  {label}: n={t}  exact {e}/{t} ({100*e/t:.0f}%)  "
                f"within1.5x {w}/{t} ({100*w/t:.0f}%)  median ratio {med:.2f}"
            )


if __name__ == "__main__":
    print_table1_2(25)
    print_table1_2(50)
    print_table5()
    print_table6()
    print_table8()
    print_table9()
