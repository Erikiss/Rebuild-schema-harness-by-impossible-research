"""Reproduce "Interpreting How Neural Nets Regress Cubic Polynomials" end-to-end.

Run:  python src/experiment.py

Steps:
  1. Build a cubic target and sample training data.
  2. Train the 1->H->1 Ramp network to regress it.
  3. Read the piecewise-linear function off the weights (breakpoints, delta-slopes).
  4. Prove the reading is exact (analytic reconstruction == network forward pass).
  5. Show the kinks congregate where the cubic curves most (|f''| large).
  6. Save figures to figures/ and metrics/report to results/.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cubic_relu_net import Cubic, RampNet
from interpret import (
    extract_neurons,
    reconstruct_pwl,
    equivalence_error,
    curvature_congregation,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
SEED = 0
HIDDEN = 24
DOMAIN = (-2.0, 2.0)
N_TRAIN = 400
STEPS = 30000
LR = 5e-3
CUBIC = Cubic(a3=1.0, a2=0.0, a1=-3.0, a0=0.0)  # f(x) = x^3 - 3x


def main() -> None:
    rng = np.random.default_rng(SEED)
    lo, hi = DOMAIN

    # 1. Data ---------------------------------------------------------------- #
    x_train = np.linspace(lo, hi, N_TRAIN)
    y_train = CUBIC(x_train)
    x_dense = np.linspace(lo, hi, 2001)
    y_dense = CUBIC(x_dense)

    # 2. Train --------------------------------------------------------------- #
    print(f"Target cubic:  f(x) = {CUBIC.as_str()}   on x in [{lo}, {hi}]")
    print(f"Network:       1 -> {HIDDEN} (Ramp) -> 1   [Adam, {STEPS} steps]")
    net = RampNet(hidden=HIDDEN, seed=SEED)
    history = net.fit(x_train, y_train, steps=STEPS, lr=LR, verbose=True)

    y_fit = net.forward(x_dense)
    mse = float(np.mean((net.forward(x_train) - y_train) ** 2))
    max_err = float(np.max(np.abs(y_fit - y_dense)))
    print(f"Final train MSE = {mse:.3e},  max |error| on dense grid = {max_err:.3e}")

    # 3. Interpret ----------------------------------------------------------- #
    neurons = extract_neurons(net)
    pwl = reconstruct_pwl(net, DOMAIN)

    # 4. Exactness check ----------------------------------------------------- #
    eq_err = equivalence_error(net, pwl, x_dense)
    print(f"Interpretation exactness: max|net - analytic CPWL| = {eq_err:.2e} "
          f"(should be ~1e-12)")

    # 5. Curvature congregation --------------------------------------------- #
    congr = curvature_congregation(net, CUBIC, DOMAIN)
    print(f"Effective kinks inside domain: {congr['n_effective_breakpoints']}")
    print(f"Curvature enrichment at kinks: {congr['curvature_enrichment']:.3f}x "
          f"(>1 => kinks favour high-curvature regions)")

    # ---- Figures ----------------------------------------------------------- #
    interior_breaks = np.array(sorted(
        n.breakpoint for n in neurons if lo < n.breakpoint < hi
    ))

    # Fig 1: the fit
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_dense, y_dense, "k-", lw=2.4, label=f"target  f(x) = {CUBIC.as_str()}")
    ax.plot(x_dense, y_fit, color="#d1495b", lw=1.6, label="Ramp-net fit (piecewise linear)")
    ax.plot(x_train[::12], y_train[::12], "o", ms=3.5, color="#2e86ab",
            alpha=0.6, label="training samples")
    for xb in interior_breaks:
        ax.axvline(xb, color="#888", lw=0.5, alpha=0.35)
    ax.scatter(interior_breaks, net.forward(interior_breaks), s=22, color="#e0a800",
               zorder=5, label="breakpoints  -b_i/w_i")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("A Ramp/ReLU net regresses the cubic as a piecewise-linear function")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "01_fit.png"), dpi=130); plt.close(fig)

    # Fig 2: individual neuron ramps
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n in neurons:
        contrib = n.v * np.maximum(0.0, n.w * x_dense + n.b)
        ax.plot(x_dense, contrib, lw=1.0, alpha=0.7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("x"); ax.set_ylabel("v_i * Ramp(w_i x + b_i)")
    ax.set_title("Each hidden neuron contributes one ramp; their sum (+c) is the fit")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_ramps.png"), dpi=130); plt.close(fig)

    # Fig 3: slope (derivative) is piecewise constant, jumping at each kink
    dx = x_dense[1] - x_dense[0]
    slope = np.gradient(y_fit, dx)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_dense, slope, color="#d1495b", lw=1.6, label="d/dx  net fit  (piecewise const.)")
    ax.plot(x_dense, CUBIC.first_derivative(x_dense),
            "k--", lw=1.4, label="f'(x) = 3x^2 - 3  (target slope)")
    for xb in interior_breaks:
        ax.axvline(xb, color="#888", lw=0.5, alpha=0.35)
    ax.set_xlabel("x"); ax.set_ylabel("slope")
    ax.set_title("Slope jumps by |w_i| v_i at each breakpoint -- that is the 'kink'")
    ax.legend(loc="upper center", fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "03_slope.png"), dpi=130); plt.close(fig)

    # Fig 4: breakpoints congregate where |f''| is large
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    curv = np.abs(CUBIC.second_derivative(x_dense))
    ax1.plot(x_dense, curv, color="#2e86ab", lw=2.0, label="|f''(x)| = |6x|  (curvature)")
    ax1.fill_between(x_dense, curv, alpha=0.12, color="#2e86ab")
    jumps = np.array([abs(n.slope_jump) for n in neurons if lo < n.breakpoint < hi])
    if len(interior_breaks):
        ax1.vlines(interior_breaks, 0, np.interp(interior_breaks, x_dense, curv),
                   color="#e0a800", lw=1.4, alpha=0.9)
        ax1.scatter(interior_breaks, np.interp(interior_breaks, x_dense, curv),
                    s=18 + 120 * jumps / (jumps.max() + 1e-9), color="#e0a800",
                    zorder=5, label="breakpoints (size ~ |slope jump|)")
    ax1.set_ylabel("curvature |f''|"); ax1.legend(fontsize=9); ax1.grid(alpha=0.2)
    ax1.set_title(f"Kinks cluster where the cubic bends most "
                  f"(enrichment {congr['curvature_enrichment']:.2f}x)")
    # density histogram of breakpoints weighted by slope jump
    if len(interior_breaks):
        ax2.hist(interior_breaks, bins=16, range=DOMAIN, weights=jumps,
                 color="#e0a800", alpha=0.8)
    ax2.set_xlabel("x"); ax2.set_ylabel("sum |slope jump|"); ax2.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "04_curvature.png"), dpi=130); plt.close(fig)

    # Fig 5: training curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(history, color="#2e86ab", lw=1.2)
    ax.set_xlabel("Adam step"); ax.set_ylabel("MSE"); ax.grid(alpha=0.2)
    ax.set_title("Training loss")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "05_training.png"), dpi=130); plt.close(fig)

    # ---- Save metrics ------------------------------------------------------ #
    metrics = {
        "config": {
            "seed": SEED, "hidden": HIDDEN, "domain": list(DOMAIN),
            "n_train": N_TRAIN, "steps": STEPS, "lr": LR,
            "cubic": CUBIC.as_str(),
        },
        "fit": {"train_mse": mse, "max_abs_error_dense": max_err,
                "final_loss": history[-1]},
        "interpretation": {
            "exactness_max_abs_diff": eq_err,
            "n_neurons": HIDDEN,
            "n_interior_breakpoints": int(len(interior_breaks)),
            "breakpoints": interior_breaks.tolist(),
            "neurons": [
                {"index": n.index, "w": n.w, "b": n.b, "v": n.v,
                 "breakpoint": n.breakpoint, "delta_slope": n.delta_slope,
                 "orientation": n.orientation, "slope_jump": n.slope_jump}
                for n in neurons
            ],
        },
        "curvature_congregation": congr,
    }
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _write_report(metrics)
    print(f"\nSaved figures -> {FIG}\nSaved metrics/report -> {RES}")


def _write_report(m: dict) -> None:
    c = m["config"]; fit = m["fit"]; it = m["interpretation"]; cg = m["curvature_congregation"]
    lines = [
        "# Reproduction results\n",
        f"**Target**: `f(x) = {c['cubic']}` on x in [{c['domain'][0]}, {c['domain'][1]}]  ",
        f"**Network**: `1 -> {c['hidden']} (Ramp) -> 1`, Adam, {c['steps']} steps, lr={c['lr']}, seed={c['seed']}\n",
        "## Fit quality",
        f"- Train MSE: `{fit['train_mse']:.3e}`",
        f"- Max abs error on dense grid: `{fit['max_abs_error_dense']:.3e}`\n",
        "## Interpretation is exact",
        f"- Rebuilding the piecewise-linear function from the weights alone "
        f"(breakpoints `-b_i/w_i`, delta-slopes `w_i v_i`) and comparing to the "
        f"network's own forward pass gives a max difference of "
        f"`{it['exactness_max_abs_diff']:.2e}` -- i.e. the mechanistic reading is "
        f"the network, not an approximation of it.",
        f"- {it['n_interior_breakpoints']} of {it['n_neurons']} neurons place their "
        f"kink inside the domain.\n",
        "## Kinks congregate in high-curvature regions",
        f"- Slope-jump-weighted mean curvature seen by the breakpoints: "
        f"`{cg['weighted_curvature_at_breakpoints']:.3f}`",
        f"- Domain-average curvature: `{cg['mean_domain_curvature']:.3f}`",
        f"- **Curvature enrichment: {cg['curvature_enrichment']:.2f}x** "
        f"(>1 means kinks favour where the cubic bends most).\n",
        "See `figures/` for the plots and `results/metrics.json` for the full per-neuron table.",
    ]
    with open(os.path.join(RES, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
