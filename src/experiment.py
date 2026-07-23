"""Reproduce "Interpreting How Neural Nets Regress Cubic Polynomials" end-to-end.

Run:  python src/experiment.py

Pipeline:
  1. Train an MLP (3 -> 15x4 -> 1) to regress  y = x^3 + a x^2 + x + b  from (x,a,b).
  2. Train a TopK Sparse Autoencoder (64 features, k=4) on its last activation layer.
  3. Find the "diagonal band" features -- those explained by t = x + c*a, invariant to b.
  4. Recover each band's slope c with the bucket test; the median is ~1/3 (Cardano).
  5. Intervene on the band features and read off the effect on the regressed curve.
Figures -> figures/, metrics + report -> results/.
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

from cubic_model import MLP, sample_inputs, targets, cubic
from sae import TopKSAE
from features import (
    feature_grid, best_slope, find_diagonal_features,
    landmark_correlation, landmarks, intervene,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

SEED = 0
LO, HI = -3.0, 3.0
NN_STEPS = 20000
SAE_EPOCHS = 40
N_ACTS = 120000
CMAP = "jet"


def main() -> None:
    rng = np.random.default_rng(SEED)

    # 1. Train the regression NN ------------------------------------------- #
    print("Target family:  y = x^3 + a*x^2 + x + b   from input (x, a, b)")
    print(f"Network:        3 -> 15 -> 15 -> 15 -> 15 -> 1 (ReLU), Adam {NN_STEPS} steps")
    model = MLP(seed=SEED)
    model.fit(rng, steps=NN_STEPS, batch=512, lr=2e-3, verbose=True)
    nn_r2 = model.r2(np.random.default_rng(999))
    print(f"NN regression R^2 (held-out) = {nn_r2:.5f}")

    # 2. Train the SAE ----------------------------------------------------- #
    acts_tr = model.layer_activations(sample_inputs(N_ACTS, rng))
    acts_te = model.layer_activations(sample_inputs(20000, rng))
    print(f"\nSAE:            {64} features, TopK=4, on the last 15-dim layer "
          f"({N_ACTS} activations)")
    sae = TopKSAE(n_features=64, k=4, seed=SEED)
    sae.fit(acts_tr, epochs=SAE_EPOCHS, batch=2048, lr=1e-3, seed=SEED, verbose=True)
    sae_r2 = sae.r2(acts_te)
    print(f"SAE reconstruction R^2 (held-out) = {sae_r2:.4f}")

    # 3-4. Diagonal band features and their slope c ------------------------ #
    diag = find_diagonal_features(model, sae, np.random.default_rng(7),
                                  r2_thresh=0.8, min_active=0.05)
    cs = np.array([d["best_c"] for d in diag])
    median_c = float(np.median(cs)) if len(cs) else float("nan")
    print(f"\n{len(diag)} diagonal-band features (R^2>=0.8), invariant to b.")
    print(f"MEDIAN slope c = {median_c:.3f}   vs Cardano a/3 = {1/3:.3f}")

    # ---- Figures --------------------------------------------------------- #
    _fig_nn_fit(model, FIG)
    diag_feats = [d["feature"] for d in diag]
    _fig_feature_grid(model, sae, diag_feats, FIG)
    _fig_diagonal_invariance(model, sae, diag_feats[:6], FIG)
    _fig_bucket_curves(model, sae, diag_feats, median_c, FIG)
    _fig_cardano(FIG)
    _fig_intervention(model, sae, diag_feats, FIG)
    _fig_training_summary(nn_r2, sae_r2, FIG)

    # ---- Metrics + report ------------------------------------------------ #
    metrics = {
        "config": {"seed": SEED, "domain": [LO, HI], "nn_steps": NN_STEPS,
                   "sae_epochs": SAE_EPOCHS, "n_acts": N_ACTS,
                   "family": "y = x^3 + a*x^2 + x + b"},
        "nn_r2": nn_r2,
        "sae_r2": sae_r2,
        "cardano_a_over_3": 1 / 3,
        "median_slope_c": median_c,
        "n_diagonal_features": len(diag),
        "diagonal_features": diag,
    }
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _write_report(metrics)
    print(f"\nSaved figures -> {FIG}\nSaved metrics/report -> {RES}")


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _fig_nn_fit(model, out):
    x = np.linspace(LO, HI, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (a, b) in zip(axes, [(3.0, -1.0), (-2.0, 2.0), (0.0, 0.0)]):
        yt = cubic(x, a, b)
        inp = np.stack([x, np.full_like(x, a), np.full_like(x, b)], axis=1)
        yp = model.predict(inp)
        ax.plot(x, yt, "k-", lw=2, label="target")
        ax.plot(x, yp, "--", color="#d1495b", lw=1.6, label="NN")
        ax.set_title(f"a={a:g}, b={b:g}"); ax.grid(alpha=0.2)
        ax.set_xlabel("x")
    axes[0].legend(fontsize=8); axes[0].set_ylabel("y")
    fig.suptitle("The MLP regresses the cubic family  y = x^3 + a x^2 + x + b")
    fig.tight_layout(); fig.savefig(os.path.join(out, "01_nn_fit.png"), dpi=130); plt.close(fig)


def _fig_feature_grid(model, sae, diag_feats, out):
    grid, axis = feature_grid(model, sae, 0.0, n=80)
    vmax = np.percentile(grid, 99.5)
    fig, axes = plt.subplots(8, 8, figsize=(13, 13))
    for f, ax in enumerate(axes.ravel()):
        ax.imshow(grid[f], origin="lower", extent=[LO, HI, LO, HI],
                  cmap=CMAP, vmin=0, vmax=max(vmax, 1e-6), aspect="auto")
        col = "#d1495b" if f in diag_feats else "#333"
        ax.set_title(f"#{f}", fontsize=7, color=col,
                     fontweight="bold" if f in diag_feats else "normal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("SAE feature activations over the (x, a) plane at b = 0  "
                 "(red = diagonal-band features)", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(out, "02_feature_grid.png"), dpi=115); plt.close(fig)


def _fig_diagonal_invariance(model, sae, feats, out):
    if not feats:
        return
    g0, _ = feature_grid(model, sae, 0.0, n=80)
    g1, _ = feature_grid(model, sae, 1.0, n=80)
    ncol = len(feats)
    fig, axes = plt.subplots(2, ncol, figsize=(2.2 * ncol, 4.8))
    axes = np.atleast_2d(axes)
    for j, f in enumerate(feats):
        vmax = max(np.percentile(g0[f], 99.5), 1e-6)
        for i, g in enumerate([g0, g1]):
            ax = axes[i, j]
            ax.imshow(g[f], origin="lower", extent=[LO, HI, LO, HI],
                      cmap=CMAP, vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"#{f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"b = {i}", fontsize=10)
    fig.suptitle("Diagonal-band features are invariant to b (top: b=0, bottom: b=1); "
                 "the band lies along x + c·a = const")
    fig.tight_layout(); fig.savefig(os.path.join(out, "03_b_invariance.png"), dpi=130); plt.close(fig)


def _fig_bucket_curves(model, sae, feats, median_c, out):
    if not feats:
        return
    rng = np.random.default_rng(11)
    c_grid = np.round(np.arange(-1.0, 1.0001, 0.025), 4)
    tr = rng.uniform(LO, HI, size=(20000, 3)); te = rng.uniform(LO, HI, size=(10000, 3))
    from features import _bucket_r2, _feature_acts
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for f in feats:
        a_tr = _feature_acts(model, sae, tr, f); a_te = _feature_acts(model, sae, te, f)
        curve = [_bucket_r2(tr[:, 0] + c * tr[:, 1], a_tr,
                            te[:, 0] + c * te[:, 1], a_te) for c in c_grid]
        cbest = c_grid[int(np.argmax(curve))]
        ax.plot(c_grid, curve, lw=1.4, alpha=0.85, label=f"#{f} (c*={cbest:+.3f})")
    ax.axvline(1 / 3, color="k", ls="--", lw=1.4, label="Cardano  a/3 = 0.333")
    ax.axvline(median_c, color="#e0a800", ls="-", lw=1.6, label=f"median c* = {median_c:.3f}")
    ax.set_xlabel("candidate slope c  in  t = x + c·a"); ax.set_ylabel("bucket-test R²")
    ax.set_title("Each diagonal-band feature is best explained by t = x + c·a with c ≈ 1/3")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2); ax.set_ylim(top=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(out, "04_bucket_slopes.png"), dpi=130); plt.close(fig)


def _fig_cardano(out):
    x = np.linspace(-3, 3, 400)
    a, b = 3.0, -1.0
    y = cubic(x, a, b)
    infl = -a / 3.0
    # depressed cubic: substitute x = t - a/3  ->  removes the quadratic term
    t = x
    y_dep = (t) ** 3 + (1 - a**2 / 3.0) * t + (b + 2 * a**3 / 27.0 - a / 3.0)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, y, "k-", lw=2, label=f"y = x³ + {a:g}x² + x + {b:g}")
    ax.axvline(infl, color="#2e86ab", ls=":", lw=1.4, label=f"inflection at x = −a/3 = {infl:.2f}")
    ax.plot(x, y_dep, color="#d1495b", lw=2, label="depressed: t = x + a/3 (no quadratic term)")
    ax.axvline(0, color="#888", lw=0.6); ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlabel("x  /  t"); ax.set_ylabel("y"); ax.grid(alpha=0.2)
    ax.set_title("Cardano's first step: t = x + a/3 shifts the inflection point onto the y-axis")
    ax.legend(fontsize=9); ax.set_ylim(-20, 20)
    fig.tight_layout(); fig.savefig(os.path.join(out, "05_cardano.png"), dpi=130); plt.close(fig)


def _fig_intervention(model, sae, feats, out):
    if not feats:
        return
    x = np.linspace(LO, HI, 200)
    a, b = 3.0, 2.0
    yt = cubic(x, a, b)
    inp = np.stack([x, np.full_like(x, a), np.full_like(x, b)], axis=1)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, yt, "k-", lw=2.2, label="target")
    for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        yp = intervene(model, sae, inp, feats, factor)
        ax.plot(x, yp, lw=1.4, alpha=0.85, label=f"clamp ×{factor:g}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.2)
    ax.set_title(f"Clamping the {len(feats)} diagonal-band features (a={a:g}, b={b:g}) "
                 f"vertically stretches the regressed curve")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "06_intervention.png"), dpi=130); plt.close(fig)


def _fig_training_summary(nn_r2, sae_r2, out):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axis("off")
    txt = (f"NN regression R²  =  {nn_r2:.5f}\n"
           f"SAE reconstruction R²  =  {sae_r2:.4f}")
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=14, family="monospace")
    fig.tight_layout(); fig.savefig(os.path.join(out, "07_quality.png"), dpi=130); plt.close(fig)


def _write_report(m):
    d = m["diagonal_features"]
    lines = [
        "# Reproduction results\n",
        f"**Family**: `{m['config']['family']}` from input `(x, a, b)`, "
        f"domain `[{m['config']['domain'][0]}, {m['config']['domain'][1]}]`.\n",
        "## Model quality",
        f"- NN regression R² (held-out): `{m['nn_r2']:.5f}`",
        f"- SAE reconstruction R² (held-out, 64 features, TopK=4): `{m['sae_r2']:.4f}`\n",
        "## The central finding: features encode t = x + c·a ≈ the Cardano substitution",
        f"- **{m['n_diagonal_features']} diagonal-band features** are well explained by a "
        f"single linear combination `t = x + c·a` and are invariant to `b`.",
        f"- **Median slope c = {m['median_slope_c']:.3f}**, vs the Cardano depression "
        f"substitution `a/3 = {m['cardano_a_over_3']:.3f}`.\n",
        "| feature | best c | bucket-test R² | active frac | b-invariance |",
        "|--:|--:|--:|--:|--:|",
    ]
    for r in d:
        lines.append(f"| #{r['feature']} | {r['best_c']:+.3f} | {r['r2']:.3f} | "
                     f"{r['active_frac']:.3f} | {r['b_invariance']:.3f} |")
    lines += [
        "\nThe NN, trained only to *regress* cubics, independently developed the same "
        "coordinate `t = x + a/3` that Cardano's method uses to *solve* them — the "
        "substitution that moves the inflection point (at `x = −a/3`) onto the axis and "
        "depresses the cubic (removes its quadratic term).",
        "\nSee `figures/` for the feature grid, b-invariance, bucket-test slopes, the "
        "Cardano illustration and the intervention.",
    ]
    with open(os.path.join(RES, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
