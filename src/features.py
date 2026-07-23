"""Feature analysis: find the "diagonal band" features and recover their slope.

The essay's central discovery: several SAE features do not care about x and a
separately, but only about a *linear combination* t = x + c*a, and they are
invariant to b. Plotting a feature's activation over the (x, a) plane at fixed b
shows a diagonal band; the slope of that band is c, and the median c across those
features turns out to be ~0.3 -- surprisingly close to the a/3 substitution of the
Cardano method for depressing a cubic.

This module provides:
  * feature_grid           -- activation of every feature over the (x, a) plane
  * bucket_test / best_slope -- the essay's test to recover the slope c of a band
  * find_diagonal_features -- features well-explained by some t = x + c*a, invariant to b
  * landmark_correlation   -- the earlier "do features track roots/critical/inflection?" probe
  * intervention           -- clamp features and read off the effect on the output
"""

from __future__ import annotations

import numpy as np

from cubic_model import MLP, cubic
from sae import TopKSAE


# --------------------------------------------------------------------------- #
#  Feature activations over the (x, a) plane at fixed b
# --------------------------------------------------------------------------- #
def feature_grid(
    model: MLP, sae: TopKSAE, b_fixed: float, n: int = 80, lo: float = -3.0, hi: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (grid, axis) where grid[f][i, j] is feature f's activation at
    a = axis[i], x = axis[j] (axis runs lo->hi). Plotted with imshow
    origin='lower', so x increases left->right and a increases bottom->top."""
    axis = np.linspace(lo, hi, n)
    xx, aa = np.meshgrid(axis, axis)               # xx varies along cols, aa along rows
    inp = np.stack([xx.ravel(), aa.ravel(), np.full(xx.size, b_fixed)], axis=1)
    z = sae.encode(model.layer_activations(inp))   # (n*n, F)
    grid = z.T.reshape(sae.n_features, n, n)
    return grid, axis


# --------------------------------------------------------------------------- #
#  The bucket test: how well is feature f explained by t = x + c*a ?
# --------------------------------------------------------------------------- #
def _bucket_r2(
    t_train: np.ndarray, act_train: np.ndarray,
    t_test: np.ndarray, act_test: np.ndarray,
    n_buckets: int = 50,
) -> float:
    lo, hi = float(t_train.min()), float(t_train.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_buckets + 1)
    bt = np.clip(np.digitize(t_train, edges) - 1, 0, n_buckets - 1)
    means = np.zeros(n_buckets)
    for k in range(n_buckets):
        sel = bt == k
        if np.any(sel):
            means[k] = act_train[sel].mean()
    bte = np.clip(np.digitize(t_test, edges) - 1, 0, n_buckets - 1)
    pred = means[bte]
    ss_res = float(np.sum((act_test - pred) ** 2))
    ss_tot = float(np.sum((act_test - act_test.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _feature_acts(model: MLP, sae: TopKSAE, inp: np.ndarray, feat: int) -> np.ndarray:
    return sae.encode(model.layer_activations(inp))[:, feat]


def best_slope(
    model: MLP, sae: TopKSAE, feat: int, rng: np.random.Generator,
    c_grid: np.ndarray | None = None, n_train: int = 20000, n_test: int = 10000,
    n_buckets: int = 50, lo: float = -3.0, hi: float = 3.0,
) -> tuple[float, float]:
    """Sweep candidate slopes c; return (best_c, best_R2) for feature ``feat``.

    Train/test combos sample (x, a, b) with b random, so a high R^2 means the
    feature is explained by t = x + c*a *alone* (invariant to b).
    """
    if c_grid is None:
        c_grid = np.round(np.arange(-1.0, 1.0001, 0.025), 4)
    tr = rng.uniform(lo, hi, size=(n_train, 3))
    te = rng.uniform(lo, hi, size=(n_test, 3))
    act_tr = _feature_acts(model, sae, tr, feat)
    act_te = _feature_acts(model, sae, te, feat)
    best_c, best_r2 = 0.0, -np.inf
    for c in c_grid:
        t_tr = tr[:, 0] + c * tr[:, 1]
        t_te = te[:, 0] + c * te[:, 1]
        r2 = _bucket_r2(t_tr, act_tr, t_te, act_te, n_buckets)
        if r2 > best_r2:
            best_c, best_r2 = float(c), r2
    return best_c, best_r2


# --------------------------------------------------------------------------- #
#  Which features are diagonal bands?
# --------------------------------------------------------------------------- #
def _grid_corr(model: MLP, sae: TopKSAE, feat: int, n: int = 60) -> float:
    """Correlation of feature f's (x,a) activation map at b=0 vs b=1 (b-invariance)."""
    g0, _ = feature_grid(model, sae, 0.0, n=n)
    g1, _ = feature_grid(model, sae, 1.0, n=n)
    v0, v1 = g0[feat].ravel(), g1[feat].ravel()
    if v0.std() < 1e-9 or v1.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(v0, v1)[0, 1])


def find_diagonal_features(
    model: MLP, sae: TopKSAE, rng: np.random.Generator,
    r2_thresh: float = 0.8, min_active: float = 0.02, lo: float = -3.0, hi: float = 3.0,
) -> list[dict]:
    """Return features well-explained by some t = x + c*a, active, and b-invariant."""
    probe = rng.uniform(lo, hi, size=(30000, 3))
    Z = sae.encode(model.layer_activations(probe))
    active_frac = (Z > 0).mean(axis=0)

    out = []
    for f in range(sae.n_features):
        if active_frac[f] < min_active:
            continue
        c, r2 = best_slope(model, sae, f, rng, lo=lo, hi=hi)
        if r2 >= r2_thresh:
            out.append({
                "feature": f,
                "best_c": c,
                "r2": r2,
                "active_frac": float(active_frac[f]),
                "b_invariance": _grid_corr(model, sae, f),
            })
    out.sort(key=lambda d: -d["r2"])
    return out


# --------------------------------------------------------------------------- #
#  Earlier probe: do features track polynomial landmark points?
# --------------------------------------------------------------------------- #
def landmarks(a: float, b: float) -> dict:
    """Roots, critical points and inflection point of x^3 + a x^2 + x + b."""
    roots = np.roots([1.0, a, 1.0, b])
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-6]
    crit = np.roots([3.0, 2.0 * a, 1.0])  # f'(x) = 3x^2 + 2a x + 1
    real_crit = [float(r.real) for r in crit if abs(r.imag) < 1e-6]
    infl = -a / 3.0                        # f''(x) = 6x + 2a = 0
    return {"roots": real_roots, "critical": real_crit, "inflection": infl}


def landmark_correlation(
    model: MLP, sae: TopKSAE, feat: int, a: float, b: float,
    kind: str = "roots", n: int = 400, width: float = 0.35, lo: float = -3.0, hi: float = 3.0,
) -> float:
    """Pearson correlation between feature f's activation over x and Gaussian
    bumps placed at the chosen landmark points (roots / critical / inflection)."""
    x = np.linspace(lo, hi, n)
    inp = np.stack([x, np.full(n, a), np.full(n, b)], axis=1)
    act = sae.encode(model.layer_activations(inp))[:, feat]
    pts = landmarks(a, b)[kind if kind != "inflection" else "inflection"]
    pts = [pts] if kind == "inflection" else pts
    bumps = np.zeros(n)
    for p in pts:
        bumps += np.exp(-0.5 * ((x - p) / width) ** 2)
    if act.std() < 1e-9 or bumps.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(act, bumps)[0, 1])


# --------------------------------------------------------------------------- #
#  Intervention: clamp features, decode, continue the forward pass
# --------------------------------------------------------------------------- #
def intervene(model: MLP, sae: TopKSAE, inp: np.ndarray, feats: list[int], factor: float) -> np.ndarray:
    """Predict y after scaling the given SAE features by ``factor`` on the last
    activation layer (decode the manipulated feature vector, then apply the
    network's output layer)."""
    act = model.layer_activations(inp)          # (N, 15)
    z = sae.encode(act)
    z = z.copy()
    z[:, feats] *= factor
    act_hat = sae.decode(z)                      # manipulated 15-dim activation
    out_std = act_hat @ model.Ws[-1] + model.bs[-1]  # network's final (output) layer
    return out_std[:, 0] * model.y_std + model.y_mean
