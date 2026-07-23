"""Self-contained checks for the SAE / Cardano reproduction.

Runnable with plain Python (no pytest):

    python tests/test_reproduction.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cubic_model import MLP, cubic, sample_inputs, targets
from sae import TopKSAE, _topk
from features import landmarks, _bucket_r2


def test_cubic_and_landmarks() -> None:
    """Polynomial evaluation and landmark formulas (inflection at -a/3)."""
    x = np.array([0.0, 1.0, -1.0])
    assert np.allclose(cubic(x, 2.0, -1.0), x**3 + 2 * x**2 + x - 1)
    lm = landmarks(2.0, -1.0)
    assert abs(lm["inflection"] - (-2.0 / 3.0)) < 1e-12
    # roots really are roots
    for r in lm["roots"]:
        assert abs(r**3 + 2 * r**2 + r - 1) < 1e-8
    print("[ok] cubic eval + landmark formulas")


def test_topk_sparsity() -> None:
    """TopK keeps at most k nonzero entries per row."""
    rng = np.random.default_rng(0)
    v = np.abs(rng.normal(size=(50, 64)))
    z = _topk(v, 4)
    assert np.all((z > 0).sum(axis=1) <= 4)
    print("[ok] TopK keeps <= k active features")


def test_nn_learns_family() -> None:
    """The MLP regresses the whole (x,a,b) cubic family to high R^2 (small config)."""
    rng = np.random.default_rng(0)
    m = MLP(seed=0)
    m.fit(rng, steps=6000, batch=512, lr=2e-3)
    r2 = m.r2(np.random.default_rng(1), n=20000)
    assert r2 > 0.99, f"NN R^2 too low ({r2:.4f})"
    print(f"[ok] NN regresses the cubic family        (R^2 {r2:.4f})")


def test_sae_reconstructs() -> None:
    """The TopK SAE reconstructs last-layer activations with high R^2."""
    rng = np.random.default_rng(0)
    m = MLP(seed=0); m.fit(rng, steps=6000, batch=512, lr=2e-3)
    tr = m.layer_activations(sample_inputs(40000, rng))
    te = m.layer_activations(sample_inputs(8000, rng))
    sae = TopKSAE(n_features=64, k=4, seed=0)
    sae.fit(tr, epochs=20, batch=2048, lr=1e-3, seed=0)
    r2 = sae.r2(te)
    assert r2 > 0.95, f"SAE R^2 too low ({r2:.4f})"
    z = sae.encode(te[:100])
    assert np.all((z > 0).sum(axis=1) <= 4)
    print(f"[ok] SAE reconstructs the last layer      (R^2 {r2:.4f})")


def test_bucket_recovers_slope() -> None:
    """The bucket test recovers the true slope c0 of a synthetic t = x + c0*a feature."""
    rng = np.random.default_rng(0)
    c0 = 0.3
    tr = rng.uniform(-3, 3, size=(20000, 2))
    te = rng.uniform(-3, 3, size=(10000, 2))
    # activation is a nonlinear function of t = x + c0*a, plus a little noise
    f = lambda t: np.maximum(0.0, t - 1.0) ** 2
    a_tr = f(tr[:, 0] + c0 * tr[:, 1]) + 0.01 * rng.normal(size=len(tr))
    a_te = f(te[:, 0] + c0 * te[:, 1]) + 0.01 * rng.normal(size=len(te))
    c_grid = np.round(np.arange(-1, 1.0001, 0.025), 4)
    r2s = [_bucket_r2(tr[:, 0] + c * tr[:, 1], a_tr, te[:, 0] + c * te[:, 1], a_te) for c in c_grid]
    best_c = c_grid[int(np.argmax(r2s))]
    assert abs(best_c - c0) <= 0.05, f"recovered c={best_c} not near {c0}"
    print(f"[ok] bucket test recovers the true slope  (c={best_c:+.3f}, truth {c0:+.3f})")


if __name__ == "__main__":
    tests = [
        test_cubic_and_landmarks,
        test_topk_sparsity,
        test_bucket_recovers_slope,
        test_nn_learns_family,
        test_sae_reconstructs,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} checks passed")
    sys.exit(1 if failed else 0)
