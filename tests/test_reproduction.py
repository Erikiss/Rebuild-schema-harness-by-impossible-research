"""Self-contained checks that the mechanistic interpretation really is exact.

Runnable with plain Python (no pytest needed):

    python tests/test_reproduction.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cubic_relu_net import Cubic, RampNet
from interpret import extract_neurons, reconstruct_pwl, equivalence_error


def _train_small() -> tuple[RampNet, Cubic, tuple[float, float]]:
    cubic = Cubic(a3=1.0, a2=0.0, a1=-3.0, a0=0.0)
    domain = (-2.0, 2.0)
    x = np.linspace(*domain, 300)
    net = RampNet(hidden=16, seed=0)
    net.fit(x, cubic(x), steps=4000, lr=5e-3)
    return net, cubic, domain


def test_forward_matches_closed_form() -> None:
    """net.forward must equal c + Σ v_i·Ramp(w_i x + b_i) exactly."""
    net, _, domain = _train_small()
    x = np.linspace(*domain, 501)
    closed = net.c + np.maximum(0.0, np.outer(x, net.w) + net.b) @ net.v
    err = float(np.max(np.abs(net.forward(x) - closed)))
    assert err < 1e-12, f"forward != closed form (err={err:.2e})"
    print(f"[ok] forward == closed form                (max diff {err:.1e})")


def test_interpretation_is_exact() -> None:
    """The analytic piecewise-linear rebuild from weights must match the net."""
    net, _, domain = _train_small()
    x = np.linspace(*domain, 4000)
    pwl = reconstruct_pwl(net, domain)
    err = equivalence_error(net, pwl, x)
    assert err < 1e-10, f"CPWL reconstruction not exact (err={err:.2e})"
    print(f"[ok] analytic CPWL == network             (max diff {err:.1e})")


def test_breakpoint_formula() -> None:
    """Each neuron's preactivation is exactly zero at its breakpoint -b/w."""
    net, _, _ = _train_small()
    for n in extract_neurons(net):
        z = n.w * n.breakpoint + n.b
        assert abs(z) < 1e-9, f"neuron {n.index}: w·β+b = {z:.2e} (expected 0)"
    print("[ok] preactivation vanishes at -b/w        (all neurons)")


def test_slope_jump_identity() -> None:
    """Slope jump across a kink equals |w_i|·v_i (delta-slope × orientation)."""
    net, _, _ = _train_small()
    for n in extract_neurons(net):
        assert np.isclose(n.slope_jump, abs(n.w) * n.v, atol=1e-12)
    print("[ok] slope jump == |w|·v                   (all neurons)")


def test_constant_neuron_included() -> None:
    """A degenerate w_i == 0 (constant) neuron must still be reconstructed exactly."""
    net = RampNet(hidden=2, seed=0)
    net.w = np.array([0.0, 1.0])   # neuron 0 is a pure constant v0*max(0,b0)
    net.b = np.array([3.0, 0.0])
    net.v = np.array([2.0, 1.0])
    net.c = 0.0
    pwl = reconstruct_pwl(net, (-2.0, 2.0))
    x = np.linspace(-5, 5, 500)
    err = float(np.max(np.abs(net.forward(x) - pwl(x))))
    assert err < 1e-12, f"constant neuron dropped from reconstruction (err={err:.2e})"
    print(f"[ok] w==0 constant neuron reconstructed    (max diff {err:.1e})")


def test_fit_is_reasonable() -> None:
    """Sanity: the small net actually learns the cubic."""
    net, cubic, domain = _train_small()
    x = np.linspace(*domain, 400)
    mse = float(np.mean((net.forward(x) - cubic(x)) ** 2))
    assert mse < 1e-2, f"fit too poor (mse={mse:.2e})"
    print(f"[ok] network learns the cubic             (mse {mse:.1e})")


if __name__ == "__main__":
    tests = [
        test_forward_matches_closed_form,
        test_interpretation_is_exact,
        test_breakpoint_formula,
        test_slope_jump_identity,
        test_constant_neuron_included,
        test_fit_is_reasonable,
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
