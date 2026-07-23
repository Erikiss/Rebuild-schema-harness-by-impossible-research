"""Interpret a trained Ramp/ReLU net: read the piecewise-linear function off the weights.

This is the reproduction of the WSRI'26 analysis "Interpreting How Neural Nets
Regress Cubic Polynomials". A single-hidden-layer Ramp network computes

    y(x) = c + sum_i  v_i * Ramp(w_i x + b_i),      Ramp(z) = max(0, z)

which is a *continuous piecewise-linear* (CPWL) function. Each hidden neuron i
contributes one kink. Following the Breakpoint / Delta-Slope / Orientation
(BDSO) reading of a shallow ReLU net we extract, for every neuron:

    breakpoint    beta_i  = -b_i / w_i        (the x where the neuron switches on/off)
    delta-slope   mu_i    =  w_i * v_i         (slope the neuron adds while active)
    orientation   s_i     =  sign(w_i)         (active to the right of beta_i if w_i>0,
                                                to the left if w_i<0)

As x increases across a breakpoint beta_i the total slope jumps by
    Delta_i = s_i * mu_i = |w_i| * v_i
so the fitted line "kinks". The intuition the article demonstrates: in a good
fit the breakpoints congregate where the target cubic curves the most, i.e.
where |f''(x)| is large, because that is where a piecewise-linear function needs
the most kinks to keep up.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cubic_relu_net import RampNet


@dataclass
class Neuron:
    index: int
    w: float
    b: float
    v: float
    breakpoint: float   # beta = -b/w
    delta_slope: float  # mu = w*v
    orientation: int    # sign(w)
    slope_jump: float   # s*mu = |w|*v  (how the total slope changes going left->right)

    @property
    def active_side(self) -> str:
        return "right" if self.orientation > 0 else "left"


def extract_neurons(net: RampNet) -> list[Neuron]:
    """Turn raw weights into interpretable (breakpoint, delta-slope, orientation) triples."""
    neurons: list[Neuron] = []
    for i in range(net.hidden):
        w, b, v = float(net.w[i]), float(net.b[i]), float(net.v[i])
        if w == 0.0:
            continue  # degenerate: a constant neuron, no kink
        beta = -b / w
        mu = w * v
        s = int(np.sign(w))
        neurons.append(
            Neuron(
                index=i,
                w=w,
                b=b,
                v=v,
                breakpoint=beta,
                delta_slope=mu,
                orientation=s,
                slope_jump=s * mu,
            )
        )
    return neurons


# --------------------------------------------------------------------------- #
#  Analytic CPWL reconstruction -- proves the interpretation is EXACT
# --------------------------------------------------------------------------- #
@dataclass
class PiecewiseLinear:
    """The function y(x) as an explicit list of (segment boundary, slope, intercept)."""

    knots: np.ndarray          # sorted interior breakpoints inside the domain
    slopes: np.ndarray         # slope on each segment (len = knots+1)
    intercepts: np.ndarray     # intercept on each segment (y = slope*x + intercept)
    domain: tuple[float, float]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # segment index for each x: number of knots strictly to the left
        idx = np.searchsorted(self.knots, x, side="right")
        idx = np.clip(idx, 0, len(self.slopes) - 1)
        return self.slopes[idx] * x + self.intercepts[idx]


def reconstruct_pwl(net: RampNet, domain: tuple[float, float]) -> PiecewiseLinear:
    """Build the explicit piecewise-linear form of the network from its weights alone.

    We evaluate slope and intercept segment by segment using only the extracted
    breakpoints/delta-slopes -- no call to ``net.forward`` -- so that comparing
    against ``net.forward`` is a genuine correctness check on the interpretation.

    ALL breakpoints are used as segment boundaries (not just those inside
    ``domain``), so the reconstruction is exact on the whole real line, not merely
    on the training window. ``domain`` is retained only as descriptive metadata.
    """
    # Only neurons with w != 0 produce a kink; a w == 0 unit is a pure constant
    # (v_i * max(0, b_i)) with no breakpoint. Dedupe identical knots so each
    # sampled midpoint lands strictly between two boundaries.
    knots = np.unique(np.array([n.breakpoint for n in extract_neurons(net)], dtype=float))

    # One representative x strictly inside every segment. For the two unbounded
    # outer segments we step one unit past the extreme knots.
    if len(knots) == 0:
        mids = np.array([0.0])
    else:
        inner = 0.5 * (knots[:-1] + knots[1:])
        mids = np.concatenate([[knots[0] - 1.0], inner, [knots[-1] + 1.0]])

    # Accumulate the affine piece on every segment from ALL hidden units,
    # including any constant (w == 0) neurons, so the reconstruction is exact.
    W, B, V = net.w, net.b, net.v
    slopes = np.empty(len(mids))
    intercepts = np.empty(len(mids))
    for k, xm in enumerate(mids):
        active = (W * xm + B) > 0.0          # v_i (w_i x + b_i) = (w_i v_i) x + v_i b_i
        slopes[k] = float(np.sum(W[active] * V[active]))
        intercepts[k] = net.c + float(np.sum(V[active] * B[active]))

    return PiecewiseLinear(knots=knots, slopes=slopes, intercepts=intercepts, domain=domain)


def equivalence_error(net: RampNet, pwl: PiecewiseLinear, x: np.ndarray) -> float:
    """Max |network(x) - analytic reconstruction(x)|. Should be ~machine epsilon."""
    return float(np.max(np.abs(net.forward(x) - pwl(x))))


# --------------------------------------------------------------------------- #
#  "Breakpoints congregate where the cubic curves most" -- quantified
# --------------------------------------------------------------------------- #
def curvature_congregation(
    net: RampNet,
    cubic,
    domain: tuple[float, float],
    weight_thresh: float = 1e-3,
) -> dict:
    """Test the article's central qualitative claim quantitatively.

    Returns a correlation between where the *effective* breakpoints sit (weighted
    by |slope jump|, since near-zero-jump neurons do nothing) and the target's
    curvature magnitude |f''(x)|. A positive correlation means kinks cluster in
    high-curvature regions.
    """
    neurons = extract_neurons(net)
    lo, hi = domain
    active = [
        n for n in neurons
        if lo < n.breakpoint < hi and abs(n.slope_jump) > weight_thresh
    ]
    betas = np.array([n.breakpoint for n in active])
    jumps = np.array([abs(n.slope_jump) for n in active])

    # Curvature |f''| at each breakpoint vs. the domain-average curvature.
    grid = np.linspace(lo, hi, 2001)
    curv = np.abs(cubic.second_derivative(grid))
    mean_curv = float(np.mean(curv))

    curv_at_breaks = np.abs(cubic.second_derivative(betas)) if len(betas) else np.array([])
    # Slope-jump-weighted mean curvature seen by the breakpoints.
    if jumps.sum() > 0:
        weighted_curv = float(np.sum(jumps * curv_at_breaks) / np.sum(jumps))
    else:
        weighted_curv = float("nan")

    # Rank-correlation-free summary: enrichment = (curvature at kinks) / (average curvature).
    enrichment = weighted_curv / mean_curv if mean_curv > 0 else float("nan")

    return {
        "n_effective_breakpoints": len(active),
        "breakpoints": betas.tolist(),
        "slope_jumps": [float(n.slope_jump) for n in active],
        "mean_domain_curvature": mean_curv,
        "weighted_curvature_at_breakpoints": weighted_curv,
        "curvature_enrichment": enrichment,  # >1  => kinks favour high-curvature regions
    }
