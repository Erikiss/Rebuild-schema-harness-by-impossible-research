"""A tiny single-hidden-layer Ramp (ReLU) regression network, in pure NumPy.

This mirrors the Wolfram Language model used in the WSRI'26 project
"Interpreting How Neural Nets Regress Cubic Polynomials":

    NetChain[{
        LinearLayer[H],          (* affine map  x  -> w x + b   *)
        ElementwiseLayer[Ramp],  (* Ramp[z] = Max[0, z] = ReLU  *)
        LinearLayer[1]           (* affine map  h  -> v . h + c  *)
    }]

The whole point of the study is that this network is *not* a black box:
for a 1-D input it computes a continuous piecewise-linear (CPWL) function

    y(x) = c + sum_i  v_i * Ramp(w_i * x + b_i)

whose kinks (breakpoints) live at  x = -b_i / w_i.  Everything downstream in
``interpret.py`` reads those parameters straight off the trained weights.

We keep the implementation dependency-light (NumPy only) and fully
deterministic so the reproduction is bit-for-bit repeatable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------- #
#  Target: a generic cubic polynomial
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cubic:
    """f(x) = a3 x^3 + a2 x^2 + a1 x + a0 (default: the classic S-curve x^3 - 3x)."""

    a3: float = 1.0
    a2: float = 0.0
    a1: float = -3.0
    a0: float = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.a3 * x**3 + self.a2 * x**2 + self.a1 * x + self.a0

    def first_derivative(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 3 a3 x^2 + 2 a2 x + a1 -- the target slope the net's staircase tracks."""
        return 3.0 * self.a3 * x**2 + 2.0 * self.a2 * x + self.a1

    def second_derivative(self, x: np.ndarray) -> np.ndarray:
        """f''(x) = 6 a3 x + 2 a2 -- curvature; a ReLU net puts kinks where |f''| is large."""
        return 6.0 * self.a3 * x + 2.0 * self.a2

    def as_str(self) -> str:
        terms = []
        for coeff, name in [(self.a3, "x^3"), (self.a2, "x^2"), (self.a1, "x"), (self.a0, "1")]:
            if coeff == 0:
                continue
            if name == "1":
                terms.append(f"{coeff:g}")
            else:
                terms.append(f"{coeff:g}*{name}")
        return " + ".join(terms).replace("+ -", "- ") or "0"


# --------------------------------------------------------------------------- #
#  The network
# --------------------------------------------------------------------------- #
@dataclass
class RampNet:
    """1 -> H -> 1 Ramp/ReLU network with an Adam trainer, written out by hand.

    Parameters (all shape (H,) except c):
        w : first-layer weights   (LinearLayer[H])
        b : first-layer biases
        v : second-layer weights  (LinearLayer[1])
        c : second-layer bias
    """

    hidden: int = 24
    seed: int = 0
    w: np.ndarray = field(default=None, repr=False)
    b: np.ndarray = field(default=None, repr=False)
    v: np.ndarray = field(default=None, repr=False)
    c: float = 0.0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        H = self.hidden
        # Spread the initial breakpoints -b/w uniformly across a nominal [-2, 2]
        # domain so training starts from a sensible piecewise-linear "comb".
        knots = np.linspace(-2.0, 2.0, H)
        self.w = rng.choice([-1.0, 1.0], size=H) * rng.uniform(0.5, 1.5, size=H)
        self.b = -self.w * knots
        self.v = rng.normal(0.0, 0.3, size=H)
        self.c = 0.0

    # ---- forward pass ----------------------------------------------------- #
    def preactivation(self, x: np.ndarray) -> np.ndarray:
        """z_i(x) = w_i x + b_i, shape (N, H)."""
        return np.outer(x, self.w) + self.b

    def hidden_activations(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.preactivation(x))  # Ramp

    def forward(self, x: np.ndarray) -> np.ndarray:
        """y(x) = c + sum_i v_i Ramp(w_i x + b_i)."""
        return self.hidden_activations(x) @ self.v + self.c

    # ---- training (full-batch Adam) --------------------------------------- #
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        steps: int = 30000,
        lr: float = 5e-3,
        verbose: bool = False,
    ) -> list[float]:
        """Minimise mean-squared error with Adam. Returns the loss history."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = {"w": self.w, "b": self.b, "v": self.v, "c": np.array(self.c)}
        m = {k: np.zeros_like(p) for k, p in params.items()}
        s = {k: np.zeros_like(p) for k, p in params.items()}
        N = x.shape[0]
        history: list[float] = []

        for t in range(1, steps + 1):
            z = np.outer(x, params["w"]) + params["b"]  # (N, H)
            h = np.maximum(0.0, z)                       # (N, H)
            pred = h @ params["v"] + params["c"]         # (N,)
            resid = pred - y                             # (N,)
            loss = float(np.mean(resid**2))
            history.append(loss)

            # Backprop of MSE = mean(resid^2)
            g = (2.0 / N) * resid                        # dL/dpred, (N,)
            grad_v = h.T @ g                             # (H,)
            grad_c = np.sum(g)                           # scalar
            dh = np.outer(g, params["v"])                # (N, H)
            dz = dh * (z > 0.0)                          # Ramp'(z) = 1[z>0]
            grad_w = dz.T @ x                            # (H,)  d(w x)/dw = x
            grad_b = np.sum(dz, axis=0)                  # (H,)

            grads = {"w": grad_w, "b": grad_b, "v": grad_v, "c": np.array(grad_c)}
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                s[k] = beta2 * s[k] + (1 - beta2) * grads[k] ** 2
                mhat = m[k] / (1 - beta1**t)
                shat = s[k] / (1 - beta2**t)
                params[k] = params[k] - lr * mhat / (np.sqrt(shat) + eps)

            if verbose and (t % max(1, steps // 10) == 0 or t == 1):
                print(f"  step {t:6d}/{steps}  mse={loss:.3e}")

        self.w, self.b, self.v = params["w"], params["b"], params["v"]
        self.c = float(params["c"])
        return history
