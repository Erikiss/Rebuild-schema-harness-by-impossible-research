"""The regression NN from "Interpreting How Neural Nets Regress Cubic Polynomials".

The network is trained to regress the two-parameter family of cubics

    y = x^3 + a*x^2 + x + b

from the input triple (x, a, b). This mirrors the original WSRI'26 setup: a
4-hidden-layer MLP of width 15 (15x15x15x15) with ReLU activations, from which we
later read the *last* 15-dimensional activation layer and train a sparse
autoencoder on it.

Pure NumPy (hand-written Adam + backprop) so the whole reproduction stays
dependency-light and fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------- #
#  Target family of cubics:  y = x^3 + a x^2 + x + b
# --------------------------------------------------------------------------- #
def cubic(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x**3 + a * x**2 + x + b


def sample_inputs(n: int, rng: np.random.Generator, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    """n rows of (x, a, b) drawn uniformly from [lo, hi]^3."""
    return rng.uniform(lo, hi, size=(n, 3))


def targets(inp: np.ndarray) -> np.ndarray:
    x, a, b = inp[:, 0], inp[:, 1], inp[:, 2]
    return cubic(x, a, b)


# --------------------------------------------------------------------------- #
#  A small MLP with ReLU, caching per-layer activations
# --------------------------------------------------------------------------- #
@dataclass
class MLP:
    """3 -> 15 -> 15 -> 15 -> 15 -> 1 ReLU regressor.

    Inputs are scaled by ``in_scale`` (a common factor, so a linear combination
    x + c*a is preserved in original coordinates) and the target is standardised
    with (y_mean, y_std) learned from a sample. ``layer_activations`` exposes the
    post-ReLU activations of every hidden layer; the SAE is trained on the last
    one (index -1, the 15-dim "layer 4").
    """

    widths: tuple[int, ...] = (3, 15, 15, 15, 15, 1)
    seed: int = 0
    in_scale: float = 1.0 / 3.0
    y_mean: float = 0.0
    y_std: float = 1.0
    Ws: list = field(default_factory=list, repr=False)
    bs: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.Ws, self.bs = [], []
        for nin, nout in zip(self.widths[:-1], self.widths[1:]):
            # He initialisation for ReLU layers
            self.Ws.append(rng.normal(0.0, np.sqrt(2.0 / nin), size=(nin, nout)))
            self.bs.append(np.zeros(nout))

    # ---- forward ---------------------------------------------------------- #
    def _forward_cache(self, inp: np.ndarray):
        """Return (output_standardised, [hidden post-ReLU activations per layer])."""
        h = inp * self.in_scale
        acts = []
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = h @ W + b
            if i < len(self.Ws) - 1:  # hidden layer -> ReLU
                h = np.maximum(0.0, z)
                acts.append(h)
            else:                     # output layer -> linear
                h = z
        return h[:, 0], acts

    def predict(self, inp: np.ndarray) -> np.ndarray:
        """De-standardised prediction of y."""
        out_std, _ = self._forward_cache(inp)
        return out_std * self.y_std + self.y_mean

    def layer_activations(self, inp: np.ndarray, layer: int = -1) -> np.ndarray:
        """Post-ReLU activations of a hidden layer (default: the last, 15-dim)."""
        _, acts = self._forward_cache(inp)
        return acts[layer]

    # ---- training --------------------------------------------------------- #
    def fit(
        self,
        rng: np.random.Generator,
        steps: int = 30000,
        batch: int = 512,
        lr: float = 2e-3,
        lo: float = -3.0,
        hi: float = 3.0,
        verbose: bool = False,
    ) -> list[float]:
        # Standardise the target from a large sample.
        big = sample_inputs(200_000, rng, lo, hi)
        y_big = targets(big)
        self.y_mean, self.y_std = float(np.mean(y_big)), float(np.std(y_big))

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW = [np.zeros_like(W) for W in self.Ws]
        vW = [np.zeros_like(W) for W in self.Ws]
        mb = [np.zeros_like(b) for b in self.bs]
        vb = [np.zeros_like(b) for b in self.bs]
        hist: list[float] = []

        for t in range(1, steps + 1):
            inp = sample_inputs(batch, rng, lo, hi)
            y = (targets(inp) - self.y_mean) / self.y_std

            # forward with cache of pre/post activations
            h = inp * self.in_scale
            zs, hs = [], [h]
            for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
                z = h @ W + b
                zs.append(z)
                h = np.maximum(0.0, z) if i < len(self.Ws) - 1 else z
                hs.append(h)
            pred = hs[-1][:, 0]
            resid = pred - y
            hist.append(float(np.mean(resid**2)))

            # backward
            g = (2.0 / batch) * resid[:, None]  # dL/d(output), (batch,1)
            gW, gb = [None] * len(self.Ws), [None] * len(self.Ws)
            for i in reversed(range(len(self.Ws))):
                gW[i] = hs[i].T @ g
                gb[i] = np.sum(g, axis=0)
                if i > 0:
                    g = (g @ self.Ws[i].T) * (zs[i - 1] > 0.0)  # through ReLU

            for i in range(len(self.Ws)):
                for m, vv, grad, param in (
                    (mW, vW, gW[i], self.Ws[i]),
                    (mb, vb, gb[i], self.bs[i]),
                ):
                    m[i] = beta1 * m[i] + (1 - beta1) * grad
                    vv[i] = beta2 * vv[i] + (1 - beta2) * grad**2
                    mhat = m[i] / (1 - beta1**t)
                    vhat = vv[i] / (1 - beta2**t)
                    param -= lr * mhat / (np.sqrt(vhat) + eps)

            if verbose and (t % max(1, steps // 10) == 0 or t == 1):
                print(f"  step {t:6d}/{steps}  mse(std)={hist[-1]:.3e}")
        return hist

    def r2(self, rng: np.random.Generator, n: int = 50_000, lo: float = -3.0, hi: float = 3.0) -> float:
        inp = sample_inputs(n, rng, lo, hi)
        y = targets(inp)
        pred = self.predict(inp)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return 1.0 - ss_res / ss_tot
