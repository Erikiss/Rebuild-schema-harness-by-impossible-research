"""A TopK Sparse Autoencoder trained on the NN's last activation layer.

Following the essay: a 64-dimensional SAE with TopK = 4 is trained on the 15-dim
activations of the network's last hidden layer. Every encoded activation is
expressed in at most 4 features, forcing the dense, polysemantic neurons to break
into sparse, (hopefully) monosemantic features.

    z = TopK_k( ReLU(W_enc (a - b_pre) + b_enc) )     (feature activations, dim 64)
    a_hat = z W_dec + b_pre                            (reconstruction, dim 15)

The pre-bias b_pre (standard for TopK SAEs, initialised to the mean activation)
is the only addition to the essay's bare ``z = ReLU(W a + b); a_hat = z D`` form;
it is what lets the reconstruction reach R^2 ~ 0.99. Pure NumPy + Adam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _topk(vals: np.ndarray, k: int) -> np.ndarray:
    """Zero out all but the top-k entries (by value) in each row of a (n, d) array."""
    if k >= vals.shape[1]:
        return vals
    kth = np.partition(vals, -k, axis=1)[:, -k:].min(axis=1, keepdims=True)
    return np.where(vals >= kth, vals, 0.0)


@dataclass
class TopKSAE:
    n_features: int = 64
    k: int = 4
    d_in: int = 15
    seed: int = 0
    W_enc: np.ndarray = field(default=None, repr=False)  # (d_in, n_features)
    b_enc: np.ndarray = field(default=None, repr=False)  # (n_features,)
    W_dec: np.ndarray = field(default=None, repr=False)  # (n_features, d_in)
    b_pre: np.ndarray = field(default=None, repr=False)  # (d_in,)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        D = rng.normal(0.0, 1.0, size=(self.n_features, self.d_in))
        D /= np.linalg.norm(D, axis=1, keepdims=True)  # unit-norm decoder rows
        self.W_dec = D
        self.W_enc = D.T.copy()                        # tied initialisation
        self.b_enc = np.zeros(self.n_features)
        self.b_pre = np.zeros(self.d_in)

    # ---- forward ---------------------------------------------------------- #
    def encode(self, a: np.ndarray) -> np.ndarray:
        pre = (a - self.b_pre) @ self.W_enc + self.b_enc
        return _topk(np.maximum(0.0, pre), self.k)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self.W_dec + self.b_pre

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(a))

    # ---- training --------------------------------------------------------- #
    def fit(
        self,
        acts: np.ndarray,
        epochs: int = 40,
        batch: int = 2048,
        lr: float = 1e-3,
        seed: int = 0,
        verbose: bool = False,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        self.b_pre = acts.mean(axis=0).copy()  # centre on the mean activation

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = {"W_enc": self.W_enc, "b_enc": self.b_enc, "W_dec": self.W_dec}
        m = {kk: np.zeros_like(p) for kk, p in params.items()}
        v = {kk: np.zeros_like(p) for kk, p in params.items()}
        n = acts.shape[0]
        hist: list[float] = []
        step = 0

        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s : s + batch]
                A = acts[idx]                                   # (B, d_in)
                step += 1

                centered = A - self.b_pre
                pre = centered @ self.W_enc + self.b_enc        # (B, F)
                relu = np.maximum(0.0, pre)
                mask = _topk(relu, self.k) > 0.0                # kept entries
                z = np.where(mask, relu, 0.0)
                recon = z @ self.W_dec + self.b_pre             # (B, d_in)

                resid = recon - A
                loss = float(np.mean(np.sum(resid**2, axis=1)))
                hist.append(loss)

                # backward (loss = mean over batch of sum-sq residual)
                B = A.shape[0]
                g_recon = (2.0 / B) * resid                     # (B, d_in)
                gW_dec = z.T @ g_recon                          # (F, d_in)
                gz = g_recon @ self.W_dec.T                     # (B, F)
                g_pre = gz * mask * (pre > 0.0)                 # through TopK+ReLU
                gW_enc = centered.T @ g_pre                     # (d_in, F)
                gb_enc = np.sum(g_pre, axis=0)                  # (F,)

                grads = {"W_enc": gW_enc, "b_enc": gb_enc, "W_dec": gW_dec}
                for kk in params:
                    m[kk] = beta1 * m[kk] + (1 - beta1) * grads[kk]
                    v[kk] = beta2 * v[kk] + (1 - beta2) * grads[kk] ** 2
                    mhat = m[kk] / (1 - beta1**step)
                    vhat = v[kk] / (1 - beta2**step)
                    params[kk] -= lr * mhat / (np.sqrt(vhat) + eps)

            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"  epoch {ep + 1:3d}/{epochs}  recon-loss={hist[-1]:.4f}")

        self.W_enc, self.b_enc, self.W_dec = params["W_enc"], params["b_enc"], params["W_dec"]
        return hist

    def r2(self, acts: np.ndarray) -> float:
        """Fraction of variance explained by the reconstruction (held-out acts)."""
        recon = self.reconstruct(acts)
        ss_res = float(np.sum((acts - recon) ** 2))
        ss_tot = float(np.sum((acts - acts.mean(axis=0)) ** 2))
        return 1.0 - ss_res / ss_tot
