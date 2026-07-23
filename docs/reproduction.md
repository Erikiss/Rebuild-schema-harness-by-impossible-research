# Interpreting How Neural Nets Regress Cubic Polynomials

*A reproduction of the WSRI'26 project by **Enrico Bottazzi***
([Wolfram Community](https://community.wolfram.com/groups/-/m/t/3763526) ·
[LessWrong](https://www.lesswrong.com/posts/ysztF7doGvEbvTMqN/how-do-neural-networks-regress-cubic-polynomials))

> **What this is.** An independent, runnable reproduction of the essay's pipeline
> and its central finding, built from the original notebook (PDF). Everything here
> is trained and measured from scratch in pure NumPy — see `src/`. Because the
> network and SAE are retrained with different tooling and random seeds, the
> *indices* of the discovered features differ from the essay's, but the
> phenomenon — a family of features that compute `t = x + c·a` with `c ≈ 1/3` —
> reproduces cleanly.

---

## The question

> *"Neural networks can easily learn to regress cubic polynomials of type
> `y = x³ + a·x² + x + b`, but how do they do that?"*

The essay's punchline: the network internally builds a variable **`t ≈ x + 0.3a`**,
surprisingly close to the substitution **`t = x + a/3`** that the **Cardano method**
uses as its first step for *solving* cubics. This reproduction rebuilds that result
end to end.

## Setup

- **Target family**: `y = x³ + a·x² + x + b`, parametrised by `(a, b)`.
- **Network**: an MLP that takes the triple `(x, a, b)` and outputs `y`, with four
  hidden layers of width 15 (`15×15×15×15`, ReLU). Trained on `(x,a,b) ∈ [−3,3]³`.
- Here it reaches **regression R² = 0.99994** on held-out inputs (`figures/01_nn_fit.png`).

## Interpretability: neurons vs. sparse autoencoders

Reading individual neurons fails, because neurons are **polysemantic** — training
pressure packs more concepts than there are neurons, so one neuron fires for
unrelated things. The essay's tool is a **Sparse Autoencoder (SAE)**: an
autoencoder that encodes the activation into a *larger* but *sparse* feature
vector, disentangling the dense neurons into (hopefully) monosemantic features.

We train a **64-feature SAE with TopK = 4** on the network's **last 15-dim
activation layer**:

```
z = TopK₄( ReLU(W_enc·(a − b_pre) + b_enc) )     # ≤ 4 active features per input
â = z·W_dec + b_pre                                # reconstruction
```

Here it reaches **reconstruction R² = 0.9923** (essay: 0.995), a close-to-lossless
compression into sparse features.

## First hypothesis: do features track "landmark points"?

A natural guess is that features fire near a polynomial's **roots**, **critical
points** (roots of `f'`), or **inflection point** (root of `f''`, at `x = −a/3`).
Measuring the correlation of each feature's activation profile against Gaussian
bumps at those landmarks gives only weak, non-localised correlations — the same
messy result the essay reports. The clean structure is elsewhere.

## The "Aha!" moment: diagonal-band features

Instead of testing a hypothesis, fix `b` and just *plot* each feature's raw
activation over the **`(x, a)` plane**. A group of features lights up in a
**diagonal band** (`figures/02_feature_grid.png`): they do not care about `x` and
`a` *separately*, only about a **linear combination `t = x + c·a`**, and they are
**invariant to `b`** — the `b=0` and `b=1` maps are identical
(`figures/03_b_invariance.png`).

To pin down the slope `c` of a band, we run the essay's **bucket test** for each
feature:

1. pick a candidate `c`; compute `t = x + c·a` for training inputs;
2. chop them into 50 buckets by `t`; average the feature's activation per bucket;
3. predict test activations from their bucket average; score with **R²**;
4. sweep `c ∈ [−1, 1]` and keep the best.

In this run, **6 diagonal-band features** (R² ≥ 0.8, active, b-invariant) emerge:

| feature | best c | bucket R² | active frac | b-invariance |
|--:|--:|--:|--:|--:|
| #26 | +0.275 | 0.990 | 0.203 | 1.000 |
| #16 | +0.300 | 0.973 | 0.260 | 0.999 |
| #55 | +0.175 | 0.946 | 0.154 | 0.996 |
| #50 | +0.325 | 0.933 | 0.109 | 0.990 |
| #43 | +0.250 | 0.901 | 0.260 | 0.996 |
| #34 | +0.575 | 0.865 | 0.289 | 0.990 |

**Median `c = 0.287`** — right next to the Cardano value `a/3 = 0.333` and the
essay's reported `≈ 0.3` (`figures/04_bucket_slopes.png`).

## Why `a/3` is special: the Cardano method

To solve `y = x³ + a·x² + x + b`, Cardano's first step substitutes `x → x − a/3`
(equivalently `t = x + a/3`), which **removes the quadratic term** and yields a
*depressed* cubic that is easier to solve. Geometrically it is a horizontal shift
placing the **inflection point** (at `x = −a/3`) onto the axis, making the curve
odd-symmetric (`figures/05_cardano.png`).

So a network trained only to **regress** cubics has independently discovered the
same coordinate a human uses to **solve** them. As the essay puts it: *the NN has
discovered a new coordinate `t = x + a/3` and activates it to regress a cubic.*

## How is the feature used? Manipulating the brain

We test the features by **intervention**: encode the last-layer activation, scale
the diagonal-band features by a factor, decode, and finish the forward pass
(`features.intervene`). Sweeping the factor `0.5 → 1.5` **vertically stretches** the
regressed curve (`figures/06_intervention.png`), matching the essay's observation
that these features act like a *multiplicative / vertical-stretch* effect rather
than a literal horizontal shift — so the Cardano reading is suggestive but not
fully confirmed by the intervention. (At factor `×1`, no manipulation, the curve
matches the target up to the SAE's own reconstruction error.)

## Results summary (seed 0)

| metric | value |
|---|---|
| NN regression R² | `0.99994` |
| SAE reconstruction R² | `0.9923` |
| diagonal-band features found | 6 |
| **median slope c** | **`0.287`** (Cardano `a/3 = 0.333`) |
| feature b-invariance (grid corr.) | `0.99–1.00` |

## Reproduce it

```bash
pip install -r requirements.txt
python src/experiment.py            # ~1 min: trains NN + SAE, runs analysis, writes figures/ + results/
python tests/test_reproduction.py   # 5/5 checks (incl. bucket test recovering a known slope)
```

The Wolfram Language version (matching the original tooling — NN in WL, SAE via an
external Python library exactly as in the essay) is
`wolfram/interpreting_cubic_nets.wls`.

## Honest notes on fidelity

- **Feature indices differ** from the essay (`#7, #16, …`); with different random
  seeds the SAE learns a different basis. The *statistic that matters* — median
  `c ≈ 1/3` across b-invariant diagonal bands — is what reproduces.
- The SAE here adds a standard pre-bias `b_pre` (the essay's bare
  `z = ReLU(Wa+b); â = zD` reaches lower R²); this is the only deviation from the
  stated SAE formula.
- One recovered band (`#34`, `c ≈ 0.575`) sits away from the cluster — real
  features are not perfectly clean; the *median* absorbs such outliers, as in the
  essay.
- We reproduce the essay's core arc (NN → SAE → diagonal bands → `t = x + a/3` →
  intervention). The later sections (SAE on middle layers, the 7×7 grid of
  single-polynomial nets, abstraction-emergence discussion) are described in the
  essay but not reproduced here.

## References

- Enrico Bottazzi, *[WSRI26] Interpreting How Neural Nets Regress Cubic
  Polynomials* — Wolfram Community, Staff Picks (July 2026):
  <https://community.wolfram.com/groups/-/m/t/3763526>
- LessWrong mirror:
  <https://www.lesswrong.com/posts/ysztF7doGvEbvTMqN/how-do-neural-networks-regress-cubic-polynomials>
- Cardano's method (depressed cubic via `x → x − a/3`), *Ars Magna* (1545).
