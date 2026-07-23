# Reproduction results

**Family**: `y = x^3 + a*x^2 + x + b` from input `(x, a, b)`, domain `[-3.0, 3.0]`.

## Model quality
- NN regression R² (held-out): `0.99994`
- SAE reconstruction R² (held-out, 64 features, TopK=4): `0.9923`

## The central finding: features encode t = x + c·a ≈ the Cardano substitution
- **6 diagonal-band features** are well explained by a single linear combination `t = x + c·a` and are invariant to `b`.
- **Median slope c = 0.287**, vs the Cardano depression substitution `a/3 = 0.333`.

| feature | best c | bucket-test R² | active frac | b-invariance |
|--:|--:|--:|--:|--:|
| #26 | +0.275 | 0.990 | 0.203 | 1.000 |
| #16 | +0.300 | 0.973 | 0.260 | 0.999 |
| #55 | +0.175 | 0.946 | 0.154 | 0.996 |
| #50 | +0.325 | 0.933 | 0.109 | 0.990 |
| #43 | +0.250 | 0.901 | 0.260 | 0.996 |
| #34 | +0.575 | 0.865 | 0.289 | 0.990 |

The NN, trained only to *regress* cubics, independently developed the same coordinate `t = x + a/3` that Cardano's method uses to *solve* them — the substitution that moves the inflection point (at `x = −a/3`) onto the axis and depresses the cubic (removes its quadratic term).

See `figures/` for the feature grid, b-invariance, bucket-test slopes, the Cardano illustration and the intervention.
