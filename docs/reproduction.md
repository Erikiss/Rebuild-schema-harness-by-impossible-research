# Interpreting How Neural Nets Regress Cubic Polynomials

*A reproduction of the WSRI'26 project by Erik*
([Wolfram Community](https://community.wolfram.com/groups/-/m/t/3763526) ·
[LessWrong](https://www.lesswrong.com/posts/ysztF7doGvEbvTMqN/how-do-neural-networks-regress-cubic-polynomials))

> **Provenance / honesty note.** This repository's execution environment blocks
> outbound web fetches (the egress proxy returns `403` for `lesswrong.com`,
> `community.wolfram.com`, and mirrors), so the original post could not be read
> verbatim. This write-up reconstructs the article from its title, the excerpts
> that surfaced through web search, and the standard theory of shallow ReLU
> networks — and, crucially, it is backed by a **runnable reproduction** (see
> `src/`) that empirically demonstrates every claim below. Where the original
> chooses specific numbers (exact polynomial, hidden width, plots) our choices
> are representative rather than guaranteed identical; the *mechanism* is the
> invariant, and it is reproduced exactly. Correct any specifics and the code
> will follow.

---

## The question

If you train a neural network to fit a smooth curve like the cubic
`f(x) = x³ − 3x`, it works — but *how*? What is the network actually computing,
and can we read that computation off the trained weights instead of treating the
network as a black box?

For a network whose activation is the **Ramp** function
`Ramp(z) = max(0, z)` (a.k.a. ReLU), the answer is unusually clean and completely
mechanistic.

## The setup

We regress a cubic with the smallest network that still tells the whole story: a
single hidden layer of Ramp units.

```
NetChain[{
    LinearLayer[H],          (* x  ->  w x + b   *)
    ElementwiseLayer[Ramp],  (* Ramp[z] = Max[0, z] *)
    LinearLayer[1]           (* h  ->  v . h + c *)
}]
```

- **Target:** `f(x) = x³ − 3x` on `x ∈ [−2, 2]` (the classic S-curve: local max at
  `x = −1`, local min at `x = +1`, inflection at `x = 0`).
- **Network:** `1 → H → 1` with `H = 24` Ramp units, trained with Adam on 400
  evenly spaced samples.

## Key insight: a Ramp net *is* a piecewise-linear function

The network computes, in closed form,

```
y(x) = c + Σ_i  v_i · Ramp(w_i · x + b_i)
```

Each term `v_i · Ramp(w_i x + b_i)` is **zero on one side** of the point where its
argument changes sign, and **linear on the other**. So the whole network is a
*continuous piecewise-linear* (CPWL) function: straight-line segments joined at
kinks. **Training a Ramp net to regress a curve is doing piecewise-linear
regression** — the optimiser is choosing where to put the kinks and how sharp to
make them.

## The interpretation: breakpoints, delta-slopes, orientation

Every hidden neuron `i` owns exactly one kink, and three numbers read straight off
its weights describe it completely:

| quantity | formula | meaning |
|---|---|---|
| **breakpoint** | `β_i = −b_i / w_i` | the `x` where neuron `i` switches on/off |
| **delta-slope** | `μ_i = w_i · v_i` | the slope the neuron adds while it is active |
| **orientation** | `s_i = sign(w_i)` | active to the **right** of `β_i` if `w_i > 0`, to the **left** if `w_i < 0` |

The mechanism of a "kink": neuron `i` is switched off on one side of `β_i` (it
contributes a flat 0) and switched on on the other (it contributes the line
`μ_i·x + v_i·b_i`). As `x` increases across `β_i`, the network's total slope jumps
by

```
Δslope_i = s_i · μ_i = |w_i| · v_i.
```

> *"When a second-layer line hits a boundary, the linear function changes, so the
> line kinks."*

Sum these ramps and you get the staircase-derivative picture: the network's slope
is **piecewise constant**, taking a step of size `|w_i|·v_i` at each breakpoint,
and that staircase approximates the target's true slope `f'(x) = 3x² − 3`
(`figures/03_slope.png`).

## This reading is exact, not an approximation

To prove the interpretation *is* the network rather than a story about it, we
rebuild the piecewise-linear function segment by segment **from the extracted
breakpoints and delta-slopes alone** (`interpret.reconstruct_pwl`) and compare it
to the network's own forward pass:

```
max |net(x) − analytic CPWL(x)|  ≈  3e-15
```

That is machine epsilon: the breakpoint/delta-slope/orientation table is a
lossless description of what the network computes.

## Where do the kinks go? Toward curvature

A straight line has zero error only where the target is straight. A
piecewise-linear fit therefore has to spend its kinks where the target *bends* —
i.e. where the second derivative `|f''(x)|` is large. For our cubic
`f'' (x) = 6x`, curvature is **smallest at the inflection `x = 0`** and **largest at
the domain edges `x = ±2`**.

That is exactly what the trained network does. Weighting each breakpoint by how
much slope it actually moves (`|Δslope_i|`), the kinks sit at a mean curvature

```
curvature enrichment = (curvature seen by kinks) / (average curvature) ≈ 1.27×
```

with a visible **gap around `x = 0`** and clustering toward the edges
(`figures/04_curvature.png`). Breakpoints congregate where the cubic curves most —
the network allocates its limited supply of kinks where they buy the most error
reduction.

## Results (this reproduction, seed 0)

| metric | value |
|---|---|
| Train MSE | `1.6 × 10⁻⁴` |
| Max abs error on dense grid | `5.7 × 10⁻²` |
| Interpretation exactness (`max|net − CPWL|`) | `3.2 × 10⁻¹⁵` |
| Effective interior kinks | 20 / 24 |
| Curvature enrichment at kinks | `1.27×` |

Figures:

- `figures/01_fit.png` — the piecewise-linear fit tracking the cubic, kinks marked.
- `figures/02_ramps.png` — the individual neuron ramps that sum to the fit.
- `figures/03_slope.png` — the piecewise-constant slope staircase vs. `f'(x)`.
- `figures/04_curvature.png` — kinks congregating where `|f''|` is large.
- `figures/05_training.png` — the training loss.

## Reproduce it

```bash
pip install numpy matplotlib
python src/experiment.py          # trains, interprets, writes figures/ and results/
```

The Wolfram Language version — matching the original project's tooling — is in
`wolfram/interpreting_cubic_nets.wls`:

```bash
wolframscript -file wolfram/interpreting_cubic_nets.wls
```

## Takeaway

A shallow Ramp/ReLU network regressing a cubic is not mysterious. It is doing
**adaptive piecewise-linear regression**: each hidden neuron is one kink located at
`−b/w`, contributing slope `w·v` on its active side. The trained weights are a
direct, exact, human-readable description of the fitted curve, and the kinks end up
concentrated where the target's curvature demands them.

## References

- [WSRI26] Interpreting How Neural Nets Regress Cubic Polynomials — Wolfram
  Community: <https://community.wolfram.com/groups/-/m/t/3763526>
- How Do Neural Networks Regress Cubic Polynomials — LessWrong:
  <https://www.lesswrong.com/posts/ysztF7doGvEbvTMqN/how-do-neural-networks-regress-cubic-polynomials>
- Background on shallow ReLU nets as splines and the breakpoint/delta-slope
  (BDSO) parametrisation: *Shallow Univariate ReLU Networks as Splines*
  (arXiv:2008.01772).
