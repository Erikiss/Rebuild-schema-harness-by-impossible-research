# Reproduction results

**Target**: `f(x) = 1*x^3 - 3*x` on x in [-2.0, 2.0]  
**Network**: `1 -> 24 (Ramp) -> 1`, Adam, 30000 steps, lr=0.005, seed=0

## Fit quality
- Train MSE: `1.640e-04`
- Max abs error on dense grid: `5.716e-02`

## Interpretation is exact
- Rebuilding the piecewise-linear function from the weights alone (breakpoints `-b_i/w_i`, delta-slopes `w_i v_i`) and comparing to the network's own forward pass gives a max difference of `3.16e-15` -- i.e. the mechanistic reading is the network, not an approximation of it.
- 20 of 24 neurons place their kink inside the domain.

## Kinks congregate in high-curvature regions
- Slope-jump-weighted mean curvature seen by the breakpoints: `7.613`
- Domain-average curvature: `6.003`
- **Curvature enrichment: 1.27x** (>1 means kinks favour where the cubic bends most).

See `figures/` for the plots and `results/metrics.json` for the full per-neuron table.
