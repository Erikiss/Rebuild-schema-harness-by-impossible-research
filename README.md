# Interpreting How Neural Nets Regress Cubic Polynomials — reproduction

A runnable reproduction of the WSRI'26 project *"Interpreting How Neural Nets
Regress Cubic Polynomials"* by Erik
([Wolfram Community](https://community.wolfram.com/groups/-/m/t/3763526) ·
[LessWrong](https://www.lesswrong.com/posts/ysztF7doGvEbvTMqN/how-do-neural-networks-regress-cubic-polynomials)).

**The claim, reproduced:** a single-hidden-layer Ramp/ReLU network that regresses a
cubic is *not* a black box. It computes a continuous piecewise-linear function

```
y(x) = c + Σ_i  v_i · Ramp(w_i · x + b_i)
```

whose kinks can be read straight off the weights — one per hidden neuron, at
`x = −b_i / w_i`, each adding slope `w_i · v_i` on its active side. Rebuilding the
function from those numbers matches the network to machine precision, and the kinks
turn out to congregate where the target cubic curves most.

Full write-up: [`docs/reproduction.md`](docs/reproduction.md).

## Quick start

```bash
pip install -r requirements.txt
python src/experiment.py     # train → interpret → figures/ + results/
python tests/test_reproduction.py   # verify the interpretation is exact
```

A Wolfram Language version matching the original project's tooling is in
[`wolfram/interpreting_cubic_nets.wls`](wolfram/interpreting_cubic_nets.wls).

## What each piece shows

| output | shows |
|---|---|
| `figures/01_fit.png` | the piecewise-linear fit tracking `f(x) = x³ − 3x`, kinks marked |
| `figures/02_ramps.png` | the individual neuron ramps summing to the fit |
| `figures/03_slope.png` | the slope staircase (jumps `\|w_i\|·v_i` at each kink) vs. `f'(x)` |
| `figures/04_curvature.png` | kinks clustering where `\|f''\|` is large |
| `results/metrics.json` | full per-neuron breakpoint / delta-slope / orientation table |
| `results/report.md` | headline numbers |

## Layout

```
src/cubic_relu_net.py   1→H→1 Ramp net (NumPy) + Adam trainer + Cubic target
src/interpret.py        breakpoints -b/w, delta-slopes w·v, exact CPWL rebuild, curvature test
src/experiment.py       end-to-end run: train, interpret, verify, plot, report
wolfram/…               the same experiment in Wolfram Language (NetChain)
tests/…                 asserts the mechanistic reading equals the network
docs/reproduction.md    the reproduced article
```

> Note on provenance: the original post could not be fetched from this environment
> (egress blocked), so the text was reconstructed from the title, search-surfaced
> excerpts, and standard shallow-ReLU theory, then verified empirically by the code
> here. See the note at the top of `docs/reproduction.md`.
