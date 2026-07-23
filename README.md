# Interpreting How Neural Nets Regress Cubic Polynomials — reproduction

A runnable reproduction of the WSRI'26 project *"Interpreting How Neural Nets
Regress Cubic Polynomials"* by **Enrico Bottazzi**
([Wolfram Community](https://community.wolfram.com/groups/-/m/t/3763526) ·
[LessWrong](https://www.lesswrong.com/posts/ysztF7doGvEbvTMqN/how-do-neural-networks-regress-cubic-polynomials)).

**The finding, reproduced:** a network trained to regress the cubic family
`y = x³ + a·x² + x + b` from the input `(x, a, b)` internally builds the coordinate
**`t = x + c·a` with `c ≈ 1/3`** — the same substitution the **Cardano method** uses
to depress a cubic. A Sparse Autoencoder on the network's last layer exposes
b-invariant "diagonal-band" features whose slope, recovered by a bucket test, has
**median `c = 0.287`** (Cardano `a/3 = 0.333`).

Full write-up: [`docs/reproduction.md`](docs/reproduction.md).

## Quick start

```bash
pip install -r requirements.txt
python src/experiment.py            # ~1 min: NN + SAE + analysis → figures/ + results/
python tests/test_reproduction.py   # 5/5 checks
```

A Wolfram Language version (NN in WL, SAE via Python, as in the original) is in
[`wolfram/interpreting_cubic_nets.wls`](wolfram/interpreting_cubic_nets.wls).

## The pipeline

1. **Regress** the cubic family with an MLP `3 → 15×4 → 1` (ReLU) — R² ≈ 0.9999.
2. **Compress** the last 15-dim activation layer with a **TopK SAE** (64 features,
   k=4) — reconstruction R² ≈ 0.99.
3. **Discover** the diagonal-band features: plot each feature over the `(x, a)`
   plane, find those explained by `t = x + c·a` and invariant to `b`.
4. **Recover** the slope `c` per feature with the bucket test → median `≈ 1/3`.
5. **Connect** to Cardano: `t = x + a/3` moves the inflection point (`x = −a/3`)
   onto the axis and depresses the cubic.
6. **Intervene**: clamping the band features vertically stretches the regressed curve.

## Figures

| output | shows |
|---|---|
| `figures/01_nn_fit.png` | the MLP regressing example cubics |
| `figures/02_feature_grid.png` | all 64 SAE features over `(x, a)`; diagonal bands in red |
| `figures/03_b_invariance.png` | the diagonal-band features are identical at `b=0` and `b=1` |
| `figures/04_bucket_slopes.png` | bucket-test R² vs candidate slope `c`, peaking near `1/3` |
| `figures/05_cardano.png` | Cardano's `t = x + a/3` depression of the cubic |
| `figures/06_intervention.png` | clamping the band features → vertical stretch |
| `results/metrics.json`, `results/report.md` | full per-feature table and headline numbers |

## Layout

```
src/cubic_model.py   MLP (NumPy) regressing y = x³ + a x² + x + b from (x,a,b)
src/sae.py           TopK Sparse Autoencoder (NumPy) + reconstruction R²
src/features.py      (x,a) feature grids, the bucket test, diagonal-band + landmark analysis, intervention
src/experiment.py    end-to-end run: NN → SAE → analysis → figures + metrics
wolfram/…            NN + analysis in Wolfram Language (SAE via Python, as in the original)
tests/…              cubic/landmark math, TopK sparsity, slope recovery, NN & SAE quality
docs/reproduction.md the reproduced essay
```

> Reproduced independently in NumPy from the original notebook; retrained weights
> mean feature *indices* differ from the essay, but the phenomenon — b-invariant
> features computing `t = x + a/3` — reproduces. See the fidelity notes in
> `docs/reproduction.md`.
