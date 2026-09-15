# Fixed-Point Reliability

**Calibrating Reference-Free Reliability Signals for Learned Image Projectors**
Final project, Modern Computer Vision (MCV), 2026.
Itay Shorian, Michael Bazkor, Boaz Cohen, Bar Milman.

## Question

Idempotent Generative Networks (IGN) and their follow-ups train maps with
$f(f(y)) \approx f(y)$. A tempting reading is that an output that is a fixed point
is also a correct reconstruction. We test whether this holds, and whether other
signals computed without the clean image are better per-image warnings.

Setting: clean image $x$, known operator $A_s$, observation $y = A_s x + n$.
All signals are per-image mean absolute values over pixels.

| Signal | Definition | Needs |
|---|---|---|
| idempotence residual | $g(y) = \lvert f(f(y)) - f(y) \rvert$ | $f, y$ |
| displacement | $d(y) = \lvert f(y) - y \rvert$ | $f, y$ |
| measurement residual | $r_A(y) = \lvert A_s f(y) - y \rvert$ | $f, y, A_s$ |
| true error (offline only) | $e(y) = \lvert f(y) - x \rvert$ | clean $x$ |

A signal is scored by Spearman's $\rho$ with $e$ and by AUROC for detecting the
worst-error quartile, with image-level bootstrap 95% CIs.

## Setup

Python 3.10+. Install PyTorch for your platform first (https://pytorch.org), then:

```bash
pip install -e ".[dev]"
pytest
```

Fashion-MNIST is downloaded to `data/` on first use.

## Reproduce

### 1. Smoke check on exact projectors (proposal, Sec. 5)

```bash
python scripts/smoke_projectors.py                              # full run, ~20 min
python scripts/smoke_projectors.py --n-eval 2000 --n-boot 200   # quick run, ~2 min
```

Projectors: `radial` (projection onto the sphere of mean training radius around the
training mean) and `pca{16,64,256}` (orthogonal projection onto the top-k principal
subspace), as in the proposal, plus two references: `identity` and `nn` (nearest
training image). Corruptions: Gaussian noise, Gaussian blur, random pixel masks and
random box masks, 13 levels in total, on the 10,000 Fashion-MNIST test images.

Outputs in `results/smoke_projectors/`: `summary.csv` (signal and error means per
condition), `metrics.csv` (Spearman and AUROC with CIs, per condition, pooled per
corruption family, and pooled over everything), figures, and `run_info.json`.

#### Results (10,000 test images, 1,000 bootstrap resamples)

The check reproduces, but it holds by construction. An exact projector satisfies
$P(P(y)) = P(y)$ for every input, so $g$ is zero up to round-off on every image and
every corruption, while the error is not. A constant signal cannot rank images:
$\rho(g, e)$ is undefined and the AUROC is 0.5 everywhere. The hypothesis can only be
tested on learned, approximately idempotent models.

"Within level" is the median [min, max] over the 13 corruption levels; "pooled" mixes
all levels and corruption types.

| projector | max $g$ | mean $e$ | $\rho(d, e)$ within level | $\rho(r_A, e)$ within level | AUROC $d$ / $r_A$ | pooled $\rho$ $d$ / $r_A$ |
|---|---|---|---|---|---|---|
| `identity` | 0 | 0.032-0.399 | undefined | +0.75 [+0.74, +0.97] (4/13 defined) | 0.50 / 0.50 | undefined / -0.33 |
| `radial` | 2e-16 | 0.050-0.234 | +0.57 [-0.46, +0.89] | +0.60 [-0.46, +0.99] | 0.84 / 0.86 | +0.57 / +0.38 |
| `pca16` | 2e-15 | 0.092-0.233 | +0.91 [+0.47, +0.99] | +0.91 [+0.47, +0.99] | 0.97 / 0.97 | +0.28 / +0.12 |
| `pca64` | 2e-15 | 0.065-0.231 | +0.84 [+0.30, +0.99] | +0.85 [+0.30, +0.99] | 0.94 / 0.92 | +0.36 / +0.25 |
| `pca256` | 2e-15 | 0.040-0.233 | +0.53 [+0.05, +0.99] | +0.73 [+0.05, +0.99] | 0.80 / 0.84 | +0.57 / +0.50 |
| `nn` | 0 | 0.065-0.236 | +0.89 [+0.48, +0.98] | +0.90 [+0.48, +0.99] | 0.95 / 0.96 | +0.21 / +0.04 |

Observations for the next phase:

1. Displacement and measurement residual rank errors well within a fixed corruption
   level when the fixed-point set is tight (`pca16`, `nn`), and worse when it is loose
   (`pca256`; `identity` gives $d \equiv 0$).
2. A badly shaped fixed-point set can reverse the signal: for `radial`,
   $\rho(d, e) = -0.46$ at noise $\sigma = 0.2$.
3. Rankings weaken as corruption grows: `pca16` under noise goes from $\rho = 0.98$
   ($\sigma = 0.1$) to $0.47$ ($\sigma = 0.5$).
4. Pooling changes the question. Within one corruption family the signals mostly detect
   severity (AUROC = 1.00 for `radial` and `pca256` over the four noise levels). Across
   families the correlations drop to 0.04-0.57, because the scale of $d$ differs between
   noise, blur and masking. Using these signals across corruptions requires calibration.
5. The measurement residual helps where displacement misleads: for blur with `pca256`,
   pooled $\rho(d, e) = -0.41$ but $\rho(r_A, e) = +0.74$.

The full run takes about 22 minutes on a laptop (GTX 1650 Ti), almost all of it in the
bootstrap.

![Example restorations](results/smoke_projectors/examples.png)

![Signals vs. true error](results/smoke_projectors/scatter.png)

![Spearman heatmap](results/smoke_projectors/spearman_heatmap.png)

### 2. Train the denoising autoencoders

```bash
python scripts/train_dae.py --lambda-id 0 --seed 0     # repeat for seeds 1, 2
python scripts/train_dae.py --lambda-id 1 --seed 0     # needs fpr.losses.idempotence_loss (open TODO)
```

`ConvAutoencoder` (602k parameters, 64-unit bottleneck, no normalization layers) is
trained with $\lvert f(y) - x \rvert_1 + \lambda_{id} L_{idem}$ on Gaussian noise with
$\sigma \sim U[0.05, 0.25]$, for 15 epochs. The last 5,000 training images are used for
validation. At test time, $\sigma = 0.1, 0.2$ are in distribution, $\sigma = 0.3, 0.5$ are
held-out noise levels, and blur and masks are unseen operators. Checkpoints go to
`checkpoints/` (git-ignored) and training curves to `results/train/`.

### 3. Evaluate models

```bash
python scripts/evaluate_models.py --device cpu     # all checkpoints, plus radial and pca64
```

In addition to $g$, $d$ and $r_A$, the evaluation computes first-order signals on the
noise levels: the divergence $\mathrm{div} = \tfrac{1}{D}\mathrm{tr}\,J_f(y)$ (Hutchinson
estimate with forward-mode AD), the linearized residual $\lvert J_f(y)(f(y) - y) \rvert$, and
Stein's unbiased risk estimate $\mathrm{SURE} = d_{mse} - \sigma^2 + 2\sigma^2\,\mathrm{div}$.
Signals are scored per corruption level, pooled per family, and pooled over all levels.
Two baselines guard against scores that do not reflect model failures:

- `b`, the mean brightness of $y$. Under masking and strong blur the error grows with the
  amount of content in the image, so brightness alone ranks errors well (Spearman up to
  0.96 within a pixel-mask level). Partial Spearman correlations given $b$ show what each
  signal adds beyond brightness.
- `<signal>@level`, the median of a signal over its corruption level. In pooled scores it
  measures how much of the ranking is severity detection.

## Repository layout

```
src/fpr/
  data.py          Fashion-MNIST loading
  degradations.py  operators A_s: noise, blur, pixel mask, box mask
  projectors.py    exact projectors: identity, radial, PCA, nearest neighbour
  models.py        convolutional denoising autoencoder and checkpoint loading
  losses.py        reconstruction and idempotence losses
  signals.py       g, d, r_A, the offline error e, and first-order signals (div, g_lin)
  metrics.py       Spearman / AUROC with clustered bootstrap CIs
  evaluation.py    shared corruption grid, per-image signals, scoring
scripts/
  smoke_projectors.py
  train_dae.py
  evaluate_models.py
  explore_sure.py  SURE vs. displacement on exact projectors (exploration)
tests/             unit tests (pytest)
results/           small result tables and figures (per-image dumps are git-ignored)
```

## Status

- [x] Evaluation pipeline and exact-projector smoke check
- [x] Training and evaluation pipeline for the autoencoders, first-order signals
- [x] Convolutional denoising autoencoders, $\lambda_{id} = 0$ (3 seeds), evaluated
- [x] Brightness baseline and partial correlations in the evaluation
- [ ] Idempotence loss and $\lambda_{id} = 1$ models (3 seeds)
- [ ] Evaluation on held-out noise levels, blur and masks
- [ ] Report (LaTeX, 2-3 pages)

## References

1. A. Shocher et al., "Idempotent Generative Network," ICLR 2024.
2. M. Al-Jaff et al., "A Non-Adversarial Approach to Idempotent Generative Modelling," ECAI 2025.
3. S. Zaman et al., "Score-based Idempotent Distillation of Diffusion Models," arXiv:2509.21470, 2025.
4. N. Durasov et al., "IT³: Idempotent Test-Time Training," ICML 2025.
