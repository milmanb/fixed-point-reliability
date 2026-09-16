# Fixed-Point Reliability

**Stable but Wrong: Fixed-Point Residuals as Reference-Free Reliability Signals for Image Restoration**
Final project, Modern Computer Vision (MCV), 2026. Proposed as "Calibrating Reference-Free
Reliability Signals for Learned Image Projectors".
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
python scripts/train_dae.py --lambda-id 0 --seed 0                      # repeat for seeds 1, 2
python scripts/train_dae.py --lambda-id 1 --lambda-warmup 5 --seed 0    # repeat for seeds 1, 2
python scripts/train_dae.py --lambda-id 1 --seed 0                      # collapses, see below
```

`ConvAutoencoder` (602k parameters, 64-unit bottleneck, no normalization layers) is
trained with $\lvert f(y) - x \rvert_1 + \lambda_{id} L_{idem}$ on Gaussian noise with
$\sigma \sim U[0.05, 0.25]$, for 15 epochs. The last 5,000 training images are used for
validation. At test time, $\sigma = 0.1, 0.2$ are in distribution, $\sigma = 0.3, 0.5$ are
held-out noise levels, and blur and masks are unseen operators. Checkpoints go to
`checkpoints/` (git-ignored) and training curves to `results/train/`.

With $\lambda_{id} = 1$ from the first step, every seed collapses within one epoch to a
constant output, the per-pixel median image: validation error 0.2107 against 0.2106 for that
image, and $g = 0$. The collapse held for all 15 epochs. `--lambda-warmup 5` ramps
$\lambda_{id}$ from 0 over five epochs and avoids it.

### 3. Evaluate models

```bash
python scripts/evaluate_models.py --device cpu     # all checkpoints, plus radial and pca64
```

In addition to $g$, $d$ and $r_A$, the evaluation computes first-order signals on the
noise levels: the divergence $\mathrm{div} = \tfrac{1}{D}\mathrm{tr}\,J_f(y)$ (Hutchinson
estimate with finite differences), the linearized residual $\lvert J_f(y)(f(y) - y) \rvert$, and
Stein's unbiased risk estimate $\mathrm{SURE} = d_{mse} - \sigma^2 + 2\sigma^2\,\mathrm{div}$.
Signals are scored per corruption level, pooled per family, and pooled over all levels.
Two baselines guard against scores that do not reflect model failures:

- `b`, the mean brightness of $y$. Under masking and strong blur the error grows with the
  amount of content in the image, so brightness alone ranks errors well (Spearman up to
  0.96 within a pixel-mask level). Partial Spearman correlations given $b$ show what each
  signal adds beyond brightness.
- `<signal>@level`, the median of a signal over its corruption level. In pooled scores it
  measures how much of the ranking is severity detection.

```bash
python scripts/plot_models.py --model dae_lam0_seed0   # figures in results/models/figures
```

```bash
python scripts/compare_models.py --groups dae_lam0 dae_lam1w5 dae_lam1
```

#### Results

Spearman $\rho$ between each signal and the true error, three seeds per group. "Within
level" is the median over the 13 corruption levels; "partial" controls for brightness;
"pooled" mixes all levels, the setting in which the corruption is unknown.

| signal | $\lambda_{id}=0$ within / noise / partial / pooled | $\lambda_{id}=1$ within / noise / partial / pooled |
|---|---|---|
| $g$ | +0.70 / +0.75 / +0.59 / +0.55 | +0.62 / +0.60 / +0.53 / +0.49 |
| $d$ | +0.76 / +0.67 / +0.57 / +0.12 | +0.82 / +0.74 / +0.65 / +0.18 |
| $r_A$ | +0.79 / +0.67 / +0.76 / +0.02 | +0.84 / +0.74 / +0.82 / +0.05 |
| SURE (noise only) | +0.73 / +0.73 / +0.73 / - | +0.77 / +0.77 / +0.78 / - |
| $b$ (baseline) | +0.51 / +0.05 / - / -0.17 | +0.41 / +0.07 / - / -0.17 |

1. **Exact projectors carry no residual information.** $g < 2\cdot10^{-15}$ everywhere,
   while the mean error is 0.04 to 0.23.
2. **On a trained model $g$ is informative**, and it is the only signal that ranks errors
   when corruptions are mixed ($\rho = 0.55$ against 0.12 for $d$). It also holds up as the
   noise grows, where $d$ collapses (0.93 at $\sigma = 0.1$ to 0.28 at $\sigma = 0.5$).
3. **$g$ is blind to the dominant failure.** Pixel masks hold 60% of the worst-quartile
   errors, and $g$ flags only 18% of them. Under 75% masking the mean error is 0.234 against
   0.043 at $\sigma = 0.1$, while mean $g$ barely moves (0.013 against 0.011).
4. **Training for idempotence makes $g$ worse as a signal.** With $\lambda_{id} = 1$ the
   residual halves (mean 0.0146 to 0.0088) and so does its usefulness: within-level
   $\rho$ falls 0.70 to 0.62, within noise levels 0.75 to 0.60, and recall of the worst
   pixel-mask errors 18% to 10%. The three seeds of each group do not overlap. Restoration
   costs 3.6% more error overall.
5. **The same training helps the other signals**, which is consistent with the projector
   experiment: a tighter fixed-point set makes displacement more informative ($d$: 0.76 to
   0.82, $r_A$: 0.79 to 0.84, SURE: 0.73 to 0.77).
6. **Without warm-up the model collapses**: a constant output with $g = 1.4\cdot10^{-6}$ and
   error 0.2095 at every corruption. Exactly idempotent, useless.
7. **Baselines matter.** Brightness alone reaches a median within-level $\rho$ of 0.51, so
   raw within-level scores overstate every signal; the partial columns correct for it.

![Idempotence residual vs. error](results/models/comparison/g_vs_error.png)

![Ranking quality per condition](results/models/comparison/rho_by_condition.png)

![Training curves](results/models/comparison/training_curves.png)

![Stable but wrong outputs](results/models/figures/failures_dae_lam0_seed0.png)

### 4. Build the report

```bash
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`latexmk` needs Perl, which is not installed on the machine used here, so the four steps
are run directly.

## Repository layout

```
src/fpr/
  data.py          Fashion-MNIST loading
  degradations.py  operators A_s: noise, blur, pixel mask, box mask
  projectors.py    exact projectors: identity, radial, PCA, nearest neighbour
  models.py        convolutional denoising autoencoder and checkpoint loading
  losses.py        reconstruction and idempotence losses
  signals.py       g, d, r_A, the offline error e, and first-order signals (div, g_lin)
  metrics.py       Spearman / AUROC / partial Spearman with clustered bootstrap CIs
  evaluation.py    shared corruption grid, per-image signals, scoring
  plotting.py      shared figure style
scripts/
  smoke_projectors.py
  train_dae.py
  evaluate_models.py
  plot_models.py
  explore_sure.py  SURE vs. displacement on exact projectors (exploration)
tests/             unit tests (pytest)
results/           small result tables and figures (per-image dumps are git-ignored)
```

## Status

- [x] Evaluation pipeline and exact-projector smoke check
- [x] Training and evaluation pipeline for the autoencoders, first-order signals
- [x] Autoencoders for $\lambda_{id} \in \{0, 1\}$, three seeds each, plus the collapsed run
- [x] Evaluation on held-out noise levels, blur and masks, with brightness and severity baselines
- [x] Report draft (LaTeX, three pages including references)
- [ ] Optional ablations: IGN-style stop-gradient routing, $\lambda_{id} = 0.1$
- [ ] 5-minute talk
- [ ] Make the repository public before submission

## References

1. A. Shocher et al., "Idempotent Generative Network," ICLR 2024.
2. M. Al-Jaff et al., "A Non-Adversarial Approach to Idempotent Generative Modelling," ECAI 2025.
3. S. Zaman et al., "Score-based Idempotent Distillation of Diffusion Models," arXiv:2509.21470, 2025.
4. N. Durasov et al., "IT³: Idempotent Test-Time Training," ICML 2025.
