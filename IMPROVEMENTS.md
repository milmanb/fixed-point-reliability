# What changed in this round, and what came out of it

This is a plain-language companion to the report. It explains the four things added on top of the
original study, why each was worth adding, and what the numbers said. Nothing here required
retraining: every result reuses the 23 checkpoints already in `checkpoints/`.

## The short version

The original study showed that the idempotence residual $g = |f(f(y)) - f(y)|$ ranks per-image
errors reasonably well inside a corruption level, but that each architecture has an unseen
corruption where the model is "stable but wrong" and $g$ misses it. That was argued with rank
correlations, which are relative: they say a signal orders images worse, not what it costs you.

The main addition turns $g$ into a **conformal error bar** with a distribution-free coverage
guarantee. The proposal's title promised calibration, conformal bounds are one way to deliver it,
and the previous report listed the missing coverage guarantee as its first open gap. The result is
sharper than the ranking version:

- The bound keeps its nominal 90% coverage **marginally**, exactly as the theory says it must.
- Broken down by corruption family, the same bound covers **100% of noise images but only 61% of
  the bottleneck model's pixel-mask images**, and 74% and 76% of the skip model's pixel- and
  box-mask images. A "90% guarantee" silently fails on nearly two in five images of the family that
  holds most of the worst errors.
- Calibrate on the two in-distribution noise levels ($\sigma$ = 0.1, 0.2), deploy on the other
  eleven, and the bottleneck model's coverage drops to **48%** for $g$ (and 19% for the
  displacement $d$ and measurement residual $r_A$). The skip model's drops to 16% for $g$, below
  $d$ and $r_A$ (26%).

So the failure is not just a weaker correlation. The guarantee breaks on unseen masks, including
each model's blind spot, and the calibration procedure gives no warning that it broke.

Two cheaper additions tested whether the blind spot is specific to the one-step residual.

- **Deeper iterates ($g_2$, $g_3$) are worse, not better** in three of the four settings — on the
  bottleneck model they rank below the one-step residual (0.63 and 0.59 against 0.70) and leave
  coverage unchanged; only the skip model with the penalty gains ($g_2$ 0.69 against 0.63). The
  report's first-order argument (Eq. 1) says why they cannot repair the blind spot: each extra
  application multiplies the displacement by the Jacobian again.
- **Ensemble disagreement is the one clear improvement**, and the most interesting result of the
  round. It ranks better than the residual within a corruption level in three of the four settings,
  and it is dramatically more robust to idempotence training — but it inherits both architectures'
  blind spots, and it ranks worse than the residual once the corruptions are pooled. Three seeds of
  one architecture fail the same way on an unseen operator, so they agree while all being wrong.

## 1. Conformal error bars (the headline)

**What it does.** Fits an isotonic map $\hat u$ from a signal to the error on a quarter of the test
images, forms nonconformity scores $e_i / \hat u(y_i)$ on a second quarter, takes the
split-conformal quantile $\hat q$ of those scores, and reports
$\hat e_{\text{hi}}(y) = \hat q \cdot \hat u(y)$ on the held-out half at $\alpha = 0.1$. The map is
fitted on images that supply no calibration score, which is what the coverage guarantee needs.

**Why.** The proposal was titled *Calibrating Reference-Free Reliability Signals*. The previous
report calibrated the signals but only scored the ranking of the calibrated values, and said so in
its limitations: "without a coverage guarantee". Conformal prediction is the standard way to close
that, and imaging-specific versions exist ([K-RCPS](https://arxiv.org/abs/2302.03791),
[SURE self-calibration](https://arxiv.org/abs/2502.05127)). Here it is used as a measuring
instrument, not as a contribution.

**What came out** (bottleneck model, $\lambda_{\text{id}} = 0$, mean over 3 seeds):

| quantity | $g$ | marginal bound | level-mean bound |
|---|---|---|---|
| mean width | 0.174 | 0.182 | **0.138** |
| pooled coverage | 0.90 | 0.90 | 0.90 |

Per-family coverage of the $g$ bound, at the same nominal 90%:

| family | bottleneck model | skip model |
|---|---|---|
| noise | 1.00 | 1.00 |
| blur | 1.00 | 0.99 |
| box mask | 0.96 | **0.76** |
| pixel mask | **0.61** | **0.74** |

Three cells are under-covered: the bottleneck model's pixel masks and both mask families of the
skip model. So the failure is not confined to the two blind spots the original report identified
by rank correlation. Per level, coverage collapses on the most severe unseen masks: 75% pixel
masks (0.25 for the bottleneck model, 0.21 for the skip model) and, for the skip model, 14-pixel
boxes (0.52). For the skip model the pixel-mask failure happens although its residual ranks those
errors well (within-level $\rho$ 0.95 to 0.97). A single normalizer fitted across all levels
mis-scales the most severe unseen masks; the blind spot adds to that but is not its only cause.
Either way, the stated guarantee is false on those families, not just the ordering worse.

Two further readings. First, for the bottleneck model $g$ buys something over ignoring the signal
entirely (0.174 vs 0.182 width at equal coverage), but for the skip model it does not (0.173 vs
0.170), and knowing which corruption you are looking at buys much more in both (0.138 and 0.129),
in line with the ranking experiments, now in the units a user cares about. Second, the marginal
guarantee is not protective, and nothing in the procedure flags the violation; you only see it if
you already know which family to break the results down by, which is precisely what you do not
know at deployment.

There is also a **SURE-self-calibrated** variant that uses no clean image anywhere, only Stein's
estimate from the noisy observation — the only variant of any of this that could actually be run at
deployment time. Since SURE estimates the per-pixel squared error, it has to be scored against that
target rather than the L1 error (getting this wrong makes the bound look catastrophically broken
when it is only on a different scale). Done correctly, on the noise family the bound on $g$
**over**-covers: 0.94 against the nominal 0.90, for 17% more width than the supervised bound on
the same target. With $d$ or $r_A$ it slightly under-covers instead (0.89). For $g$, erring
conservative is the right direction to err, so dropping the clean images from calibration is
affordable here.

Code: `src/fpr/conformal.py`, `scripts/conformal.py`. Outputs: `results/conformal/`.

## 2. Risk-coverage and selective risk

**What it does.** Rejects the highest-signal images first and reports the mean error of what is
left, summarized as the normalized area under the risk-coverage curve (nAURC): the mean selective
risk over all coverages, as in Geifman et al. (ICLR 2019), scaled so that 0 means ranking by the
true error and 1 means a random order.

**Why.** Spearman $\rho$ and AUROC do not tell you whether abstention is worth doing. Selective
risk is the operational number, and recent work pairs abstention with conformal risk control
([Selective Conformal Risk Control](https://arxiv.org/abs/2512.12844)).

**What came out** (bottleneck, $\lambda_{\text{id}} = 0$):

| signal | nAURC | mean error if you reject the worst 20% |
|---|---|---|
| oracle (rank by true error) | 0.00 | 0.064 |
| $g$ | **0.41** | 0.084 |
| $g_2$ | 0.45 | 0.083 |
| disagreement | 0.48 | 0.085 |
| $d$ | 0.63 | 0.092 |
| $r_A$ | 0.70 | 0.100 |
| contraction ratio $q$ | 0.92 | 0.091 |
| brightness $b$ | 1.31 | 0.097 |
| keep everything | — | 0.092 |

By nAURC $g$ is the best reference-free signal here and closes 59% of the gap between a random
order and the oracle. At 80% coverage it achieves under a third of the oracle's risk reduction
(0.092 to 0.084, against 0.064), and $g_2$ is marginally better there. Three results are worth
flagging:

- At 80% coverage $r_A$ and brightness are **worse than rejecting nothing**, so a plausible-looking
  signal can actively hurt. By nAURC $r_A$ still beats a random order, and brightness does not;
  its pooled AUROC (0.37) already says it is worse than random.
- Idempotence training makes selective risk worse too: nAURC rises from 0.41 to 0.50 on the
  bottleneck model and 0.51 to 0.65 on the skip model. That is the report's main claim reproduced
  in a third independent metric.
- Disagreement is *worse* than $g$ here (0.48 vs 0.41) even though it ranks better within 12 of
  the 13 levels. Pooling is what costs it: its level-to-level scale tracks the error less well
  (its level medians alone give a pooled $\rho$ of 0.29, against 0.46 for $g$), so mixing
  corruptions hurts it more. Worth knowing if you plan to abstain without knowing the corruption.

Code: `src/fpr/selective.py`, `scripts/selective.py`. Outputs: `results/selective/`.

## 3. Ensemble disagreement (the positive finding)

**What it does.** For the three seeds of a configuration, $\text{dis}(y)$ is their mean absolute
deviation from their own mean prediction. Reference-free, and free in training terms because the
seeds already exist.

**Why.** A 2026 paper on language models argues that
[self-consistency, a proxy for aleatoric uncertainty, collapses on confident errors, where a model
repeats the same wrong answer, while cross-model disagreement captures the epistemic
part](https://arxiv.org/abs/2604.17112). That gives a falsifiable prediction for this project's
failure mode: an output that is stable but wrong is a confident error, so disagreement should catch
what $g$ misses.

**What came out.** Partly confirmed, and the split is informative.

Where it wins, it wins clearly. Within-level Spearman $\rho$ against the true error, median over
the 13 levels, mean over seeds:

| model | $g$ | $\text{dis}$ |
|---|---|---|
| bottleneck, $\lambda_{\text{id}} = 0$ | 0.70 | **0.79** |
| bottleneck, $\lambda_{\text{id}} = 1$ | 0.62 | **0.76** |
| skip, $\lambda_{\text{id}} = 0$ | **0.86** | 0.84 |
| skip, $\lambda_{\text{id}} = 1$ | 0.63 | **0.88** |

The robustness pattern is the real result. Training for idempotence costs $g$ 0.09 on the bottleneck
model and 0.23 on the skip model; disagreement loses 0.03 and *gains* 0.04. That makes sense —
disagreement is not the quantity being minimized, so optimizing the residual cannot launder it.
This is the clearest practical recommendation to come out of the round: if a model has been trained
for idempotence, do not read its residual, read the spread of its seeds.

Where it fails, it fails for a reason worth stating: **disagreement inherits both blind spots.**

| blind spot | $g$ | $\text{dis}$ | chance |
|---|---|---|---|
| bottleneck, pixel masks: coverage | 0.61 | 0.59 | (nominal 0.90) |
| bottleneck, pixel masks: worst errors flagged | 18% | 20% | 25% |
| skip, box masks: coverage | 0.76 | 0.79 | (nominal 0.90) |
| skip, box masks: worst errors flagged | 0.4% | 1.2% | 25% |

All three seeds turn sparse dots into the same dim garment, so they agree with each other while all
being wrong. Disagreement measures epistemic uncertainty only if the members actually disagree, and
three seeds of one architecture on one training distribution are too correlated for that. (It is
genuinely better on the bottleneck model's box masks — flagging 85% of the worst errors against
$g$'s 74% — just not on either model's actual blind spot.)

So the source paper's claim holds in ranking and in robustness to the penalty, but not as a repair
for an unseen-operator blind spot. Homogeneous ensembles inherit shared failure modes — a
limitation that paper itself raises, and this is a concrete instance of it.

Code: `src/fpr/ensemble.py`.

## 4. Deeper iterates $g_2$, $g_3$ and the contraction ratio

**What it does.** $g_k = |f^{k+1}(y) - f^k(y)|$, plus $q = g_2/g_1$. Two extra forward passes.

**Why.** $g$ only ever probes one extra application, while recursive self-consistency methods
iterate. The report's first-order argument (Eq. 1: $g \approx |J_f \delta|$) extends to deeper
iterates, which give roughly $|J_f^k \delta|$ and therefore cannot recover information the Jacobian
has already cancelled. This had not been measured.

**What came out.** Confirmed for the blind spot: conformal coverage on pixel masks is unchanged
(0.60 vs 0.61). In ranking, deeper iterates are mostly worse, not merely uninformative:
within-level $\rho$ on the bottleneck model falls with depth — $g$ 0.70, $g_2$ 0.63, $g_3$ 0.59 —
and only the skip model with the penalty gains ($g_2$ 0.69 against 0.63). The contraction ratio $q$
is useless ($\rho = 0.01$, and $-0.16$ under idempotence training; nAURC 0.92). Each extra
application multiplies by the Jacobian again, which attenuates the displacement without revealing
anything new, so iterating the map is not a way out of the blind spot.

Code: `iterate_steps` in `compute_signals` (`src/fpr/signals.py`), passed on by
`per_image_signals` (`src/fpr/evaluation.py`) and `--iterate-steps`; `iterate_signals` is a
standalone helper the tests check against it.

## What this changes about the project's conclusion

The original conclusion — a small idempotence residual is weak evidence of a correct restoration —
survives and gets stronger. Three things are new:

1. The failure is quantified in the units a deployment would use. "The guarantee you would quote is
   marginal, it holds, and it is false on unseen masks, including the family you care about" is a
   more actionable statement
   than "the rank correlation drops here", and it cannot be dismissed as a small-effect artifact.
2. One escape route is closed by measurement rather than argument: iterating the map further does
   not repair the blind spot, as the first-order argument suggests, and in three of the four
   settings it ranks worse.
3. One signal beats the residual at ranking. Cross-seed disagreement ranks better within a level in
   three of the four settings and is much more robust to the idempotence penalty, which is a
   positive recommendation the original study did not have. It is not a fix for the blind spot, and
   it is worse once the corruptions are pooled, so the two signals are complementary rather than
   interchangeable.

The practical upshot: a fixed-point residual is usable as a within-corruption ranking signal, and
defensible as an error bar only over a corruption distribution you have calibrated on. Neither is
the claim that "a fixed point is a correct output".

## Reproducing

The per-image dumps are git-ignored because they are large (about 120 MB and 50 MB), so
regenerate them first:

```bash
pip install -e ".[dev]"
pytest

# per-image signals including dis, g2, g3, q (hours; the bottleneck run also does the projectors,
# which the summary needs, so do not pass a bare --projectors there)
python scripts/evaluate_models.py --pattern "dae_*.pt"  --ensemble --no-metrics \
    --device cpu --jobs 4 --threads 2 --out results/models
python scripts/evaluate_models.py --pattern "unet_*.pt" --projectors --ensemble --no-metrics \
    --device cpu --jobs 4 --threads 2 --out results/models_skip

# bootstrap scores for the new signals, merged into the existing metrics.csv
python scripts/evaluate_models.py --from-per-image --signals dis g2 g3 q \
    --device cpu --jobs 4 --out results/models
python scripts/evaluate_models.py --from-per-image --signals dis g2 g3 q \
    --device cpu --jobs 4 --out results/models_skip

# the new analyses (minutes; numpy and pandas only)
python scripts/conformal.py
python scripts/selective.py
python scripts/calibrate.py
python scripts/report_table.py
```

`torch` and `numpy` must be ABI-compatible: with `torch 2.3.1`, for instance, the project needs
`numpy < 2`, or every tensor-to-numpy conversion fails with "Numpy is not available". The models
are small, so a CPU run only costs wall-clock time.

## File map of the additions

```
src/fpr/conformal.py      split conformal bounds, shift test, SURE self-calibration
src/fpr/selective.py      risk-coverage curve, AURC, selective risk
src/fpr/ensemble.py       cross-seed disagreement
src/fpr/signals.py        + compute_signals gained iterate_steps (g_k, q); iterate_signals helper
src/fpr/evaluation.py     + per_image_signals passes iterate_steps to every model
scripts/conformal.py      coverage/width/shift/SURE tables -> results/conformal/
scripts/selective.py      selective risk tables and figure -> results/selective/
scripts/evaluate_models.py  + --ensemble --no-metrics --signals --from-per-image --models
                            --iterate-steps
scripts/report_table.py   + table_coverage.tex, r_A and dis columns in table_main.tex
tests/test_improvements.py  coverage, AURC, disagreement and multi-step residual tests
```
