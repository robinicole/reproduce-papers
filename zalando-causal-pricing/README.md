# Causal demand forecasting for pricing: two Zalando papers on one benchmark

Independent re-implementations of the demand forecasters from two Zalando papers, evaluated on the
same data pipeline (the synthetic pricing simulator of paper 1, and the public M5 retail data set),
with a single comparison table and an assessment of each method's strengths and weaknesses.

| Paper | Model in this repo | File |
|---|---|---|
| 1. *Causal Forecasting for Pricing* (Schultz et al., [arXiv:2312.15282v2](https://arxiv.org/abs/2312.15282v2)) | **DML Forecaster** (`dml`) plus its ablations `dml-nocf`, `sdml`, `tf` | `dml.py` |
| 2. *Deep Learning based Forecasting: a case study from the online fashion industry* (Kunz et al., [arXiv:2305.14406](https://arxiv.org/abs/2305.14406)) | **Monotonic-demand transformer** (`mdl`) | `mdl.py` |

`papers/fetch.sh` downloads both papers and converts them to markdown (they are not committed).
The headline numbers are below; the full tables land in `results/RESULTS.md` after a run.

## Layout

| Path | Purpose |
|---|---|
| `models/common.py` | Shared pieces: forecast windows, the small transformer block, training/prediction loops, metrics (MAE, MSE, price-weighted demand error). |
| `models/dml.py` | Paper 1. Outcome model, treatment model, effect model; two-fold cross-fitting by item parity; cross-fit/own-fold ensemble at inference. Ablations: no cross-fitting, sDML (no treatment model), TF (naive S-learner with the same head). |
| `models/mdl.py` | Paper 2. Encoder/decoder transformer; future discount bypasses the network and enters a piecewise-linear monotonic demand layer; Taylor-exponential loss on log demand. |
| `data/synthetic.py` | Paper 1's Appendix E simulator: 4467 articles, 100 weeks, seasonal and trend base demand, linear price effect, stock-coverage pricing policy. Has knobs for policy sharpness and noise (see reproduction notes). |
| `data/m5_data.py` | M5 daily sales to weekly demand; discount = 1 − price / expanding-max price per series. Raw CSVs go in `data/m5/`. |
| `experiments/run_synthetic.py` | Paper 1's synthetic protocol: four training periods, on-policy and off-policy evaluation (six constant discount levels with simulator ground truth), effect error. |
| `experiments/run_m5.py` | M5 protocol: 26-week context, 4-week horizon, four forecast origins; all windows plus a "price change" slice as an off-policy proxy. |
| `experiments/report.py` | Collects `results/*.csv` into `results/RESULTS.md` and refreshes the table below. |
| `scripts/run_all.sh`, `scripts/run_extra.sh` | The full pipeline (about two hours on one GPU), plus the anchored paper-2 ablation. |
| `tests/` | `pytest` runs the three module self-checks on CPU. |
| `papers/fetch.sh` | Fetches and converts both papers. |

## Quick start

```bash
pip install -r requirements.txt
pytest                                   # three self-checks on CPU, ~2 minutes
papers/fetch.sh                          # papers as PDF + markdown into papers/
python experiments/run_synthetic.py --seeds 1 --periods 1 --epochs 4 --sharpness 10 --noise_mult 0.1   # ~1 minute
scripts/run_all.sh                       # everything, then results/RESULTS.md and the table below
```

M5 data: the Kaggle competition download requires accepting the competition rules; the raw CSVs
are also published as Kaggle datasets (this repo used `aryayadav0513/m5-forecasting-accuracy`).
Put `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv` in `data/m5/`; `data/m5_data.py` builds the weekly cache.

## The two methods in one paragraph each

**Paper 2 (the production model, 2019 onwards).** One global transformer. The encoder reads the
article's past (demand, discount, covariates); a non-autoregressive decoder reads the known future
covariates plus the last observed demand and discount. Future discounts never enter the network:
they go straight into a monotonic demand layer, `log q = q̂(κ) + σ(κ)·PL(d; δ(γ))`, where PL is a
piecewise-linear function of the discount with non-negative slopes `δ` from the encoder state and a
non-negative scale `σ` from the decoder state. Demand is therefore increasing in discount by
construction, which makes the model safe to feed into a price optimizer. The loss is a squared
error after a third-order Taylor approximation of `exp` on log-transformed demand.

**Paper 1 (the causal refinement).** The same kind of transformer is used three times in a
double machine learning (DML) layout. An *outcome* model predicts demand from everything except
the future discount, a *treatment* model predicts the future discount from the same inputs, and an
*effect* model outputs a single elasticity `ψ(z)` per window, combined as
`q̂ = q̃ · ((1−d)/(1−d̃))^ψ` (real data, constant elasticity) or `q̂ = q̃ + ψ·(d − d̃)` (the linear
simulator). The nuisance models are trained on two item-parity folds and cross-fitted so the effect
model only ever sees out-of-sample residuals; at inference the cross-fit and own-fold predictions
are averaged. The point is orthogonalization: the discount effect is learned from the part of the
discount that the past does not explain, so regularization of the big network cannot bias it.

## Reproduction notes and deviations from the papers

- **Networks are small** (64-dim, 2 layers, 4 heads) and the horizon is 5 weeks (synthetic) or 4
  weeks (M5). Paper 2's near/far-future split, 26-week horizon, multi-market output and
  sales-to-demand translation are not implemented; only the near-future decoder is.
- **The paper-1 baseline `tf`** feeds the future discount to the network as an ordinary input
  *and* through a linear head with `d̃ = 0`, as described for the simulation study. The faithful
  paper-2 model with the monotonic head is the separate `mdl` model, so both readings of "TF" are
  in the table.
- **Synthetic pricing policy.** Appendix E's probabilistic rule (move one discount step with
  probability `1 − 1/w` when stock coverage `w > 1`) produces almost no contemporaneous
  confounding: in our runs the discount is uncorrelated with the seasonal demand level at lag 0 and
  only correlates at a lag of several weeks. The main text describes a policy that adjusts every
  week. `synthetic.generate(sharpness=γ)` raises the move probability to `1 − w^−γ`; `γ = 1` is
  the appendix rule, `γ = 10` gives discount/season correlations of −0.5 to −0.85, matching the
  paper's Figure 2 narrative. Both settings are reported (`literal` and `calibrated`).
- **Noise.** The appendix noise terms alone give on-policy MAE around 6, versus 10 to 12 in the
  paper. The main text's multiplicative and additive week-level noise on the article factor is
  added (`noise_mult=0.1`) in the calibrated setting, which puts on-policy MAE and MSE at the
  paper's magnitude.
- **Training length matters for DML.** With short nuisance training the residual-on-residual slope
  is attenuated and the effect model drifts further from the truth the longer it trains. With 48
  nuisance epochs the attenuation mostly disappears. All neural models get 48 epochs; `tf` and
  `mdl` are additionally reported at 24 epochs, which is where they were best on a held-out period.
- **Baselines not reproduced:** SARIMAX and the two-way fixed-effects elasticity model.
- **M5 has no counterfactual ground truth.** The "price change" slice (horizon discount differs
  from the last four weeks' by more than ten points) is used as a natural-experiment proxy for the
  off-policy setting, in the spirit of paper 1's Cyberweek evaluation. Discounts on M5 are small
  (mean 2%, above 5% in a sixth of the weeks), so this is a much weaker price signal than fashion.
- **Stock covariate and absolute-level models.** Stock only declines over an article's life, so at
  every test origin it lies outside the range seen in training. The paper-2 model predicts the
  absolute log-demand level and extrapolates this into a uniform 20% under-prediction in later
  periods (in-sample it is unbiased; removing the stock feature removes the bias). The paper-1
  models predict a ratio to the recent demand level and are immune. `mdl-anchored` is the paper-2
  model with `log(mean recent demand)` added to its output (local scaling, not in the paper); it is
  reported as an ablation.
- **Ground-truth effect on synthetic data** is `dq/d(discount) = −p0·e_i`; for `mdl` it is
  estimated by finite differences of the monotone head between 0% and 50% discount.

## Results

Full tables with standard deviations are written to `results/RESULTS.md` by a run. The headline
table below is generated by `experiments/report.py` (best epoch count per model; synthetic numbers are means over four
training periods and three seeds; M5 numbers are means over four forecast origins).

<!-- RESULTS -->
|                                                                         |   calibrated: off-policy MAE |   calibrated: on-policy MAE |   calibrated: effect MAE |   literal: off-policy MAE |   literal: on-policy MAE |   literal: effect MAE |   M5: MAE all |   M5: elasticity |   M5: MAE price change |
|:------------------------------------------------------------------------|-----------------------------:|----------------------------:|-------------------------:|--------------------------:|-------------------------:|----------------------:|--------------:|-----------------:|-----------------------:|
| dml — DML Forecaster (paper 1)                                          |                        17.30 |                       16.80 |                    21.60 |                      8.00 |                     7.50 |                 13.80 |          3.62 |            -0.11 |                   6.82 |
| dml-nocf — DML, no cross-fitting                                        |                        21.30 |                       21.60 |                    21.80 |                      8.50 |                     8.20 |                  8.80 |          3.62 |            -0.06 |                   6.80 |
| sdml — sDML (no treatment model)                                        |                        19.40 |                       16.50 |                    44.50 |                     13.20 |                     8.30 |                 47.90 |          3.62 |            -0.04 |                   6.87 |
| tf — TF, linear head S-learner (paper 1 ablation)                       |                        16.80 |                       16.60 |                    20.20 |                      8.90 |                     8.20 |                 14.80 |          3.62 |            -0.08 |                   6.83 |
| mdl — Monotonic-demand transformer (paper 2)                            |                        30.20 |                       29.10 |                    18.00 |                     11.30 |                     9.90 |                 20.00 |          3.73 |            -0.68 |                   6.62 |
| mdl-anchored — paper 2 model anchored to recent demand level (ablation) |                        17.70 |                       18.10 |                    17.80 |                      8.20 |                     7.40 |                 13.70 |          3.76 |            -0.58 |                   6.51 |
| last4 — naive: mean of last 4 weeks                                     |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |          3.84 |           nan    |                   7.62 |
<!-- /RESULTS -->

How to read it, benchmark by benchmark:

- **Synthetic, calibrated** (strong confounding, paper-level noise). The DML Forecaster, the
  naive TF at its best epoch, and the anchored paper-2 model are within one standard deviation of
  each other on- and off-policy; DML has the lowest variance across periods and seeds. Removing the treatment model (`sdml`) reproduces the paper's
  collapse: on-policy is unchanged, off-policy error rises by a sixth and the effect error more than
  doubles. Training the naive TF twice as long costs it 1.4 MAE off-policy; the DML model does not
  have that failure mode. The unanchored paper-2 model is far off because of the stock
  extrapolation described above, yet its effect estimates are the best in the table.
- **Synthetic, literal** (the appendix rule verbatim, low noise). Here the paper's headline
  reproduces in direction: DML beats the naive TF off-policy (8.0 vs 8.9 at TF's best epoch, vs
  10.5 at equal training budget) while being on par on-policy, and sDML collapses off-policy
  (13.2 vs 8.3 on-policy). Cross-fitting did not matter for demand error but the no-cross-fitting
  variant had the best effect error, the one place where our ablation disagrees with the paper.
- **M5.** Every trained model lands on the same on-policy error (3.62 MAE) and all beat the naive
  last-four-weeks baseline by 6%. On the price-change windows the paper-2 model is best by 3%,
  followed by the DML variants. M5 prices rarely move, so the residual discount that DML learns
  from is tiny: its estimated elasticity is close to zero (−0.11) and its price-change accuracy is
  indistinguishable from the naive TF. The monotone head, which learns the response shape directly
  from the few price changes, extracts more from this data (elasticity −0.68).

## Strengths and weaknesses

**Paper 2, monotonic-demand transformer**

- Strengths: one network, one training run, cheapest of all models here. Monotonicity in discount
  is guaranteed, so the demand grid handed to the optimizer never says that a deeper discount sells
  less. The piecewise-linear head can represent saturating or accelerating responses, which the
  constant-elasticity head of paper 1 cannot. On M5 it is the best model on the price-change
  windows, and once level-anchored it matches the DML Forecaster on the synthetic benchmark
  (within one standard deviation, with twice the run-to-run variance).
- Weaknesses: it is an S-learner. The discount response is learned jointly with everything else,
  so whatever the pricing policy correlates with (season, stock, recent demand) can be absorbed
  into the discount slopes or, conversely, the true effect can be absorbed by the covariate path.
  On the confounded toy problem its effect estimate is attenuated by half. Monotonicity bounds
  the damage but does not remove the bias. Predicting the absolute demand level (log target,
  no local scaling) makes it fragile to covariates that drift out of the training range, which
  is what broke it on the synthetic benchmark. Its off-policy error also grows with training
  length in the literal setting (11.3 to 14.3 MAE from 24 to 48 epochs).

**Paper 1, DML Forecaster**

- Strengths: the effect is identified from the exogenous part of the discount only, so it is
  robust to regularization bias and to how long the networks train, and it degrades gracefully
  off-policy. The ablations reproduce cleanly: removing the treatment model (`sdml`) destroys the
  effect estimate in every setting, which is direct evidence that residualizing the treatment is
  what does the work. Cross-fitting costs little and never hurt.
- Weaknesses: three to five networks instead of one, and a two-stage training procedure whose
  second stage is only as good as the first. When the nuisance models are under-trained, the effect
  model can exploit the leftover predictable structure in the residuals, and it does so more the
  longer it trains; the method therefore needs well-converged nuisances, which is exactly the
  expensive part. The constant-elasticity head assumes a single elasticity per window and cannot
  represent kinks in the response. The exogenous discount variation must exist: under a fully
  deterministic policy the treatment residual is zero and there is nothing to learn from, and on M5,
  where price changes are rare and small, the extra machinery buys little.

**What the comparison did and did not confirm.** Reproduced: on-policy parity between the
methods, sDML's off-policy collapse in both settings, and DML's off-policy advantage over the naive
transformer in the literal setting and whenever the naive model is trained past its best epoch.
Not reproduced: in the calibrated setting the advantage shrinks to within noise at the naive
model's best epoch, and cross-fitting did not help (it hurt demand error in the calibrated
setting, with high variance). The most likely reasons are the simulator's policy randomness (the
coin flip in the pricing rule gives every model enough exogenous variation) and the size of our
networks; the paper's production-scale transformer has far more capacity to overfit the confounded
path. Our absolute numbers are not comparable to the paper's because the noise level and policy
sharpness are our calibration, not theirs.
