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
| `models/schema.py` | The data contract: `Panel` (Nixtla's variable taxonomy), `make_windows`, and the `Windows` batch container. |
| `models/heads.py` | The two demand heads every model shares (additive for the linear simulator, constant-elasticity for real data), with the bounds that keep them finite, plus the finite-difference effect for S-learners. |
| `models/dml.py` | Paper 1. `DML` runs the procedure once over pluggable learners: item-parity folds, cross-fitted residuals, one effect model, the cross-fit/own-fold ensemble. Ablations are arguments (`cross_fit=False`, `treatment_model=False`). Holds the paper's torch learners and its naive `TFForecaster` baseline. |
| `models/mdl.py` | Paper 2. Encoder/decoder transformer; future discount bypasses the network into a piecewise-linear monotonic demand layer; Taylor-exponential loss on log demand; optional level anchoring. |
| `models/lgbm.py` | LightGBM learners on the same windows: an S-learner, and DML nuisance/effect learners (direct or autoregressive). Every fit early-stops and logs its tree count. |
| `models/twfe.py` | Two-way fixed-effects Poisson elasticity, paper 1's econometric baseline. |
| `models/registry.py` | Every model by name: constructor, display name, whether it trains by epochs. The one place to add a model. |
| `models/nn.py`, `models/metrics.py` | The transformer block with its training loop; MAE, MSE and the price-weighted demand error. |
| `data/synthetic.py` | Paper 1's Appendix E simulator, with knobs for policy sharpness and noise (see reproduction notes). |
| `data/m5_data.py` | M5 daily sales to weekly demand; discount = 1 − price / expanding-max price per series. Raw CSVs go in `data/m5/`. |
| `experiments/grid.py` | The loop both runners share: cells × evaluate → one CSV with provenance columns (setting, budgets, git hash); finished cells are skipped, so every run resumes. |
| `experiments/run_synthetic.py`, `experiments/run_m5.py` | The two protocols, as a panel, a cell list and an evaluate function each. |
| `experiments/pipeline.py` | The whole study as a list of cells; runs whatever `results/` is missing. `scripts/run_all.sh` calls it. |
| `experiments/report.py` | `results/*.csv` → `results/RESULTS.md` and the headline table below, labelled from columns. |
| `experiments/elasticity_m5.py` | The M5 elasticity reference and each forecaster's estimate against it. |
| `tests/` | `pytest`: every registered model on a confounded toy problem, the schema contract, reproducibility. About three minutes on CPU. |
| `papers/fetch.sh` | Fetches and converts both papers. |

## Quick start

```bash
pip install -r requirements.txt
pytest                                                                        # ~3 minutes on CPU
papers/fetch.sh                                                               # papers as PDF + markdown
python experiments/run_synthetic.py --setting calibrated --seeds 1 --periods 1 --epochs 4 --models dml,tf   # ~1 minute
scripts/run_all.sh                                                            # the whole study, resumable
```

Adding a model is one entry in `models/registry.py`; adding a dataset is a `Panel` plus a cell
list and an evaluate function (see the two runners); adding a budget is a line in
`experiments/pipeline.py`. Results carry their provenance as columns, so the report never
decodes a filename and a rerun only computes cells that are missing.

M5 data: the Kaggle competition download requires accepting the competition rules; the raw CSVs
are also published as Kaggle datasets (this repo used `aryayadav0513/m5-forecasting-accuracy`).
Put `calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv` in `data/m5/`; `data/m5_data.py` builds the weekly cache.

## Using your own data

Both papers' models read a shared `Panel`, which uses the variable taxonomy Nixtla uses, so a new
dataset is a schema declaration rather than a code change:

| Type | Meaning | Where it is read |
|---|---|---|
| target | the series being forecast | context, transformed by `y_past_transform` |
| treatment | the variable you intervene on | context, and the horizon as the intervention |
| historical exogenous | known only up to the forecast origin | context only |
| future exogenous | known through the horizon | context and horizon |
| static categorical / numeric | constant per series | every window (3-D numerics are read at the origin) |

A Nixtla-style long frame (`unique_id`, `ds`, `y`) goes straight in:

```python
from schema import Panel, make_windows

panel = Panel.from_long(
    df, id_col="unique_id", time_col="ds", target_col="y",
    treatment_col="discount",                 # the thing you price
    hist_exog=["stock", "web_traffic"],       # past only
    futr_exog=["snap", "is_holiday", "week_sin", "week_cos"],   # known ahead
    static_cat=["store", "category"], static_num=["base_price"],
)
W = make_windows(panel, origins=range(26, 250, 3), C=26, H=4)
```

Nothing else is wired to the column list: embedding sizes come from `panel.n_cat`, and the network
input widths come from the window shapes. Missing series-timestamp rows are filled and marked
invalid, so a ragged panel is safe. Two knobs carry the assumptions that would otherwise be silent:
`y_past_transform` (defaults to `log1p`, right for counts, wrong for a target that goes negative)
and `drop_inactive` (drops windows whose context target sums to zero, right for sales, wrong for a
zero-mean series). `tests/test_schema.py` pins this contract, including that historical exogenous
never reach the horizon block.

What the two shipped datasets declare: the simulator passes stock as historical exogenous and a
30-week calendar as future exogenous; M5 passes a 52-week calendar as future exogenous and uses
availability as the validity mask. M5's SNAP and event flags are genuinely known ahead and would be
added to `futr_exog` with no other change.

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
  `mdl` are additionally reported at 24 epochs. That 24-epoch row is **not** selected on a held-out
  period, so treat the equal-budget (48 epoch) rows as the fair comparison and the 24-epoch rows as
  evidence about sensitivity to training length.
- **Baselines not reproduced:** SARIMAX and the two-way fixed-effects elasticity model.
- **Budgets and convergence.** Neural models on M5 are reported at 4 and 12 epochs, because the
  4-epoch runs turned out to be undertrained on the price-change slice. LightGBM uses early stopping
  against a tree cap, and the cap must be checked rather than assumed: on synthetic data every fit
  stops between 200 and 700 trees, but on M5's 1.7M windows an initial 2000-tree cap was still
  binding (fits ran to 1995-2000), and raising it to 12000 let them stop at 3400 to 9000. Every fit
  logs its `best_iteration_`; a row whose trees sit at the cap is a truncated model, not a result.
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
table below is generated by `experiments/report.py` (every trained variant, no selection by test
error; synthetic numbers are means over four
training periods and three seeds; M5 numbers are means over four forecast origins).

<!-- RESULTS -->
|                                                                                                 |   calibrated: off-policy MAE |   calibrated: on-policy MAE |   calibrated: effect MAE |   literal: off-policy MAE |   literal: on-policy MAE |   literal: effect MAE |   multi: off-policy MAE |   multi: on-policy MAE |   multi: effect MAE |   M5: MAE all |   M5: MAE price change |   M5: elasticity |
|:------------------------------------------------------------------------------------------------|-----------------------------:|----------------------------:|-------------------------:|--------------------------:|-------------------------:|----------------------:|------------------------:|-----------------------:|--------------------:|--------------:|-----------------------:|-----------------:|
| dml (48 ep) — DML Forecaster (paper 1)                                                          |                        17.30 |                       16.80 |                    21.60 |                      8.00 |                     7.50 |                 13.80 |                   81.60 |                  42.50 |                0.80 |        nan    |                 nan    |           nan    |
| dml (48 ep) (discount only) — DML Forecaster (paper 1)                                          |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                   62.10 |                  43.50 |                0.60 |        nan    |                 nan    |           nan    |
| dml (12 ep) — DML Forecaster (paper 1)                                                          |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.59 |                   6.74 |            -0.10 |
| dml (4 ep) — DML Forecaster (paper 1)                                                           |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.62 |                   6.84 |            -0.15 |
| dml-lgbm — DML layout, direct multi-horizon LightGBM nuisances                                  |                        19.10 |                       18.10 |                    18.20 |                      7.00 |                     6.50 |                  9.80 |                   75.60 |                  38.10 |                0.40 |          3.62 |                   7.13 |            -0.42 |
| dml-lgbm (discount only) — DML layout, direct multi-horizon LightGBM nuisances                  |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                   52.10 |                  44.00 |                0.40 |        nan    |                 nan    |           nan    |
| dml-lgbm-ar — DML layout, autoregressive LightGBM nuisances                                     |                        23.80 |                       19.80 |                    32.90 |                     14.30 |                     9.60 |                 37.50 |                   97.00 |                  36.90 |                0.60 |          3.66 |                   6.69 |            -0.23 |
| dml-lgbm-ar (discount only) — DML layout, autoregressive LightGBM nuisances                     |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                   67.30 |                  51.60 |                0.50 |        nan    |                 nan    |           nan    |
| dml-nocf (48 ep) — DML, no cross-fitting                                                        |                        21.30 |                       21.60 |                    21.80 |                      8.50 |                     8.20 |                  8.80 |                   95.50 |                  45.30 |                1.10 |        nan    |                 nan    |           nan    |
| dml-nocf (48 ep) (discount only) — DML, no cross-fitting                                        |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                   71.30 |                  44.50 |                0.80 |        nan    |                 nan    |           nan    |
| dml-nocf (12 ep) — DML, no cross-fitting                                                        |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.60 |                   6.71 |            -0.09 |
| dml-nocf (4 ep) — DML, no cross-fitting                                                         |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.61 |                   6.80 |            -0.11 |
| last4 — naive: mean of last 4 weeks                                                             |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.84 |                   7.62 |           nan    |
| lgbm — direct multi-horizon LightGBM (S-learner, paper 2 baseline)                              |                        24.50 |                       19.50 |                    39.10 |                      8.10 |                     6.20 |                 18.60 |                   96.60 |                  31.50 |                1.30 |          3.61 |                   5.10 |            -2.93 |
| lgbm (discount only) — direct multi-horizon LightGBM (S-learner, paper 2 baseline)              |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                   82.20 |                  37.60 |                1.20 |        nan    |                 nan    |           nan    |
| mdl (24 ep) — Monotonic-demand transformer (paper 2)                                            |                        30.20 |                       29.10 |                    18.00 |                     11.30 |                     9.90 |                 20.00 |                  nan    |                 nan    |              nan    |        nan    |                 nan    |           nan    |
| mdl (48 ep) — Monotonic-demand transformer (paper 2)                                            |                        31.90 |                       31.60 |                    14.60 |                     14.30 |                    13.40 |                 18.40 |                   59.50 |                  29.50 |                0.30 |        nan    |                 nan    |           nan    |
| mdl (48 ep) (discount only) — Monotonic-demand transformer (paper 2)                            |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                   45.80 |                  36.10 |                0.30 |        nan    |                 nan    |           nan    |
| mdl (12 ep) — Monotonic-demand transformer (paper 2)                                            |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.88 |                   6.47 |            -0.37 |
| mdl (4 ep) — Monotonic-demand transformer (paper 2)                                             |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.75 |                   6.56 |            -0.69 |
| mdl-anchored (24 ep) — paper 2 model anchored to recent demand level (ablation)                 |                        17.70 |                       18.10 |                    17.80 |                      8.40 |                     7.40 |                 14.60 |                  nan    |                 nan    |              nan    |        nan    |                 nan    |           nan    |
| mdl-anchored (48 ep) — paper 2 model anchored to recent demand level (ablation)                 |                        19.80 |                       20.70 |                    17.50 |                      8.20 |                     7.40 |                 13.70 |                   59.60 |                  27.90 |                0.40 |        nan    |                 nan    |           nan    |
| mdl-anchored (48 ep) (discount only) — paper 2 model anchored to recent demand level (ablation) |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                   44.90 |                  35.80 |                0.40 |        nan    |                 nan    |           nan    |
| mdl-anchored (12 ep) — paper 2 model anchored to recent demand level (ablation)                 |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.84 |                   6.13 |            -0.77 |
| mdl-anchored (4 ep) — paper 2 model anchored to recent demand level (ablation)                  |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.76 |                   6.55 |            -0.80 |
| sdml (48 ep) — sDML (no treatment model)                                                        |                        19.40 |                       16.50 |                    44.50 |                     13.20 |                     8.30 |                 47.90 |                   85.20 |                  44.50 |                0.70 |        nan    |                 nan    |           nan    |
| sdml (48 ep) (discount only) — sDML (no treatment model)                                        |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  101.70 |                  46.60 |                1.60 |        nan    |                 nan    |           nan    |
| sdml (12 ep) — sDML (no treatment model)                                                        |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.60 |                   6.76 |            -0.00 |
| sdml (4 ep) — sDML (no treatment model)                                                         |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.62 |                   6.87 |            -0.03 |
| tf (24 ep) — TF, linear head S-learner (paper 1 ablation)                                       |                        16.80 |                       16.60 |                    20.20 |                      8.90 |                     8.20 |                 14.80 |                  nan    |                 nan    |              nan    |        nan    |                 nan    |           nan    |
| tf (48 ep) — TF, linear head S-learner (paper 1 ablation)                                       |                        18.20 |                       17.80 |                    19.50 |                     10.50 |                     9.70 |                 13.50 |                   76.10 |                  30.80 |                1.00 |        nan    |                 nan    |           nan    |
| tf (48 ep) (discount only) — TF, linear head S-learner (paper 1 ablation)                       |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                   63.80 |                  38.60 |                0.40 |        nan    |                 nan    |           nan    |
| tf (12 ep) — TF, linear head S-learner (paper 1 ablation)                                       |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.60 |                   6.68 |            -0.23 |
| tf (4 ep) — TF, linear head S-learner (paper 1 ablation)                                        |                       nan    |                      nan    |                   nan    |                    nan    |                   nan    |                nan    |                  nan    |                 nan    |              nan    |          3.61 |                   6.75 |            -0.09 |
<!-- /RESULTS -->

How to read it, benchmark by benchmark:

- **Synthetic, calibrated** (strong confounding, paper-level noise). The DML Forecaster, the
  anchored paper-2 model and the naive TF are within one standard deviation of each other on- and
  off-policy; DML has the lowest variance across periods and seeds. At equal budget (48 epochs) DML
  is ahead of TF (17.3 vs 18.2 off-policy); TF only draws level when its shorter 24-epoch run is
  used, and that shorter run was not chosen on validation data. Removing the treatment model (`sdml`) reproduces the paper's
  collapse: on-policy is unchanged, off-policy error rises by a sixth and the effect error more than
  doubles. Training the naive TF twice as long costs it 1.4 MAE off-policy; the DML model does not
  have that failure mode. The unanchored paper-2 model is far off because of the stock
  extrapolation described above, yet its effect estimates are the best in the table.
- **Synthetic, literal** (the appendix rule verbatim, low noise). Here the paper's headline
  reproduces in direction: DML beats the naive TF off-policy at equal budget (8.0 vs 10.5) and also
  against TF's shorter 24-epoch run (8.9), while being on par on-policy, and sDML collapses off-policy
  (13.2 vs 8.3 on-policy). Cross-fitting did not matter for demand error but the no-cross-fitting
  variant had the best effect error, the one place where our ablation disagrees with the paper.
- **M5.** At equal training budget every transformer lands on the same overall error, 3.59 to 3.60
  MAE, and all beat the naive last-four-weeks baseline by 6%. The price-change slice separates them
  sharply, and not in the papers' favour: a plain LightGBM S-learner reaches 5.10 MAE there against
  6.13 for the best transformer, with less than half the squared error (151 against 311). It is also
  the only model reporting a retail-plausible elasticity, at -2.9 against -0.1 to -0.8 for the rest.
  The causal machinery does not pay on this data and can actively hurt: DML with LightGBM nuisances
  reaches only 7.13 on that slice, worse than the naive transformer, and its price-weighted error
  there exceeds 1.0, which is worse than forecasting nothing. M5 prices barely move, with a mean
  discount of 2.3% and only a sixth of weeks above 5%, so the treatment residual that DML
  orthogonalizes against is mostly noise, and dividing by it manufactures variance.

### Which elasticity to believe on M5

M5 has no ground-truth elasticity, so the forecasters' estimates cannot be scored. Two independent
fixed-effects estimators agree on a reference: PPML gives -0.417 and a log-log within-estimator on
positive-sales weeks gives -0.463, over 6.7M item-weeks. Category heterogeneity dwarfs that pooled
number: foods -0.581, hobbies -0.215, household -0.093.

Against that reference the forecasters split in a way that inverts the accuracy ranking:

| Model | Implied elasticity | Gap vs PPML | MAE where price moved |
|---|---|---|---|
| LightGBM S-learner | -2.93 | -2.51 | **5.10** |
| Paper-2, anchored | -0.79 | -0.37 | 6.34 |
| Paper-2 monotone head | -0.53 | -0.11 | 6.52 |
| DML with LightGBM | **-0.42** | -0.00 | 7.13 |
| Naive TF / DML / sDML | -0.16 to -0.02 | +0.25 to +0.40 | 6.71 to 6.82 |

The best forecaster on price-change windows carries an elasticity seven times the reference, and the
model that reproduces the reference to two decimals is the worst forecaster. The S-learner attributes
everything that moves with price, such as promotion timing and display, to price itself, which
predicts recurring promotions well and answers the counterfactual question badly. The DML
transformers fail the opposite way, attenuating toward zero because M5 leaves almost no exogenous
price variation to orthogonalize against. Neither failure is visible in forecast error alone.

Caveat that binds every row: M5 publishes no promotion or display flags, so all of these estimates,
the reference included, omit a variable known to move with price.

## Three treatments at once: discount, list price, stock (branch `multi-treatment`)

The papers intervene on one variable, the discount. A retailer also sets the list price and, through
replenishment, the stock, and all three respond to demand. Treating one as the treatment and the
others as covariates is not a smaller version of the same problem: the omitted ones are confounded
by the same policy, so the discount effect absorbs part of theirs.

**Design.** A `Panel` now carries a vector of treatments, each declared as `(name, kind, sign)`.
The demand heads are partially linear in the treatments' effect-space coordinates: the discount
enters as `log(1-d)` under the multiplicative head exactly as before, log price and log stock as
they are, each with a sign-constrained activation. DML residualizes all three jointly and learns
one effect vector per window, which is the textbook multi-treatment partially linear model. Trees
do the effect stage by backfitting the weighted-regression trick over treatments. The S-learners
and paper 2's model take the extra treatments as inputs and report finite-difference effects per
treatment; paper 2's monotone head stays discount-only, since that is what the paper specifies.

**Simulator.** `synthetic.generate_multi` keeps Appendix E's base demand and makes it log-linear
in three treatments: a list price with elasticity ε, a discount with elasticity ε·(1+λ) where λ is
a *promo premium* beyond the pure price effect, and an availability multiplier that falls once
stock drops below a threshold. The discount follows the coverage rule, the list price moves with
sell-through against plan, and stock jumps on replenishment when sales run ahead, so all three are
confounded with demand and with each other (within-item correlations near ±0.5).

**Hypothesis and protocol.** Two questions, each with ground truth: does the three-treatment model
recover the promo premium (a discount coefficient larger in magnitude than the price coefficient)
and the stock effect, each separately, while all three move? And how biased is a discount-only
model on the same data, with the list price as a future exogenous and stock as a historical one?
`run_synthetic.py --setting multi --treatments all|discount` reports on-policy error, off-policy
error on a joint grid and one treatment at a time, and per-treatment effect error.

<!-- MULTI-RESULTS -->
**Results** (12 runs per cell: four periods, three seeds, 48 epochs; full tables in `results/RESULTS.md`).

All three treatments declared:

| Model | On-policy MAE | Joint off-policy MAE | Discount effect error | Price effect error | Stock effect error |
|---|---|---|---|---|---|
| Paper-2 monotone head | 29.5 | **59.5 ± 2.8** | **0.33** | 1.21 | **0.13** |
| Paper-2, anchored | **27.9** | 59.6 | 0.41 | 1.19 | 0.13 |
| DML with LightGBM | 38.1 | 75.7 | 0.38 | 1.94 | 0.15 |
| Naive TF | 30.9 | 76.1 | 0.99 | 0.71 | 0.84 |
| DML Forecaster | 42.5 | 81.6 ± 5.3 | 0.76 | 1.20 | 0.23 |
| sDML | 44.5 | 85.2 | 0.74 | **0.56** | 0.18 |
| DML, no cross-fitting | 45.3 | 95.5 | 1.11 | 1.11 | 0.15 |
| LightGBM S-learner | 31.5 | 96.6 | 1.31 | 1.29 | 0.23 |
| DML, autoregressive LightGBM | 36.9 | 97.0 | 0.58 | 2.30 | 0.16 |

True magnitudes: discount coefficient about 2.1, price coefficient about 1.4, stock slope 0.5 below
the availability threshold and 0 above it.

The same models with the discount as the only treatment, the list price as a future exogenous and
stock as a historical exogenous:

| Model | Discount effect error, all three | Discount effect error, discount only | Discount off-policy MAE, all three | discount only |
|---|---|---|---|---|
| Paper-2 monotone head | 0.33 | 0.34 | 40.7 | 45.8 |
| DML with LightGBM | 0.38 | 0.38 | 48.3 | 52.1 |
| Naive TF | 0.99 | **0.39** | 67.3 | 63.8 |
| DML Forecaster | 0.76 | **0.58** | 72.4 | 62.1 |
| sDML | 0.74 | 1.62 | 72.0 | 101.7 |

**What it shows.**

1. *With three confounded treatments, paper 2's model beats the DML Forecaster on every axis*, the
   reverse of the single-treatment study, and the DML transformer trails even the naive one on the
   joint grid. The spread across runs (2.8 against a 22-point gap) rules out noise. Two mechanisms
   are visible in the numbers: an outcome model that never sees horizon treatments loses about 13
   points of on-policy accuracy once stock jumps and the list price steps inside the horizon (sDML
   pays the same, so it is not the treatment model), and three treatments that co-move at
   correlations near 0.5 leave little independent variation after residualization to identify
   three coefficients per window from five noisy steps.

2. *Declaring more treatments costs DML more than it buys.* The transformer DML's discount effect
   is better when price and stock are covariates (0.58) than when they are residualized as
   treatments (0.76), and the naive transformer's is two and a half times better (0.39 against
   0.99). This is the variance side of the bias-variance trade: jointly residualizing three
   collinear treatments inflates the effect estimates more than the omitted-treatment bias it
   removes. The models with a structural discount head are indifferent to the declaration (0.33
   and 0.34; 0.38 and 0.38), which is the behaviour one would want.

3. *The omitted-treatment bias is real but needs the treatment model to be missing.* sDML, which
   has no treatment model, goes from 0.74 to 1.62 when stock and price stop being treatments: the
   horizon stock is then unobserved and its effect loads onto the discount. Every model with a
   treatment model, or a structural head, is protected.

4. *The list-price effect is unidentified on this data by every model.* The best error is 0.56
   against a true magnitude of 1.4. Within-item log-price variation has a standard deviation of
   0.09 and moves with the discount, so the promo premium the simulator was built to expose is not
   separable here. That is a statement about the simulator's list-price policy, which is too slow
   and too coupled to the discount to give the price effect its own variation, not about the
   estimators; a study of the premium needs a price policy with an independent component.

5. *The stock effect is recoverable but rarely active.* Its truth is near zero in early periods and
   grows late in the season; the monotone head and the DML variants track it (0.13 to 0.23), the
   naive transformer does not (0.84).

Taken together: for a multi-treatment pricing problem on this evidence, use paper 2's monotone
head, declare the extra treatments as inputs rather than residualized treatments, and expect the
list-price elasticity to need an identification strategy of its own.
<!-- /MULTI-RESULTS -->

## Data leakage audit

Checked deliberately, since every headline here is an out-of-sample claim.

Clean:
- **Temporal separation.** Synthetic training origins end at `b-H`, so the last training target week
  is `b` and the first test target week is `b+1`. On M5 the last training target week is 257 and the
  first test target week is 261, a four-week gap. Test contexts reuse weeks that were training
  targets, which is legitimate: they are observed history at the forecast origin.
- **Context-only conditioning.** The per-window `scale`, every lag feature and the stock covariate
  are read from the context slice `t-C+1 : t+1`. The horizon slice supplies only the target and the
  discount, which is the treatment.
- **No global statistics.** Nothing is standardized with dataset-wide means or variances; all
  transforms are deterministic (`log1p`, calendar sine and cosine).
- **Fold hygiene.** DML cross-fitting splits on item parity; the LightGBM early-stopping hold-out
  splits on `(item // 2) % 10`, deliberately independent of that parity so no nuisance model is
  early-stopped on its own cross-fitting partner.

Found and fixed:
- **Look-ahead base price on M5.** The static price feature was the expanding-max base price at the
  *end of the series*, so it encoded price highs set after the forecast origin for 0.4% to 5.7% of
  series, depending on origin, with a mean 5.5% gap where it differed. It is now the expanding max
  at each window's own origin (`make_windows` accepts per-week static numerics). The M5 results were
  regenerated after this fix.
- **Epoch count chosen on test error.** The headline table used to report each model at whichever
  epoch count minimized test off-policy error, which is selection on the evaluation set. It now
  reports every variant, and the prose compares at equal budget.

Known and accepted:
- **The window filter reads horizon availability** (`valid[:, t-C+1 : t+1+H].all(1)`), which is
  selection on the future. Empirically it drops nothing at the M5 test origins (30370 and 30450
  windows kept either way), so it does not affect any reported number; it is kept because training
  targets must exist.
- **Calibration knobs** (policy sharpness, noise level) were tuned to match the paper's reported
  error magnitude, using the simulator rather than the test split, but they are a researcher choice
  and the `literal` setting is reported alongside as the unturned baseline.

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

**LightGBM, as the tree baseline paper 2 benchmarks against**

- Strengths: on M5 it is the best model on exactly the windows the whole exercise cares about, the
  ones where price moves, by 17% on error and more than half on squared error, and it is the only
  model whose implied elasticity is plausible for retail. It trains on CPU in minutes, and its
  convergence is observable rather than a matter of faith.
- Weaknesses: it is an S-learner with no structural protection, and on the confounded simulator that
  shows, with the worst effect estimate of any non-degenerate model at 39.1. It extrapolates badly
  to discount levels it has not seen, which is precisely the off-policy question. Its strong M5
  showing and its weak synthetic showing are consistent: real M5 price variation is small but
  genuine, while the simulator's is large and deliberately confounded.

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

**The structure helps or hurts depending on how much exogenous price variation exists.** Putting
LightGBM inside the DML layout on synthetic data cuts its off-policy error from 24.5 to 19.1 and its
effect error from 39.1 to 18.2, the clearest evidence in this study that orthogonalization, not
architecture, is what buys off-policy robustness. The same swap on M5 makes things worse, 5.10 to
7.13 on the price-change slice. The difference is the denominator: DML divides by a treatment
residual, which is informative when prices move and is noise when they do not.

**What the comparison did and did not confirm.** Reproduced: on-policy parity between the
methods, sDML's off-policy collapse in both settings, and DML's off-policy advantage over the naive
transformer in the literal setting and at equal training budget in both settings. Not reproduced:
in the calibrated setting the advantage shrinks to within noise once the naive model is given a
shorter 24-epoch budget, and cross-fitting did not help (it hurt demand error in the calibrated
setting, with high variance). The most likely reasons are the simulator's policy randomness (the
coin flip in the pricing rule gives every model enough exogenous variation) and the size of our
networks; the paper's production-scale transformer has far more capacity to overfit the confounded
path. Our absolute numbers are not comparable to the paper's because the noise level and policy
sharpness are our calibration, not theirs.
