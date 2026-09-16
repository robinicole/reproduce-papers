# Expected result pattern

Reference numbers from the build of 2026-09-15 (4 periods × 3 seeds on synthetic data, 4 origins
on M5; MAE unless stated). Compare `results/RESULTS.md` against the checks, not the digits: about one
standard deviation of movement between runs is normal (std is in the per-setting tables).

## Synthetic, calibrated setting (sharpness 10, noise 0.1)

| model | off-policy | on-policy | effect MAE |
|---|---|---|---|
| dml (48 ep) | 17.3 ± 1.5 | 16.8 ± 1.9 | 21.6 |
| tf (24 ep) | 16.8 ± 1.8 | 16.6 ± 2.1 | 20.2 |
| tf (48 ep) | 18.2 ± 3.1 | 17.8 ± 3.0 | 19.5 |
| sdml (48 ep) | 19.4 ± 1.6 | 16.5 ± 1.2 | 44.5 |
| dml-nocf (48 ep) | 21.3 ± 4.7 | 21.6 ± 5.4 | 21.8 |
| mdl (24 / 48 ep) | 30.2 / 31.9 | 29.1 / 31.6 | 18.0 / 14.6 |
| mdl-anchored (24 / 48 ep) | 17.7 ± 3.0 / 19.8 ± 4.5 | 18.1 / 20.7 | 17.8 / 17.5 |

Checks:
- `dml`, `tf (24 ep)` and `mdl-anchored` off-policy within one std of each other. Failure usually
  means the nuisance epochs were cut (attenuated effect) or jobs ran concurrently on the GPU.
- `sdml` on-policy ≈ `dml` on-policy, but `sdml` effect MAE more than double `dml`'s and its
  off-policy error at least 2 MAE above its on-policy error. This is the paper's ablation and the
  most robust check in the study; if it fails the effect stage is broken.
- `tf (48 ep)` off-policy worse than `tf (24 ep)`.
- `mdl` (unanchored) off- and on-policy far above the rest with a huge std (≈13): the stock
  covariate drifts out of the training range and the absolute-level model extrapolates. Its effect
  MAE is nevertheless the best or near-best. `mdl-anchored` closes the gap to within one std of
  `dml`, with about twice `dml`'s std, and is worse at 48 epochs than at 24.

## Synthetic, literal setting (sharpness 1, no noise)

| model | off-policy | on-policy | effect MAE |
|---|---|---|---|
| dml (48 ep) | 8.0 ± 1.8 | 7.5 ± 1.8 | 13.8 |
| tf (24 ep) | 8.9 ± 1.9 | 8.2 ± 1.8 | 14.8 |
| tf (48 ep) | 10.5 ± 2.0 | 9.7 ± 2.1 | 13.5 |
| sdml (48 ep) | 13.2 ± 2.0 | 8.3 ± 1.6 | 47.9 |
| dml-nocf (48 ep) | 8.5 ± 1.3 | 8.2 ± 1.2 | 8.8 |
| mdl (24 / 48 ep) | 11.3 / 14.3 | 9.9 / 13.4 | 20.0 / 18.4 |
| mdl-anchored (24 / 48 ep) | 8.4 / 8.2 | 7.4 / 7.4 | 14.6 / 13.7 |

Checks:
- `dml` and `mdl-anchored` have the lowest off-policy errors, below `tf` at both epoch counts (the paper's headline direction).
- `sdml` off-policy ≥ 1.5 × its on-policy error.
- Errors are roughly half the calibrated setting's (this setting has no extra noise).

## M5 (weekly, 4-week horizon)

| model | MAE all | MAE price-change windows | elasticity |
|---|---|---|---|
| dml / dml-nocf / sdml / tf | 3.62 | 6.80 to 6.87 | −0.04 to −0.11 |
| mdl | 3.73 | 6.62 | −0.68 |
| mdl-anchored | 3.76 | 6.51 | −0.58 |
| last4 (naive) | 3.84 | 7.62 | – |

Checks:
- All trained models beat `last4` on both slices.
- The four paper-1 variants tie on `MAE all` to two decimals; the two paper-2 variants are the
  best on the price-change slice by a few percent and report a clearly negative elasticity, while the DML
  variants' elasticities sit near zero (prices on M5 rarely move, so the residual discount is
  tiny). The price-change slice has about 130 to 140 windows per origin; if it is empty, the
  discount definition in `data/m5_data.py` (expanding-max base price) was changed.
