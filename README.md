# reproduce-papers

Independent reproductions of published forecasting papers, one folder each, every folder
self-contained with its own README, data preparation, experiments and tests.

| Folder | Papers | What it reproduces |
|---|---|---|
| [`zalando-causal-pricing/`](zalando-causal-pricing/) | Schultz et al., *Causal Forecasting for Pricing* ([arXiv:2312.15282](https://arxiv.org/abs/2312.15282)); Kunz et al., *Deep Learning based Forecasting: a case study from the online fashion industry* ([arXiv:2305.14406](https://arxiv.org/abs/2305.14406)) | The DML Forecaster and the monotonic-demand transformer on the papers' synthetic pricing simulator and on M5, with one comparison table and a strengths-and-weaknesses assessment. |

Papers, data and results are fetched or generated locally and are not committed; each folder's
README says how. `CLAUDE.md` holds the conventions for agents working in this repo.
