---
name: reproduce-zalando-causal-pricing
description: Reproduce the two-paper Zalando demand forecasting benchmark (DML Forecaster vs monotonic-demand transformer) on synthetic and M5 data, end to end, and verify the result pattern.
disable-model-invocation: true
---

# Reproduce the Zalando demand forecasting benchmark

Reproduces `zalando-causal-pricing/` in this repo: paper 1 (arXiv 2312.15282v2, DML Forecaster)
and paper 2 (arXiv 2305.14406, monotonic-demand transformer), evaluated on paper 1's synthetic
simulator and on M5, with one comparison table. The repo's `README.md` is the source of truth for
what each file does; this skill is the order of operations, the gotchas the code does not confess,
and the numbers that tell you the run went right. Expected numbers live in
[`expected-results.md`](expected-results.md); read it at step 6, not before.

Set `ROOT=<repo>/zalando-causal-pricing` and run everything from there. Only the GPU note below
assumes this machine.

## 1. Environment

Needs Python 3.11, `numpy`, `pandas`, `torch` (CUDA), `tabulate`, `pandoc`, `curl`, and the
`kaggle` CLI with `~/.kaggle/kaggle.json`. Verify with:

```bash
cd $ROOT && python3 -c "import torch, pandas, numpy; print(torch.cuda.is_available())" && pandoc --version | head -1 && kaggle --version
```

Done when: prints `True`, a pandoc version, and a kaggle version. Shell output on this machine
carries harmless `FakeAssocArray ... command not found` lines from the zsh profile; ignore them.

**GPU is shared.** An unrelated Ollama `llama-server` (started weeks ago, ~20 GB) may occupy most
of the 24 GB card. Leave it alone. Run one training job at a time; three concurrent jobs made
every fit ten times slower and caused out-of-memory errors. The self-checks run on CPU.

## 2. Papers (only if `$ROOT/papers/*.md` is missing)

```bash
cd $ROOT && papers/fetch.sh
```

It downloads both PDFs and LaTeX sources and converts the LaTeX with pandoc (the HTML route drops
equations into raw spans). Gotcha it already handles: pandoc 2.9 **hangs and eats memory without
limit** on the custom macros in 2312.15282v2's preamble, so the script pre-expands them with sed
and passes `-f latex-latex_macros`. Done when: it prints about 30 headings for paper 1 and 28 for
paper 2. The outputs are git-ignored on purpose.

## 3. M5 data (only if `$ROOT/data/m5/weekly.npz` or the three CSVs are missing)

The competition download (`kaggle competitions download -c m5-forecasting-accuracy`) fails with
`403 ... accept this competition's rules` unless the rules were accepted in the browser. Use a
dataset mirror instead; two carried the raw CSVs when this was built:

```bash
mkdir -p $ROOT/data/m5 && cd $ROOT/data/m5
kaggle datasets download -d aryayadav0513/m5-forecasting-accuracy -p . && unzip -o -q *.zip && mv -f m5-forecasting-accuracy/*.csv . 2>/dev/null; rm -rf m5-forecasting-accuracy *.zip
```

(Fallback mirror: `luisfelipevendramim/m5-forecasting-dataset`. `marcogorelli/m5-forecasting-data`
is a parquet of sales only, without prices; it is useless here.) Then build the weekly cache:

```bash
cd $ROOT && python3 -W ignore data/m5_data.py
```

Done when: it prints `q shape: (30490, 277)`, share of cells available ≈ 0.79, mean discount ≈
0.023, and both asserts pass silently.

## 4. Self-checks

```bash
cd $ROOT && python3 -m pytest -q
```

Done when: all tests pass (about three minutes on CPU). They fit every registered model on a
confounded toy problem and check what each design implies: DML recovers the effect, sDML loses it,
the monotone head is monotone, the DML layout beats the same trees used naively, PPML recovers a
known elasticity, and a run with the same seed reproduces exactly.
Gotcha: the checks train on ~5600 windows; with the default batch of 1024 they need the epoch
counts written in the files (20 for DML), fewer looks degenerate (effect MAE ≈ mean effect).

## 5. Full pipeline

```bash
cd $ROOT && mkdir -p results && nohup scripts/run_all.sh > results/logs_pipeline.txt 2>&1 &
```

`scripts/run_all.sh` runs `experiments/pipeline.py`: both synthetic settings with every model at 48
epochs and the transformers also at 24, then M5 with the transformers at 4 and 12 epochs and the
tree models at a 12000-tree cap, then the report. A few hours on a free GPU. Every finished cell is
a row in `results/synthetic.csv` or `results/m5.csv`, and the pipeline skips cells already there,
so a crash (a `Traceback` in `results/logs_pipeline.txt`) is fixed and relaunched without losing
work. LightGBM tree counts are logged per fit; a count at the cap means a truncated model.

Why these settings (the code does not say): the appendix pricing rule alone yields almost no
contemporaneous confounding and on-policy errors half the paper's, so every model identifies the
effect and nothing separates them; sharpness 10 and noise 0.1 bring the discount/season
correlation and the error magnitude to the paper's. Nuisance models need the 48 epochs: with 8,
the residual-on-residual slope is attenuated by a fifth and the effect model drifts further from
truth the longer it trains.

Done when: `pipeline.py` exits cleanly and `results/RESULTS.md` has a headline table with a row for
every model in `models/registry.py` plus `last4`, and no `nan` in the synthetic columns.

## 6. Verify the result pattern

Open [`expected-results.md`](expected-results.md) and compare against `results/RESULTS.md`. The pattern,
not the digits, is the criterion; seeds and GPU nondeterminism move numbers by about one standard
deviation. Done when every check in that file holds. If one fails, the file names the usual cause.

## 7. Hand over

`README.md` embeds the headline table between `<!-- RESULTS -->` markers; `experiments/report.py`
rewrites it, so the prose after the table is what to re-read: every number quoted there must match
the fresh table. Done when the README prose and `results/RESULTS.md` agree and `pytest` passes.
