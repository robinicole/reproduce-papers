#!/bin/bash
# Full pipeline: synthetic (two settings) + M5, all models, then the comparison table.
set -e
cd "$(dirname "$0")/.."  # project root
P="python3 -W ignore"
ALL=tf,dml,dml-nocf,sdml,mdl
for cfg in "calibrated 10 0.1" "literal 1 0"; do
  set -- $cfg
  $P experiments/run_synthetic.py --seeds 3 --periods 4 --epochs 8 --nuisance_epochs 48 --models $ALL --sharpness $2 --noise_mult $3 --out results/results_synthetic_$1.csv --resume > results/logs_synthetic_$1.txt 2>&1
  $P experiments/run_synthetic.py --seeds 3 --periods 4 --epochs 8 --nuisance_epochs 24 --models tf,mdl --sharpness $2 --noise_mult $3 --out results/results_synthetic_$1_tfmdl24.csv --resume > results/logs_synthetic_$1_24.txt 2>&1
done
$P experiments/run_m5.py --seeds 1 --epochs 4 --effect_epochs 3 --models $ALL --out results/results_m5.csv > results/logs_m5.txt 2>&1
$P experiments/report.py > results/logs_report.txt 2>&1
echo ALLDONE > results/pipeline_done.txt
