#!/bin/bash
# Follow-up: the anchored paper-2 ablation, appended to the same result files, then the report.
cd "$(dirname "$0")/.."  # project root
while [ ! -f results/pipeline_done.txt ]; do sleep 30; done
P="python3 -W ignore"
for cfg in "calibrated 10 0.1" "literal 1 0"; do
  set -- $cfg
  $P experiments/run_synthetic.py --seeds 3 --periods 4 --epochs 8 --nuisance_epochs 48 --models mdl-anchored --sharpness $2 --noise_mult $3 --out results/results_synthetic_$1.csv --resume > results/logs_extra_$1.txt 2>&1
  $P experiments/run_synthetic.py --seeds 3 --periods 4 --epochs 8 --nuisance_epochs 24 --models mdl-anchored --sharpness $2 --noise_mult $3 --out results/results_synthetic_$1_tfmdl24.csv --resume > results/logs_extra_$1_24.txt 2>&1
done
$P experiments/run_m5.py --seeds 1 --epochs 4 --effect_epochs 3 --models mdl-anchored --out results/results_m5_anchored.csv > results/logs_m5_extra.txt 2>&1
$P experiments/report.py > results/logs_report.txt 2>&1
echo ALLDONE > results/final_done.txt
