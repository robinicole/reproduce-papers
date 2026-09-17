"""The whole study, as the list of cells that produced RESULTS.md. Runs whatever is missing.

usage: python experiments/pipeline.py [--dry]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
from registry import MODELS  # noqa: E402

ALL = ",".join(MODELS)
NEURAL = ",".join(m for m, s in MODELS.items() if s.neural)
TREES = ",".join(m for m, s in MODELS.items() if not s.neural)

STEPS = [
    # synthetic: every model at the paper-scale budget, transformers also at the shorter budget
    *[["experiments/run_synthetic.py", "--setting", s, "--models", ALL, "--epochs", "48", "--effect_epochs", "8"]
      for s in ("calibrated", "literal")],
    *[["experiments/run_synthetic.py", "--setting", s, "--models", "tf,mdl,mdl-anchored", "--epochs", "24", "--effect_epochs", "8"]
      for s in ("calibrated", "literal")],
    # multi-treatment synthetic: every model with all three treatments, and again discount-only on the same data
    ["experiments/run_synthetic.py", "--setting", "multi", "--treatments", "all", "--models", ALL, "--epochs", "48", "--effect_epochs", "8"],
    ["experiments/run_synthetic.py", "--setting", "multi", "--treatments", "discount", "--models", ALL, "--epochs", "48", "--effect_epochs", "8"],
    # M5: transformers at two budgets; trees with a cap that is not binding on 1.7M windows
    ["experiments/run_m5.py", "--models", NEURAL, "--epochs", "4", "--effect_epochs", "3"],
    ["experiments/run_m5.py", "--models", NEURAL, "--epochs", "12", "--effect_epochs", "8"],
    ["experiments/run_m5.py", "--models", TREES, "--max_trees", "12000"],
    ["experiments/report.py"],
]

if __name__ == "__main__":
    dry = argparse.ArgumentParser().add_argument("--dry", action="store_true") and "--dry" in sys.argv
    for step in STEPS:
        print("python3", *step, flush=True)
        if not dry:
            subprocess.run([sys.executable, "-W", "ignore", *step], cwd=ROOT, check=True)
