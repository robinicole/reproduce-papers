"""Runs each module's built-in self-check (its __main__ block) on CPU. About two minutes in total.

synthetic: simulator produces 4467 valid articles.
dml:       DML recovers the discount effect on a confounded toy problem (effect MAE < half the mean effect).
mdl:       monotone in discount; effect attenuated under confounding but not degenerate.
lgbm:      fits on-policy; effect attenuated under confounding but not degenerate.
twfe:      PPML recovers a known elasticity from a confounded panel where pooled OLS is biased.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore")


@pytest.mark.parametrize("script", ["data/synthetic.py", "models/dml.py", "models/mdl.py", "models/lgbm.py", "models/twfe.py"])
def test_selfcheck(script):
    r = subprocess.run([sys.executable, str(ROOT / script)], env=ENV, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
