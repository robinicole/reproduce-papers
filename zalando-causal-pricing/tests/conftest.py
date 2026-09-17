"""Shared fixtures: a small confounded pricing panel with a known effect, for both demand heads."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("models", "data", "experiments")]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # tests run on CPU

from schema import Panel, make_windows, week_feats  # noqa: E402

C, H, N, T = 10, 3, 400, 40


def toy(head, seed=0):
    """Season drives both demand and discount (discount is high when the season is low), so a
    naive learner is confounded. add: q = base + eff*d; mult: q = base*(1-d)^eff with eff < 0."""
    rng = np.random.default_rng(seed)
    season = np.sin(2 * np.pi * np.arange(T) / 20)[None, :]
    base = rng.uniform(20, 60, (N, 1)) * (1 + 0.5 * season)
    d = np.clip(0.3 - 0.25 * season + rng.normal(0, 0.05, (N, T)), 0, 0.5)
    if head == "add":
        eff = rng.uniform(20, 60, N)
        q = base + eff[:, None] * d + rng.normal(0, 2, (N, T))
    else:
        eff = -rng.uniform(0.5, 2.5, N)
        q = base * (1 - d) ** eff[:, None] + rng.normal(0, 2, (N, T))
    P = Panel(y=np.clip(q, 0, None).astype(np.float32), treatment=d.astype(np.float32),
              futr=np.broadcast_to(week_feats(np.arange(T), 20), (N, T, 3)))
    Wtr, Wt = make_windows(P, range(C - 1, T - H, 2), C, H), make_windows(P, [T - H - 1], C, H)
    return P, Wtr, Wt, eff[Wt.item]


@pytest.fixture(scope="session")
def toy_add():
    return toy("add")


@pytest.fixture(scope="session")
def toy_mult():
    return toy("mult")
