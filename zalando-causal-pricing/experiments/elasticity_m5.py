"""What is the price elasticity on M5, and which model's estimate should you trust?

M5 has no ground-truth elasticity, so the forecasters' estimates cannot be scored directly. This
script produces two pieces of evidence:

  1. A two-way fixed-effects Poisson (PPML) reference elasticity, overall and per category. This is
     paper 1's econometric baseline and the standard identification strategy for this question.
  2. Each forecaster's implied elasticity next to its accuracy on the windows where price actually
     moved, which is the only place an elasticity can show up in a forecast.

usage: python experiments/elasticity_m5.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("models", "data", "experiments")]

import numpy as np
import pandas as pd

import m5_data
import twfe

TRAIN_END = 257  # last week used as a training target by run_m5.py


def twoway_demean(x, mask, iters=60):
    """Absorb item and week effects by alternating projections on the masked panel."""
    x = (x * mask).astype(np.float64)
    ri, ci = np.maximum(mask.sum(1), 1), np.maximum(mask.sum(0), 1)
    for _ in range(iters):
        x -= (x.sum(1) / ri)[:, None] * mask
        x -= (x.sum(0) / ci)[None, :] * mask
    return x


def ols_within(q, price, mask):
    """Log-log two-way FE OLS on weeks with positive sales. Biased by dropping zeros, so it is a
    direction check on PPML rather than a competing estimate."""
    m = mask & (q > 0)
    xd = twoway_demean(np.where(m, np.log(np.clip(price, 1e-6, None)), 0.0), m)
    yd = twoway_demean(np.where(m, np.log(np.clip(q, 1e-9, None)), 0.0), m)
    return float((xd * yd).sum() / max((xd * xd).sum(), 1e-12))


def main():
    d = m5_data.load()
    q, price, avail = d["q"].astype(np.float64), d["price"].astype(np.float64), d["avail"]
    mask = avail & np.isfinite(price) & (price > 0)
    cat = np.array([i.split("_")[0] for i in d["ids"]])

    rows = []
    for name, m in [("all weeks", mask), ("training weeks only", mask & (np.arange(q.shape[1]) <= TRAIN_END))]:
        eps, se, it = twfe.fit(q, price, m)
        rows.append({"sample": name, "n_items": int((m.any(1)).sum()), "cells": int(m.sum()),
                     "elasticity": round(eps, 3), "se": round(se, 3), "iters": it})
    for c in sorted(set(cat)):
        m = mask & (cat == c)[:, None]
        eps, se, it = twfe.fit(q, price, m)
        rows.append({"sample": f"category {c}", "n_items": int((cat == c).sum()), "cells": int(m.sum()),
                     "elasticity": round(eps, 3), "se": round(se, 3), "iters": it})
    rows.append({"sample": "all weeks (log-log OLS, positive sales only)", "n_items": int(mask.any(1).sum()),
                 "cells": int((mask & (q > 0)).sum()), "elasticity": round(ols_within(q, price, mask), 3),
                 "se": float("nan"), "iters": 0})
    ref = pd.DataFrame(rows)
    print("\n=== PPML reference elasticity (two-way fixed effects: item and week) ===")
    print(ref.to_string(index=False))

    # forecasters' implied elasticities against their accuracy where price moved
    f = ROOT / "results" / "m5.csv"
    if f.exists():
        from registry import label
        df = pd.read_csv(f)
        df["model"] = [label(m, e) for m, e in zip(df["model"], df["epochs"])]
        a = df[df["slice"] == "all"].groupby("model")["mean_psi"].mean()
        pc = df[df["slice"] == "price_change"].groupby("model")[["MAE", "demand_err"]].mean()
        out = pc.join(a.rename("elasticity")).dropna(subset=["elasticity"]).sort_values("MAE")
        ppml = ref.loc[ref["sample"] == "all weeks", "elasticity"].iloc[0]
        out["gap vs PPML"] = (out["elasticity"] - ppml).round(2)
        print(f"\n=== forecaster elasticities vs PPML reference ({ppml}) ===")
        print(out.round(3).to_string())
    return ref


if __name__ == "__main__":
    main()
