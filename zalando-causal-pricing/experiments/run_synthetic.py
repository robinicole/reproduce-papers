"""Synthetic-data experiment of arXiv:2312.15282v2, Sec. 4 'Experiments on Synthetic Data'.

Train on weeks [a, b], forecast b+1..b+5.  On-policy: realised discounts.  Off-policy: constant
discount in {0, .1, .., .5} with counterfactual ground truth from the simulator.  Effect metrics
compare psi(z) with the true dq/d(discount) = -p0 * e_i.

usage: python experiments/run_synthetic.py [--seeds 3] [--epochs 8] [--models tf,dml,dml-nocf,sdml] [--periods 4]
"""
import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("models", "data", "experiments")]

import numpy as np
import pandas as pd

import synthetic
from common import make_windows, metrics
from dml import Forecaster
from mdl import MDLForecaster

C, H = 20, 5
PERIODS = [(20, 65), (30, 75), (40, 85), (50, 94)]  # weeks are 0-indexed; last horizon = weeks 95..99
LEVELS = np.arange(0, 0.51, 0.1)


def arrays(df):
    piv = lambda c: df.pivot(index="item", columns="week", values=c).to_numpy(np.float32)
    first = df.groupby("item").first()
    return dict(q=piv("demand"), d=piv("discount"), stock=piv("stock"), qb=piv("base_demand"),
                p0=first["p0"].to_numpy(np.float32), e=first["effect"].to_numpy(np.float32),
                static_cat=first[["cat_d", "cat_k"]].to_numpy(np.int64),
                static_num=np.log(first[["p0"]].to_numpy(np.float32)))


def windows(A, origins):
    return make_windows(A["q"], A["d"], [A["stock"]], np.arange(synthetic.T), A["static_cat"], A["static_num"],
                        origins, C, H, period=30)


def build(kind, head, loss, epochs, effect_epochs, seed):
    """kind 'mdl' = arXiv:2305.14406 model; everything else = arXiv:2312.15282v2 DML Forecaster and its ablations."""
    if kind.startswith("mdl"):
        return MDLForecaster(head, epochs=epochs, seed=seed, log=lambda s: None, anchor=kind == "mdl-anchored")
    return Forecaster(kind, head, loss, epochs=epochs, effect_epochs=effect_epochs, seed=seed, log=lambda s: None)


def evaluate(model, A, Wt, origin):
    it = Wt.item
    qb = A["qb"][it, origin + 1:origin + 1 + H]
    p0, e = A["p0"][it], A["e"][it]
    true_eff = -p0 * e
    pred, psi = model.predict(Wt)
    out = {f"on_{k}": v for k, v in metrics(pred, Wt.y).items()}
    off = []
    for lvl in LEVELS:
        p, _ = model.predict(Wt.with_discount(lvl))
        y = np.clip(qb + (p0 * (1 - lvl) * e)[:, None], 0, None)
        off.append(metrics(p, y))
    out.update({f"off_{k}": np.mean([o[k] for o in off]) for k in off[0]})
    out["eff_MAE"] = np.abs(psi - true_eff).mean()
    out["eff_MSE"] = ((psi - true_eff) ** 2).mean()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--models", default="tf,dml,dml-nocf,sdml")
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "results_synthetic.csv"))
    ap.add_argument("--sharpness", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true", help="skip (period, seed, model) rows already in --out")
    ap.add_argument("--noise_mult", type=float, default=0.0)
    ap.add_argument("--nuisance_epochs", type=int, default=None)
    args = ap.parse_args()

    df = synthetic.generate(sharpness=args.sharpness, noise_mult=args.noise_mult)
    A = arrays(df)
    n_cat = [45, 15]
    import os
    rows = pd.read_csv(args.out).to_dict("records") if args.resume and os.path.exists(args.out) else []
    done = {(r["period"], r["seed"], r["model"]) for r in rows}
    for a, b in PERIODS[:args.periods]:
        Wtr = windows(A, range(a + C - 1, b - H + 1))
        Wt = windows(A, [b])
        for seed in range(args.seeds):
            for kind in args.models.split(","):
                if (f"{a}-{b}", seed, kind) in done:
                    continue
                t0 = time.time()
                m = build(kind, "add", "l2", args.nuisance_epochs or args.epochs, args.epochs, seed).fit(Wtr, n_cat)
                r = dict(period=f"{a}-{b}", seed=seed, model=kind, **evaluate(m, A, Wt, b), secs=round(time.time() - t0))
                rows.append(r)
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    res = pd.DataFrame(rows)
    cols = ["off_MAE", "on_MAE", "off_MSE", "on_MSE", "eff_MAE", "eff_MSE"]
    agg = res.groupby("model")[cols].agg(["mean", "std"]).round(1)
    pd.set_option("display.width", 200)
    print("\n=== mean ± std over periods x seeds (paper Table 1 layout) ===")
    print(agg.to_string())
