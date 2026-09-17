"""M5 weekly-data experiment: DML Forecaster vs TF/sDML baselines vs a naive last4 baseline.

usage: python run_m5.py [--seeds 1] [--epochs 3] [--models tf,dml,dml-nocf,sdml] [--subsample 1.0]
"""
import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("models", "data", "experiments")]

import numpy as np
import pandas as pd

import m5_data
from common import Panel, make_windows, metrics, week_feats
from run_synthetic import build

C, H = 26, 4
TEST = [260, 264, 268, 272]


def panel(data, static_num):
    """M5 in the shared schema. The 52-week calendar is future exogenous; availability marks weeks
    the series is not sellable. SNAP and event flags are future exogenous too and would go in
    `futr` alongside the calendar columns."""
    n, T = data["q"].shape
    return Panel(y=data["q"], treatment=data["d"],
                 futr=np.broadcast_to(week_feats(np.arange(T), 52), (n, T, 3)),
                 static_cat=data["static_cat"], static_num=static_num, valid=data["avail"],
                 names={"futr": ["week", "sin", "cos"],
                        "static_cat": ["dept", "cat", "store", "state"], "static_num": ["log_base_price"]})


def windows(data, static_num, origins):
    return make_windows(panel(data, static_num) if not isinstance(data, Panel) else data, origins, C, H)


def slice_masks(Wt):
    ctx_d = Wt.past[:, -4:, 1].mean(1)
    hor_d = Wt.d_fut.mean(1)
    price_change = np.abs(hor_d - ctx_d) > 0.10
    promo_start = price_change & (hor_d > ctx_d)
    return {"all": np.ones(len(Wt), bool), "price_change": price_change, "promo_start": promo_start}


def last4_pred(Wt):
    return np.repeat(np.expm1(Wt.past[:, -4:, 0]).mean(1, keepdims=True), H, axis=1)


def eval_rows(name, seed, origin, Wt, pred, psi, secs):
    price = np.exp(Wt.static_num[:, 0])
    psi = np.full(len(Wt), np.nan) if psi is None else psi
    rows = []
    for slc, mask in slice_masks(Wt).items():
        if mask.sum() == 0:
            continue
        m = metrics(pred[mask], Wt.y[mask], price=price[mask])
        rows.append(dict(model=name, seed=seed, origin=origin, slice=slc, n=int(mask.sum()),
                         MAE=m["MAE"], MSE=m["MSE"], demand_err=m["demand_err"],
                         mean_psi=float(np.nanmean(psi[mask])), secs=secs))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--effect_epochs", type=int, default=3)
    ap.add_argument("--models", default="tf,dml,dml-nocf,sdml")
    ap.add_argument("--out", default=str(ROOT / "results" / "results_m5.csv"))
    ap.add_argument("--subsample", type=float, default=1.0)
    args = ap.parse_args()

    data = m5_data.load()
    n = data["q"].shape[0]
    if args.subsample < 1.0:
        idx = np.random.default_rng(0).choice(n, size=int(n * args.subsample), replace=False)
        for k in ("q", "d", "avail", "price", "base_price_last", "static_cat", "ids"):
            data[k] = data[k][idx]

    # base price = expanding max of price up to and including each week: known at the forecast origin.
    # (Using the series-final base price here would leak post-origin price highs for up to 5.7% of series.)
    base = np.fmax.accumulate(data["price"], axis=1)
    static_num = np.log(np.nan_to_num(base, nan=1.0)).clip(-2, 6)[..., None].astype(np.float32)
    P = panel(data, static_num)
    n_cat = P.n_cat  # derived from the panel, so a new schema needs no edit here

    Wtr = windows(P, None, range(C - 1, 256, 3))
    print(f"training windows: {len(Wtr)}")

    test_w = {origin: windows(P, None, [origin]) for origin in TEST}
    for origin, Wt in test_w.items():
        s = slice_masks(Wt)
        print(f"origin {origin}: n={len(Wt)} price_change={s['price_change'].sum()} promo_start={s['promo_start'].sum()}")

    rows = []
    for name in args.models.split(",") + ["last4"]:
        for seed in range(args.seeds):
            t0 = time.time()
            model = None if name == "last4" else build(
                name, "mult", "l1", args.epochs, args.effect_epochs, seed).fit(Wtr, n_cat)
            for origin, Wt in test_w.items():
                pred, psi = last4_pred(Wt), None
                if model is not None:
                    pred, psi = model.predict(Wt)
                rows += eval_rows(name, seed, origin, Wt, pred, psi, round(time.time() - t0, 1))
            pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"done {name}", flush=True)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for metric in ("MAE", "demand_err"):
        piv = res.pivot_table(index="model", columns="slice", values=metric, aggfunc="mean").round(3)
        print(f"\n=== {metric} (mean over seeds x origins) ===")
        print(piv.to_string())
