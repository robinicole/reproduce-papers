"""M5 weekly protocol: 26-week context, 4-week horizon, four forecast origins at the end of the
series. Reports every test window and a "price change" slice (horizon discount differs from the
last four weeks' by more than ten points), the natural-experiment proxy for off-policy accuracy.

usage: python experiments/run_m5.py [--models dml,lgbm] [--epochs 4] [--max_trees 12000] [--subsample 0.05]
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("models", "data", "experiments")]

import numpy as np

import m5_data
from grid import run_grid
from metrics import metrics
from registry import MODELS, Config, build
from schema import Panel, make_windows, week_feats

C, H = 26, 4
TEST = [260, 264, 268, 272]  # forecast origins; the last horizon ends at week 276, the final index
TRAIN_ORIGINS = range(C - 1, 256, 3)


def panel(data):
    """M5 in the shared schema. The 52-week calendar is future exogenous; the base price is the
    expanding max of price up to each week, so it is known at the origin (a series-final base price
    would leak later price highs). SNAP and event flags are future exogenous too and would be
    added to `futr` beside the calendar columns."""
    n, T = data["q"].shape
    base = np.fmax.accumulate(data["price"], axis=1)
    return Panel(y=data["q"], treatment=data["d"],
                 futr=np.broadcast_to(week_feats(np.arange(T), 52), (n, T, 3)),
                 static_cat=data["static_cat"],
                 static_num=np.log(np.nan_to_num(base, nan=1.0)).clip(-2, 6)[..., None].astype(np.float32),
                 valid=data["avail"],
                 names={"futr": ["week", "sin", "cos"], "static_cat": ["dept", "cat", "store", "state"],
                        "static_num": ["log_base_price"]})


def slices(Wt):
    ctx_d, hor_d = Wt.past[:, -4:, 1].mean(1), Wt.d_fut[..., 0].mean(1)  # treatment 0 is the discount
    price_change = np.abs(hor_d - ctx_d) > 0.10
    return {"all": np.ones(len(Wt), bool), "price_change": price_change, "promo_start": price_change & (hor_d > ctx_d)}


def last4(Wt):
    """Naive baseline: the mean of the last four context weeks, repeated over the horizon."""
    return np.repeat(np.expm1(Wt.past[:, -4:, 0]).mean(1, keepdims=True), H, axis=1)


def rows(origin, Wt, pred, psi):
    price = np.exp(Wt.static_num[:, 0])
    psi = np.full(len(Wt), np.nan) if psi is None else np.asarray(psi).reshape(len(Wt), -1)[:, 0]
    out = []
    for name, mask in slices(Wt).items():
        if mask.any():
            out.append(dict(origin=origin, slice=name, n=int(mask.sum()), mean_psi=float(np.nanmean(psi[mask])),
                            **metrics(pred[mask], Wt.y[mask], price=price[mask])))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--effect_epochs", type=int, default=3)
    ap.add_argument("--max_trees", type=int, default=2000)
    ap.add_argument("--subsample", type=float, default=1.0, help="fraction of series, for smoke runs")
    ap.add_argument("--out", default=str(ROOT / "results" / "m5.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = m5_data.load()
    if args.subsample < 1:
        keep = np.random.default_rng(0).random(len(data["q"])) < args.subsample
        data = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in data.items()}
    P = panel(data)
    Wtr = make_windows(P, TRAIN_ORIGINS, C, H)
    tests = {o: make_windows(P, [o], C, H) for o in TEST}
    logging.info("training windows: %d", len(Wtr))
    for o, Wt in tests.items():
        logging.info("origin %d: %s", o, {k: int(m.sum()) for k, m in slices(Wt).items()})
    cfg = Config(head="mult", loss="l1", epochs=args.epochs, effect_epochs=args.effect_epochs, max_trees=args.max_trees)
    provenance = dict(epochs=args.epochs, effect_epochs=args.effect_epochs, max_trees=args.max_trees)
    cells = [dict(model=m, seed=s) for m in args.models.split(",") + ["last4"] for s in range(args.seeds)]

    def run(cell):
        if cell["model"] == "last4":
            return [r for o, Wt in tests.items() for r in rows(o, Wt, last4(Wt), None)]
        model = build(cell["model"], Config(**{**cfg.__dict__, "seed": cell["seed"]}), P.n_cat).fit(Wtr)
        return [r for o, Wt in tests.items() for r in rows(o, Wt, *model.predict(Wt))]

    run_grid(args.out, cells, run, provenance)
