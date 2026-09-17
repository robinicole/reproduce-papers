"""Paper 1's synthetic protocol (arXiv:2312.15282v2, Sec. 4): train on weeks [a, b], forecast the
next five. On-policy: realised discounts. Off-policy: constant discount in {0, .., .5} with the
simulator's counterfactual ground truth. Effect: psi against the true dq/d(discount) = -p0 * e_i.

usage: python experiments/run_synthetic.py --setting calibrated [--models dml,tf] [--epochs 48] [--seeds 3]
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("models", "data", "experiments")]

import numpy as np

import synthetic
from grid import run_grid
from metrics import metrics
from registry import MODELS, Config, build
from schema import Panel, make_windows, week_feats

C, H = 20, 5
PERIODS = [(20, 65), (30, 75), (40, 85), (50, 94)]  # weeks are 0-indexed; last horizon = weeks 95..99
LEVELS = np.arange(0, 0.51, 0.1)
SETTINGS = {  # see README, "Reproduction notes"
    "calibrated": dict(sharpness=10.0, noise_mult=0.1),  # main-text policy, paper-level noise
    "literal": dict(sharpness=1.0, noise_mult=0.0),      # appendix rule verbatim
}


def arrays(df):
    piv = lambda c: df.pivot(index="item", columns="week", values=c).to_numpy(np.float32)
    first = df.groupby("item").first()
    return dict(q=piv("demand"), d=piv("discount"), stock=piv("stock"), qb=piv("base_demand"),
                p0=first["p0"].to_numpy(np.float32), e=first["effect"].to_numpy(np.float32),
                static_cat=first[["cat_d", "cat_k"]].to_numpy(np.int64),
                static_num=np.log(first[["p0"]].to_numpy(np.float32)))


def panel(A):
    """The simulator in the shared schema: stock is historical exogenous (known only up to the
    origin), the 30-week calendar is future exogenous, category codes and base price are static."""
    n, T = A["q"].shape
    return Panel(y=A["q"], treatment=A["d"], hist=np.log1p(A["stock"])[..., None],
                 futr=np.broadcast_to(week_feats(np.arange(T), 30), (n, T, 3)),
                 static_cat=A["static_cat"], static_num=A["static_num"],
                 names={"hist": ["stock"], "futr": ["week", "sin", "cos"],
                        "static_cat": ["cat_d", "cat_k"], "static_num": ["log_p0"]})


def evaluate(model, A, Wt, origin):
    """On-policy error, off-policy error over the discount grid, and effect error vs ground truth."""
    it = Wt.item
    qb, p0, e = A["qb"][it, origin + 1:origin + 1 + H], A["p0"][it], A["e"][it]
    pred, psi = model.predict(Wt)
    out = {f"on_{k}": v for k, v in metrics(pred, Wt.y).items()}
    off = []
    for lvl in LEVELS:
        p, _ = model.predict(Wt.with_discount(lvl))
        off.append(metrics(p, np.clip(qb + (p0 * (1 - lvl) * e)[:, None], 0, None)))
    out.update({f"off_{k}": np.mean([o[k] for o in off]) for k in off[0]})
    out["eff_MAE"] = np.abs(psi + p0 * e).mean()
    out["eff_MSE"] = ((psi + p0 * e) ** 2).mean()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=SETTINGS, default="calibrated")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=48)
    ap.add_argument("--effect_epochs", type=int, default=8)
    ap.add_argument("--max_trees", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results" / "synthetic.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    A = arrays(synthetic.generate(**SETTINGS[args.setting]))
    P = panel(A)
    cfg = Config(head="add", loss="l2", epochs=args.epochs, effect_epochs=args.effect_epochs, max_trees=args.max_trees)
    provenance = dict(setting=args.setting, **SETTINGS[args.setting], epochs=args.epochs,
                      effect_epochs=args.effect_epochs, max_trees=args.max_trees)
    windows = {}  # per period: (train, test)
    cells = [dict(period=f"{a}-{b}", seed=s, model=m) for a, b in PERIODS[:args.periods]
             for s in range(args.seeds) for m in args.models.split(",")]

    def run(cell):
        a, b = map(int, cell["period"].split("-"))
        if cell["period"] not in windows:
            windows[cell["period"]] = (make_windows(P, range(a + C - 1, b - H + 1), C, H), make_windows(P, [b], C, H))
        Wtr, Wt = windows[cell["period"]]
        model = build(cell["model"], Config(**{**cfg.__dict__, "seed": cell["seed"]}), P.n_cat).fit(Wtr)
        return [evaluate(model, A, Wt, b)]

    run_grid(args.out, cells, run, provenance)
