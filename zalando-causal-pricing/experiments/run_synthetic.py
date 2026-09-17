"""Paper 1's synthetic protocol (arXiv:2312.15282v2, Sec. 4): train on weeks [a, b], forecast the
next five. On-policy: realised treatments. Off-policy: counterfactual treatments with the
simulator's ground truth. Effects: psi against the true effect-space coefficients.

Settings:
  calibrated / literal  one treatment (the discount), linear demand, additive head (the paper's study)
  multi                 three confounded treatments (discount, log list price, log stock), log-linear
                        demand, multiplicative head. --treatments all models all three; --treatments
                        discount is the same data seen by a discount-only model, with the list price as
                        a future exogenous and stock as a historical exogenous (the omitted-treatment
                        comparison).

usage: python experiments/run_synthetic.py --setting multi --treatments all [--models dml,tf] [--epochs 48]
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
    "multi": dict(sharpness=10.0, noise_mult=0.1),       # three treatments, see synthetic.generate_multi
}
TREATMENTS = (("discount", "discount", +1), ("log_price", "linear", -1), ("log_stock", "linear", +1))


def arrays(df):
    piv = lambda c: df.pivot(index="item", columns="week", values=c).to_numpy(np.float32)
    first = df.groupby("item").first()
    A = dict(q=piv("demand"), d=piv("discount"), stock=piv("stock"), qb=piv("base_demand"),
             static_cat=first[["cat_d", "cat_k"]].to_numpy(np.int64), static_num=np.log(first[["p0"]].to_numpy(np.float32)),
             p0=first["p0"].to_numpy(np.float32))
    if "log_price" in df:
        A.update(lp=piv("log_price"), eps_p=first["eps_p"].to_numpy(np.float32), eps_d=first["eps_d"].to_numpy(np.float32),
                 kappa=first["kappa"].to_numpy(np.float32), gamma=float(first["gamma"].iloc[0]))
    else:
        A["e"] = first["effect"].to_numpy(np.float32)
    return A


def panel(A, treatments="all"):
    """The simulator in the shared schema. Single-treatment data: stock is historical exogenous, the
    30-week calendar future exogenous. Multi-treatment data: discount, log list price and log stock
    are the treatments, or (treatments='discount') the discount alone with the list price as future
    exogenous (it is planned ahead) and stock as historical exogenous."""
    n, T = A["q"].shape
    cal = np.broadcast_to(week_feats(np.arange(T), 30), (n, T, 3))
    common = dict(y=A["q"], static_cat=A["static_cat"], static_num=A["static_num"])
    if "lp" not in A or treatments == "discount":
        futr = np.concatenate([cal, A["lp"][..., None]], -1) if "lp" in A else cal
        return Panel(treatment=A["d"], hist=np.log1p(A["stock"])[..., None], futr=futr, **common,
                     names={"hist": ["log_stock"], "futr": ["week", "sin", "cos"] + (["log_price"] if "lp" in A else [])})
    return Panel(treatment=np.stack([A["d"], A["lp"], np.log1p(A["stock"])], -1), treatments=TREATMENTS, futr=cal, **common,
                 names={"treatments": [t[0] for t in TREATMENTS], "futr": ["week", "sin", "cos"]})


def evaluate_single(model, A, Wt, origin):
    """On-policy, off-policy over the discount grid, and effect error vs the true dq/d(discount)."""
    it = Wt.item
    qb, p0, e = A["qb"][it, origin + 1:origin + 1 + H], A["p0"][it], A["e"][it]
    pred, psi = model.predict(Wt)
    psi = psi[:, 0]
    out = {f"on_{k}": v for k, v in metrics(pred, Wt.y).items()}
    off = [metrics(model.predict(Wt.with_discount(lvl))[0], np.clip(qb + (p0 * (1 - lvl) * e)[:, None], 0, None)) for lvl in LEVELS]
    out.update({f"off_{k}": np.mean([o[k] for o in off]) for k in off[0]})
    out["eff_MAE"], out["eff_MSE"] = np.abs(psi + p0 * e).mean(), ((psi + p0 * e) ** 2).mean()
    return out


def evaluate_multi(model, A, Wt, origin, treatments):
    """On-policy; off-policy jointly over a grid of all three treatments and one treatment at a time;
    effect error per treatment against the true effect-space coefficients."""
    it, hor = Wt.item, slice(origin + 1, origin + 1 + H)
    qb, kappa = A["qb"][it, hor], A["kappa"][it]
    eps_p, eps_d, gamma = A["eps_p"][it], A["eps_d"][it], A["gamma"]
    d_real, lp_real, s_real = A["d"][it, hor], A["lp"][it, hor], A["stock"][it, hor]
    truth = lambda d, lp, s: synthetic.counterfactual_multi(qb, eps_p[:, None], eps_d[:, None], kappa[:, None], gamma, d, lp, s)
    pred, psi = model.predict(Wt)
    out = {f"on_{k}": v for k, v in metrics(pred, Wt.y).items()}
    # off-policy: intervene on the discount (all models), and on price / stock (multi-treatment models)
    disc = [metrics(model.predict(Wt.with_treatment(lvl, 0))[0], truth(lvl, lp_real, s_real)) for lvl in LEVELS]
    out["off_MAE_discount"] = np.mean([o["MAE"] for o in disc])
    if treatments == "all":
        price = [metrics(model.predict(Wt.with_treatment(lp_real + dl, 1))[0], truth(d_real, lp_real + dl, s_real)) for dl in (-0.1, 0.1)]
        stock = [metrics(model.predict(Wt.with_treatment(np.log1p(f * kappa)[:, None] * np.ones_like(s_real), 2))[0], truth(d_real, lp_real, f * kappa[:, None]))
                 for f in (0.5, 3.0)]
        out["off_MAE_log_price"], out["off_MAE_log_stock"] = np.mean([o["MAE"] for o in price]), np.mean([o["MAE"] for o in stock])
        joint = [metrics(model.predict(Wt.with_treatment(lvl, 0).with_treatment(lp_real + dl, 1).with_treatment(np.log1p(f * kappa)[:, None] * np.ones_like(s_real), 2))[0],
                         truth(lvl, lp_real + dl, f * kappa[:, None]))
                 for lvl in (0.0, 0.25, 0.5) for dl in (-0.1, 0.0, 0.1) for f in (0.5, 3.0)]
        out.update({f"off_{k}": np.mean([o[k] for o in joint]) for k in joint[0]})
        true = synthetic.true_effects_multi(eps_p, eps_d, kappa, gamma, s_real)
        for k, name in enumerate(["discount", "log_price", "log_stock"]):
            out[f"eff_MAE_{name}"] = np.abs(psi[:, k] - true[:, k]).mean()
        out["eff_MAE"], out["eff_MSE"] = out["eff_MAE_discount"], ((psi[:, 0] - true[:, 0]) ** 2).mean()
    else:  # discount-only model on the same data: its off-policy and effect are the discount's
        out.update({f"off_{k}": np.mean([o[k] for o in disc]) for k in disc[0]})
        out["eff_MAE"], out["eff_MSE"] = np.abs(psi[:, 0] - eps_d).mean(), ((psi[:, 0] - eps_d) ** 2).mean()
        out["eff_MAE_discount"] = out["eff_MAE"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=SETTINGS, default="calibrated")
    ap.add_argument("--treatments", choices=["all", "discount"], default="all", help="multi setting only")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=48)
    ap.add_argument("--effect_epochs", type=int, default=8)
    ap.add_argument("--max_trees", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results" / "synthetic.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    multi = args.setting == "multi"
    gen = synthetic.generate_multi if multi else synthetic.generate
    A = arrays(gen(**SETTINGS[args.setting]))
    P = panel(A, args.treatments if multi else "discount")
    head, loss = ("mult", "l1") if multi else ("add", "l2")
    cfg = Config(head=head, loss=loss, epochs=args.epochs, effect_epochs=args.effect_epochs, max_trees=args.max_trees,
                 treatments=P.treatments)
    provenance = dict(setting=args.setting, treatments=args.treatments if multi else "discount", **SETTINGS[args.setting],
                      epochs=args.epochs, effect_epochs=args.effect_epochs, max_trees=args.max_trees)
    windows = {}  # per period: (train, test)
    cells = [dict(period=f"{a}-{b}", seed=s, model=m) for a, b in PERIODS[:args.periods]
             for s in range(args.seeds) for m in args.models.split(",")]

    def run(cell):
        a, b = map(int, cell["period"].split("-"))
        if cell["period"] not in windows:
            windows[cell["period"]] = (make_windows(P, range(a + C - 1, b - H + 1), C, H), make_windows(P, [b], C, H))
        Wtr, Wt = windows[cell["period"]]
        model = build(cell["model"], Config(**{**cfg.__dict__, "seed": cell["seed"]}), P.n_cat).fit(Wtr)
        return [evaluate_multi(model, A, Wt, b, args.treatments) if multi else evaluate_single(model, A, Wt, b)]

    run_grid(args.out, cells, run, provenance)
