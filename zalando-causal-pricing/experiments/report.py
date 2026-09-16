"""Collect every results_*.csv into one comparison table (RESULTS.md).

Synthetic rows follow the paper's Table 1 layout (MAE/MSE on- and off-policy, effect MAE/MSE),
M5 rows report MAE / MSE / demand error on all windows and on the price-change slice.
Model names: dml (paper 1), mdl (paper 2), tf / sdml / dml-nocf (paper-1 ablations), last4 (naive).
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

SYN_COLS = ["off_MAE", "on_MAE", "off_MSE", "on_MSE", "eff_MAE", "eff_MSE"]
NAMES = {"dml": "DML Forecaster (paper 1)", "dml-nocf": "DML, no cross-fitting", "sdml": "sDML (no treatment model)",
         "tf": "TF, linear head S-learner (paper 1 ablation)", "mdl": "Monotonic-demand transformer (paper 2)", "mdl-anchored": "paper 2 model anchored to recent demand level (ablation)", "last4": "naive: mean of last 4 weeks"}


def load(pattern, suffix_epochs):
    """Concatenate CSVs; files named *_<variant>NN.csv get their model tagged with the epoch count."""
    dfs = []
    for f in sorted(glob.glob(str(RES / pattern))):
        df = pd.read_csv(f)
        tag = f.rsplit("_", 1)[-1].replace(".csv", "")
        if tag[-2:].isdigit():  # e.g. tf24, mdl48
            df["model"] = df["model"] + f" ({tag[-2:]} ep)"
        elif suffix_epochs:
            df["model"] = df["model"] + f" ({suffix_epochs} ep)"
        dfs.append(df)
    return pd.concat(dfs) if dfs else None


def fmt(mean, std):
    return mean.round(1).astype(str) + " ± " + std.round(1).astype(str)


def synthetic_table(df):
    g = df.groupby("model")[SYN_COLS]
    out = fmt(g.mean(), g.std())
    out.insert(0, "runs", g.size())
    return out


def m5_table(df):
    d = df[df["slice"].isin(["all", "price_change"])]
    g = d.groupby(["model", "slice"])[["MAE", "MSE", "demand_err"]].mean().round(3).unstack("slice")
    g.columns = [f"{m} ({s})" for m, s in g.columns]
    return g


def headline():
    """One table, both papers, all benchmarks (best-epoch variant of each model)."""
    rows = {}
    for setting in ["calibrated", "literal"]:
        df = load(f"results_synthetic_{setting}*.csv", 48)
        if df is None:
            continue
        g = df.groupby("model")[SYN_COLS].mean()
        g["base"] = [m.split(" (")[0] for m in g.index]
        best = g.loc[g.groupby("base")["off_MAE"].idxmin()].set_index("base")  # best epoch count per model
        for m, r in best.iterrows():
            rows.setdefault(m, {})
            rows[m][f"{setting}: off-policy MAE"] = round(r["off_MAE"], 1)
            rows[m][f"{setting}: on-policy MAE"] = round(r["on_MAE"], 1)
            rows[m][f"{setting}: effect MAE"] = round(r["eff_MAE"], 1)
    df = load("results_m5*.csv", None)
    if df is not None:
        for m, r in df[df["slice"] == "all"].groupby("model")[["MAE", "mean_psi"]].mean().iterrows():
            rows.setdefault(m, {})["M5: MAE all"] = round(r["MAE"], 2)
            rows[m]["M5: elasticity"] = round(r["mean_psi"], 2)
        for m, r in df[df["slice"] == "price_change"].groupby("model")[["MAE"]].mean().iterrows():
            rows.setdefault(m, {})["M5: MAE price change"] = round(r["MAE"], 2)
    order = [m for m in NAMES if m in rows]
    t = pd.DataFrame.from_dict(rows, orient="index").loc[order]
    t.index = [f"{m} — {NAMES[m]}" for m in t.index]
    return t.to_markdown(floatfmt=".2f")


def inject_readme(table):
    r = open(ROOT / "README.md").read()
    a, b = "<!-- RESULTS -->", "<!-- /RESULTS -->"
    if b not in r:
        r = r.replace(a, a + "\n" + b)
    pre, rest = r.split(a); _, post = rest.split(b)
    open(ROOT / "README.md", "w").write(pre + a + "\n" + table + "\n" + b + post)


if __name__ == "__main__":
    md = ["# Results\n", "Model key: " + "; ".join(f"`{k}` = {v}" for k, v in NAMES.items()) + "\n"]
    head = headline()
    md += ["\n## Headline comparison (best epoch count per model; synthetic = mean over 4 periods x 3 seeds)\n", head, ""]
    inject_readme(head)
    for setting, desc in [("calibrated", "policy sharpness 10, multiplicative noise 0.1 (matches the paper's error magnitude)"),
                          ("literal", "appendix rule verbatim (sharpness 1), no extra noise")]:
        df = load(f"results_synthetic_{setting}*.csv", 48)
        if df is None:
            continue
        md += [f"\n## Synthetic data, {setting} setting\n", desc + ". Mean ± std over 4 training periods x seeds. Off-policy = constant discount in {0,...,0.5} with simulator ground truth; effect = dq/d(discount) vs true -p0*e_i.\n",
               synthetic_table(df).to_markdown(), ""]
    df = load("results_m5*.csv", None)
    if df is not None:
        md += ["\n## M5 (weekly, 30490 series, horizon 4 weeks, 4 forecast origins)\n",
               "`all` = every test window; `price_change` = windows whose horizon discount differs from the last 4 weeks' by more than 10 points (off-policy proxy). demand_err = price-weighted relative RMSE (paper 1, eq. demand error).\n",
               m5_table(df).to_markdown(), ""]
        psi = df[df["slice"] == "all"].groupby("model")["mean_psi"].mean().dropna().round(2)
        md += ["\nMean estimated price elasticity on M5 test windows (negative = demand falls with price):\n", psi.to_frame("elasticity").to_markdown(), ""]
    open(RES / "RESULTS.md", "w").write("\n".join(md))
    print("\n".join(md))
