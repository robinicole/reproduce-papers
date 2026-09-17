"""results/synthetic.csv + results/m5.csv -> results/RESULTS.md, and the headline table into README.

Rows are labelled from columns (model, epochs), never from filenames. Every trained variant is
shown; nothing is selected by test error.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
RES = ROOT / "results"

import pandas as pd

from registry import display, label

SYN_COLS = ["off_MAE", "on_MAE", "off_MSE", "on_MSE", "eff_MAE", "eff_MSE"]
M5_COLS = ["MAE", "MSE", "demand_err"]


def load(name):
    f = RES / f"{name}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df["label"] = [label(m, e) for m, e in zip(df["model"], df["epochs"])]
    if "treatments" in df:  # the omitted-treatment comparison in the multi setting
        df.loc[(df["setting"] == "multi") & (df["treatments"] == "discount"), "label"] += " (discount only)"
    return df


def synthetic_table(df):
    extra = [c for c in df.columns if c.startswith(("eff_MAE_", "off_MAE_")) and df[c].notna().any()]
    g = df.groupby("label")[SYN_COLS + extra]
    out = g.mean().round(1).astype(str) + " ± " + g.std().round(1).astype(str)
    out.insert(0, "runs", g.size())
    return out


def m5_table(df):
    d = df[df["slice"].isin(["all", "price_change"])]
    g = d.groupby(["label", "slice"])[M5_COLS].mean().round(3).unstack("slice")
    g.columns = [f"{m} ({s})" for m, s in g.columns]
    return g


def headline(syn, m5):
    """One table: synthetic columns per setting, M5 columns joined on the model name."""
    rows = {}
    if syn is not None:
        for setting, d in syn.groupby("setting"):
            for lab, r in d.groupby("label")[SYN_COLS].mean().iterrows():
                rows.setdefault(lab, {"model": lab.split(" (")[0]}).update({
                    f"{setting}: off-policy MAE": round(r["off_MAE"], 1), f"{setting}: on-policy MAE": round(r["on_MAE"], 1),
                    f"{setting}: effect MAE": round(r["eff_MAE"], 1)})
    if m5 is not None:
        a = m5[m5["slice"] == "all"].groupby("label")[["MAE", "mean_psi"]].mean()
        pc = m5[m5["slice"] == "price_change"].groupby("label")["MAE"].mean()
        for lab in a.index:
            rows.setdefault(lab, {"model": lab.split(" (")[0]}).update({
                "M5: MAE all": round(a.loc[lab, "MAE"], 2), "M5: MAE price change": round(pc.get(lab, float("nan")), 2),
                "M5: elasticity": round(a.loc[lab, "mean_psi"], 2)})
    t = pd.DataFrame.from_dict(rows, orient="index").sort_values("model", kind="stable")
    t.index = [f"{lab} — {display(m)}" for lab, m in zip(t.index, t["model"])]
    return t.drop(columns="model").to_markdown(floatfmt=".2f")


def inject_readme(table):
    r = open(ROOT / "README.md").read()
    a, b = "<!-- RESULTS -->", "<!-- /RESULTS -->"
    pre, rest = r.split(a)
    _, post = rest.split(b)
    open(ROOT / "README.md", "w").write(pre + a + "\n" + table + "\n" + b + post)


if __name__ == "__main__":
    syn, m5 = load("synthetic"), load("m5")
    head = headline(syn, m5)
    inject_readme(head)
    md = ["# Results\n", "\n## Headline comparison (every trained variant; synthetic = mean over periods x seeds)\n", head, ""]
    if syn is not None:
        for setting, d in syn.groupby("setting"):
            note = ("Three confounded treatments (discount, log list price, log stock), log-linear demand. off_MAE = joint off-policy grid; "
                    "off_MAE_<t> = intervening on one treatment; eff_MAE_<t> = effect-space coefficient error per treatment. "
                    "'(discount only)' rows see the same data with a single treatment.\n" if setting == "multi" else
                    "Mean ± std over 4 training periods x seeds. Off-policy = constant discount in {0,...,0.5} with simulator ground truth; effect = dq/d(discount) vs true -p0*e_i.\n")
            md += [f"\n## Synthetic data, {setting} setting\n", note,
                   synthetic_table(d).to_markdown(), ""]
    if m5 is not None:
        md += ["\n## M5 (weekly, 30490 series, horizon 4 weeks, 4 forecast origins)\n",
               "`all` = every test window; `price_change` = windows whose horizon discount differs from the last 4 weeks' by more than 10 points (off-policy proxy). demand_err = price-weighted relative RMSE (paper 1, eq. demand error).\n",
               m5_table(m5).to_markdown(), "",
               "\nMean estimated price elasticity on M5 test windows (negative = demand falls with price):\n",
               m5[m5["slice"] == "all"].groupby("label")["mean_psi"].mean().dropna().round(2).to_frame("elasticity").to_markdown(), ""]
    RES.mkdir(exist_ok=True)
    open(RES / "RESULTS.md", "w").write("\n".join(md))
    print("\n".join(md))
