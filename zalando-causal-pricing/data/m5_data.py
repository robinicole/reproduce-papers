"""Prepare M5 forecasting-accuracy data as weekly demand/price arrays."""
import numpy as np
from pathlib import Path

M5 = Path(__file__).resolve().parent / "m5"
import pandas as pd

DATA = "data/m5"
N_DAYS = 1941


def _build():
    cal = pd.read_csv(f"{DATA}/calendar.csv")
    cal["dnum"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal = cal[cal["dnum"] <= N_DAYS].sort_values("dnum")
    wk_codes = cal["wm_yr_wk"].to_numpy()

    # contiguous run-length groups of wm_yr_wk over the (chronologically sorted) days
    change = np.flatnonzero(np.diff(wk_codes) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(wk_codes)]))
    complete = (ends - starts) == 7
    starts, ends = starts[complete], ends[complete]
    weeks = wk_codes[starts]  # chronological, one entry per complete week
    n_weeks = len(weeks)

    sales = pd.read_csv(f"{DATA}/sales_train_evaluation.csv")
    day_cols = [f"d_{i}" for i in range(1, N_DAYS + 1)]
    daily = sales[day_cols].to_numpy(dtype=np.float32)

    q = np.empty((len(sales), n_weeks), dtype=np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        q[:, i] = daily[:, s:e].sum(axis=1)

    ids = sales["id"].to_numpy(dtype=str)
    static_cols = ["dept_id", "cat_id", "store_id", "state_id"]
    static_cat = np.empty((len(sales), 4), dtype=np.int64)
    n_cat = []
    for j, col in enumerate(static_cols):
        codes = pd.Categorical(sales[col])
        static_cat[:, j] = codes.codes
        n_cat.append(len(codes.categories))

    key = sales["store_id"] + "_" + sales["item_id"]

    sp = pd.read_csv(f"{DATA}/sell_prices.csv")
    sp = sp.drop_duplicates(subset=["store_id", "item_id", "wm_yr_wk"], keep="first")
    sp_key = sp["store_id"] + "_" + sp["item_id"]
    piv = sp.assign(_key=sp_key).pivot(index="_key", columns="wm_yr_wk", values="sell_price")
    price = piv.reindex(index=key, columns=weeks).to_numpy(dtype=np.float32)

    base_price = np.fmax.accumulate(price, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1.0 - price / base_price
    d = np.where(np.isnan(price), 0.0, d).astype(np.float32)
    d = np.clip(d, 0.0, 0.95)

    avail = ~np.isnan(price)
    base_price_last = base_price[:, -1].astype(np.float32)

    return dict(
        q=q, price=price, d=d, avail=avail,
        base_price_last=base_price_last, static_cat=static_cat,
        n_cat=n_cat, ids=ids, weeks=weeks.astype(np.int64),
    )


def load(cache=str(M5 / "weekly.npz")):
    import os

    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return dict(
            q=z["q"], price=z["price"], d=z["d"], avail=z["avail"],
            base_price_last=z["base_price_last"], static_cat=z["static_cat"],
            n_cat=list(z["n_cat"]), ids=z["ids"], weeks=z["weeks"],
        )

    out = _build()
    np.savez(
        cache, q=out["q"], price=out["price"], d=out["d"], avail=out["avail"],
        base_price_last=out["base_price_last"], static_cat=out["static_cat"],
        n_cat=np.array(out["n_cat"]), ids=out["ids"], weeks=out["weeks"],
    )
    return out


if __name__ == "__main__":
    data = load()
    q, price, d, avail = data["q"], data["price"], data["d"], data["avail"]

    print("q shape:", q.shape)
    print("price shape:", price.shape)
    print("d shape:", d.shape)
    print("avail shape:", avail.shape)
    print("static_cat shape:", data["static_cat"].shape, "n_cat:", data["n_cat"])
    print("n complete weeks:", len(data["weeks"]))
    print("share of cells available:", avail.mean())
    print("mean discount (available cells):", d[avail].mean())
    print("share of available cells with discount > 0.05:", (d[avail] > 0.05).mean())
    print("mean weekly demand (available cells):", q[avail].mean())

    assert (d >= 0).all() and (d <= 0.95).all()
    assert (q >= 0).all()
