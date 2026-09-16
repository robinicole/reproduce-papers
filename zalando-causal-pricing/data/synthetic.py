"""Synthetic pricing data, following Appendix E of arXiv:2312.15282v2 (Causal Forecasting for Pricing).

q_it = q_b_it + p_it * e_i           (e_i < 0: demand falls with price)
q_b  = (0.15 tau + 0.25 s + 1) * c   (trend, seasonality, article factor)
Pricing policy steers stock coverage towards 1 with 10% discount steps in [0, 0.5].
"""
import numpy as np
import pandas as pd

T = 100
HORIZON_MAX_DISCOUNT = 0.5


def generate(n_items=4467, n_candidates=18000, seed=0, sharpness=1.0, noise_mult=0.0, noise_add=0.0):
    """sharpness: exponent on the policy's move probability. 1.0 = the appendix rule
    P(up)=1-1/w; larger values approach the main text's deterministic 'adjust every week' rule."""
    rng = np.random.default_rng(seed)
    n = n_candidates
    t = np.arange(T)

    # article factor c_it
    cat_d = rng.integers(0, 45, n)
    cat_k = rng.integers(0, 15, n)
    alpha = rng.normal(10, 3, 45)[cat_d]
    beta = rng.normal(300, 50, 15)[cat_k]
    a = alpha[:, None] + rng.normal(0, 1, (n, T))
    b = beta[:, None] + rng.normal(0, 5, (n, T))
    c = 0.05 * a**2 + 0.25 * a + 0.5 * b

    # seasonality: k split into 6 groups, integer shift in [-15, 15] per group
    shift = rng.integers(-15, 16, 6)[cat_k % 6]
    s = np.sin(2 * np.pi * (t[None, :] + shift[:, None]) / 30)

    # noisy linear trend
    gamma = rng.uniform(-0.02, 0.02, n)
    sig_tau = rng.uniform(0, 0.15, n)
    tau = rng.normal(t[None, :] * gamma[:, None], sig_tau[:, None])

    # main-text noise model: c * lambda_it + eta_it (multiplicative and additive week-level noise)
    c = c * (1 + rng.normal(0, noise_mult, (n, T))) + rng.normal(0, noise_add, (n, T))
    q_b = (0.15 * tau + 0.25 * s + 1) * c

    # treatment effect (negative: higher price -> lower demand)
    e_b = np.maximum(1.3, rng.lognormal(0.75, 0.125, n))
    e = -e_b * 0.15 * a.mean(1)

    q_bar = q_b.mean(1)
    p0 = np.abs(rng.normal(q_bar / 3, q_bar / 1.5))
    p0 = np.maximum(p0, 1.0)

    # stock to clear at 14% avg discount
    z0 = (q_b + 0.86 * p0[:, None] * e[:, None]).sum(1)

    # simulate policy
    q = np.zeros((n, T))
    d = np.zeros((n, T))
    stock = np.zeros((n, T))
    j = np.zeros(n, dtype=int)
    z = z0.copy()
    for wk in range(T):
        if wk >= 4:
            m = 4 * z / np.maximum(q[:, wk - 4:wk].sum(1), 1e-6)
            w = m / (T - wk)
            lam = rng.uniform(0, 1, n)
            up = (w > 1) & (lam > w ** -sharpness)
            down = (w < 1) & (lam > w ** sharpness)
            j = np.clip(j + up.astype(int) - down.astype(int), 0, 5)
        d[:, wk] = 0.1 * j
        stock[:, wk] = z
        q[:, wk] = q_b[:, wk] + p0 * (1 - d[:, wk]) * e
        z = z - q[:, wk]

    # filter: only articles with non-negative demand for any discount in [0, 0.5]
    q_min = q_b + p0[:, None] * e[:, None]  # demand at full price (lowest)
    keep = np.where((q_min.min(1) >= 0) & (q_b.min(1) > 0))[0][:n_items]

    q = np.round(q[keep]).clip(0)
    df = pd.DataFrame({
        "item": np.repeat(np.arange(len(keep)), T),
        "week": np.tile(t, len(keep)),
        "demand": q.ravel(),
        "discount": d[keep].ravel(),
        "stock": np.maximum(stock[keep], 0).ravel(),
        "base_demand": q_b[keep].ravel(),
        "cat_d": np.repeat(cat_d[keep], T),
        "cat_k": np.repeat(cat_k[keep], T),
        "p0": np.repeat(p0[keep], T),
        "effect": np.repeat(e[keep], T),  # dq/dp
    })
    return df


def counterfactual(df, discount):
    """True demand if every week in df had `discount` (scalar or array aligned with df)."""
    return (df["base_demand"] + df["p0"] * (1 - discount) * df["effect"]).clip(lower=0)


if __name__ == "__main__":
    df = generate()
    n = df["item"].nunique()
    assert n >= 4000, n
    assert (df["demand"] >= 0).all()
    assert df["discount"].between(0, 0.5).all()
    print(f"{n} items, mean demand {df.demand.mean():.1f}, mean discount {df.discount.mean():.3f}")
    print(df.groupby("week")["discount"].mean().round(3).iloc[::10].to_dict())
    # confounding check: discount should be higher when seasonal base demand is low
    print("corr(discount, base_demand/p0-scaled):", np.corrcoef(df.discount, df.base_demand / df.groupby("item").base_demand.transform("mean"))[0, 1].round(3))
