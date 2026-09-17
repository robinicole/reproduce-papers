"""Synthetic pricing data.

generate       paper 1's Appendix E simulator: linear price effect, one treatment (the discount),
               stock-coverage pricing policy.  q_it = q_b_it + p_it * e_i
generate_multi three confounded treatments on the same base demand, log-linear:
               q = q_b * (p0_t / p0)^eps_p * (1 - d)^eps_d * a(stock),   eps_d = eps_p * (1 + premium)
               a(s) = min(1, ((1+s)/(1+kappa))^gamma): availability falls once stock is below kappa.
               The list price, the discount and replenishment are all set by policies that react to
               demand, so all three are confounded. The hypothesis it lets you test: does a model
               recover a discount response that is stronger than the list-price response (the promo
               premium), and the stock effect, separately, when all three move at once?
Both return a long DataFrame (item, week, ...) with the ground truth needed for counterfactuals.
"""
import numpy as np
import pandas as pd

T = 100
HORIZON_MAX_DISCOUNT = 0.5


def _base(rng, n, noise_mult, noise_add):
    """Base demand q_b (n, T), its category codes and article factor, as in Appendix E."""
    t = np.arange(T)
    cat_d, cat_k = rng.integers(0, 45, n), rng.integers(0, 15, n)
    alpha, beta = rng.normal(10, 3, 45)[cat_d], rng.normal(300, 50, 15)[cat_k]
    a = alpha[:, None] + rng.normal(0, 1, (n, T))
    b = beta[:, None] + rng.normal(0, 5, (n, T))
    c = 0.05 * a ** 2 + 0.25 * a + 0.5 * b
    c = c * (1 + rng.normal(0, noise_mult, (n, T))) + rng.normal(0, noise_add, (n, T))
    shift = rng.integers(-15, 16, 6)[cat_k % 6]
    s = np.sin(2 * np.pi * (t[None, :] + shift[:, None]) / 30)
    gamma, sig = rng.uniform(-0.02, 0.02, n), rng.uniform(0, 0.15, n)
    tau = rng.normal(t[None, :] * gamma[:, None], sig[:, None])
    return (0.15 * tau + 0.25 * s + 1) * c, cat_d, cat_k, a


def _discount_step(j, w, lam, sharpness):
    """The appendix's probabilistic coverage rule, with a sharpness exponent (1 = verbatim)."""
    up = (w > 1) & (lam > w ** -sharpness)
    down = (w < 1) & (lam > w ** sharpness)
    return np.clip(j + up.astype(int) - down.astype(int), 0, 5)


def generate(n_items=4467, n_candidates=18000, seed=0, sharpness=1.0, noise_mult=0.0, noise_add=0.0):
    """sharpness: exponent on the policy's move probability. 1.0 = the appendix rule
    P(up)=1-1/w; larger values approach the main text's deterministic 'adjust every week' rule."""
    rng = np.random.default_rng(seed)
    n = n_candidates
    q_b, cat_d, cat_k, a = _base(rng, n, noise_mult, noise_add)
    e_b = np.maximum(1.3, rng.lognormal(0.75, 0.125, n))
    e = -e_b * 0.15 * a.mean(1)
    q_bar = q_b.mean(1)
    p0 = np.maximum(np.abs(rng.normal(q_bar / 3, q_bar / 1.5)), 1.0)
    z0 = (q_b + 0.86 * p0[:, None] * e[:, None]).sum(1)
    q, d, stock = np.zeros((n, T)), np.zeros((n, T)), np.zeros((n, T))
    j, z = np.zeros(n, dtype=int), z0.copy()
    for wk in range(T):
        if wk >= 4:
            w = 4 * z / np.maximum(q[:, wk - 4:wk].sum(1), 1e-6) / (T - wk)
            j = _discount_step(j, w, rng.uniform(0, 1, n), sharpness)
        d[:, wk], stock[:, wk] = 0.1 * j, z
        q[:, wk] = q_b[:, wk] + p0 * (1 - d[:, wk]) * e
        z = z - q[:, wk]
    q_min = q_b + p0[:, None] * e[:, None]
    keep = np.where((q_min.min(1) >= 0) & (q_b.min(1) > 0))[0][:n_items]
    return pd.DataFrame({
        "item": np.repeat(np.arange(len(keep)), T), "week": np.tile(np.arange(T), len(keep)),
        "demand": np.round(q[keep]).clip(0).ravel(), "discount": d[keep].ravel(),
        "stock": np.maximum(stock[keep], 0).ravel(), "base_demand": q_b[keep].ravel(),
        "cat_d": np.repeat(cat_d[keep], T), "cat_k": np.repeat(cat_k[keep], T),
        "p0": np.repeat(p0[keep], T), "effect": np.repeat(e[keep], T)})


def counterfactual(df, discount):
    """True demand of `generate` if every week in df had `discount`."""
    return (df["base_demand"] + df["p0"] * (1 - discount) * df["effect"]).clip(lower=0)


def generate_multi(n_items=4467, n_candidates=9000, seed=0, sharpness=10.0, noise_mult=0.1, promo_premium=0.5,
                   gamma=0.5, kappa_frac=0.15):
    rng = np.random.default_rng(seed)
    n = n_candidates
    q_b, cat_d, cat_k, _ = _base(rng, n, noise_mult, 0.0)
    q_b = np.clip(q_b, 1.0, None)
    eps_p = rng.uniform(-2.0, -0.8, n)                 # list-price elasticity
    eps_d = eps_p * (1 + promo_premium)                # discounts bite harder: the hypothesis
    p0 = np.maximum(np.abs(rng.normal(q_b.mean(1) / 3, q_b.mean(1) / 1.5)), 1.0)
    z0 = 0.9 * q_b.sum(1)                              # planned to sell out at a mild average discount
    kappa = kappa_frac * z0                            # availability threshold
    avail = lambda s: np.minimum(1.0, ((1 + s) / (1 + kappa)) ** gamma)
    q, d, lp, stock = (np.zeros((n, T)) for _ in range(4))
    j, z, price = np.zeros(n, dtype=int), z0.copy(), p0.copy()
    plan = np.cumsum(q_b, 1) * 0.9                     # planned cumulative sales
    for wk in range(T):
        if wk >= 4:
            w = 4 * z / np.maximum(q[:, wk - 4:wk].sum(1), 1e-6) / (T - wk)
            j = _discount_step(j, w, rng.uniform(0, 1, n), sharpness)
            # list-price policy: ahead of plan -> raise, behind -> cut (a slow, demand-driven confounder)
            sold = q[:, :wk].sum(1) / np.maximum(plan[:, wk - 1], 1e-6)
            move = rng.uniform(0, 1, n) < 0.15
            price = price * np.where(move & (sold > 1.1), 1.05, 1.0) * np.where(move & (sold < 0.9), 0.95, 1.0)
            price = np.clip(price, 0.7 * p0, 1.3 * p0)
            # replenishment: selling fast -> restock (stock is a treatment, and confounded with demand)
            z = z + np.where((w < 0.7) & (rng.uniform(0, 1, n) < 0.3), 0.2 * z0, 0.0)
        d[:, wk], lp[:, wk], stock[:, wk] = 0.1 * j, np.log(price / p0), z
        q[:, wk] = q_b[:, wk] * np.exp(eps_p * lp[:, wk]) * (1 - d[:, wk]) ** eps_d * avail(z)
        q[:, wk] = np.minimum(q[:, wk], np.maximum(z, 0))  # cannot sell more than is in stock
        z = np.maximum(z - q[:, wk], 0)
    keep = np.arange(min(n, n_items))
    return pd.DataFrame({
        "item": np.repeat(keep, T), "week": np.tile(np.arange(T), len(keep)),
        "demand": np.round(q[keep]).clip(0).ravel(), "discount": d[keep].ravel(),
        "log_price": lp[keep].ravel(), "stock": stock[keep].ravel(), "base_demand": q_b[keep].ravel(),
        "cat_d": np.repeat(cat_d[keep], T), "cat_k": np.repeat(cat_k[keep], T),
        "p0": np.repeat(p0[keep], T), "eps_p": np.repeat(eps_p[keep], T), "eps_d": np.repeat(eps_d[keep], T),
        "kappa": np.repeat(kappa[keep], T), "gamma": gamma})


def counterfactual_multi(qb, eps_p, eps_d, kappa, gamma, discount, log_price, stock):
    """True demand of `generate_multi` at the given treatments (arrays broadcastable to qb's shape)."""
    a = np.minimum(1.0, ((1 + stock) / (1 + kappa)) ** gamma)
    return qb * np.exp(eps_p * log_price) * (1 - discount) ** eps_d * a


def true_effects_multi(eps_p, eps_d, kappa, gamma, stock):
    """Effect-space coefficients per window: on log(1-d), on log price, and the local slope on
    log(1+stock), which is gamma below the availability threshold and 0 above it."""
    slope = gamma * (stock < kappa[:, None]).mean(1)
    return np.stack([eps_d, eps_p, slope], -1)


if __name__ == "__main__":
    df = generate()
    n = df["item"].nunique()
    assert n >= 4000, n
    assert (df["demand"] >= 0).all() and df["discount"].between(0, 0.5).all()
    print(f"generate: {n} items, mean demand {df.demand.mean():.1f}, mean discount {df.discount.mean():.3f}")
    m = generate_multi(n_items=2000, n_candidates=2000)
    print(f"generate_multi: mean demand {m.demand.mean():.1f}, discount {m.discount.mean():.3f}, "
          f"log price sd {m.log_price.std():.3f}, share weeks below availability threshold {(m.stock < m.kappa).mean():.2f}")
    piv = lambda c: m.pivot(index="item", columns="week", values=c).to_numpy()
    print("within-item corr(discount, log price) %.2f, corr(discount, log stock) %.2f" % (
        np.corrcoef((piv("discount") - piv("discount").mean(1, keepdims=True)).ravel(), (piv("log_price") - piv("log_price").mean(1, keepdims=True)).ravel())[0, 1],
        np.corrcoef((piv("discount") - piv("discount").mean(1, keepdims=True)).ravel(), (np.log1p(piv("stock")) - np.log1p(piv("stock")).mean(1, keepdims=True)).ravel())[0, 1]))
