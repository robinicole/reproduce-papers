"""Two-way fixed-effects Poisson (PPML) price elasticity: paper 1's econometric baseline.

    log E[q_it] = eps * log(price_it) + u_i + c_t

Identified from within-item price variation net of common week shocks. Poisson rather than log-log
OLS because M5 demand is a count with many zeros, which dropping or offsetting would bias.

Estimated by block coordinate ascent on the Poisson likelihood: the fixed effects have closed-form
updates given eps (row and column log-ratios of observed to expected), and eps takes a Newton step
given the fixed effects. The panel is a dense (item, week) grid, so each update is a row or column
sum.

This is a reference value, not ground truth. It assumes price changes are as good as random once
item and week effects are absorbed. Promotions timed to expected demand violate that, and the bias
runs toward zero, so treat it as a lower bound on the magnitude.
"""
import numpy as np


def ppml(q, logp, mask, iters=200, tol=1e-8):
    """q, logp, mask: (n_items, n_weeks). Returns (elasticity, iterations_used)."""
    q, logp = q * mask, logp * mask
    row_y, col_y = q.sum(1), q.sum(0)
    a, g, b = np.zeros(q.shape[0]), np.zeros(q.shape[1]), 0.0
    for it in range(iters):
        for _ in range(2):  # fixed effects: closed form given b
            e = np.exp(b * logp + g[None, :]) * mask
            a = np.log(np.clip(row_y, 1e-12, None) / np.clip(e.sum(1), 1e-12, None))
            e = np.exp(b * logp + a[:, None]) * mask
            g = np.log(np.clip(col_y, 1e-12, None) / np.clip(e.sum(0), 1e-12, None))
        mu = np.exp(b * logp + a[:, None] + g[None, :]) * mask
        score = ((q - mu) * logp).sum()
        hess = (mu * logp ** 2).sum()
        step = score / max(hess, 1e-12)
        b += np.clip(step, -1.0, 1.0)
        if abs(step) < tol:
            return b, it + 1
    return b, iters


def cluster_se(q, logp, mask, b, a, g):
    """Standard error clustered by item: sandwich with per-item summed scores."""
    mu = np.exp(b * logp + a[:, None] + g[None, :]) * mask
    bread = 1.0 / max((mu * logp ** 2).sum(), 1e-12)
    s_i = ((q * mask - mu) * logp).sum(1)  # score summed within item
    return float(np.sqrt(bread * (s_i ** 2).sum() * bread))


def fit(q, price, mask, iters=200):
    """Convenience wrapper returning elasticity, clustered SE and the fitted effects."""
    logp = np.where(mask, np.log(np.clip(price, 1e-6, None)), 0.0)
    b, used = ppml(q, logp, mask, iters)
    # recover the effects at the solution for the standard error
    a, g = np.zeros(q.shape[0]), np.zeros(q.shape[1])
    row_y, col_y = (q * mask).sum(1), (q * mask).sum(0)
    for _ in range(50):
        e = np.exp(b * logp + g[None, :]) * mask
        a = np.log(np.clip(row_y, 1e-12, None) / np.clip(e.sum(1), 1e-12, None))
        e = np.exp(b * logp + a[:, None]) * mask
        g = np.log(np.clip(col_y, 1e-12, None) / np.clip(e.sum(0), 1e-12, None))
    return b, cluster_se(q * mask, logp, mask, b, a, g), used


if __name__ == "__main__":
    # self-check: recover a known elasticity from a simulated confounded panel
    rng = np.random.default_rng(0)
    n, T, true_eps = 800, 60, -2.0
    item = rng.normal(2.5, 0.5, n)[:, None]
    week = np.sin(2 * np.pi * np.arange(T) / 26)[None, :] * 0.4
    # price responds to the week effect (confounding) plus item-level noise
    logp = 1.0 + 0.3 * rng.normal(0, 1, (n, 1)) - 0.5 * week + rng.normal(0, 0.15, (n, T))
    mu = np.exp(item + week + true_eps * logp)
    q = rng.poisson(mu).astype(np.float64)
    mask = np.ones_like(q, bool)
    eps, se, used = fit(q, np.exp(logp), mask)
    print(f"true {true_eps}, PPML {eps:.3f} (se {se:.3f}) in {used} iterations")
    assert abs(eps - true_eps) < 0.1, eps
    # naive pooled regression ignoring the fixed effects should be visibly biased
    x, y = logp.ravel(), np.log1p(q.ravel())
    naive = np.polyfit(x, y, 1)[0]
    print(f"naive pooled log-log slope {naive:.3f} (biased by the confounding)")
    assert abs(naive - true_eps) > abs(eps - true_eps), "fixed effects should help"
