"""Direct multi-horizon LightGBM baseline on the same windows as the transformers.

One regressor per horizon step h, features = flattened context (demand, discount, covariates, week),
static categories and numerics, horizon week features, and the horizon discounts (future discount is a plain
input: an S-learner, like paper 2's model but without the monotone head). Target = demand / recent scale.
Paper 2 benchmarks against "a tree-based model using LightGBM" without stating its design; this is the
non-autoregressive reading, matching both papers' horizon-independent decoders.
"""
import os

import numpy as np
import lightgbm as lgb


# Cap, not a target: check best_iteration_ in the logs. On the synthetic data every fit early-stops
# between 200 and 700 trees; on M5's 1.7M sparse windows a 2000 cap was still binding, so raise it there.
MAX_TREES, PATIENCE = int(os.environ.get("LGBM_MAX_TREES", 2000)), 50


def fit_es(reg, X, y, item, cat, weight=None):
    """Fit with early stopping on a hold-out of 10% of the items; returns the regressor.
    reg.best_iteration_ is the convergence record: well below MAX_TREES means converged."""
    val = ((item // 2) % 10) == 0  # independent of the parity folds
    reg.fit(X[~val], y[~val], sample_weight=None if weight is None else weight[~val], categorical_feature=cat,
            eval_set=[(X[val], y[val])], eval_sample_weight=None if weight is None else [weight[val]],
            callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return reg


class LGBMForecaster:
    """Same interface as dml.Forecaster: fit(W, n_cat), predict(W) -> (demand (N,H), effect (N,)).
    After fit, .best_iters lists the early-stopped tree count of every regressor."""

    def __init__(self, head="mult", n_estimators=MAX_TREES, lr=0.05, seed=0, log=print, **_):
        self.head, self.n, self.lr, self.seed, self.log = head, n_estimators, lr, seed, log

    def _X(self, W, d=None):
        d = W.d_fut if d is None else np.full_like(W.d_fut, d)
        n = len(W)
        return np.concatenate([W.static_cat, W.static_num, W.past.reshape(n, -1), W.fut.reshape(n, -1), d], 1).astype(np.float32)

    def fit(self, W, n_cat):
        X, cat = self._X(W), list(range(W.static_cat.shape[1]))
        self.models, self.best_iters = [], []
        for h in range(W.y.shape[1]):
            m = lgb.LGBMRegressor(objective="l1" if self.head == "mult" else "l2", n_estimators=self.n, learning_rate=self.lr,
                                  num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=self.seed + h, verbose=-1)
            self.models.append(fit_es(m, X, W.y[:, h] / W.scale, W.item, cat))
            self.best_iters.append(m.best_iteration_)
            self.log(f"lgbm step {h + 1}/{W.y.shape[1]}: {m.best_iteration_} trees")
        return self

    def _demand(self, W, d=None):
        X = self._X(W, d)
        return np.clip(np.stack([m.predict(X) for m in self.models], 1) * W.scale[:, None], 0, None)

    def predict(self, W):
        pred = self._demand(W)
        if self.head == "add":   # dq/dd over the off-policy range
            eff = (self._demand(W, 0.5) - self._demand(W, 0.0)).mean(1) / 0.5
        else:                    # elasticity dlog q / dlog(1-d) near full price
            q0, q1 = self._demand(W, 0.0) + 1e-3, self._demand(W, 0.1) + 1e-3
            eff = (np.log(q1) - np.log(q0)).mean(1) / np.log(0.9)
        return pred, eff


class DMLLGBMForecaster:
    """Paper 1's DML layout with LightGBM everywhere: per-step outcome and treatment regressors on z (no
    horizon discount), two-fold cross-fitting on item parity, and an effect regressor fitted by weighted
    least squares of the outcome residual on the treatment residual (R-learner form):
      'add':  rq = psi(z) * rd            -> target rq/rd, weight rd^2
      'mult': log(q/q~) = psi(z) * log((1-d)/(1-d~))  -> same with the log ratios, psi clipped <= 0
    Inference averages the cross-fit and own-fold paths like dml.Forecaster.
    ar=True: autoregressive nuisances. One one-step outcome and one one-step treatment regressor, rolled
    over the horizon by appending their own predicted (log demand, discount) to the context window."""

    def __init__(self, head="mult", n_estimators=MAX_TREES, lr=0.05, seed=0, log=print, ar=False, **_):
        self.head, self.n, self.lr, self.seed, self.log, self.ar = head, n_estimators, lr, seed, log, ar
        self.best_iters = {}

    def _X1(self, past, W, h):  # one-step features: context + covariates of the next step
        n = len(W)
        return np.concatenate([W.static_cat, W.static_num, past.reshape(n, -1), W.fut[:, h, :]], 1).astype(np.float32)

    def _Xz(self, W):
        n = len(W)
        return np.concatenate([W.static_cat, W.static_num, W.past.reshape(n, -1), W.fut.reshape(n, -1)], 1).astype(np.float32)

    def _reg(self, obj, seed):
        return lgb.LGBMRegressor(objective=obj, n_estimators=self.n, learning_rate=self.lr, num_leaves=63, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)

    def fit(self, W, n_cat):
        H, cat = W.y.shape[1], list(range(W.static_cat.shape[1]))
        self.folds = [W.item % 2 == 0, W.item % 2 == 1]
        self.q, self.d = [], []
        qobj = "l1" if self.head == "mult" else "l2"
        for k, mask in enumerate(self.folds):
            Wk = W.subset(np.where(mask)[0]); y, d, sc = Wk.y, Wk.d_fut, Wk.scale
            if self.ar:  # one-step models on step 1 of every window
                X = self._X1(Wk.past, Wk, 0)
                self.q.append([fit_es(self._reg(qobj, self.seed), X, y[:, 0] / sc, Wk.item, cat)])
                self.d.append([fit_es(self._reg("l2", self.seed + 10), X, d[:, 0], Wk.item, cat)])
            else:
                X = self._Xz(Wk)
                self.q.append([fit_es(self._reg(qobj, self.seed + h), X, y[:, h] / sc, Wk.item, cat) for h in range(H)])
                self.d.append([fit_es(self._reg("l2", self.seed + 10 + h), X, d[:, h], Wk.item, cat) for h in range(H)])
            self.best_iters[f"outcome fold {k}"] = [m.best_iteration_ for m in self.q[k]]
            self.best_iters[f"treatment fold {k}"] = [m.best_iteration_ for m in self.d[k]]
            self.log(f"nuisance fold {k}: outcome {self.best_iters[f'outcome fold {k}']} treatment {self.best_iters[f'treatment fold {k}']} trees")
        q_t, d_t = self._nuisance(W, cross=True)
        if self.head == "add":
            rq, rd = W.y - q_t, W.d_fut - d_t
        else:
            rq, rd = np.log((W.y + 1) / (q_t + 1)), np.log((1 - W.d_fut) / np.clip(1 - d_t, 0.05, None))
        w = rd ** 2
        target = np.where(np.abs(rd) > 1e-6, rq / np.where(np.abs(rd) > 1e-6, rd, 1), 0.0)
        Z = np.repeat(self._Xz(W), W.y.shape[1], 0)  # one row per (window, step); psi is per window
        self.psi = fit_es(self._reg("l2", self.seed + 20), Z, target.ravel(), np.repeat(W.item, W.y.shape[1]), cat, w.ravel())
        self.best_iters["effect"] = self.psi.best_iteration_
        self.log(f"effect model: {self.psi.best_iteration_} trees")
        return self

    def _nuisance(self, W, cross):
        q_t, d_t = np.zeros_like(W.y), np.zeros_like(W.d_fut)
        for k, mask in enumerate([W.item % 2 == 0, W.item % 2 == 1]):
            j = (1 - k) if cross else k
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            Wk = W.subset(idx)
            if self.ar:
                past = Wk.past.copy()
                for h in range(W.y.shape[1]):
                    X = self._X1(past, Wk, h)
                    # clamp the fed-back level: unbounded recursion compounds into overflow on sparse series
                    qh = np.clip(self.q[j][0].predict(X) * Wk.scale, 0, 20 * Wk.scale)
                    dh = np.clip(self.d[j][0].predict(X), -0.5, 0.95)
                    q_t[idx, h], d_t[idx, h] = qh, dh
                    tok = past[:, -1, :].copy()  # extras (e.g. stock) held at their last value
                    tok[:, 0], tok[:, 1], tok[:, -3:] = np.log1p(qh), dh, Wk.fut[:, h, :]
                    past = np.concatenate([past[:, 1:], tok[:, None, :]], 1)
                continue
            X = self._Xz(Wk)
            q_t[idx] = np.clip(np.stack([m.predict(X) for m in self.q[j]], 1) * W.scale[idx, None], 0, None)
            d_t[idx] = np.stack([m.predict(X) for m in self.d[j]], 1)
        return q_t, d_t

    MAX_LOG_ADJ = 3.0  # price adjustment capped at a factor of exp(3) either way

    def _head(self, q_t, d_t, d, psi):
        if self.head == "add":
            return q_t + psi[:, None] * (d - d_t)
        # exp form so the adjustment can be bounded: a tree-based psi is not smooth like the paper's
        # softplus head, and an unbounded exponent blows up on sparse series (M5 dml-lgbm-ar diverged).
        adj = psi[:, None] * np.log((1 - d) / np.clip(1 - d_t, 0.05, None))
        return q_t * np.exp(np.clip(adj, -self.MAX_LOG_ADJ, self.MAX_LOG_ADJ))

    def predict(self, W):
        psi = self.psi.predict(self._Xz(W))
        # sign as in the paper's activations; elasticity magnitude capped at a plausible retail range
        psi = np.maximum(psi, 0) if self.head == "add" else np.clip(psi, -5.0, 0.0)
        preds = [self._head(*self._nuisance(W, cross), W.d_fut, psi) for cross in (True, False)]
        if self.head == "mult":
            out = np.exp(np.mean([np.log(np.clip(p, 1e-6, None)) for p in preds], 0))
        else:
            out = np.mean(preds, 0)
        return np.clip(out, 0, None), psi


if __name__ == "__main__":
    from common import make_windows
    rng = np.random.default_rng(0)
    n, T, C, H = 400, 40, 10, 3
    season = np.sin(2 * np.pi * np.arange(T) / 20)[None, :]
    base = rng.uniform(20, 60, (n, 1)) * (1 + 0.5 * season)
    d = np.clip(0.3 - 0.25 * season + rng.normal(0, 0.05, (n, T)), 0, 0.5)
    eff = rng.uniform(20, 60, n)
    q = base + eff[:, None] * d + rng.normal(0, 2, (n, T))
    mk = lambda o: make_windows(q.astype(np.float32), d.astype(np.float32), [], np.arange(T), np.zeros((n, 1), np.int64),
                                np.zeros((n, 1), np.float32), o, C, H, period=20)
    m = LGBMForecaster("add", log=lambda s: None).fit(mk(range(C - 1, T - H, 2)), [1])
    Wt = mk([T - H - 1])
    pred, e = m.predict(Wt)
    err = np.abs(e - eff[Wt.item]).mean()
    print("on-policy MAE %.2f, effect MAE %.1f (mean effect %.1f)" % (np.abs(pred - Wt.y).mean(), err, eff.mean()))
    assert np.abs(pred - Wt.y).mean() < 6, "poor on-policy fit"
    assert err < 0.8 * eff.mean(), err  # S-learner on confounded data: attenuated, not degenerate
    m2 = DMLLGBMForecaster("add", log=lambda s: None).fit(mk(range(C - 1, T - H, 2)), [1])
    pred2, e2 = m2.predict(Wt)
    err2 = np.abs(e2 - eff[Wt.item]).mean()
    print("DML-LightGBM: on-policy MAE %.2f, effect MAE %.1f" % (np.abs(pred2 - Wt.y).mean(), err2))
    assert err2 < err, "DML with LightGBM nuisances should beat the S-learner on the effect"
    m3 = DMLLGBMForecaster("add", log=lambda s: None, ar=True).fit(mk(range(C - 1, T - H, 2)), [1])
    pred3, e3 = m3.predict(Wt)
    err3 = np.abs(e3 - eff[Wt.item]).mean()
    print("DML-LightGBM autoregressive: on-policy MAE %.2f, effect MAE %.1f" % (np.abs(pred3 - Wt.y).mean(), err3))
    assert err3 < err, "autoregressive DML-LightGBM should beat the S-learner on the effect"
