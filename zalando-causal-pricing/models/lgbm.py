"""LightGBM learners on the same windows as the transformers, for a vector of treatments.

LGBMSLearner   direct multi-horizon S-learner: one regressor per horizon step, the horizon treatments
               as plain inputs. The tree baseline paper 2 benchmarks against, in the
               non-autoregressive reading that matches both papers' horizon-independent decoders.
LGBMNuisance   outcome and treatment regressors on z (no horizon treatments) for one DML fold, either
               direct (one per step and treatment) or autoregressive (one-step models rolled over
               the horizon, feeding back their own predicted demand and treatments).
LGBMEffect     psi by weighted least squares of the outcome residual on the treatment residuals
               (R-learner form, backfitted over treatments), in the head's effect space.

Every fit early-stops on a hold-out of 10% of the items against a tree cap, and logs the tree count
it stopped at: a count sitting at the cap is a truncated model, not a converged one.
"""
import logging

import lightgbm as lgb
import numpy as np

from heads import clip_psi, implied_effect, link

log = logging.getLogger(__name__)
PATIENCE = 50


def _regressor(objective, seed, max_trees, lr=0.05):
    # deterministic + a fixed histogram mode: by default LightGBM picks row- or column-wise
    # histograms by timing them, so the same fit could differ between runs of one process.
    return lgb.LGBMRegressor(objective=objective, n_estimators=max_trees, learning_rate=lr, num_leaves=63,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
                             deterministic=True, force_col_wise=True)


def fit_es(reg, X, y, item, cat, weight=None):
    """Fit with early stopping on a hold-out of 10% of the items, chosen by a rule independent of
    the DML parity folds so no nuisance model is stopped on its own cross-fitting partner."""
    val = ((item // 2) % 10) == 0
    reg.fit(X[~val], y[~val], sample_weight=None if weight is None else weight[~val], categorical_feature=cat,
            eval_set=[(X[val], y[val])], eval_sample_weight=None if weight is None else [weight[val]],
            callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return reg


def _cat(W):
    return list(range(W.static_cat.shape[1]))


def _z(W):
    """Features without the horizon treatments: statics, flattened context, horizon covariates."""
    n = len(W)
    return np.concatenate([W.static_cat, W.static_num, W.past.reshape(n, -1), W.fut.reshape(n, -1)], 1).astype(np.float32)


def _z_and_d(W):
    return np.concatenate([_z(W), W.d_fut.reshape(len(W), -1)], 1)


def _one_step(past, W, h):
    n = len(W)
    return np.concatenate([W.static_cat, W.static_num, past.reshape(n, -1), W.fut[:, h, :]], 1).astype(np.float32)


class LGBMSLearner:
    def __init__(self, head, spec, seed=0, max_trees=2000):
        self.head, self.spec, self.seed, self.max_trees = head, spec, seed, max_trees

    def fit(self, W):
        X, obj = _z_and_d(W), "l1" if self.head == "mult" else "l2"
        self.models = []
        for h in range(W.y.shape[1]):
            m = fit_es(_regressor(obj, self.seed + h, self.max_trees), X, W.y[:, h] / W.scale, W.item, _cat(W))
            self.models.append(m)
            log.info("lgbm step %d/%d: %d trees", h + 1, W.y.shape[1], m.best_iteration_)
        return self

    def _demand(self, W):
        X = _z_and_d(W)
        return np.clip(np.stack([m.predict(X) for m in self.models], 1) * W.scale[:, None], 0, None)

    def predict(self, W):
        return self._demand(W), implied_effect(self.head, self.spec, W, self._demand)


class LGBMNuisance:
    def __init__(self, head, spec, seed=0, max_trees=2000, autoregressive=False, fold=0):
        self.head, self.K, self.seed, self.max_trees, self.ar, self.fold = head, len(spec), seed, max_trees, autoregressive, fold

    def fit(self, W):
        H, cat, obj = W.y.shape[1], _cat(W), "l1" if self.head == "mult" else "l2"
        r = lambda o, s: _regressor(o, s, self.max_trees)
        if self.ar:  # one-step models, trained on step 1 of every window
            X = _one_step(W.past, W, 0)
            self.q = [fit_es(r(obj, self.seed), X, W.y[:, 0] / W.scale, W.item, cat)]
            self.d = [[fit_es(r("l2", self.seed + 10 + 100 * k), X, W.d_fut[:, 0, k], W.item, cat)] for k in range(self.K)]
        else:
            X = _z(W)
            self.q = [fit_es(r(obj, self.seed + h), X, W.y[:, h] / W.scale, W.item, cat) for h in range(H)]
            self.d = [[fit_es(r("l2", self.seed + 10 + h + 100 * k), X, W.d_fut[:, h, k], W.item, cat) for h in range(H)]
                      for k in range(self.K)]
        log.info("nuisance fold %d: outcome %s treatment %s trees", self.fold,
                 [m.best_iteration_ for m in self.q], [[m.best_iteration_ for m in dk] for dk in self.d])
        return self

    def predict(self, W):
        if not self.ar:
            X = _z(W)
            q_t = np.clip(np.stack([m.predict(X) for m in self.q], 1) * W.scale[:, None], 0, None)
            return q_t, np.stack([np.stack([m.predict(X) for m in dk], 1) for dk in self.d], -1)
        q_t, d_t, past = np.zeros_like(W.y), np.zeros_like(W.d_fut), W.past.copy()
        for h in range(W.y.shape[1]):
            X = _one_step(past, W, h)
            # clamp the fed-back level: unbounded recursion compounds into overflow on sparse series
            q_t[:, h] = np.clip(self.q[0].predict(X) * W.scale, 0, 20 * W.scale)
            d_t[:, h] = np.stack([dk[0].predict(X) for dk in self.d], -1)
            d_t[:, h, 0] = np.clip(d_t[:, h, 0], -0.5, 0.95)  # the first treatment is the discount fraction
            tok = past[:, -1, :].copy()  # historical exogenous held at their last value
            tok[:, 0], tok[:, 1:1 + self.K], tok[:, -W.fut.shape[-1]:] = np.log1p(q_t[:, h]), d_t[:, h], W.fut[:, h, :]
            past = np.concatenate([past[:, 1:], tok[:, None, :]], 1)
        return q_t, d_t


class LGBMEffect:
    """psi(z) per treatment by weighted regression of the outcome residual on that treatment's
    residual, holding the others' fitted contributions fixed (backfitting; one round when K = 1)."""

    def __init__(self, head, spec, seed=0, max_trees=2000, rounds=3):
        self.head, self.spec, self.seed, self.max_trees, self.rounds = head, spec, seed, max_trees, rounds

    def fit(self, W, q_t, d_t):
        H, K = W.y.shape[1], len(self.spec)
        rq = (W.y - q_t) if self.head == "add" else np.log((W.y + 1) / (q_t + 1))
        rd = link(self.head, self.spec, W.d_fut) - link(self.head, self.spec, d_t)          # (N, H, K)
        Z = np.repeat(_z(W), H, 0)                                                            # one row per (window, step)
        item = np.repeat(W.item, H)
        rq, rd = rq.reshape(-1), rd.reshape(-1, K)
        self.models, contrib = [None] * K, np.zeros((len(rq), K))
        for _ in range(self.rounds if K > 1 else 1):
            for k in range(K):
                partial = rq - contrib.sum(1) + contrib[:, k]
                ok = np.abs(rd[:, k]) > 1e-6
                target = np.where(ok, partial / np.where(ok, rd[:, k], 1.0), 0.0)
                self.models[k] = fit_es(_regressor("l2", self.seed + 20 + k, self.max_trees), Z, target, item, _cat(W), rd[:, k] ** 2)
                contrib[:, k] = self.models[k].predict(Z) * rd[:, k]
        log.info("effect model: %s trees", [m.best_iteration_ for m in self.models])
        return self

    def predict(self, W):
        Z = _z(W)
        return clip_psi(self.head, self.spec, np.stack([m.predict(Z) for m in self.models], -1))
