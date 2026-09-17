"""LightGBM learners on the same windows as the transformers.

LGBMSLearner   direct multi-horizon S-learner: one regressor per horizon step, the horizon discounts
               as plain inputs. The tree baseline paper 2 benchmarks against, in the
               non-autoregressive reading that matches both papers' horizon-independent decoders.
LGBMNuisance   outcome and treatment regressors on z (no horizon discount) for one DML fold, either
               direct (one per step) or autoregressive (one-step models rolled over the horizon,
               feeding back their own predicted demand and discount).
LGBMEffect     psi by weighted least squares of the outcome residual on the treatment residual
               (R-learner form), with the log-ratio version for the multiplicative head.

Every fit early-stops on a hold-out of 10% of the items against a tree cap, and logs the tree count
it stopped at: a count sitting at the cap is a truncated model, not a converged one.
"""
import logging

import lightgbm as lgb
import numpy as np

from heads import clip_psi, implied_effect

log = logging.getLogger(__name__)
PATIENCE = 50


def _regressor(objective, seed, max_trees, lr=0.05):
    return lgb.LGBMRegressor(objective=objective, n_estimators=max_trees, learning_rate=lr, num_leaves=63,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)


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
    """Features without the horizon discount: statics, flattened context, horizon covariates."""
    n = len(W)
    return np.concatenate([W.static_cat, W.static_num, W.past.reshape(n, -1), W.fut.reshape(n, -1)], 1).astype(np.float32)


def _z_and_d(W, d=None):
    d = W.d_fut if d is None else np.full_like(W.d_fut, d)
    return np.concatenate([_z(W), d], 1)


def _one_step(past, W, h):
    n = len(W)
    return np.concatenate([W.static_cat, W.static_num, past.reshape(n, -1), W.fut[:, h, :]], 1).astype(np.float32)


class LGBMSLearner:
    def __init__(self, head, seed=0, max_trees=2000):
        self.head, self.seed, self.max_trees = head, seed, max_trees

    def fit(self, W):
        X, obj = _z_and_d(W), "l1" if self.head == "mult" else "l2"
        self.models = []
        for h in range(W.y.shape[1]):
            m = fit_es(_regressor(obj, self.seed + h, self.max_trees), X, W.y[:, h] / W.scale, W.item, _cat(W))
            self.models.append(m)
            log.info("lgbm step %d/%d: %d trees", h + 1, W.y.shape[1], m.best_iteration_)
        return self

    def _demand(self, W, d=None):
        X = _z_and_d(W, d)
        return np.clip(np.stack([m.predict(X) for m in self.models], 1) * W.scale[:, None], 0, None)

    def predict(self, W):
        return self._demand(W), implied_effect(self.head, lambda d: self._demand(W, d))


class LGBMNuisance:
    def __init__(self, head, seed=0, max_trees=2000, autoregressive=False, fold=0):
        self.head, self.seed, self.max_trees, self.ar, self.fold = head, seed, max_trees, autoregressive, fold

    def fit(self, W):
        H, cat, obj = W.y.shape[1], _cat(W), "l1" if self.head == "mult" else "l2"
        r = lambda o, s: _regressor(o, s, self.max_trees)
        if self.ar:  # one-step models, trained on step 1 of every window
            X = _one_step(W.past, W, 0)
            self.q = [fit_es(r(obj, self.seed), X, W.y[:, 0] / W.scale, W.item, cat)]
            self.d = [fit_es(r("l2", self.seed + 10), X, W.d_fut[:, 0], W.item, cat)]
        else:
            X = _z(W)
            self.q = [fit_es(r(obj, self.seed + h), X, W.y[:, h] / W.scale, W.item, cat) for h in range(H)]
            self.d = [fit_es(r("l2", self.seed + 10 + h), X, W.d_fut[:, h], W.item, cat) for h in range(H)]
        log.info("nuisance fold %d: outcome %s treatment %s trees", self.fold,
                 [m.best_iteration_ for m in self.q], [m.best_iteration_ for m in self.d])
        return self

    def predict(self, W):
        if not self.ar:
            X = _z(W)
            q_t = np.clip(np.stack([m.predict(X) for m in self.q], 1) * W.scale[:, None], 0, None)
            return q_t, np.stack([m.predict(X) for m in self.d], 1)
        q_t, d_t, past = np.zeros_like(W.y), np.zeros_like(W.d_fut), W.past.copy()
        for h in range(W.y.shape[1]):
            X = _one_step(past, W, h)
            # clamp the fed-back level: unbounded recursion compounds into overflow on sparse series
            q_t[:, h] = np.clip(self.q[0].predict(X) * W.scale, 0, 20 * W.scale)
            d_t[:, h] = np.clip(self.d[0].predict(X), -0.5, 0.95)
            tok = past[:, -1, :].copy()  # historical exogenous held at their last value
            tok[:, 0], tok[:, 1], tok[:, -W.fut.shape[-1]:] = np.log1p(q_t[:, h]), d_t[:, h], W.fut[:, h, :]
            past = np.concatenate([past[:, 1:], tok[:, None, :]], 1)
        return q_t, d_t


class LGBMEffect:
    def __init__(self, head, seed=0, max_trees=2000):
        self.head, self.seed, self.max_trees = head, seed, max_trees

    def fit(self, W, q_t, d_t):
        if self.head == "add":
            rq, rd = W.y - q_t, W.d_fut - d_t
        else:
            rq, rd = np.log((W.y + 1) / (q_t + 1)), np.log((1 - W.d_fut) / np.clip(1 - d_t, 0.05, None))
        ok = np.abs(rd) > 1e-6
        target = np.where(ok, rq / np.where(ok, rd, 1.0), 0.0)
        Z = np.repeat(_z(W), W.y.shape[1], 0)  # one row per (window, step); psi is per window
        self.model = fit_es(_regressor("l2", self.seed + 20, self.max_trees), Z, target.ravel(),
                            np.repeat(W.item, W.y.shape[1]), _cat(W), (rd ** 2).ravel())
        log.info("effect model: %d trees", self.model.best_iteration_)
        return self

    def predict(self, W):
        return clip_psi(self.head, self.model.predict(_z(W)))
