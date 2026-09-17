"""Paper 1's DML Forecaster (arXiv:2312.15282v2) as one procedure over pluggable learners.

    DML(head, nuisance, effect, cross_fit)
      nuisance(fold) -> learner with fit(W_fold) and predict(W) -> (q~, d~)
      effect()       -> learner with fit(W, q~, d~) and predict(W) -> psi
    fit:     nuisances on two item-parity folds, cross-fitted residuals, one effect model
    predict: demand head on the cross-fit and own-fold nuisance paths, ensembled (paper 1 eq. ensemble)

Ablations are constructor arguments rather than model kinds: cross_fit=False is "DML no cf", a
nuisance learner with treatment_model=False is sDML. The torch learners below are the paper's
transformers; models/lgbm.py provides tree-based ones. TFForecaster is the paper's naive baseline.
"""
import logging

import numpy as np
import torch
import torch.nn as nn

from heads import activate_psi, demand, ensemble
from nn import loss_fn, outcome_act, predict, seqnet, train

log = logging.getLogger(__name__)


class DML:
    def __init__(self, head, nuisance, effect, cross_fit=True):
        self.head, self.make_nuisance, self.make_effect, self.cross_fit = head, nuisance, effect, cross_fit

    def _folds(self, W):
        return [W.item % 2 == 0, W.item % 2 == 1] if self.cross_fit else [np.ones(len(W), bool)]

    def fit(self, W):
        self.nuisances = []
        for k, mask in enumerate(self._folds(W)):
            log.info("nuisance fold %d", k)
            self.nuisances.append(self.make_nuisance(k).fit(W.subset(np.where(mask)[0])))
        q_t, d_t = self._nuisance(W, cross=True)  # with one fold this is the in-sample path
        log.info("effect model")
        self.effect = self.make_effect().fit(W, q_t, d_t)
        return self

    def _nuisance(self, W, cross):
        """Nuisance predictions for W. cross=True routes each fold through the nuisances that did
        not train on it; that is what the effect model learns from and the 'cf' path at inference."""
        q_t, d_t = np.zeros_like(W.y), np.zeros_like(W.d_fut)
        folds = self._folds(W)
        for k, mask in enumerate(folds):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            j = (len(folds) - 1 - k) if cross else k
            q_t[idx], d_t[idx] = self.nuisances[j].predict(W.subset(idx))
        return q_t, d_t

    def predict(self, W):
        """(demand forecast (N, H), effect psi (N,))"""
        psi = self.effect.predict(W)
        paths = [True, False] if self.cross_fit else [False]
        preds = [demand(self.head, *self._nuisance(W, c), W.d_fut, psi) for c in paths]
        return np.clip(ensemble(self.head, preds), 0, None), psi


# ----------------------------------------------------------------------------- torch learners
class TorchNuisance:
    """Outcome and treatment transformers for one fold. Neither sees the horizon discount."""

    def __init__(self, n_cat, head, loss, epochs, lr, seed, net=None, treatment_model=True, fold=0):
        self.n_cat, self.loss, self.epochs, self.lr = n_cat, loss_fn(loss), epochs, lr
        self.seed, self.net, self.treatment_model, self.fold = seed, net or {}, treatment_model, fold

    _q = staticmethod(lambda m, b: outcome_act(m(b), b["scale"]))

    def fit(self, W):
        L = self.loss
        torch.manual_seed(self.seed + self.fold)  # construction seeded too, so runs reproduce exactly
        self.q_net = train(seqnet(W, self.n_cat, "seq", **self.net), W, lambda m, b: L(self._q(m, b), b["y"]),
                           self.epochs, self.lr, seed=self.seed + self.fold)
        self.d_net = None
        if self.treatment_model:
            torch.manual_seed(self.seed + 10 + self.fold)
            self.d_net = train(seqnet(W, self.n_cat, "seq", **self.net), W, lambda m, b: L(m(b), b["d_fut"]),
                               self.epochs, self.lr, seed=self.seed + 10 + self.fold)
        return self

    def predict(self, W):
        q_t = predict(self.q_net, W, self._q)
        d_t = predict(self.d_net, W, lambda m, b: m(b)) if self.d_net is not None else np.zeros_like(W.d_fut)
        return q_t, d_t


class TorchEffect:
    """One elasticity per window from a transformer, trained through the demand head on the
    cross-fitted nuisance predictions (L1 loss, as in the paper)."""

    def __init__(self, n_cat, head, epochs, lr, seed, net=None, loss="l1"):
        self.n_cat, self.head, self.epochs, self.lr, self.seed, self.net = n_cat, head, epochs, lr, seed, net or {}
        self.loss = loss_fn(loss)

    def _psi(self, m, b):
        return activate_psi(self.head, m(b), b["scale"])

    def fit(self, W, q_t, d_t):
        Wx = W.subset(slice(None))
        Wx.extras = {"q_t": q_t, "d_t": d_t}
        step = lambda m, b: self.loss(demand(self.head, b["q_t"], b["d_t"], b["d_fut"], self._psi(m, b)), b["y"])
        torch.manual_seed(self.seed + 20)
        self.net_ = train(seqnet(W, self.n_cat, "scalar", **self.net), Wx, step, self.epochs, self.lr, seed=self.seed + 20)
        return self

    def predict(self, W):
        return predict(self.net_, W, self._psi)


class TFForecaster:
    """Paper 1's naive baseline: the future discount is an ordinary network input (an S-learner)
    and the head adds psi * d on top with d~ = 0."""

    def __init__(self, n_cat, head, loss, epochs, lr, seed, net=None):
        self.n_cat, self.head, self.loss, self.epochs, self.lr, self.seed = n_cat, head, loss_fn(loss), epochs, lr, seed
        self.net = net or {}

    def _forward(self, b):
        b = dict(b, fut=torch.cat([b["fut"], b["d_fut"][..., None]], -1))
        qt = outcome_act(self.q_net(b), b["scale"])
        psi = activate_psi(self.head, self.psi_net(b), b["scale"])
        return demand(self.head, qt, torch.zeros_like(b["d_fut"]), b["d_fut"], psi), psi

    def fit(self, W):
        torch.manual_seed(self.seed)
        self.q_net = seqnet(W, self.n_cat, "seq", extra_fut=1, **self.net)
        self.psi_net = seqnet(W, self.n_cat, "scalar", extra_fut=1, **self.net)
        log.info("TF: end-to-end")
        train(nn.ModuleList([self.q_net, self.psi_net]), W, lambda _, b: self.loss(self._forward(b)[0], b["y"]),
              self.epochs, self.lr, seed=self.seed)
        return self

    def predict(self, W):
        pred = predict(self.q_net, W, lambda m, b: self._forward(b)[0])
        psi = predict(self.psi_net, W, lambda m, b: self._forward(b)[1])
        return pred, psi
