"""DML Forecaster (arXiv:2312.15282v2) plus the TF / sDML baselines from the paper.

Three sub-models, all the same small transformer:
  outcome   q~ = f(z)   (softplus, scaled by past demand)   -- never sees future discount
  treatment d~ = m(z)   (linear)                             -- never sees future discount
  effect    psi(z)      (pooled scalar)
Heads:
  'mult' (real data):  q^ = q~ * ((1-d)/(1-d~))^psi,  psi = -softplus   (eq. effecthead)
  'add'  (synthetic):  q^ = q~ + psi * (d - d~),      psi =  softplus   (eq. simeffecthead)
Two-fold cross-fitting on item parity; inference averages the cross-fit and the
own-fold prediction (geometric mean for 'mult', arithmetic for 'add').
"""
import numpy as np
import torch
import torch.nn as nn

from common import DEV, SeqNet, Windows, head, loss_fn, make_windows, outcome_act, predict, train

# ----------------------------------------------------------------------------- forecasters
class Forecaster:
    """kind in {'dml', 'dml-nocf', 'sdml', 'tf'}; head in {'mult', 'add'}."""

    def __init__(self, kind="dml", head="mult", nuisance_loss="l1", epochs=10, effect_epochs=10, lr=1e-3,
                 seed=0, net=None, log=print):
        self.kind, self.head, self.nloss = kind, head, loss_fn(nuisance_loss)
        self.epochs, self.effect_epochs, self.lr, self.seed, self.log = epochs, effect_epochs, lr, seed, log
        self.net = net or {}

    def _new(self, W, out, extra_fut=0):
        return SeqNet(W.past.shape[-1], W.fut.shape[-1] + extra_fut, self.n_cat, W.static_num.shape[-1],
                      W.past.shape[1], W.fut.shape[1], out, **self.net).to(DEV)

    def _tf_forward(self, b):
        """Naive S-learner: future discount is a plain input feature (eq. current_estimand); head adds psi*d."""
        b = dict(b, fut=torch.cat([b["fut"], b["d_fut"][..., None]], -1))
        qt = outcome_act(self.q_net(b), b["scale"])
        return head(self.head, qt, torch.zeros_like(b["d_fut"]), b["d_fut"], self.psi_net(b), b["scale"])

    # nuisance steps
    def _q_step(self, m, b):
        return self.nloss(outcome_act(m(b), b["scale"]), b["y"])

    def _d_step(self, m, b):
        return self.nloss(m(b), b["d_fut"])

    def _q_pred(self, m, b):
        return outcome_act(m(b), b["scale"])

    def fit(self, W, n_cat):
        self.n_cat = n_cat
        L = self.log
        if self.kind == "tf":
            self.q_net, self.psi_net = self._new(W, "seq", extra_fut=1), self._new(W, "scalar", extra_fut=1)
            both = nn.ModuleList([self.q_net, self.psi_net])

            def step(_, b):
                pred, _ = self._tf_forward(b)
                return self.nloss(pred, b["y"])
            L("TF: end-to-end"); train(both, W, step, self.epochs, self.lr, seed=self.seed, log=L)
            return self

        # folds: cross-fitting on item parity, or a single fold for -nocf
        self.n_folds = 1 if self.kind == "dml-nocf" else 2
        self.q_nets, self.d_nets = [], []
        for k, mask in enumerate(self._folds(W)):
            Wk = W.subset(np.where(mask)[0])
            L(f"outcome model fold {k}"); self.q_nets.append(train(self._new(W, "seq"), Wk, self._q_step, self.epochs, self.lr, seed=self.seed + k, log=L))
            if self.kind != "sdml":
                L(f"treatment model fold {k}"); self.d_nets.append(train(self._new(W, "seq"), Wk, self._d_step, self.epochs, self.lr, seed=self.seed + 10 + k, log=L))

        return self.fit_effect(W)

    def fit_effect(self, W, loss="l1", epochs=None):
        """Stage 2: cross-fitted nuisance predictions -> single effect model psi(z)."""
        q_t, d_t = self._nuisance(W, cross=True)
        self.psi_net = self._new(W, "scalar")
        q_t_t, d_t_t = torch.as_tensor(q_t, device=DEV), torch.as_tensor(d_t, device=DEV)

        class Wp(Windows):  # windows carrying their nuisance predictions
            def tensors(s, idx):
                b = super().tensors(idx)
                b["q_t"], b["d_t"] = q_t_t[idx], d_t_t[idx]
                return b
        Wpp = Wp(W.past, W.fut, W.d_fut, W.y, W.static_cat, W.static_num, W.scale, W.item)
        lf = loss_fn(loss)

        def step(m, b):
            pred, _ = head(self.head, b["q_t"], b["d_t"], b["d_fut"], m(b), b["scale"])
            return lf(pred, b["y"])
        self.log("effect model"); train(self.psi_net, Wpp, step, epochs or self.effect_epochs, self.lr, seed=self.seed + 20, log=self.log)
        return self

    def _folds(self, W):
        return [np.ones(len(W), bool)] if self.n_folds == 1 else [W.item % 2 == 0, W.item % 2 == 1]

    def _nuisance(self, W, cross):
        """Outcome/treatment predictions. cross=True: even items <- odd-trained nets (cross-fitting)."""
        q_t, d_t = np.zeros_like(W.y), np.zeros_like(W.d_fut)
        for k, mask in enumerate(self._folds(W)):
            j = (self.n_folds - 1 - k) if cross else k
            idx = np.where(mask)[0]
            Wk = W.subset(idx)
            q_t[idx] = predict(self.q_nets[j], Wk, self._q_pred)
            if self.d_nets:
                d_t[idx] = predict(self.d_nets[j], Wk, lambda m, b: m(b))
        return q_t, d_t

    def predict(self, W):
        """Returns (demand forecast (N,H), effect psi (N,))."""
        if self.kind == "tf":
            pred = predict(self.q_net, W, lambda m, b: self._tf_forward(b)[0])
            psi = predict(self.psi_net, W, lambda m, b: self._tf_forward(b)[1])
            return pred, psi
        d, scale = torch.as_tensor(W.d_fut, device=DEV), torch.as_tensor(W.scale, device=DEV)
        psi_raw = torch.as_tensor(predict(self.psi_net, W, lambda m, b: m(b)), device=DEV)
        preds = []
        for cross in ([False] if self.kind == "dml-nocf" else [True, False]):
            q_t, d_t = self._nuisance(W, cross)
            p, psi = head(self.head, torch.as_tensor(q_t, device=DEV), torch.as_tensor(d_t, device=DEV), d, psi_raw, scale)
            preds.append(p)
        if self.head == "mult":  # geometric mean (eq. ensemble)
            out = torch.exp(sum(torch.log(p.clamp(min=1e-6)) for p in preds) / len(preds))
        else:
            out = sum(preds) / len(preds)
        return out.cpu().numpy(), psi.cpu().numpy()


if __name__ == "__main__":
    # self-check: linear-in-discount toy data with a confounded policy; DML must recover the effect
    rng = np.random.default_rng(0)
    n, T, C, H = 400, 40, 10, 3
    season = np.sin(2 * np.pi * np.arange(T) / 20)[None, :]
    base = rng.uniform(20, 60, (n, 1)) * (1 + 0.5 * season)
    d = np.clip(0.3 - 0.25 * season + rng.normal(0, 0.05, (n, T)), 0, 0.5)  # discount high when base low
    eff = rng.uniform(20, 60, n)
    q = base + eff[:, None] * d + rng.normal(0, 2, (n, T))
    W = make_windows(q.astype(np.float32), d.astype(np.float32), [], np.arange(T), np.zeros((n, 1), np.int64),
                     np.zeros((n, 1), np.float32), range(C - 1, T - H, 2), C, H, period=20)
    m = Forecaster("dml", "add", "l2", epochs=20, effect_epochs=20, log=lambda s: None).fit(W, [1])
    Wt = make_windows(q.astype(np.float32), d.astype(np.float32), [], np.arange(T), np.zeros((n, 1), np.int64),
                      np.zeros((n, 1), np.float32), [T - H - 1], C, H, period=20)
    _, psi = m.predict(Wt)
    err = np.abs(psi - eff[Wt.item]).mean()
    print("effect MAE", err, "(mean effect", eff.mean(), ")")
    assert err < 0.5 * eff.mean(), err
