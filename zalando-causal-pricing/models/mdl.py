"""Zalando's production demand model (arXiv:2305.14406, Kunz et al.): encoder/decoder transformer with a
Monotonic Demand Layer.  This is the 'TF' model that arXiv:2312.15282v2 uses as its baseline.

  encoder  <- past (demand, discount, covariates);  decoder <- future covariates + last observed (d_t, q_t)
  future discount bypasses encoder/decoder and enters the head only:
      log1p q^(d) = q_hat(kappa_T) + sigma(kappa_T) * PL(d; delta(gamma))          (eq. monotonic_demand_response)
  PL is piecewise linear on segments of width 0.1 with slopes delta = softplus(phi0(encoder state)) >= 0,
  sigma = softplus(phi1(decoder state)), so demand is monotonically increasing in discount by construction.
  loss = (v(pred) - v(log1p q))^2 with v(x) = 1 + x + x^2/2 + x^3/6  (eq. loss).
Near/far-future split (5 vs 20 weeks) is not implemented: our horizons are <= 5 weeks (near future only).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEV, predict, train

SEG = 0.1  # discount segment width


class MDLNet(nn.Module):
    def __init__(self, f_past, f_fut, n_cat, n_num, C, H, d_model=64, n_layers=2, n_heads=4, dropout=0.1, n_seg=7):
        super().__init__()
        self.C, self.H, self.n_seg = C, H, n_seg
        self.past_in = nn.Linear(f_past, d_model)
        self.fut_in = nn.Linear(f_fut + 2, d_model)  # + last observed discount and log demand
        self.pos_past = nn.Parameter(torch.randn(C, d_model) * 0.02)
        self.pos_fut = nn.Parameter(torch.randn(H, d_model) * 0.02)
        self.cat_emb = nn.ModuleList([nn.Embedding(n, d_model) for n in n_cat])
        self.num_in = nn.Linear(n_num, d_model) if n_num else None
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True), n_layers)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True), n_layers)
        self.norm_e, self.norm_d = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.phi0 = nn.Linear(d_model, n_seg)  # slopes delta from encoder state
        self.phi1 = nn.Linear(d_model, 2)      # (q_hat, sigma) from decoder state

    def states(self, b):
        s = sum(e(b["static_cat"][:, i]) for i, e in enumerate(self.cat_emb))
        if self.num_in is not None:
            s = s + self.num_in(b["static_num"])
        s = s[:, None, :]
        gamma = self.norm_e(self.enc(self.past_in(b["past"]) + self.pos_past + s))
        last = b["past"][:, -1, :2]  # (log1p demand, discount) at t
        fut = torch.cat([b["fut"], last[:, None, :].expand(-1, self.H, -1)], -1)
        kappa = self.norm_d(self.dec(self.fut_in(fut) + self.pos_fut + s, gamma))  # non-autoregressive: no mask
        delta = F.softplus(self.phi0(gamma.mean(1)))  # (N, n_seg), same for all future weeks
        qs = self.phi1(kappa)                          # (N, H, 2)
        return qs[..., 0], F.softplus(qs[..., 1]), delta

    def forward(self, b, d):
        """log1p-demand prediction at discounts d (N, H)."""
        q_hat, sigma, delta = self.states(b)
        m = (d / SEG).floor().clamp(0, self.n_seg - 1).long()                       # segment index
        cum = torch.cat([torch.zeros_like(delta[:, :1]), delta.cumsum(1)], 1)       # (N, n_seg+1)
        full = cum.gather(1, m)                                                      # completed segments
        partial = (d - m * SEG) / SEG * delta.gather(1, m)                           # fraction of current segment
        return q_hat + sigma * (full + partial)


def v(x):  # third-order Taylor expansion of exp
    return 1 + x + x ** 2 / 2 + x ** 3 / 6


class MDLForecaster:
    """Same interface as dml.Forecaster: fit(W, n_cat), predict(W) -> (demand (N,H), effect (N,))."""

    def __init__(self, head="mult", epochs=10, lr=1e-3, seed=0, net=None, log=print, anchor=False, **_):
        """anchor=True adds log(mean recent demand) to the log-demand output (local scaling, not in the paper):
        the network then predicts a ratio to the recent level instead of an absolute level."""
        self.head, self.epochs, self.lr, self.seed, self.log, self.net = head, epochs, lr, seed, log, net or {}
        self.anchor = anchor

    def fit(self, W, n_cat):
        self.model = MDLNet(W.past.shape[-1], W.fut.shape[-1], n_cat, W.static_num.shape[-1],
                            W.past.shape[1], W.fut.shape[1], **self.net).to(DEV)

        if self.anchor:
            fwd = self.model.forward
            self.model.forward = lambda b, d: fwd(b, d) + torch.log(b["scale"])[:, None]

        def step(m, b):
            return ((v(m(b, b["d_fut"])) - v(torch.log1p(b["y"]))) ** 2).mean()
        self.log("MDL: encoder/decoder + monotonic demand layer")
        train(self.model, W, step, self.epochs, self.lr, seed=self.seed, log=self.log)
        return self

    def _demand(self, W, d=None):
        return predict(self.model, W, lambda m, b: torch.expm1(m(b, b["d_fut"] if d is None else torch.full_like(b["d_fut"], d))).clamp(min=0))

    def predict(self, W):
        pred = self._demand(W)
        if self.head == "add":   # dq/dd, finite difference over the off-policy range
            eff = (self._demand(W, 0.5) - self._demand(W, 0.0)).mean(1) / 0.5
        else:                    # price elasticity dlog q / dlog(1-d) near full price
            q0, q1 = self._demand(W, 0.0) + 1e-3, self._demand(W, 0.1) + 1e-3
            eff = (np.log(q1) - np.log(q0)).mean(1) / np.log(0.9)
        return pred, eff


if __name__ == "__main__":
    from common import Panel, make_windows, week_feats
    rng = np.random.default_rng(0)
    n, T, C, H = 400, 40, 10, 3
    season = np.sin(2 * np.pi * np.arange(T) / 20)[None, :]
    base = rng.uniform(20, 60, (n, 1)) * (1 + 0.5 * season)
    d = np.clip(0.3 - 0.25 * season + rng.normal(0, 0.05, (n, T)), 0, 0.5)
    eff = rng.uniform(20, 60, n)
    q = base + eff[:, None] * d + rng.normal(0, 2, (n, T))
    P = Panel(y=q.astype(np.float32), treatment=d.astype(np.float32),
              futr=np.broadcast_to(week_feats(np.arange(T), 20), (n, T, 3)))
    mk = lambda o: make_windows(P, o, C, H)
    m = MDLForecaster("add", epochs=8, log=lambda s: None).fit(mk(range(C - 1, T - H, 2)), [1])
    Wt = mk([T - H - 1])
    pred, e = m.predict(Wt)
    lo, hi = m._demand(Wt, 0.0), m._demand(Wt, 0.5)
    assert (hi >= lo - 1e-4).all(), "monotonicity violated"
    err = np.abs(e - eff[Wt.item]).mean()
    print("on-policy MAE %.2f, effect MAE %.1f (mean effect %.1f)" % (np.abs(pred - Wt.y).mean(), err, eff.mean()))
    assert err < 0.8 * eff.mean(), err  # attenuated under confounding by design (this is paper 1's point), but not degenerate
