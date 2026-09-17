"""Zalando's production demand model (arXiv:2305.14406, Kunz et al.): encoder/decoder transformer with a
Monotonic Demand Layer. This is the 'TF' model that arXiv:2312.15282v2 uses as its baseline.

  encoder  <- past (demand, treatments, covariates);  decoder <- future covariates + last observed step
  the future DISCOUNT bypasses encoder/decoder and enters the head only:
      log1p q^(d) = q_hat(kappa_T) + sigma(kappa_T) * PL(d; delta(gamma))          (eq. monotonic_demand_response)
  PL is piecewise linear on segments of width 0.1 with slopes delta = softplus(phi0(encoder state)) >= 0,
  sigma = softplus(phi1(decoder state)), so demand is monotonically increasing in discount by construction.
  loss = (v(pred) - v(log1p q))^2 with v(x) = 1 + x + x^2/2 + x^3/6  (eq. loss).
Any other treatment (base price, stock) has no structural head in the paper; it enters the decoder
as a future covariate, and its effect is read off by finite differences like the S-learners'.
anchor=True adds log(mean recent demand) to the output (local scaling, not in the paper): the network then
predicts a ratio to the recent level, which is what keeps it from extrapolating a drifting covariate.
Near/far-future split (5 vs 20 weeks) is not implemented: our horizons are <= 5 weeks (near future only).
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from heads import implied_effect
from nn import DEV, predict, train

log = logging.getLogger(__name__)
SEG = 0.1  # discount segment width


class MDLNet(nn.Module):
    def __init__(self, f_past, f_fut, n_cat, n_num, C, H, n_treat=1, d_model=64, n_layers=2, n_heads=4, dropout=0.1, n_seg=7):
        super().__init__()
        self.C, self.H, self.n_seg, self.n_treat = C, H, n_seg, n_treat
        self.past_in = nn.Linear(f_past, d_model)
        self.fut_in = nn.Linear(f_fut + 1 + n_treat, d_model)  # + last observed (log demand, treatments)
        self.pos_past = nn.Parameter(torch.randn(C, d_model) * 0.02)
        self.pos_fut = nn.Parameter(torch.randn(H, d_model) * 0.02)
        self.cat_emb = nn.ModuleList([nn.Embedding(n, d_model) for n in n_cat])
        self.num_in = nn.Linear(n_num, d_model) if n_num else None
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True), n_layers)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True), n_layers)
        self.norm_e, self.norm_d = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.phi0 = nn.Linear(d_model, n_seg)  # slopes delta from encoder state
        self.phi1 = nn.Linear(d_model, 2)      # (q_hat, sigma) from decoder state

    def states(self, b, fut):
        s = sum(e(b["static_cat"][:, i]) for i, e in enumerate(self.cat_emb))
        if self.num_in is not None:
            s = s + self.num_in(b["static_num"])
        s = s[:, None, :]
        gamma = self.norm_e(self.enc(self.past_in(b["past"]) + self.pos_past + s))
        last = b["past"][:, -1, :1 + self.n_treat]  # (log1p demand, treatments) at t
        fut = torch.cat([fut, last[:, None, :].expand(-1, self.H, -1)], -1)
        kappa = self.norm_d(self.dec(self.fut_in(fut) + self.pos_fut + s, gamma))  # non-autoregressive: no mask
        delta = F.softplus(self.phi0(gamma.mean(1)))  # (N, n_seg), same for all future weeks
        qs = self.phi1(kappa)                          # (N, H, 2)
        return qs[..., 0], F.softplus(qs[..., 1]), delta

    def forward(self, b, d, fut):
        """log1p-demand prediction at discounts d (N, H), given the decoder covariates fut."""
        q_hat, sigma, delta = self.states(b, fut)
        m = (d / SEG).floor().clamp(0, self.n_seg - 1).long()                       # segment index
        cum = torch.cat([torch.zeros_like(delta[:, :1]), delta.cumsum(1)], 1)       # (N, n_seg+1)
        full = cum.gather(1, m)                                                      # completed segments
        partial = (d - m * SEG) / SEG * delta.gather(1, m)                           # fraction of current segment
        return q_hat + sigma * (full + partial)


def v(x):  # third-order Taylor expansion of exp
    return 1 + x + x ** 2 / 2 + x ** 3 / 6


class MDLForecaster:
    """fit(W) / predict(W) -> (demand (N,H), effects (N,K)); effects are read off by finite differences
    since the model has no explicit elasticity parameter."""

    def __init__(self, n_cat, head="mult", spec=(("discount", "discount", +1),), epochs=10, lr=1e-3, seed=0, net=None, anchor=False):
        self.n_cat, self.head, self.spec, self.epochs, self.lr, self.seed = n_cat, head, spec, epochs, lr, seed
        self.net, self.anchor = net or {}, anchor
        kinds = [kind for _, kind, _ in spec]
        self.di = kinds.index("discount") if "discount" in kinds else None
        self.others = [k for k in range(len(spec)) if k != self.di]  # treatments without a structural head

    def _split(self, b):
        """(discount for the monotone head, decoder covariates with the other treatments appended)."""
        d = b["d_fut"][..., self.di] if self.di is not None else torch.zeros_like(b["d_fut"][..., 0])
        fut = torch.cat([b["fut"], b["d_fut"][..., self.others]], -1) if self.others else b["fut"]
        return d, fut

    def _log_demand(self, m, b):
        d, fut = self._split(b)
        out = m(b, d, fut)
        return out + torch.log(b["scale"])[:, None] if self.anchor else out

    def fit(self, W):
        torch.manual_seed(self.seed)
        self.model = MDLNet(W.past.shape[-1], W.fut.shape[-1] + len(self.others), self.n_cat, W.static_num.shape[-1],
                            W.past.shape[1], W.fut.shape[1], n_treat=len(self.spec), **self.net).to(DEV)
        step = lambda m, b: ((v(self._log_demand(m, b)) - v(torch.log1p(b["y"]))) ** 2).mean()
        log.info("MDL: encoder/decoder + monotonic demand layer%s", " (anchored)" if self.anchor else "")
        train(self.model, W, step, self.epochs, self.lr, seed=self.seed)
        return self

    def _demand(self, W):
        return predict(self.model, W, lambda m, b: torch.expm1(self._log_demand(m, b)).clamp(min=0))

    def predict(self, W):
        return self._demand(W), implied_effect(self.head, self.spec, W, self._demand)
