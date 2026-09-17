"""The demand heads every model shares, generalised to a vector of treatments, with the bounds that
keep them finite.

Each treatment is declared in the Panel as (name, kind, sign):
  kind "discount"  a fraction in [0, 1). Under the multiplicative head it enters as log(1-d), so the
                   coefficient is an elasticity; under the additive head it enters as is.
  kind "linear"    already in effect space (log price, log stock, ...): enters as is under both heads.
  sign             the expected direction of demand's response in natural units (+1: demand rises
                   with the treatment). The head translates it to effect space per kind and head.

With x = link(treatment) the heads are partially linear in the treatments (paper 1 eq. effecthead and
eq. simeffecthead, summed over treatments):
  add   q = q~ + sum_k psi_k (x_k - x~_k)
  mult  q = q~ * exp(clip(sum_k psi_k (x_k - x~_k), ±MAX_LOG_ADJ))
Shapes: q~, d~, d are (N, H[, K]); psi is (N, K). Works on torch tensors and numpy arrays alike.
"""
import numpy as np
import torch
import torch.nn.functional as F

MAX_LOG_ADJ = 3.0  # price adjustment capped at a factor of exp(3) either way
PSI_MAX = 5.0      # magnitude cap for learners that output psi directly (trees), multiplicative head
MIN_PRICE_RATIO = 0.05
DISCOUNT = (("discount", "discount", +1),)  # the default, single-treatment spec


def signs(head, spec):
    """Effect-space sign per treatment: log(1-d) falls as the discount rises, so a demand-raising
    discount has a negative coefficient under the multiplicative head."""
    return np.array([(-s if head == "mult" else s) if kind == "discount" else s for _, kind, s in spec], np.float32)


def link(head, spec, d):
    """Natural units -> effect space. Only the discount kind under the multiplicative head changes."""
    if head != "mult" or not any(kind == "discount" for _, kind, _ in spec):
        return d
    cols = []
    for k, (_, kind, _) in enumerate(spec):
        x = d[..., k]
        if kind == "discount":
            x = torch.log((1 - x).clamp(min=MIN_PRICE_RATIO)) if torch.is_tensor(x) else np.log(np.clip(1 - x, MIN_PRICE_RATIO, None))
        cols.append(x)
    return torch.stack(cols, -1) if torch.is_tensor(d) else np.stack(cols, -1)


def demand(head, spec, q_tilde, d_tilde, d, psi):
    dx = link(head, spec, d) - link(head, spec, d_tilde)
    adj = (dx * psi[:, None, :]).sum(-1)
    if head == "add":
        return q_tilde + adj
    if torch.is_tensor(adj):
        return q_tilde * torch.exp(adj.clamp(-MAX_LOG_ADJ, MAX_LOG_ADJ))
    return q_tilde * np.exp(np.clip(adj, -MAX_LOG_ADJ, MAX_LOG_ADJ))


def activate_psi(head, spec, raw, scale):
    """Network output (N, K) -> signed effects, as in paper 1's activations: softplus with the
    expected sign; the additive head is in demand units, so it scales with the recent level."""
    s = torch.as_tensor(signs(head, spec), device=raw.device)
    psi = s * F.softplus(raw)
    return psi * scale[:, None] if head == "add" else psi


def clip_psi(head, spec, psi):
    """Sign and range for learners that output psi directly rather than through an activation."""
    s = signs(head, spec)
    psi = np.where(s > 0, np.maximum(psi, 0.0), np.minimum(psi, 0.0))
    return np.clip(psi, -PSI_MAX, PSI_MAX) if head == "mult" else psi


def ensemble(head, preds):
    """Combine the cross-fit and own-fold forecasts: geometric mean for the multiplicative head."""
    if head == "mult":
        return np.exp(np.mean([np.log(np.clip(p, 1e-6, None)) for p in preds], 0))
    return np.mean(preds, 0)


def implied_effect(head, spec, W, predict):
    """Effects of a model with no explicit psi, by finite differences one treatment at a time with
    the others at their realised values. predict(W) -> (N, H) demand. Returns (N, K) in effect space.
    Discount: between 0 and 0.5 (add) or 0 and 0.1 (mult), as before. Linear kinds: a step of 0.1."""
    out = []
    for k, (_, kind, _) in enumerate(spec):
        if kind == "discount":
            lo, hi = (0.0, 0.5) if head == "add" else (0.0, 0.1)
            p0, p1 = predict(W.with_treatment(lo, k)), predict(W.with_treatment(hi, k))
            dx = (hi - lo) if head == "add" else np.log(1 - hi)
        else:
            base = W.d_fut[..., k]
            p0, p1 = predict(W), predict(W.with_treatment(base + 0.1, k))
            dx = 0.1
        eff = (p1 - p0).mean(1) / dx if head == "add" else (np.log(p1 + 1e-3) - np.log(p0 + 1e-3)).mean(1) / dx
        out.append(eff)
    return np.stack(out, -1)
