"""The demand heads every model shares, in one place, with the bounds that keep them finite.

add  (linear simulator, paper 1 eq. simeffecthead):  q = q~ + psi * (d - d~)
mult (real data, paper 1 eq. effecthead):            q = q~ * ((1-d)/(1-d~))^psi, psi <= 0
The multiplicative head is evaluated as exp(psi * log ratio) with the exponent clamped, because an
unbounded exponent from a tree-based psi overflowed on sparse series. Works on torch tensors (for
training losses) and numpy arrays (for inference) alike.
"""
import numpy as np
import torch
import torch.nn.functional as F

MAX_LOG_ADJ = 3.0  # price adjustment capped at a factor of exp(3) either way
PSI_MIN = -5.0     # elasticity magnitude cap for learners that output psi directly (trees)
MIN_PRICE_RATIO = 0.05


def demand(head, q_tilde, d_tilde, d, psi):
    """psi is (N,), the rest (N, H)."""
    if head == "add":
        return q_tilde + psi[:, None] * (d - d_tilde)
    if torch.is_tensor(q_tilde):
        ratio = (1 - d) / (1 - d_tilde).clamp(min=MIN_PRICE_RATIO)
        return q_tilde * torch.exp((psi[:, None] * torch.log(ratio)).clamp(-MAX_LOG_ADJ, MAX_LOG_ADJ))
    ratio = (1 - d) / np.clip(1 - d_tilde, MIN_PRICE_RATIO, None)
    return q_tilde * np.exp(np.clip(psi[:, None] * np.log(ratio), -MAX_LOG_ADJ, MAX_LOG_ADJ))


def activate_psi(head, raw, scale):
    """Network output -> signed effect, as in paper 1's activations: a negative elasticity for the
    multiplicative head, a non-negative slope in demand units per unit discount for the additive one."""
    return -F.softplus(raw) if head == "mult" else F.softplus(raw) * scale


def clip_psi(head, psi):
    """Sign and range for learners that output psi directly rather than through an activation."""
    return np.maximum(psi, 0.0) if head == "add" else np.clip(psi, PSI_MIN, 0.0)


def ensemble(head, preds):
    """Combine the cross-fit and own-fold forecasts: geometric mean for the multiplicative head."""
    if head == "mult":
        return np.exp(np.mean([np.log(np.clip(p, 1e-6, None)) for p in preds], 0))
    return np.mean(preds, 0)


def implied_effect(head, predict_at):
    """Effect of a model with no explicit psi, by finite differences. predict_at(d) -> (N, H) demand
    at a constant discount d. add: dq/dd over the off-policy range; mult: elasticity near full price."""
    if head == "add":
        return (predict_at(0.5) - predict_at(0.0)).mean(1) / 0.5
    q0, q1 = predict_at(0.0) + 1e-3, predict_at(0.1) + 1e-3
    return (np.log(q1) - np.log(q0)).mean(1) / np.log(0.9)
