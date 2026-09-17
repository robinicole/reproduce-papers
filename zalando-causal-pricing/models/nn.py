"""The small transformer both papers' models are built from, and the training/prediction loops."""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
log = logging.getLogger(__name__)


class SeqNet(nn.Module):
    """Encoder-only transformer over [context tokens | horizon tokens].
    out='seq' -> n_out values per horizon step; out='scalar' -> n_out pooled values.
    With n_out=1 the trailing axis is squeezed, so (N, H) and (N,)."""

    def __init__(self, f_past, f_fut, n_cat, n_num, C, H, out="seq", d_model=64, n_layers=2, n_heads=4, dropout=0.1, n_out=1):
        super().__init__()
        self.C, self.H, self.out = C, H, out
        self.past_in = nn.Linear(f_past, d_model)
        self.fut_in = nn.Linear(f_fut, d_model)
        self.pos = nn.Parameter(torch.randn(C + H, d_model) * 0.02)
        self.cat_emb = nn.ModuleList([nn.Embedding(n, d_model) for n in n_cat])
        self.num_in = nn.Linear(n_num, d_model) if n_num else None
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.n_out = n_out
        self.head = nn.Linear(d_model, n_out)

    def forward(self, b):
        x = torch.cat([self.past_in(b["past"]), self.fut_in(b["fut"])], 1) + self.pos
        s = sum(e(b["static_cat"][:, i]) for i, e in enumerate(self.cat_emb))
        if self.num_in is not None:
            s = s + self.num_in(b["static_num"])
        h = self.norm(self.enc(x + s[:, None, :]))
        y = self.head(h[:, self.C:].mean(1) if self.out == "scalar" else h[:, self.C:])
        return y.squeeze(-1) if self.n_out == 1 else y


def seqnet(W, n_cat, out, extra_fut=0, n_out=1, **net):
    """A SeqNet sized from the windows it will see."""
    return SeqNet(W.past.shape[-1], W.fut.shape[-1] + extra_fut, n_cat, W.static_num.shape[-1],
                  W.past.shape[1], W.fut.shape[1], out, n_out=n_out, **net).to(DEV)


def outcome_act(raw, scale):
    """Positive demand as a multiple of the recent level."""
    return F.softplus(raw) * scale[:, None]


def loss_fn(name):
    return {"l1": lambda p, y: (p - y).abs().mean(), "l2": lambda p, y: ((p - y) ** 2).mean()}[name]


def train(model, W, step_fn, epochs, lr, batch=1024, seed=0):
    """Generic loop: step_fn(model, batch) -> loss. AdamW + one-cycle schedule."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(W)
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = W.tensors(perm[i:i + batch])
            loss = step_fn(model, b)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(b["y"])
        log.debug("epoch %d/%d loss %.4f", ep + 1, epochs, tot / n)
    model.eval()
    return model


@torch.no_grad()
def predict(model, W, fn, batch=4096):
    """fn(model, batch) -> tensor; returns the numpy concatenation over W."""
    model.eval()
    return np.concatenate([fn(model, W.tensors(slice(i, i + batch))).cpu().numpy() for i in range(0, len(W), batch)])
