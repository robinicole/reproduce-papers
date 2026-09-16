"""Shared pieces for both papers: forecast windows, the transformer block, train/predict loops, metrics."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
class Windows:
    """Fixed-length forecast windows. All arrays are numpy; converted to torch lazily."""

    def __init__(self, past, fut, d_fut, y, static_cat, static_num, scale, item):
        self.past, self.fut, self.d_fut, self.y = past, fut, d_fut, y
        self.static_cat, self.static_num, self.scale, self.item = static_cat, static_num, scale, item

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        return Windows(*[getattr(self, k)[idx] for k in
                         ("past", "fut", "d_fut", "y", "static_cat", "static_num", "scale", "item")])

    def with_discount(self, d):
        w = self.subset(slice(None))
        w.d_fut = np.broadcast_to(np.asarray(d, dtype=np.float32), self.d_fut.shape).copy()
        return w

    KEYS = ("past", "fut", "d_fut", "y", "static_cat", "static_num", "scale")

    def tensors(self, idx):
        """Batch as device tensors. Arrays are cached on the device once (if they fit) and indexed there."""
        if not hasattr(self, "_dev"):
            small = sum(getattr(self, k).nbytes for k in self.KEYS) < 0.6e9
            self._dev = {k: torch.as_tensor(getattr(self, k), device=DEV) for k in self.KEYS} if small else None
        if self._dev is None:
            return {k: torch.as_tensor(getattr(self, k)[idx], device=DEV) for k in self.KEYS}
        if not isinstance(idx, slice):
            idx = torch.as_tensor(idx, device=DEV)
        return {k: v[idx] for k, v in self._dev.items()}


def week_feats(week, period):
    w = np.asarray(week, dtype=np.float32)
    return np.stack([w / 100.0, np.sin(2 * np.pi * w / period), np.cos(2 * np.pi * w / period)], -1)


def make_windows(q, d, extra, week, static_cat, static_num, origins, C, H, period=52, with_y=True, valid=None):
    """q, d, extra[k]: (n_items, T) arrays; week: (T,); origins: forecast-origin indices (last observed step).
    valid: optional (n_items, T) bool; windows touching an invalid step are dropped.
    Returns one window per (item, origin). Windows whose context has no sales are dropped."""
    n, T = q.shape
    P, Fu, D, Y, SC, SN, S, I = [], [], [], [], [], [], [], []
    wf = week_feats(week, period)
    for t in origins:
        ctx, hor = slice(t - C + 1, t + 1), slice(t + 1, t + 1 + H)
        qc = q[:, ctx]
        scale = qc.mean(1) + 1.0
        past = np.concatenate([np.log1p(qc)[..., None], d[:, ctx][..., None]]
                              + [np.log1p(e[:, ctx])[..., None] for e in extra]
                              + [np.broadcast_to(wf[ctx], (n, C, 3))], -1)
        fut = np.broadcast_to(wf[hor], (n, H, 3)).copy() if with_y else np.broadcast_to(wf[t + 1:t + 1 + H], (n, H, 3)).copy()
        keep = qc.sum(1) > 0
        if valid is not None:
            keep &= valid[:, t - C + 1:t + 1 + H].all(1)
        P.append(past[keep]); Fu.append(fut[keep]); D.append(d[keep][:, hor] if with_y else np.zeros((keep.sum(), H), np.float32))
        Y.append(q[keep][:, hor] if with_y else np.zeros((keep.sum(), H), np.float32))
        SC.append(static_cat[keep]); SN.append(static_num[keep]); S.append(scale[keep]); I.append(np.where(keep)[0])
    cat = lambda xs, dt=np.float32: np.ascontiguousarray(np.concatenate(xs)).astype(dt)
    return Windows(cat(P), cat(Fu), cat(D), cat(Y), cat(SC, np.int64), cat(SN), cat(S), cat(I, np.int64))


# ----------------------------------------------------------------------------- model
class SeqNet(nn.Module):
    """Encoder-only transformer over [context tokens | horizon tokens].
    out='seq' -> one value per horizon step; out='scalar' -> pooled scalar."""

    def __init__(self, f_past, f_fut, n_cat, n_num, C, H, out="seq", d_model=64, n_layers=2, n_heads=4, dropout=0.1):
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
        self.head = nn.Linear(d_model, 1)

    def forward(self, b):
        x = torch.cat([self.past_in(b["past"]), self.fut_in(b["fut"])], 1) + self.pos
        s = sum(e(b["static_cat"][:, i]) for i, e in enumerate(self.cat_emb))
        if self.num_in is not None:
            s = s + self.num_in(b["static_num"])
        h = self.norm(self.enc(x + s[:, None, :]))
        if self.out == "scalar":
            return self.head(h[:, self.C:].mean(1)).squeeze(-1)
        return self.head(h[:, self.C:]).squeeze(-1)


def outcome_act(raw, scale):
    return F.softplus(raw) * scale[:, None]


def head(kind, q_tilde, d_tilde, d, psi_raw, scale):
    if kind == "mult":
        psi = -F.softplus(psi_raw)
        return q_tilde * ((1 - d) / (1 - d_tilde).clamp(min=0.05)) ** psi[:, None], psi
    psi = F.softplus(psi_raw) * scale  # demand units per unit discount
    return q_tilde + psi[:, None] * (d - d_tilde), psi


def loss_fn(name):
    return {"l1": lambda p, y: (p - y).abs().mean(), "l2": lambda p, y: ((p - y) ** 2).mean()}[name]


def train(model, W, step_fn, epochs, lr, batch=1024, seed=0, log=None):
    """Generic loop: step_fn(model, batch) -> loss. AdamW + cosine schedule."""
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
        if log:
            log(f"  epoch {ep + 1}/{epochs} loss {tot / n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict(model, W, fn, batch=4096):
    """fn(model, batch) -> tensor; returns numpy concatenation."""
    model.eval()
    return np.concatenate([fn(model, W.tensors(slice(i, i + batch))).cpu().numpy() for i in range(0, len(W), batch)])


# ----------------------------------------------------------------------------- metrics
def metrics(pred, y, price=None):
    err = pred - y
    out = {"MAE": np.abs(err).mean(), "MSE": (err ** 2).mean()}
    if price is not None:
        w = np.broadcast_to(price[:, None], y.shape)
        out["demand_err"] = np.sqrt((w * err ** 2).sum() / (w * y ** 2).sum())
    return out
