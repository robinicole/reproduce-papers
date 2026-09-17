"""Shared pieces for both papers: the data schema, forecast windows, the transformer block,
train/predict loops, and metrics."""
from dataclasses import dataclass

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


@dataclass
class Panel:
    """A dense (series, time) panel, in the variable taxonomy Nixtla uses.

    y           (n, T)            target, the thing being forecast
    treatment   (n, T)            the variable you intervene on (discount). Known through the
                                  horizon, like a future exogenous, but kept separate because the
                                  causal models residualize it and forecast under changes to it.
    hist        (n, T, n_hist)    HISTORICAL exogenous: known only up to the forecast origin, so it
                                  is read from the context and never from the horizon (stock, past
                                  prices, past traffic). Pass it already transformed.
    futr        (n, T, n_futr)    FUTURE exogenous: known through the horizon (calendar features,
                                  planned promotions, SNAP/holiday flags). Read from both slices.
    static_cat  (n, k)            STATIC categorical, as integer codes (category, store, region).
    static_num  (n, k) | (n,T,k)  STATIC numeric. A 3-D array is read at each window's own origin,
                                  which is how a "static" that actually drifts (a base price) stays
                                  causal.
    valid       (n, T)            False where the series does not exist or is not sellable.
    y_past_transform              applied to the target's own lags only. log1p suits counts; pass
                                  a different callable (or None) for a target that can go negative.
    drop_inactive                 drop windows whose context target sums to <= 0. That means "no
                                  sales in the context" for a count target, but it would discard
                                  most windows for a target that can be negative or zero-mean, so
                                  set it False for those.

    Build it with `from_long` for a Nixtla-style long DataFrame, or construct it directly from
    arrays. Feature order in a window is [transform(y), treatment, *hist, *futr] over the context
    and [*futr] over the horizon, so adding a column to `hist` or `futr` is all a new schema needs.
    """
    y: np.ndarray
    treatment: np.ndarray
    hist: np.ndarray = None
    futr: np.ndarray = None
    static_cat: np.ndarray = None
    static_num: np.ndarray = None
    valid: np.ndarray = None
    ids: np.ndarray = None
    times: np.ndarray = None
    names: dict = None
    y_past_transform: object = np.log1p
    drop_inactive: bool = True

    def __post_init__(self):
        n, T = self.y.shape
        z = lambda a, k: np.zeros((n, T, k), np.float32) if a is None else np.asarray(a, np.float32)
        self.y = np.asarray(self.y, np.float32)
        self.treatment = np.asarray(self.treatment, np.float32)
        self.hist, self.futr = z(self.hist, 0), z(self.futr, 0)
        if self.static_cat is None:
            self.static_cat = np.zeros((n, 1), np.int64)
        if self.static_num is None:
            self.static_num = np.zeros((n, 1), np.float32)
        self.static_cat = np.asarray(self.static_cat, np.int64)
        self.static_num = np.asarray(self.static_num, np.float32)
        if self.valid is None:
            self.valid = np.ones((n, T), bool)
        self.names = self.names or {}

    @property
    def n_cat(self):
        """Embedding sizes for the static categorical columns."""
        return [int(self.static_cat[:, i].max()) + 1 for i in range(self.static_cat.shape[1])]

    @classmethod
    def from_long(cls, df, id_col="unique_id", time_col="ds", target_col="y", treatment_col=None,
                  static_cat=(), static_num=(), hist_exog=(), futr_exog=(), valid_col=None,
                  y_past_transform=np.log1p, drop_inactive=True):
        """Build from a long DataFrame in Nixtla layout (one row per series-timestamp).

        Missing series-timestamp combinations are filled and marked invalid, so a ragged panel is
        fine. Categorical statics are factorized to codes. Everything else is taken as float.
        """
        import pandas as pd
        ids = np.sort(df[id_col].unique())
        times = np.sort(df[time_col].unique())
        n, T = len(ids), len(times)
        full = pd.MultiIndex.from_product([ids, times], names=[id_col, time_col])
        g = df.set_index([id_col, time_col]).reindex(full)
        grid = lambda c: g[c].to_numpy(dtype=np.float64).reshape(n, T)
        y = np.nan_to_num(grid(target_col))
        valid = ~np.isnan(grid(valid_col)) if valid_col else ~np.isnan(grid(target_col))
        if valid_col:
            valid = np.nan_to_num(grid(valid_col)).astype(bool)
        treat = np.nan_to_num(grid(treatment_col)) if treatment_col else np.zeros((n, T))
        stack = lambda cols: (np.stack([np.nan_to_num(grid(c)) for c in cols], -1)
                              if len(cols) else np.zeros((n, T, 0), np.float32))
        first = df.drop_duplicates(id_col).set_index(id_col).reindex(ids)
        s_cat = (np.stack([pd.factorize(first[c])[0] for c in static_cat], -1)
                 if len(static_cat) else np.zeros((n, 1), np.int64))
        s_num = (first[list(static_num)].to_numpy(np.float32)
                 if len(static_num) else np.zeros((n, 1), np.float32))
        return cls(y=y, treatment=treat, hist=stack(hist_exog), futr=stack(futr_exog),
                   static_cat=s_cat, static_num=np.nan_to_num(s_num), valid=valid, ids=ids, times=times,
                   names={"hist": list(hist_exog), "futr": list(futr_exog),
                          "static_cat": list(static_cat), "static_num": list(static_num),
                          "target": target_col, "treatment": treatment_col},
                   y_past_transform=y_past_transform, drop_inactive=drop_inactive)


def make_windows(panel, origins, C, H, with_y=True):
    """Cut a Panel into fixed-length forecast windows, one per (series, origin).

    origins are indices of the last observed step. Context is origin-C+1 : origin+1, horizon is
    origin+1 : origin+1+H. Windows whose context has no activity, or which touch an invalid step,
    are dropped.
    """
    y, d = panel.y, panel.treatment
    n, T = y.shape
    tf = panel.y_past_transform or (lambda a: a)
    P, Fu, D, Y, SC, SN, S, I = [], [], [], [], [], [], [], []
    for t in origins:
        ctx, hor = slice(t - C + 1, t + 1), slice(t + 1, t + 1 + H)
        yc = y[:, ctx]
        scale = yc.mean(1) + 1.0
        past = np.concatenate([tf(yc)[..., None], d[:, ctx][..., None],
                               panel.hist[:, ctx], panel.futr[:, ctx]], -1)
        fut = panel.futr[:, hor] if with_y else panel.futr[:, t + 1:t + 1 + H]
        sn = panel.static_num[:, t] if panel.static_num.ndim == 3 else panel.static_num
        keep = yc.sum(1) > 0 if panel.drop_inactive else np.ones(n, bool)
        if panel.valid is not None:
            keep &= panel.valid[:, t - C + 1:t + 1 + H].all(1)
        P.append(past[keep]); Fu.append(fut[keep].copy())
        D.append(d[keep][:, hor] if with_y else np.zeros((keep.sum(), H), np.float32))
        Y.append(y[keep][:, hor] if with_y else np.zeros((keep.sum(), H), np.float32))
        SC.append(panel.static_cat[keep]); SN.append(sn[keep]); S.append(scale[keep])
        I.append(np.where(keep)[0])
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
