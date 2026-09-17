"""The data schema is adaptable: a foreign dataset in Nixtla long layout becomes windows correctly.

Checks the contract that makes a new dataset cheap to add: historical exogenous are read only from
the context, future exogenous from both slices, statics are carried per series, and a ragged panel
(missing series-timestamp rows) is filled and excluded rather than silently misaligned.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from common import Panel, make_windows

C, H, N, T = 4, 2, 6, 12


def long_df(drop=()):
    """A Nixtla-style long frame: unique_id, ds, y, plus one column of each exogenous type."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N):
        for t in range(T):
            if (i, t) in drop:  # drop is a set of (series_index, timestamp)
                continue
            rows.append({"unique_id": f"s{i}", "ds": t,
                         "y": float(10 + i + t), "discount": round(0.05 * (t % 4), 3),
                         "stock": float(500 - 10 * t), "snap": float(t % 7 == 0),
                         "store": f"store_{i % 3}", "base_price": 20.0 + i})
    return pd.DataFrame(rows)


def build(df):
    return Panel.from_long(df, treatment_col="discount", hist_exog=["stock"], futr_exog=["snap"],
                           static_cat=["store"], static_num=["base_price"])


def test_variable_types_land_in_the_right_slices():
    p = build(long_df())
    W = make_windows(p, [7], C, H)
    assert len(W) == N
    # past = [log1p(y), treatment, *hist, *futr]; horizon block carries future exogenous only
    assert W.past.shape == (N, C, 4) and W.fut.shape == (N, H, 1)
    y = np.array([[10.0 + i + t for t in range(4, 8)] for i in range(N)], np.float32)
    assert np.allclose(W.past[:, :, 0], np.log1p(y), atol=1e-5)          # target lags, transformed
    assert np.allclose(W.past[:, :, 1], p.treatment[:, 4:8], atol=1e-6)  # treatment history
    assert np.allclose(W.past[:, :, 2], p.hist[:, 4:8, 0], atol=1e-6)    # historical exogenous
    assert np.allclose(W.past[:, :, 3], p.futr[:, 4:8, 0], atol=1e-6)    # future exogenous, context
    assert np.allclose(W.fut[:, :, 0], p.futr[:, 8:10, 0], atol=1e-6)    # future exogenous, horizon
    assert np.allclose(W.d_fut, p.treatment[:, 8:10], atol=1e-6)         # the intervention
    assert np.allclose(W.y, p.y[:, 8:10], atol=1e-6)


def test_historical_exogenous_never_reaches_the_horizon():
    """Stock must not appear in the horizon block, or the model could read the future."""
    p = build(long_df())
    W = make_windows(p, [7], C, H)
    for h in range(W.fut.shape[-1]):
        for step in range(H):
            assert not np.allclose(W.fut[:, step, h], p.hist[:, 8 + step, 0], atol=1e-6)


def test_ragged_panel_is_excluded_not_misaligned():
    """A series missing a timestamp inside the window is dropped; the others keep their values."""
    p = build(long_df(drop=[(2, 6)]))
    assert not p.valid[2, 6]
    W = make_windows(p, [7], C, H)
    assert len(W) == N - 1 and 2 not in set(W.item.tolist())
    full = make_windows(build(long_df()), [7], C, H)
    keep = [i for i in range(N) if i != 2]
    assert np.allclose(W.past, full.past[keep], atol=1e-6)


def test_statics_and_embedding_sizes_come_from_the_data():
    p = build(long_df())
    assert p.n_cat == [3]                                   # three distinct stores, no hand-set size
    assert p.static_num.shape == (N, 1)
    W = make_windows(p, [7], C, H)
    assert W.static_cat.shape == (N, 1) and W.static_num.shape == (N, 1)


@pytest.mark.parametrize("hist,futr,width", [([], ["snap"], 3), (["stock"], [], 3),
                                             ([], [], 2), (["stock"], ["snap", "y"], 5)])
def test_feature_width_tracks_the_declared_schema(hist, futr, width):
    """Adding or removing an exogenous column is all a new dataset needs; nothing else is wired."""
    p = Panel.from_long(long_df(), treatment_col="discount", hist_exog=hist, futr_exog=futr,
                        static_cat=["store"], static_num=["base_price"])
    W = make_windows(p, [7], C, H)
    assert W.past.shape[-1] == width and W.fut.shape[-1] == len(futr)


def test_target_transform_is_configurable_for_non_count_targets():
    """log1p suits counts; a target that goes negative needs a different transform."""
    df = long_df()
    df["y"] = df["y"] - 20.0                                # now negative
    p = Panel.from_long(df, treatment_col="discount", static_cat=["store"], y_past_transform=None,
                        drop_inactive=False)  # the count-style activity filter would eat these
    W = make_windows(p, [7], C, H)
    assert len(W) == N and np.isfinite(W.past).all()
    assert np.allclose(W.past[:, :, 0], p.y[:, 4:8], atol=1e-6)
