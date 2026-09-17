"""Accuracy metrics: MAE, MSE, and paper 1's price-weighted relative RMSE (the demand error)."""
import numpy as np


def metrics(pred, y, price=None):
    err = pred - y
    out = {"MAE": np.abs(err).mean(), "MSE": (err ** 2).mean()}
    if price is not None:
        w = np.broadcast_to(price[:, None], y.shape)
        out["demand_err"] = np.sqrt((w * err ** 2).sum() / (w * y ** 2).sum())
    return out
