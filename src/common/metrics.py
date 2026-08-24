import numpy as np
from typing import Optional, Sequence, Union


def accuracy_score(y_true: Sequence, y_pred: Sequence) -> float:
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_t == y_p))


def f1_score(y_true: Sequence, y_pred: Sequence) -> float:
    y_t, y_p = np.asarray(y_true).astype(int).ravel(), np.asarray(y_pred).astype(int).ravel()
    tp = np.sum((y_t == 1) & (y_p == 1))
    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def root_mean_squared_error(y_true: Sequence, y_pred: Sequence) -> float:
    y_t, y_p = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mean_squared_error(y_true: Sequence, y_pred: Sequence, squared: bool = True) -> float:
    y_t, y_p = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
    mse = float(np.mean((y_t - y_p) ** 2))
    return mse if squared else float(np.sqrt(mse))


def r2_score(y_true: Sequence, y_pred: Sequence) -> float:
    y_t, y_p = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 1.0


def mean_absolute_scaled_error(y_true: Sequence, y_pred: Sequence, y_train: Sequence, period: int = 1) -> float:
    y_t, y_p, y_tr = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel(), np.asarray(y_train, dtype=float).ravel()
    scale = np.mean(np.abs(y_tr[period:] - y_tr[:-period])) or 1e-8
    return float(np.mean(np.abs(y_t - y_p)) / scale)


def asymmetric_weighted_loss(y_true: Sequence, y_pred: Sequence, under_prediction_weight: float = 4.0, over_prediction_weight: float = 1.0, **kwargs) -> float:
    under_w = kwargs.get("under_weight", under_prediction_weight)
    over_w = kwargs.get("over_weight", over_prediction_weight)
    y_t, y_p = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
    diff = y_t - y_p
    weights = np.where(diff > 0, under_w, over_w)
    return float(np.mean(weights * (diff ** 2)))