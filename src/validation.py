from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(
    df: pd.DataFrame,
    valid_weeks: int = 6,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Пустой датафрейм")
    cutoff = df[date_col].max() - pd.Timedelta(weeks=valid_weeks)
    train = df[df[date_col] <= cutoff].copy()
    valid = df[df[date_col] > cutoff].copy()
    if train.empty or valid.empty:
        raise ValueError("Недостаточно истории для разбиения по времени")
    return train, valid


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"MAE": mae(y_true, y_pred), "MAPE": mape(y_true, y_pred)}