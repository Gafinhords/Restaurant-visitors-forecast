from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor

from src.features import FEATURE_COLS


def build_ridge(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=random_state)),
    ])


def build_lgbm(random_state: int = 42) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        random_state=random_state,
        verbose=-1,
    )


def train_model(
    train_df: pd.DataFrame,
    model,
    feature_cols: list[str] | None = None,
    target: str = "guests",
):
    feature_cols = feature_cols or FEATURE_COLS
    train = train_df.dropna(subset=feature_cols + [target]).copy()
    model.fit(train[feature_cols], train[target])
    return model, feature_cols


def predict_model(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    mask = out[feature_cols].notna().all(axis=1)
    out["y_pred"] = np.nan
    if mask.any():
        out.loc[mask, "y_pred"] = model.predict(out.loc[mask, feature_cols])
    return out


def blend(y_pred_a: np.ndarray, y_pred_b: np.ndarray, weight_a: float = 0.4) -> np.ndarray:
    return weight_a * y_pred_a + (1.0 - weight_a) * y_pred_b

from pathlib import Path
import joblib


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")
    return joblib.load(path)