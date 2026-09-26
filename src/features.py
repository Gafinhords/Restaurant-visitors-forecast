from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)


def make_features(
    df: pd.DataFrame,
    target: str = "guests",
    id_col: str = "restaurant_id",
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.sort_values([id_col, date_col]).copy()

    out["dow"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["week_of_year"] = out[date_col].dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["dow"].isin((4, 5, 6)).astype(int)
    out["is_saturday"] = (out["dow"] == 5).astype(int)
    out["is_sunday"] = (out["dow"] == 6).astype(int)
    if "holiday_flg" in out.columns:
        out["is_holiday"] = pd.to_numeric(out["holiday_flg"], errors="coerce").fillna(0).astype(int)
    else:
        out["is_holiday"] = 0
    out["day_after_holiday"] = (
        out.groupby(id_col, observed=True)["holiday_flg"].shift(1).fillna(0).astype(int)
    )

    for lag in LAGS:
        out[f"lag_{lag}"] = out.groupby(id_col, observed=True)[target].shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = out.groupby(id_col, observed=True)[target].shift(1)
        out[f"roll_mean_{window}"] = (
            shifted.groupby(out[id_col], observed=True)
            .rolling(window, min_periods=max(2, window // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[f"roll_std_{window}"] = (
            shifted.groupby(out[id_col], observed=True)
            .rolling(window, min_periods=max(2, window // 2))
            .std()
            .reset_index(level=0, drop=True)
        )
    for col in FEATURE_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    return out


FEATURE_COLS = [
    "dow", "month", "day", "week_of_year",
    "is_weekend", "is_saturday", "is_sunday",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "roll_std_7", "roll_std_14", "roll_std_28", "is_holiday"
]