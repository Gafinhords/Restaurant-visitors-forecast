from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_SCHEMA = {
    "date": "datetime64[ns]",
    "restaurant_id": "int64",
    "guests": "int64",
    "revenue": "float64",
}

AVG_CHECK = 2500.0
RANDOM_STATE = 42


def load_raw(visits_path: str | Path, dates_path: str | Path) -> pd.DataFrame:

    visits = pd.read_csv(visits_path, parse_dates=["visit_date"]).rename(
        columns={
            "air_store_id": "restaurant_id",
            "visit_date": "date",
            "visitors": "guests",
        }
    )
    dates = pd.read_csv(dates_path, parse_dates=["calendar_date"]).rename(
        columns={"calendar_date": "date"}
    )

    df = visits.merge(dates[["date", "holiday_flg"]], on="date", how="left")

    df["restaurant_id"] = df["restaurant_id"].astype("category").cat.codes.astype("int64")

    rng = np.random.default_rng(RANDOM_STATE)
    df["revenue"] = (
        df["guests"].astype(float) * AVG_CHECK * rng.normal(1.0, 0.05, len(df))
    ).astype("float64")

    df = df[["date", "restaurant_id", "guests", "revenue", "holiday_flg"]]
    return df.astype(RAW_SCHEMA | {"holiday_flg": "int64"}).sort_values(
        ["restaurant_id", "date"]
    ).reset_index(drop=True)


def restore_calendar(df: pd.DataFrame, fill_value: int = 0) -> pd.DataFrame:
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    full_idx = pd.MultiIndex.from_product(
        [df["restaurant_id"].unique(), full_range],
        names=["restaurant_id", "date"],
    )
    full = pd.DataFrame(index=full_idx).reset_index()
    merged = full.merge(df, on=["restaurant_id", "date"], how="left")
    merged["guests"] = merged["guests"].fillna(fill_value).astype(int)
    merged["revenue"] = merged["revenue"].fillna(0.0)
    merged["holiday_flg"] = merged["holiday_flg"].fillna(0).astype(int)
    return merged.sort_values(["restaurant_id", "date"]).reset_index(drop=True)


def filter_stores(
    df: pd.DataFrame,
    max_missing_ratio: float = 0.1,
    min_history_days: int = 400,
) -> pd.DataFrame:

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    full_idx = pd.MultiIndex.from_product(
        [df["restaurant_id"].unique(), full_range],
        names=["restaurant_id", "date"],
    )
    full = pd.DataFrame(index=full_idx).reset_index()
    merged = full.merge(
        df[["restaurant_id", "date", "guests"]],
        on=["restaurant_id", "date"],
        how="left",
    )
    merged["is_missing"] = merged["guests"].isna()

    by_store = merged.groupby("restaurant_id")["is_missing"].mean()
    good = by_store[by_store < max_missing_ratio].index

    real = merged[~merged["is_missing"] & merged["restaurant_id"].isin(good)]
    span = real.groupby("restaurant_id")["date"].agg(["min", "max"])
    span["days"] = (span["max"] - span["min"]).dt.days
    full_stores = span[span["days"] > min_history_days].index

    return df[df["restaurant_id"].isin(full_stores)].copy().reset_index(drop=True)