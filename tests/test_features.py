import pandas as pd
import numpy as np
from src.features import make_features


def test_lag_does_not_see_future():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "restaurant_id": 1,
        "guests": np.arange(10, dtype=float),
        "holiday_flg": 0,
    })
    out = make_features(df)
    assert out.loc[5, "lag_1"] == 4.0
    assert out.loc[9, "lag_7"] == 2.0


def test_first_lag_is_nan():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "restaurant_id": 1,
        "guests": [10, 20, 30, 40, 50],
        "holiday_flg": 0,
    })
    out = make_features(df)
    assert pd.isna(out.loc[0, "lag_1"])


def test_features_are_numeric():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=30, freq="D"),
        "restaurant_id": 1,
        "guests": np.random.default_rng(42).integers(10, 50, 30).astype(float),
        "holiday_flg": 0,
    })
    out = make_features(df)
    from src.features import FEATURE_COLS
    for col in FEATURE_COLS:
        assert pd.api.types.is_numeric_dtype(out[col]), f"{col} не числовой"