from __future__ import annotations

import pandas as pd

def naive_lag7(
    df: pd.DataFrame,
    target: str = "guests",
    date_col: str = "date",
    id_col: str = "restaurant_id",
    lag: int = 7,
) -> pd.DataFrame:
    out = df.sort_values([id_col, date_col]).copy()
    out["y_pred"] = out.groupby(id_col, observed=True)[target].shift(lag)
    return out