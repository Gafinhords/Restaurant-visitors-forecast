from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baseline import naive_lag7
from src.data import filter_stores, load_raw, restore_calendar
from src.features import FEATURE_COLS, make_features
from src.model import blend, build_lgbm, build_ridge, predict_model, train_model
from src.validation import score, time_split
from src.model import save_model


from pathlib import Path
Path("reports").mkdir(exist_ok=True)


def evaluate(name: str, y_true, y_pred) -> dict:
    m = score(y_true, y_pred)
    print(f"{name:<22} MAE = {m['MAE']:.2f}   MAPE = {m['MAPE']:.2f}%")
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visits", type=Path, default="data/raw/air_visit_data.csv")
    parser.add_argument("--dates", type=Path, default="data/raw/date_info.csv")
    parser.add_argument("--valid-weeks", type=int, default=6)
    args = parser.parse_args()

    df = load_raw(args.visits, args.dates)
    df = filter_stores(df)
    df = restore_calendar(df)
    df.to_csv("data/processed/df_clean.csv", index=False)
    scored = naive_lag7(df)
    train_raw, _ = time_split(df, valid_weeks=args.valid_weeks)
    baseline_valid = scored[scored["date"] > train_raw["date"].max()].dropna(subset=["y_pred"])

    featured = make_features(df)
    train, valid = time_split(featured, valid_weeks=args.valid_weeks)
    valid_model = valid.dropna(subset=FEATURE_COLS + ["guests"]).copy()

    print("=== Валидация ===")
    print(f"Train: {train['date'].min().date()} … {train['date'].max().date()}")
    print(f"Valid: {valid['date'].min().date()} … {valid['date'].max().date()}")
    print(f"Valid с признаками: {len(valid_model)}\n")

    print("=== Метрики ===")
    evaluate("Baseline lag_7",
             baseline_valid["guests"].values,
             baseline_valid["y_pred"].values)

    ridge, cols = train_model(train, build_ridge())
    ridge_pred = predict_model(ridge, valid_model, cols)["y_pred"].values
    evaluate("Ridge", valid_model["guests"].values, ridge_pred)

    lgbm, _ = train_model(train, build_lgbm())
    lgbm_pred = predict_model(lgbm, valid_model, cols)["y_pred"].values
    evaluate("LightGBM", valid_model["guests"].values, lgbm_pred)

    print("\n=== Диагностика LightGBM ===")
    valid_model = valid_model.assign(
        lgbm_pred=lgbm_pred,
        ridge_pred=ridge_pred,
        abs_err_lgbm=lambda d: (d["guests"] - d["lgbm_pred"]).abs(),
    )
    avg = df.groupby("restaurant_id")["guests"].mean()
    valid_model["avg_store"] = valid_model["restaurant_id"].map(avg)

    print("По размеру ресторана:")
    for thr in [0, 10, 20, 30]:
        sub = valid_model[valid_model["avg_store"] > thr]
        if len(sub) == 0:
            continue
        mae_l = sub["abs_err_lgbm"].mean()
        mape_l = (sub["abs_err_lgbm"] / sub["guests"].clip(lower=1)).mean() * 100
        mae_r = (sub["guests"] - sub["ridge_pred"]).abs().mean()
        print(f"  avg_store>{thr:>2}: n={len(sub):>4}  LGBM MAE={mae_l:.2f} MAPE={mape_l:.1f}%  |  Ridge MAE={mae_r:.2f}")

    print("\nПо праздникам:")
    for h, g in valid_model.groupby("holiday_flg"):
        mae_l = g["abs_err_lgbm"].mean()
        print(f"  holiday={h}: n={len(g):>4}  LGBM MAE={mae_l:.2f}")

    # --- Важности признаков ---
    imp = pd.Series(lgbm.feature_importances_, index=cols).sort_values(ascending=False)
    print("\n=== Топ-10 признаков (LightGBM) ===")
    print(imp.head(10))

    fig, ax = plt.subplots(figsize=(10, 5))
    imp.head(15).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Важности признаков (LightGBM)")
    ax.set_xlabel("Важность")
    plt.tight_layout()
    fig.savefig("reports/feature_importance.png", dpi=100)
    print("\nГрафик сохранён в reports/feature_importance.png")

    save_model(lgbm, "models/lgbm.pkl")
    save_model(ridge, "models/ridge.pkl")
    print("Модели сохранены: models/lgbm.pkl, models/ridge.pkl")

    df.to_csv("data/processed/df_clean.csv", index=False)
    print(f"История сохранена: data/processed/df_clean.csv ({len(df)} строк)")

if __name__ == "__main__":
    main()