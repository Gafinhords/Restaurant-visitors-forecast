from __future__ import annotations

import argparse
from pathlib import Path

from src.baseline import naive_lag7
from src.data import filter_stores, load_raw, restore_calendar
from src.validation import score, time_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visits", type=Path, default="data/raw/air_visit_data.csv")
    parser.add_argument("--dates", type=Path, default="data/raw/date_info.csv")
    parser.add_argument("--valid-weeks", type=int, default=6)
    args = parser.parse_args()

    df = load_raw(args.visits, args.dates)
    print(f"Загружено: {len(df)} строк, {df['restaurant_id'].nunique()} ресторанов")

    df = filter_stores(df)
    print(f"После фильтрации: {len(df)} строк, {df['restaurant_id'].nunique()} ресторанов")

    df = restore_calendar(df)
    print(f"После восстановления календаря: {len(df)} строк")

    train, valid = time_split(df, valid_weeks=args.valid_weeks)
    print(f"Train: {train['date'].min().date()} … {train['date'].max().date()} ({len(train)} строк)")
    print(f"Valid: {valid['date'].min().date()} … {valid['date'].max().date()} ({len(valid)} строк)")

    scored = naive_lag7(df)
    valid_pred = scored[scored["date"] > train["date"].max()].dropna(subset=["y_pred"])

    metrics = score(valid_pred["guests"].values, valid_pred["y_pred"].values)
    print(f"\nBaseline lag_7:")
    print(f"  MAE  = {metrics['MAE']:.2f} гостей")
    print(f"  MAPE = {metrics['MAPE']:.2f} %")
    avg = df.groupby("restaurant_id")["guests"].mean()
    valid_pred = valid_pred.assign(avg_store=valid_pred["restaurant_id"].map(avg))

    for threshold in [0, 3, 5, 10, 20, 30]:
        sub = valid_pred[valid_pred["avg_store"] > threshold]
        if len(sub) == 0:
            continue
        m = score(sub["guests"].values, sub["y_pred"].values)
        print(f"avg_store > {threshold:>2}: n={len(sub):>5}, MAE={m['MAE']:.2f}, MAPE={m['MAPE']:.2f}%")
    valid_pred = valid_pred.assign(
        dow=valid_pred["date"].dt.dayofweek,
        abs_err=(valid_pred["guests"] - valid_pred["y_pred"]).abs(),
    )
    print("По дням недели:")
    print(valid_pred.groupby("dow").agg(
        MAE=("abs_err", "mean"),
        n=("abs_err", "size"),
    ).round(2))

    print("\nПо holiday_flg:")
    print(valid_pred.groupby("holiday_flg").agg(
        MAE=("abs_err", "mean"),
        n=("abs_err", "size"),
    ).round(2))

    print("\nХудшие прогнозы:")
    print(valid_pred.nlargest(15, "abs_err")[
              ["restaurant_id", "date", "guests", "y_pred", "abs_err", "holiday_flg"]
          ])

if __name__ == "__main__":
    main()