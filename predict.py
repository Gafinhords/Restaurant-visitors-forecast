from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, make_features
from src.model import load_model


def forecast(
    history: pd.DataFrame,
    restaurant_id: int,
    start_date: pd.Timestamp,
    model,
    feature_cols: list[str],
    horizon: int = 7,
) -> pd.DataFrame:

    hist = history[history["restaurant_id"] == restaurant_id].copy()
    hist = hist.sort_values("date").reset_index(drop=True)

    if hist.empty:
        raise ValueError(f"Ресторан {restaurant_id} не найден в истории")

    last_date = hist["date"].max()
    if last_date >= start_date:
        raise ValueError(
            f"История содержит даты >= {start_date.date()}. "
            f"Отрежь данные до {start_date.date()} перед прогнозом."
        )

    predictions = []

    for step in range(horizon):
        target_date = start_date + pd.Timedelta(days=step)

        new_row = pd.DataFrame([{
            "date": target_date,
            "restaurant_id": restaurant_id,
            "guests": np.nan,
            "holiday_flg": 0,
        }])
        extended = pd.concat([hist, pd.DataFrame(predictions), new_row], ignore_index=True)
        extended = extended.sort_values("date").reset_index(drop=True)
        extended["date"] = pd.to_datetime(extended["date"])
        extended["restaurant_id"] = extended["restaurant_id"].astype("int64")
        extended["guests"] = pd.to_numeric(extended["guests"], errors="coerce")
        extended["holiday_flg"] = extended["holiday_flg"].fillna(0).astype("int64")
        if "revenue" in extended.columns:
            extended["revenue"] = pd.to_numeric(extended["revenue"], errors="coerce")
        extended = extended.sort_values("date").reset_index(drop=True)
        print("extended dtypes:")
        print(extended.dtypes)
        print("extended head:")
        print(extended.tail(3))
        featured = make_features(extended)
        row = featured[featured["date"] == target_date].iloc[-1]

        if row[feature_cols].isna().any():
            missing = row[feature_cols][row[feature_cols].isna()].index.tolist()
            raise ValueError(f"Недостаточно истории для {target_date.date()}. "
                             f"NaN в признаках: {missing}")
        row_df = row[feature_cols].to_frame().T
        print("row_df dtypes:")
        print(row_df.dtypes)
        print("row_df values:")
        print(row_df.values)
        row_df = row[feature_cols].to_frame().T.astype("float64")
        y_pred = float(model.predict(row_df)[0])

        predictions.append({
            "date": target_date,
            "restaurant_id": restaurant_id,
            "guests": y_pred,
            "holiday_flg": 0,
        })

    result = pd.DataFrame(predictions)
    return result[["date", "restaurant_id", "guests"]].rename(
        columns={"guests": "y_pred"}
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогноз потока гостей на 7 дней вперёд."
    )
    parser.add_argument("--date", required=True, type=str,
                        help="Первый день прогноза, формат YYYY-MM-DD")
    parser.add_argument("--restaurant", required=True, type=int,
                        help="ID ресторана")
    parser.add_argument("--model", type=Path, default="models/lgbm.pkl",
                        help="Путь к сохранённой модели")
    parser.add_argument("--data", type=Path, default="data/processed/df_clean.csv",
                        help="Путь к истории")
    parser.add_argument("--out", type=Path, default=None,
                        help="Куда сохранить CSV (опционально)")
    parser.add_argument("--horizon", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        start_date = pd.Timestamp(args.date)
    except Exception as e:
        raise SystemExit(f"Некорректная дата '{args.date}': {e}")

    if not args.data.exists():
        raise SystemExit(f"Файл истории не найден: {args.data}")
    if not args.model.exists():
        raise SystemExit(f"Файл модели не найден: {args.model}")

    history = pd.read_csv(args.data, parse_dates=["date"])
    history = history[history["date"] < start_date].copy()

    if history.empty:
        raise SystemExit(f"Нет истории до {start_date.date()}")

    model = load_model(args.model)
    feature_cols = FEATURE_COLS

    try:
        result = forecast(history, args.restaurant, start_date,
                          model, feature_cols, horizon=args.horizon)
    except ValueError as e:
        raise SystemExit(f"Ошибка прогноза: {e}")

    print(f"Прогноз для ресторана {args.restaurant}")
    dow_ru = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    for _, row in result.iterrows():
        dow = dow_ru[row["date"].dayofweek]
        print(f"  {row['date'].date()} ({dow})  {row['y_pred']:.0f} гостей")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"\nСохранено в {args.out}")


if __name__ == "__main__":
    main()