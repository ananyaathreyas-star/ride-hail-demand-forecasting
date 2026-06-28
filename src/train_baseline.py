import argparse
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODELS_DIR
from src.data_pipeline import load_hourly_demand
from src.features import model_frame

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1_hour",
    "lag_2_hours",
    "rolling_3_hour_mean",
    "rolling_12_hour_mean",
]


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_baseline(test_frac: float = 0.2):
    df = model_frame(load_hourly_demand())
    split_time = df["datetime_hour"].quantile(1 - test_frac)
    train = df[df["datetime_hour"] < split_time]
    test = df[df["datetime_hour"] >= split_time]

    if train.empty or test.empty:
        raise ValueError("Not enough rows after feature engineering. Try increasing --limit in src.data_pipeline.")

    X_train, y_train = train[FEATURE_COLS], train["trip_count"]
    X_test, y_test = test[FEATURE_COLS], test["trip_count"]

    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    naive_preds = test["lag_1_hour"].values

    model_metrics = _metrics(y_test, preds)
    naive_metrics = _metrics(y_test, naive_preds)
    metrics = {
        **model_metrics,
        "naive_mae": naive_metrics["mae"],
        "naive_rmse": naive_metrics["rmse"],
        "naive_r2": naive_metrics["r2"],
        "mae_improvement_pct": float((naive_metrics["mae"] - model_metrics["mae"]) / naive_metrics["mae"] * 100),
        "test_rows": int(len(test)),
        "train_rows": int(len(train)),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "baseline_random_forest.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS, "metrics": metrics}, f)

    predictions = test[["zone_id", "datetime_hour", "trip_count"]].copy()
    predictions["prediction"] = preds
    predictions["naive_prediction"] = naive_preds
    predictions.to_csv(MODELS_DIR / "baseline_predictions.csv", index=False)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline demand forecasting model.")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()
    print(train_baseline(test_frac=args.test_frac))
