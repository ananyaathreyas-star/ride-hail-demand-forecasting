import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime_hour"] = pd.to_datetime(out["datetime_hour"])
    out = out.sort_values(["zone_id", "datetime_hour"])
    out["hour"] = out["datetime_hour"].dt.hour
    out["day_of_week"] = out["datetime_hour"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    grouped = out.groupby("zone_id")["trip_count"]
    out["lag_1_hour"] = grouped.shift(1)
    out["lag_2_hours"] = grouped.shift(2)
    out["lag_24_hours"] = grouped.shift(24)
    out["rolling_3_hour_mean"] = grouped.shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    out["rolling_12_hour_mean"] = grouped.shift(1).rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
    return out


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_lag_features(df)
    feature_cols = [
        "hour",
        "day_of_week",
        "is_weekend",
        "lag_1_hour",
        "lag_2_hours",
        "rolling_3_hour_mean",
        "rolling_12_hour_mean",
    ]
    return out.dropna(subset=feature_cols + ["trip_count"]).reset_index(drop=True)
