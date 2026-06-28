import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import AGG_TABLE, DB_PATH, DEFAULT_DATASET_ID, PROCESSED_DIR, RAW_DIR, SOCRATA_DOMAIN

# Keep downloader columns intentionally minimal so the project works with both
# Chicago Taxi Trips and TNP/ride-hail datasets when their shared fields exist.
COLUMNS = [
    "trip_start_timestamp",
    "pickup_community_area",
    "dropoff_community_area",
    "trip_seconds",
    "trip_miles",
    "fare",
    "tips",
    "trip_total",
]


def _month_windows(start_date: str, end_date: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    starts = pd.date_range(start=start, end=end, freq="MS")
    windows = []
    for month_start in starts:
        month_end = min(month_start + pd.offsets.MonthBegin(1), end + pd.Timedelta(days=1))
        windows.append((month_start, month_end))
    return windows


def download_chicago_trip_sample(
    dataset_id: str = DEFAULT_DATASET_ID,
    limit: int = 500_000,
    output_path: Optional[Path] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
) -> Path:
    """Download a broad, month-stratified sample of Chicago public trip records.

    A plain `$order=trip_start_timestamp LIMIT n` query only pulls the first few days of
    the dataset. This function spreads the requested limit across month windows so the
    dashboard has useful temporal coverage for trends, events, and train/test splits.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or RAW_DIR / f"chicago_trip_sample_{dataset_id}_{start_date}_{end_date}_{limit}.csv"

    select = ",".join(COLUMNS)
    url = f"https://{SOCRATA_DOMAIN}/resource/{dataset_id}.json"
    windows = _month_windows(start_date, end_date)
    per_window = max(limit // len(windows), 1)
    remainder = limit - per_window * len(windows)

    frames: list[pd.DataFrame] = []
    for i, (window_start, window_end) in enumerate(windows):
        window_limit = per_window + (1 if i < remainder else 0)
        where = (
            "pickup_community_area IS NOT NULL "
            "AND trip_start_timestamp IS NOT NULL "
            f"AND trip_start_timestamp >= '{window_start:%Y-%m-%dT%H:%M:%S}' "
            f"AND trip_start_timestamp < '{window_end:%Y-%m-%dT%H:%M:%S}'"
        )
        params = {
            "$select": select,
            "$limit": window_limit,
            "$order": "trip_start_timestamp",
            "$where": where,
        }
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        records = response.json()
        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        raise ValueError("No records returned. Try another dataset ID or date range.")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_path, index=False)
    return output_path


def clean_and_aggregate(raw_csv: Path) -> pd.DataFrame:
    """Convert raw trip records into hourly demand by pickup community area."""
    df = pd.read_csv(raw_csv)
    df["trip_start_timestamp"] = pd.to_datetime(df["trip_start_timestamp"], errors="coerce")
    df["pickup_community_area"] = pd.to_numeric(df["pickup_community_area"], errors="coerce")
    df = df.dropna(subset=["trip_start_timestamp", "pickup_community_area"])

    df["zone_id"] = df["pickup_community_area"].astype(int)
    df["datetime_hour"] = df["trip_start_timestamp"].dt.floor("h")
    df["day_of_week"] = df["datetime_hour"].dt.dayofweek
    df["hour"] = df["datetime_hour"].dt.hour
    df["date"] = df["datetime_hour"].dt.date.astype(str)

    hourly = (
        df.groupby(["zone_id", "datetime_hour", "date", "day_of_week", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "trip_count"})
        .sort_values(["zone_id", "datetime_hour"])
    )
    hourly["datetime_hour"] = hourly["datetime_hour"].astype(str)
    return hourly


def write_sqlite(hourly: pd.DataFrame, db_path: Path = DB_PATH) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        hourly.to_sql(AGG_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{AGG_TABLE}_zone_time ON {AGG_TABLE}(zone_id, datetime_hour)")
    return db_path


def load_hourly_demand(db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {AGG_TABLE}", conn, parse_dates=["datetime_hour"])


def build_database(
    dataset_id: str,
    limit: int,
    raw_csv: Optional[str] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
) -> Path:
    if raw_csv:
        raw_path = Path(raw_csv)
    else:
        raw_path = download_chicago_trip_sample(
            dataset_id=dataset_id,
            limit=limit,
            start_date=start_date,
            end_date=end_date,
        )
    hourly = clean_and_aggregate(raw_path)
    return write_sqlite(hourly)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SQLite database from Chicago public trip records.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--limit", type=int, default=500_000)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--raw-csv", default=None, help="Use an existing raw CSV instead of downloading.")
    args = parser.parse_args()

    db = build_database(args.dataset_id, args.limit, args.raw_csv, args.start_date, args.end_date)
    print(f"Wrote database to {db}")
