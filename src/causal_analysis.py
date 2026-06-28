import pandas as pd


def _window_split(df: pd.DataFrame, event_date: str, window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["datetime_hour"] = pd.to_datetime(work["datetime_hour"])
    event_ts = pd.Timestamp(event_date)
    before = work[
        (work["datetime_hour"] >= event_ts - pd.Timedelta(days=window_days))
        & (work["datetime_hour"] < event_ts)
    ].copy()
    after = work[
        (work["datetime_hour"] >= event_ts)
        & (work["datetime_hour"] < event_ts + pd.Timedelta(days=window_days))
    ].copy()
    before["period"] = "Before"
    after["period"] = "After"
    return before, after


def before_after_event_impact(
    df: pd.DataFrame,
    event_date: str,
    zone_id: int | None = None,
    window_days: int = 14,
) -> dict:
    """Compare average hourly demand before vs. after an event date."""
    before, after = _window_split(df, event_date, window_days)
    if zone_id is not None:
        before = before[before["zone_id"] == zone_id]
        after = after[after["zone_id"] == zone_id]

    before_mean = before["trip_count"].mean()
    after_mean = after["trip_count"].mean()
    absolute_change = after_mean - before_mean
    percent_change = absolute_change / before_mean * 100 if before_mean and before_mean > 0 else None

    return {
        "event_date": str(pd.Timestamp(event_date).date()),
        "zone_id": zone_id,
        "window_days": window_days,
        "before_avg_hourly_trips": None if pd.isna(before_mean) else float(before_mean),
        "after_avg_hourly_trips": None if pd.isna(after_mean) else float(after_mean),
        "absolute_change": None if pd.isna(absolute_change) else float(absolute_change),
        "percent_change": None if percent_change is None or pd.isna(percent_change) else float(percent_change),
        "before_rows": int(len(before)),
        "after_rows": int(len(after)),
    }


def difference_in_differences_event_impact(
    df: pd.DataFrame,
    event_date: str,
    treated_zone_id: int,
    control_zone_ids: list[int] | None = None,
    window_days: int = 14,
) -> dict:
    """Simple DiD estimate using the selected zone as treatment and other zones as controls."""
    before, after = _window_split(df, event_date, window_days)
    work = pd.concat([before, after], ignore_index=True)

    if control_zone_ids is None:
        control_zone_ids = sorted([int(z) for z in work["zone_id"].dropna().unique() if int(z) != int(treated_zone_id)])

    treated = work[work["zone_id"] == treated_zone_id].copy()
    control = work[work["zone_id"].isin(control_zone_ids)].copy()

    treated_before = treated.loc[treated["period"] == "Before", "trip_count"].mean()
    treated_after = treated.loc[treated["period"] == "After", "trip_count"].mean()
    control_before = control.loc[control["period"] == "Before", "trip_count"].mean()
    control_after = control.loc[control["period"] == "After", "trip_count"].mean()

    treated_change = treated_after - treated_before
    control_change = control_after - control_before
    did = treated_change - control_change

    summary = pd.DataFrame([
        {"Group": "Treated zone", "Period": "Before", "Average hourly trips": treated_before},
        {"Group": "Treated zone", "Period": "After", "Average hourly trips": treated_after},
        {"Group": "Control zones", "Period": "Before", "Average hourly trips": control_before},
        {"Group": "Control zones", "Period": "After", "Average hourly trips": control_after},
    ])

    return {
        "event_date": str(pd.Timestamp(event_date).date()),
        "treated_zone_id": int(treated_zone_id),
        "control_zone_count": int(len(control_zone_ids)),
        "window_days": int(window_days),
        "treated_before": None if pd.isna(treated_before) else float(treated_before),
        "treated_after": None if pd.isna(treated_after) else float(treated_after),
        "control_before": None if pd.isna(control_before) else float(control_before),
        "control_after": None if pd.isna(control_after) else float(control_after),
        "treated_change": None if pd.isna(treated_change) else float(treated_change),
        "control_change": None if pd.isna(control_change) else float(control_change),
        "did_estimate": None if pd.isna(did) else float(did),
        "summary": summary,
        "treated_rows": int(len(treated)),
        "control_rows": int(len(control)),
    }
