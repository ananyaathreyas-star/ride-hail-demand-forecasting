from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.causal_analysis import before_after_event_impact, difference_in_differences_event_impact
from src.config import DB_PATH, MODELS_DIR
from src.data_pipeline import build_database, load_hourly_demand

st.set_page_config(page_title="Ride-Hail Demand Forecasting", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --sand: #f4f1ea;
        --sage: #a3b19b;
        --sienna: #d48c46;
        --ink: #111111;
        --soft-ink: #3a332c;
        --card: #fffdf8;
    }

    .stApp {
        background: var(--sand);
        color: var(--ink);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.015em;
        font-weight: 700 !important;
    }

    p, li, label, span, div, .stMarkdown {
        color: var(--ink);
    }

    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid rgba(163, 177, 155, 0.55);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 10px 28px rgba(58, 51, 44, 0.06);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--soft-ink) !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-family: 'Cormorant Garamond', serif;
        font-weight: 700;
    }

    [data-testid="stMetricDelta"] {
        color: var(--sienna) !important;
    }

    [data-testid="stSidebar"] {
        background: #ebe6db;
        border-right: 1px solid rgba(163, 177, 155, 0.75);
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(163, 177, 155, 0.65);
        border-radius: 14px;
        background: rgba(255, 253, 248, 0.78);
    }

    .stCaption, caption, small {
        color: var(--soft-ink) !important;
    }

    a { color: var(--sienna) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Ride-Hail Demand Forecasting & Event Impact Dashboard")
st.caption("Analyzing 500k+ Chicago public trips across 2020–2023 to forecast demand and measure event impact")

EVENT_PRESETS = {
    "COVID emergency declaration — 2020-03-13": {"date": "2020-03-13", "note": "Shock to citywide mobility after the national emergency declaration."},
    "Major winter storm — 2022-12-22": {"date": "2022-12-22", "note": "Severe winter weather disrupted Chicago travel before Christmas."},
    "Lollapalooza 2023 starts — 2023-08-03": {"date": "2023-08-03", "note": "Large downtown event likely to shift demand patterns near central zones."},
    "Thanksgiving travel week — 2023-11-22": {"date": "2023-11-22", "note": "Holiday travel period with airport and nightlife demand shifts."},
    "Custom date": {"date": None, "note": "Choose any date within the available data window."},
}


@st.cache_data
def cached_data():
    return load_hourly_demand(DB_PATH)


@st.cache_data(show_spinner=False)
def load_community_area_geojson():
    url = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson?$limit=100"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def community_area_lookup() -> pd.DataFrame:
    geojson = load_community_area_geojson()
    return pd.DataFrame(
        [
            {
                "zone_id": int(feature["properties"].get("area_numbe")),
                "zone_id_str": feature["properties"].get("area_numbe"),
                "community": str(feature["properties"].get("community", "Unknown")).title(),
            }
            for feature in geojson["features"]
            if feature["properties"].get("area_numbe") is not None
        ]
    )


def pct_delta(current: float, previous: float | None) -> str | None:
    if previous is None or previous == 0 or pd.isna(previous):
        return None
    return f"{(current - previous) / previous * 100:+.1f}% vs prior window"


if not DB_PATH.exists():
    with st.spinner("Building starter SQLite database from Chicago public trip data. This can take a few minutes on first deploy..."):
        try:
            build_database(dataset_id="wrvz-psew", limit=500_000, start_date="2020-01-01", end_date="2023-12-31")
        except Exception as exc:
            st.error(f"Could not build the database automatically: {exc}")
            st.info("Run locally with: python3 -m src.data_pipeline --limit 500000 --start-date 2020-01-01 --end-date 2023-12-31")
            st.stop()

hourly = cached_data()
hourly["datetime_hour"] = pd.to_datetime(hourly["datetime_hour"])

date_min = hourly["datetime_hour"].min().date()
date_max = hourly["datetime_hour"].max().date()
total_hourly_rows = len(hourly)
total_trips = int(hourly["trip_count"].sum())
total_zones = hourly["zone_id"].nunique()
coverage_days = max((date_max - date_min).days + 1, 1)

pred_path = MODELS_DIR / "baseline_predictions.csv"
preds = pd.read_csv(pred_path, parse_dates=["datetime_hour"]) if pred_path.exists() else None
if preds is not None and len(preds):
    if "naive_prediction" not in preds.columns:
        preds["naive_prediction"] = np.nan
    errors = preds["trip_count"] - preds["prediction"]
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt((errors**2).mean()))
    ss_res = float((errors**2).sum())
    ss_tot = float(((preds["trip_count"] - preds["trip_count"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    if preds["naive_prediction"].notna().any():
        naive_errors = preds["trip_count"] - preds["naive_prediction"]
        naive_mae = float(naive_errors.abs().mean())
        naive_rmse = float(np.sqrt((naive_errors**2).mean()))
        naive_ss_res = float((naive_errors**2).sum())
        naive_r2 = 1 - naive_ss_res / ss_tot if ss_tot else np.nan
        mae_improvement = (naive_mae - mae) / naive_mae * 100 if naive_mae else np.nan
    else:
        naive_mae = naive_rmse = naive_r2 = mae_improvement = np.nan
else:
    mae = rmse = r2 = naive_mae = naive_rmse = naive_r2 = mae_improvement = np.nan

with st.expander("About this project", expanded=False):
    st.markdown(
        f"""
        This project turns public Chicago trip records into an interactive demand forecasting dashboard. It aggregates raw trips into hourly pickup-zone demand, trains forecasting models, and gives product teams a way to inspect trends, prediction quality, and event impact.

        **Tech stack:** `Python` · `PyTorch` · `SQLite` · `Streamlit`

        **Pipeline:** Chicago API → SQLite → Feature Engineering → LSTM / Baseline Model → Dashboard

        **Current dataset:** {total_trips:,} trips aggregated into {total_hourly_rows:,} hourly zone rows across {total_zones} pickup zones. Available dates: **{date_min} to {date_max}** ({coverage_days:,} calendar days). The event module reports both before/after change and a simple difference-in-differences estimate using other pickup zones as controls.
        """
    )
    metric_table = pd.DataFrame([
        {"Metric": "Rows processed", "Value": f"{total_trips:,} raw trips / {total_hourly_rows:,} hourly rows"},
        {"Metric": "Zones covered", "Value": f"{total_zones}"},
        {"Metric": "Date coverage", "Value": f"{date_min} to {date_max}"},
        {"Metric": "Model MAE", "Value": "not trained" if np.isnan(mae) else f"{mae:.2f} trips/hour"},
        {"Metric": "Naive MAE", "Value": "not trained" if np.isnan(naive_mae) else f"{naive_mae:.2f} trips/hour"},
        {"Metric": "Model R²", "Value": "not trained" if np.isnan(r2) else f"{r2:.3f}"},
    ])
    st.dataframe(metric_table, hide_index=True, width="stretch")

try:
    area_names = community_area_lookup()
except Exception:
    area_names = pd.DataFrame(columns=["zone_id", "zone_id_str", "community"])

with st.sidebar:
    st.header("Filters")
    zones = sorted(int(z) for z in hourly["zone_id"].unique())
    zone_name_map = dict(zip(area_names["zone_id"], area_names["community"])) if not area_names.empty else {}
    zone_options = [{"zone_id": zone, "label": f"{zone_name_map.get(zone, 'Community Area')} ({zone})"} for zone in zones]
    default_idx = next((idx for idx, item in enumerate(zone_options) if item["zone_id"] == 8), 0)
    selected_zone = st.selectbox("Pickup community area", zone_options, index=default_idx, format_func=lambda item: item["label"])
    zone_id = selected_zone["zone_id"]
    st.caption(f"Data window: {date_min} → {date_max}")
    start_date = st.date_input("Start date", value=date_min, min_value=date_min, max_value=date_max, key="start_date")
    end_date = st.date_input("End date", value=date_max, min_value=date_min, max_value=date_max, key="end_date")
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()
    st.caption(f"Selected analysis window: {(end_date - start_date).days + 1:,} day(s)")

start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
filtered = hourly[(hourly["zone_id"] == zone_id) & (hourly["datetime_hour"] >= start) & (hourly["datetime_hour"] < end)]
selected_trips = int(filtered["trip_count"].sum()) if len(filtered) else 0

previous_start = start - (end - start)
previous = hourly[(hourly["zone_id"] == zone_id) & (hourly["datetime_hour"] >= previous_start) & (hourly["datetime_hour"] < start)]
prev_trips = int(previous["trip_count"].sum()) if len(previous) else None
prev_rows = len(previous) if len(previous) else None
prev_avg = float(previous["trip_count"].mean()) if len(previous) else None
prev_peak = float(previous["trip_count"].max()) if len(previous) else None

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Trips", f"{selected_trips:,}", pct_delta(selected_trips, prev_trips))
col2.metric("Hourly rows", f"{len(filtered):,}", pct_delta(len(filtered), prev_rows))
col3.metric("Avg hourly trips", f"{filtered['trip_count'].mean():.1f}" if len(filtered) else "n/a", pct_delta(float(filtered["trip_count"].mean()) if len(filtered) else np.nan, prev_avg))
col4.metric("Peak hourly trips", f"{filtered['trip_count'].max():,.0f}" if len(filtered) else "n/a", pct_delta(float(filtered["trip_count"].max()) if len(filtered) else np.nan, prev_peak))
col5.metric("Model R²", "n/a" if np.isnan(r2) else f"{r2:.3f}", None if np.isnan(naive_r2) else f"naive {naive_r2:.3f}")

if len(filtered) < 2:
    st.warning("This zone/date selection has very little data. Expand the date window or choose another pickup area.")

st.subheader("Chicago demand map")
map_data = hourly[(hourly["datetime_hour"] >= start) & (hourly["datetime_hour"] < end)].groupby("zone_id", as_index=False)["trip_count"].sum()
map_data["zone_id_str"] = map_data["zone_id"].astype(str)
try:
    geojson = load_community_area_geojson()
    area_lookup = community_area_lookup()[["zone_id_str", "community"]]
    valid_area_ids = set(area_lookup["zone_id_str"])
    map_data = map_data[map_data["zone_id_str"].isin(valid_area_ids)].merge(area_lookup, on="zone_id_str", how="left")
    fig_map = px.choropleth_mapbox(
        map_data,
        geojson=geojson,
        locations="zone_id_str",
        featureidkey="properties.area_numbe",
        color="trip_count",
        color_continuous_scale=[[0, "#d9dfd2"], [0.35, "#a3b19b"], [0.7, "#d48c46"], [1, "#9c4f24"]],
        mapbox_style="carto-positron",
        zoom=10.2,
        center={"lat": 41.84, "lon": -87.68},
        opacity=0.82,
        labels={"trip_count": "Trips", "community": "Community area"},
        hover_name="community",
        hover_data={"zone_id_str": False, "trip_count": ":,", "community": False},
        title="Total trips by Chicago community area in selected date window",
    )
    fig_map.update_layout(
        margin={"r": 0, "t": 42, "l": 0, "b": 0},
        height=610,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#111111", "family": "Inter"},
        coloraxis_colorbar={"title": {"text": "Trips", "font": {"color": "#111111"}}, "tickfont": {"color": "#111111"}},
    )
    st.plotly_chart(fig_map, width="stretch")
except Exception as exc:
    st.warning(f"Map could not load from Chicago boundary GeoJSON: {exc}")

st.subheader("Hourly demand with forecast overlay")
if preds is not None:
    pred_zone = preds[preds["zone_id"] == zone_id]
    overlay = filtered[["datetime_hour", "trip_count"]].merge(pred_zone[["datetime_hour", "prediction"]], on="datetime_hour", how="left")
    if overlay["prediction"].notna().any():
        fig = px.line(overlay, x="datetime_hour", y=["trip_count", "prediction"], title=f"Actual vs. predicted hourly demand in zone {zone_id}", labels={"value": "Trips", "variable": "Series"})
    else:
        fig = px.line(filtered, x="datetime_hour", y="trip_count", title=f"Hourly trips in zone {zone_id}")
else:
    fig = px.line(filtered, x="datetime_hour", y="trip_count", title=f"Hourly trips in zone {zone_id}")
st.plotly_chart(fig, width="stretch")

st.subheader("Demand patterns")
if len(filtered):
    heat = filtered.pivot_table(index="day_of_week", columns="hour", values="trip_count", aggfunc="mean").fillna(0)
    fig_heat = px.imshow(heat, labels={"x": "Hour", "y": "Day of week", "color": "Avg trips"}, title="Average hourly demand by weekday/hour", aspect="auto")
    st.plotly_chart(fig_heat, width="stretch")
else:
    st.info("No data in selected range.")

st.subheader("Model performance")
if preds is not None:
    perf1, perf2, perf3, perf4 = st.columns(4)
    perf1.metric("Model MAE", f"{mae:.2f}")
    perf2.metric("Naive MAE", "n/a" if np.isnan(naive_mae) else f"{naive_mae:.2f}")
    perf3.metric("R²", f"{r2:.3f}" if not np.isnan(r2) else "n/a")
    perf4.metric("MAE improvement", "n/a" if np.isnan(mae_improvement) else f"{mae_improvement:+.1f}%")
    comparison = pd.DataFrame([
        {"Model": "Random forest", "MAE": mae, "RMSE": rmse, "R²": r2},
        {"Model": "Naive previous-hour", "MAE": naive_mae, "RMSE": naive_rmse, "R²": naive_r2},
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    plot_limit = float(np.nanpercentile(np.concatenate([preds["trip_count"].values, preds["prediction"].values]), 99))
    scatter_df = preds[(preds["trip_count"] <= plot_limit) & (preds["prediction"] <= plot_limit)].copy()
    scatter = px.scatter(scatter_df, x="trip_count", y="prediction", title=f"Predicted vs. actual demand (clipped at 99th percentile: ≤{plot_limit:.0f} trips)", labels={"trip_count": "Actual trips", "prediction": "Predicted trips"}, opacity=0.45)
    scatter.add_shape(type="line", x0=0, y0=0, x1=plot_limit, y1=plot_limit, line=dict(color="gray", dash="dash"))
    scatter.update_xaxes(range=[0, plot_limit])
    scatter.update_yaxes(range=[0, plot_limit])
    st.plotly_chart(scatter, width="stretch")
else:
    st.info("Train the baseline model to see forecasts: python3 -m src.train_baseline")

st.subheader("Event impact analysis")
st.write("Choose a real event preset or switch to custom date. Presets lock the event date so the event label and analysis date cannot drift out of sync.")
impact_col1, impact_col2, impact_col3 = st.columns([1.5, 1, 1])
with impact_col1:
    event_label = st.selectbox("Event preset", list(EVENT_PRESETS.keys()))
with impact_col2:
    preset_date = EVENT_PRESETS[event_label]["date"]
    if preset_date:
        event_date = pd.to_datetime(preset_date).date()
        st.metric("Event date", str(event_date))
    else:
        event_date = st.date_input("Custom event date", value=min(max(start_date, date_min), date_max), min_value=date_min, max_value=date_max, key="custom_event_date")
with impact_col3:
    window_days = st.slider("Window size in days", 7, 60, 28)

st.caption(EVENT_PRESETS[event_label]["note"])
impact = before_after_event_impact(hourly, str(event_date), zone_id=zone_id, window_days=window_days)
did = difference_in_differences_event_impact(hourly, str(event_date), treated_zone_id=zone_id, window_days=window_days)
metric_cols = st.columns(5)
metric_cols[0].metric("Before avg", "n/a" if impact["before_avg_hourly_trips"] is None else f"{impact['before_avg_hourly_trips']:.1f}")
metric_cols[1].metric("After avg", "n/a" if impact["after_avg_hourly_trips"] is None else f"{impact['after_avg_hourly_trips']:.1f}")
metric_cols[2].metric("Change", "n/a" if impact["absolute_change"] is None else f"{impact['absolute_change']:+.1f}")
metric_cols[3].metric("% change", "n/a" if impact["percent_change"] is None else f"{impact['percent_change']:+.1f}%")
metric_cols[4].metric("DiD estimate", "n/a" if did["did_estimate"] is None else f"{did['did_estimate']:+.2f}")

if impact["before_avg_hourly_trips"] is not None and impact["after_avg_hourly_trips"] is not None:
    impact_df = pd.DataFrame({"Period": ["Before", "After"], "Average hourly trips": [impact["before_avg_hourly_trips"], impact["after_avg_hourly_trips"]]})
    fig_impact = px.bar(impact_df, x="Period", y="Average hourly trips", title=f"Before/after demand around {event_date}", text="Average hourly trips", color="Period")
    fig_impact.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_impact.update_layout(showlegend=False, height=420, yaxis_range=[0, max(impact_df["Average hourly trips"].max() * 1.2, 1)])
    st.plotly_chart(fig_impact, width="stretch")
else:
    st.info("Not enough treated-zone data around that event date/window. Try a wider window or a different zone.")

if did["did_estimate"] is not None:
    did_summary = did["summary"].copy()
    did_summary["Average hourly trips"] = did_summary["Average hourly trips"].astype(float)
    fig_did = px.bar(did_summary, x="Period", y="Average hourly trips", color="Group", barmode="group", title="DiD components: Before/After × Treated/Control", text="Average hourly trips")
    fig_did.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_did.update_layout(height=460, yaxis_range=[0, max(did_summary["Average hourly trips"].max() * 1.2, 1)])
    st.plotly_chart(fig_did, width="stretch")
    st.caption(f"DiD = (treated after − treated before) − (control after − control before). Controls: {did['control_zone_count']} other zones; window: ±{window_days} days.")
else:
    st.info("Not enough data to compute DiD for this event/window.")

with st.expander("Method note"):
    st.markdown(
        """
        The DiD estimate is a simple exploratory causal statistic, not a final academic causal claim.
        It uses the selected pickup zone as the treated unit and all other zones as controls, then compares
        how the treated zone changed relative to the control-zone average over the same event window.
        """
    )
