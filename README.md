<div align="center">

# Ride-Hail Demand Forecasting & Event Impact Dashboard

### A product-style mobility analytics dashboard for forecasting Chicago trip demand and measuring event impact.

<img alt="Python" src="https://img.shields.io/badge/Python-3.12-111111?style=for-the-badge&logo=python&logoColor=white">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Time%20Series-d48c46?style=for-the-badge&logo=pytorch&logoColor=white">
<img alt="SQLite" src="https://img.shields.io/badge/SQLite-Local%20Database-a3b19b?style=for-the-badge&logo=sqlite&logoColor=white">
<img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-f4f1ea?style=for-the-badge&logo=streamlit&logoColor=111111">

<br>

<b>Palette:</b> Sand · Desert Sage · Burnt Sienna · Black editorial typography

</div>

---

## Dashboard Preview

![Ride-hail dashboard map preview](assets/dashboard-map-preview.jpg)

<div align="center">
  <i>Interactive Streamlit dashboard with readable community-area filters, demand KPIs, and a Chicago demand choropleth.</i>
</div>

---

## About This Project

> **Plain-English summary:** This dashboard analyzes public Chicago trip records to forecast hourly pickup demand by community area. It combines SQL-style data storage, time-series feature engineering, model evaluation, and event-impact analysis into one product-style Streamlit app.

<table>
  <tr>
    <td><b>What it does</b></td>
    <td><b>Why it matters</b></td>
  </tr>
  <tr>
    <td>Maps trip demand across Chicago community areas</td>
    <td>Makes spatial demand patterns easy to understand at a glance</td>
  </tr>
  <tr>
    <td>Forecasts hourly pickup demand</td>
    <td>Shows how well recent demand can predict future operational load</td>
  </tr>
  <tr>
    <td>Compares model performance against a naive baseline</td>
    <td>Proves the forecasting model adds value beyond a simple previous-hour rule</td>
  </tr>
  <tr>
    <td>Measures before/after event impact</td>
    <td>Turns real-world events into interpretable product analytics</td>
  </tr>
  <tr>
    <td>Adds a simple DiD estimate using control zones</td>
    <td>Frames the event analysis like a causal inference workflow</td>
  </tr>
</table>

The project mirrors a real mobility product data science workflow:

<div align="center">

**Ingest trip data → Engineer demand features → Evaluate forecasts → Visualize operations → Estimate event impact**

</div>

---

## Product Questions This Answers

- **Where is ride demand concentrated across Chicago community areas?**
- **How does demand change over time by zone, weekday, and hour?**
- **How accurately can a model forecast hourly pickup demand?**
- **How much did major events, such as COVID lockdowns or storms, shift demand?**
- **Did the selected zone change more than comparable control zones?**

---

## Pipeline

<div align="center">

| Stage | Output |
|---|---|
| Chicago Data Portal API | Public trip records |
| SQLite | Local trip-demand database |
| Feature engineering | Hourly zone-level lag and rolling features |
| Forecasting models | Random Forest baseline + PyTorch LSTM scaffold |
| Streamlit dashboard | Map, KPIs, forecasts, model evaluation, event impact |

</div>

```text
Chicago Data Portal API
        ↓
SQLite trip-demand database
        ↓
Hourly zone-level feature engineering
        ↓
Forecasting models: Random Forest baseline + PyTorch LSTM scaffold
        ↓
Streamlit dashboard: map, KPIs, forecasts, model evaluation, event impact
```

---

## Current Results

These metrics are from the broad-date local run using a **500k-row Chicago public trip sample** across **2020–2023**.

| Metric | Value |
|---|---:|
| Raw rows processed | 500,000 trips |
| Hourly zone rows | 58,572 |
| Date coverage | 2020-01-01 to 2023-12-01 |
| Pickup zones covered | 77 |
| Baseline train rows | 46,709 |
| Baseline test rows | 11,709 |
| Random Forest MAE | 4.46 trips/hour |
| Random Forest RMSE | 13.80 trips/hour |
| Random Forest R² | 0.882 |
| Naive previous-hour MAE | 5.12 trips/hour |
| MAE improvement vs. naive | 13.0% |
| Event analysis | Before/after change + simple DiD vs. control zones |

---

## Dashboard Features

### Demand Exploration

- Chicago community-area choropleth map colored by selected-window demand
- Readable community area names in the map tooltip and sidebar dropdown
- KPI cards for trips, hourly rows, average hourly demand, peak demand, and model R²
- Prior-window delta indicators for quick trend comparison
- Weekday/hour demand heatmap

### Forecasting and Model Evaluation

- Forecast-vs-actual overlay for the selected pickup community area
- Model performance table comparing Random Forest against a naive previous-hour baseline
- Predicted-vs-actual scatter plot with outlier-safe axis clipping
- PyTorch LSTM training scaffold for time-series experimentation

### Event Impact and Causal Framing

- Real event presets, including COVID emergency declaration, winter storm, Lollapalooza, and Thanksgiving travel week
- Locked preset dates to prevent event/date mismatch
- Before/after event impact bar chart
- Simple difference-in-differences estimate using other pickup zones as controls
- Clear method note explaining that DiD is exploratory, not a final causal proof

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | Chicago Data Portal / Socrata API, Requests |
| Storage | SQLite |
| Data processing | Pandas, NumPy |
| Forecasting | scikit-learn Random Forest, PyTorch LSTM scaffold |
| Visualization | Streamlit, Plotly, Chicago boundary GeoJSON |
| Deployment target | Streamlit Cloud |

---

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build the Database

Recommended broad-date run:

```bash
python3 -m src.data_pipeline --limit 500000 --start-date 2020-01-01 --end-date 2023-12-31
```

If the Chicago API endpoint changes, pass a different Socrata dataset ID:

```bash
python3 -m src.data_pipeline --dataset-id YOUR_DATASET_ID --limit 500000 --start-date 2020-01-01 --end-date 2023-12-31
```

## Train Models

Random Forest baseline with naive comparison:

```bash
python3 -m src.train_baseline
```

PyTorch LSTM for one Chicago community area:

```bash
python3 -m src.train_lstm --zone-id 8 --epochs 10
```

## Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Streamlit Cloud Deployment

The repo keeps generated data and large model files out of version control. On first hosted launch, the app automatically builds a starter SQLite database from the Chicago public dataset if `data/processed/ride_hail.db` is missing.

Deploy with:

| Setting | Value |
|---|---|
| Repository | `ananyaathreyas-star/ride-hail-demand-forecasting` |
| Branch | `main` |
| Main file path | `app/streamlit_app.py` |

---

## Project Structure

```text
ride_hail_demand_forecasting/
├── app/
│   └── streamlit_app.py
├── assets/
│   └── dashboard-map-preview.jpg
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── causal_analysis.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── features.py
│   ├── train_baseline.py
│   └── train_lstm.py
├── requirements.txt
└── README.md
```
