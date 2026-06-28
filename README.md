# Ride-Hail Demand Forecasting & Event Impact Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Time%20Series-orange)
![SQLite](https://img.shields.io/badge/SQLite-Local%20Database-lightgrey)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## What I built and why

This project analyzes public Chicago trip data to forecast hourly pickup demand by community area and measure how demand changes around major events. The goal is to mirror the kind of product data science workflow used by mobility teams: ingest raw trip data, create reliable demand features, evaluate model accuracy, and surface operational insights in an interactive dashboard.

The dashboard is designed for a hiring manager or product data scientist to quickly answer: where is demand concentrated, how well does the model forecast demand, and what changed before vs. after a selected event?

## Dashboard preview

Add a screenshot or GIF after running locally:

```bash
streamlit run app/streamlit_app.py
```

Suggested repo asset path:

```text
assets/dashboard-preview.gif
```

Then embed it here:

```md
![Dashboard preview](assets/dashboard-preview.gif)
```

## Pipeline

```text
Chicago API → SQLite → Feature Engineering → LSTM / Baseline Model → Streamlit Dashboard
```

## Current metrics

These are from the local broad-date run using a 500k-row Chicago public trip sample across 2020–2023.

| Metric | Value |
|---|---:|
| Raw rows processed | 500,000 trips |
| Hourly zone rows | 58,572 |
| Date coverage | 2020-01-01 to 2023-12-01 |
| Pickup zones covered | 77 |
| Baseline train rows | 46,709 |
| Baseline test rows | 11,709 |
| Random forest MAE | 4.46 trips/hour |
| Random forest RMSE | 13.80 trips/hour |
| Random forest R² | 0.882 |
| Naive previous-hour MAE | 5.12 trips/hour |
| MAE improvement vs. naive | 13.0% |
| Causal/event measure | Before/after change + simple DiD vs. control zones |

## Tech stack

- **Python** for data processing and modeling
- **Pandas / NumPy** for cleaning, aggregation, and feature engineering
- **SQLite** for local SQL storage
- **scikit-learn** for the baseline forecast model
- **PyTorch** for the LSTM time-series model scaffold
- **Streamlit + Plotly** for the dashboard

## Dashboard features

- Demand KPIs with prior-window delta indicators
- Clean date window controls in the sidebar
- Chicago community-area choropleth demand map
- Hourly demand trend chart
- Weekday/hour demand heatmap
- Forecast-vs-actual overlay
- Model performance metrics: MAE, RMSE, R², and naive baseline comparison
- Predicted-vs-actual scatter plot with outlier-safe axis clipping
- Event presets with locked dates plus a separate custom-date mode
- Before/after bar chart for event impact
- Simple difference-in-differences estimate using other pickup zones as controls
- Expandable project explanation for recruiters/hiring managers

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build the database

Recommended broad-date run:

```bash
python3 -m src.data_pipeline --limit 500000 --start-date 2020-01-01 --end-date 2023-12-31
```

If the Chicago API endpoint changes, pass a different Socrata dataset ID:

```bash
python3 -m src.data_pipeline --dataset-id YOUR_DATASET_ID --limit 500000 --start-date 2020-01-01 --end-date 2023-12-31
```

## Train models

Baseline model:

```bash
python3 -m src.train_baseline
```

PyTorch LSTM for one Chicago community area:

```bash
python3 -m src.train_lstm --zone-id 8 --epochs 10
```

## Run dashboard

```bash
streamlit run app/streamlit_app.py
```

## Streamlit Cloud deployment

The `data/` and large model files are gitignored so the repo stays lightweight. On first hosted launch, the Streamlit app automatically builds a starter SQLite database from the stable Chicago dataset if the database is missing. For the most impressive deployed version, run a larger local pipeline, decide whether to persist a lightweight processed artifact, and add a dashboard screenshot/GIF to the README.

## Project structure

```text
ride_hail_demand_forecasting/
├── app/
│   └── streamlit_app.py
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

## Resume bullet

Built an end-to-end ride-hail demand forecasting dashboard using Python, SQLite, PyTorch, and Streamlit, processing 500k public Chicago trip records into hourly zone-level demand forecasts and evaluating event impacts with before/after and difference-in-differences analysis.

## Next improvements

- Add confidence intervals around forecasts
- Add a richer DiD design with manually selected comparable control zones
- Add weather, airport delay, or transit disruption features
- Deploy to Streamlit Cloud and add the public URL
