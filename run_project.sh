#!/usr/bin/env bash
set -euo pipefail

python3 -m src.data_pipeline --limit 100000
python3 -m src.train_baseline
streamlit run app/streamlit_app.py
