from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DB_PATH = PROCESSED_DIR / "ride_hail.db"
AGG_TABLE = "hourly_demand"

# Chicago Data Portal public trip dataset.
# Default uses Taxi Trips (2013-2023), which is stable and has the same zone/time fields.
# To use TNP/ride-hail data instead, pass a current Socrata dataset ID with --dataset-id.
DEFAULT_DATASET_ID = "wrvz-psew"
SOCRATA_DOMAIN = "data.cityofchicago.org"
