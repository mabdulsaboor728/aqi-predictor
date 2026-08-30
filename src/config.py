"""Central configuration for the AQI predictor project."""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- location
CITY = "Islamabad"
LATITUDE = 33.6844
LONGITUDE = 73.0479
TIMEZONE = "UTC"          # keep everything UTC internally; localise only for display
LOCAL_TZ = "Asia/Karachi"

# ---------------------------------------------------------------- backfill window
# Historical Forecast API coverage starts ~2022. Leave a margin.
BACKFILL_START = "2022-08-01"
BACKFILL_END = None        # None -> yesterday

# ---------------------------------------------------------------- API endpoints
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WX_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------------- variables
# Pollutants + the target. These are ONLY available up to "now" at inference time.
AQ_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "aerosol_optical_depth",
    "us_aqi",
]

# Weather. These ARE available for future hours via the forecast API,
# which is what makes a 72h horizon possible at all.
WX_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "boundary_layer_height",
    "shortwave_radiation",
]

TARGET = "us_aqi"

# ---------------------------------------------------------------- modelling
HORIZONS_H = [24, 48, 72]   # direct multi-horizon: one model per horizon