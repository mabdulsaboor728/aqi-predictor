# Islamabad AQI Forecast

Three-day air quality forecasting for Islamabad, running unattended. An hourly
job ingests data, rebuilds features and publishes a forecast; a daily job
retrains and registers models; a Streamlit dashboard shows the result.

**[Live dashboard →](https://aqi-predictor-by-mas728.streamlit.app)** ·
**[Full report →](REPORT.md)**

---

## Results

Rolling twelve-month holdout, never seen during training:

| Horizon | Model | RMSE | MAE | R² | vs. persistence |
|---|---|---|---|---|---|
| +24h | HistGradientBoosting | 13.56 | 10.20 | 0.807 | 16.9% better |
| +48h | XGBoost | 17.49 | 13.55 | 0.679 | 19.2% better |
| +72h | XGBoost | 19.01 | 14.86 | 0.620 | 22.4% better |

Persistence — assuming tomorrow matches today — is the baseline. It is a hard
bar at 24 hours and has no skill left at 72 (R² −0.148), which is why the
model's advantage grows with horizon.

A second model per horizon handles warnings. The accuracy-optimised forecast
detects 48% of Unhealthy hours three days out; a 90th-percentile quantile model
detects 92%, at the cost of more false alarms. Different jobs need different
models.

## How it works

```
Open-Meteo (CAMS air quality + weather forecast)
        │
        ▼
  hourly pipeline ──── Hopsworks feature store ──── daily retrain
        │                                                │
        │                                          model registry
        ▼                                                │
  batch inference ◄──────────────────────────────────────┘
        │
        ▼
  forecast JSON ────► Streamlit dashboard
```

Forecasts are computed hourly in batch and written to a JSON artifact rather
than served from a live endpoint. The forecast only changes once an hour, so an
endpoint would add infrastructure without reducing any latency a user
experiences.

## Three things worth knowing

**The target is built from rolling averages.** US AQI is the maximum of
per-pollutant sub-indices, each using its own averaging window — PM2.5 over 24
hours, ozone over 8. Matching those windows in the features raised the
correlation between PM2.5 and winter AQI from 0.573 to 0.966. That single
change mattered more than any model choice.

**Features are split by availability, not by type.** Anything named `_t` comes
from pollutant history up to the moment the forecast is issued. Anything named
`_f_h24` is weather or calendar at the target hour, which is available because
weather forecasts exist. Using future pollutant data would be leakage; using a
future weather forecast is not.

**Leakage is tested, not asserted.** Deleting every row after time `t` and
recomputing all 62 origin features leaves all 62 unchanged. The serving feature
row matches the corresponding training row to 4 × 10⁻¹² across all 91 features.

## Repository

```
.github/workflows/   hourly.yml, daily.yml
pipelines/           CI entrypoints
src/
  config.py          location, horizons, variables
  data/              ingestion, cleaning, feature store
  features/          feature construction + leakage checks
  models/            training, inference, SHAP, sequence models
  app/               Streamlit dashboard
models/              6 artifacts (3 point heads, 3 alert heads) + metadata
reports/             metrics, figures, forecast history
```

## Running it

```bash
pip install -r requirements-dev.txt
export HOPSWORKS_API_KEY="..."      # free tier at hopsworks.ai

python -m src.data.fetch_openmeteo   # backfill from Aug 2022
python -m src.data.clean
python -m src.features.build_features
python -m src.models.train
python -m src.models.predict
```

The dashboard needs no credentials — it reads committed JSON and the public
Open-Meteo API:

```bash
pip install -r requirements.txt
streamlit run src/app/dashboard.py
```

## Known limitations

The weather features are drawn from an archive that stitches together the early
hours of successive forecast runs, so training sees more accurate weather than
production will. Reported scores are therefore somewhat optimistic. This is
train/serve skew rather than target leakage, and [the report](REPORT.md#111-forecast-lead-time-skew--the-most-significant-limitation)
covers it in detail alongside five other limitations.

The target is a CAMS model estimate at roughly 40 km resolution, not a ground
station reading — regional air quality rather than street-level.

---

Built as an internship project for 10Pearls. Data from
[Open-Meteo](https://open-meteo.com); air quality from the Copernicus CAMS
global model.
