"""
Step 1 - Data ingestion.

Pulls two hourly time series for the configured city and writes them to data/raw/:

  1. Air quality (CAMS global)   -> pollutants + us_aqi   [the target lives here]
  2. Historical Forecast weather -> meteorological drivers

Why the *Historical Forecast* API and not the ERA5 archive:
at inference time we will only ever have *forecast* weather for t+24..t+72.
If we train on reanalysis (ERA5) and serve on forecasts, the model sees a
different input distribution in production and accuracy drops. The historical
forecast archive is what the forecast actually said at the time, so train and
serve stay consistent.

Run:
    python -m src.data.fetch_openmeteo
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from src import config as cfg

SESSION = requests.Session()
CHUNK_DAYS = 365          # split long ranges so no single request times out
MAX_RETRIES = 4


# --------------------------------------------------------------------------- #
# low-level fetch
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict) -> dict:
    """GET with simple exponential backoff. Open-Meteo rate-limits free usage."""
    for attempt in range(MAX_RETRIES):
        resp = SESSION.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f"  HTTP {resp.status_code} - retrying in {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {url}")


def _hourly_to_frame(payload: dict) -> pd.DataFrame:
    """Open-Meteo returns column-oriented arrays under 'hourly'."""
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _date_chunks(start: str, end: str, size_days: int = CHUNK_DAYS):
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    while s <= e:
        stop = min(s + timedelta(days=size_days - 1), e)
        yield s.isoformat(), stop.isoformat()
        s = stop + timedelta(days=1)


def _fetch_series(url: str, variables: list[str], start: str, end: str,
                  extra: dict | None = None, label: str = "") -> pd.DataFrame:
    frames = []
    for c_start, c_end in _date_chunks(start, end):
        print(f"  [{label}] {c_start} -> {c_end}")
        params = {
            "latitude": cfg.LATITUDE,
            "longitude": cfg.LONGITUDE,
            "hourly": ",".join(variables),
            "start_date": c_start,
            "end_date": c_end,
            "timezone": "UTC",
        }
        if extra:
            params.update(extra)
        frames.append(_hourly_to_frame(_get_json(url, params)))
        time.sleep(1.0)          # be polite to a free API

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# public entrypoints
# --------------------------------------------------------------------------- #
def fetch_air_quality(start: str, end: str) -> pd.DataFrame:
    return _fetch_series(
        cfg.AQ_URL,
        cfg.AQ_VARS,
        start,
        end,
        extra={"domains": "cams_global"},   # Europe-only domain has no Pakistan coverage
        label="air-quality",
    )


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    return _fetch_series(cfg.WX_HIST_URL, cfg.WX_VARS, start, end, label="weather")


def main() -> None:
    start = cfg.BACKFILL_START
    end = cfg.BACKFILL_END or (date.today() - timedelta(days=1)).isoformat()
    print(f"Backfilling {cfg.CITY} ({cfg.LATITUDE}, {cfg.LONGITUDE})  {start} -> {end}\n")

    aq = fetch_air_quality(start, end)
    aq.to_parquet(cfg.DATA_RAW / "air_quality_raw.parquet", index=False)
    print(f"\nair quality : {aq.shape[0]:>6} rows, {aq.shape[1]} cols")

    wx = fetch_weather(start, end)
    wx.to_parquet(cfg.DATA_RAW / "weather_raw.parquet", index=False)
    print(f"weather     : {wx.shape[0]:>6} rows, {wx.shape[1]} cols")

    # inner join on the hourly timestamp - both are UTC, both hourly
    merged = aq.merge(wx, on="time", how="inner", validate="one_to_one")
    merged.to_parquet(cfg.DATA_RAW / "merged_raw.parquet", index=False)

    print(f"merged      : {merged.shape[0]:>6} rows, {merged.shape[1]} cols")
    print(f"range       : {merged['time'].min()} -> {merged['time'].max()}")
    print(f"target null : {merged[cfg.TARGET].isna().mean():.2%}")
    print(f"\nsaved to {cfg.DATA_RAW}")


if __name__ == "__main__":
    main()