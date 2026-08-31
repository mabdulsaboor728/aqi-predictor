"""
Hourly pipeline - the CI entrypoint run by .github/workflows/hourly.yml.

    feature store history
              +                -> clean -> features -> push tail -> forecast
    last N days from the API

Why it is shaped this way
-------------------------
A CI runner starts with an empty filesystem, so data/interim/clean.parquet does
not exist. Two bad options and one good one:

  * re-fetch four years from Open-Meteo every hour  - slow, and abusive of a
    free API
  * cache the parquet between runs                  - caches expire and go
    stale silently
  * treat the FEATURE STORE as the state            - read history from
    Hopsworks, fetch only recent hours from the API, merge, rebuild

The third is what a feature store is for, so that is what this does.

Rolling features need history: origin windows reach back 168h and weather
windows 72h. We therefore rebuild features over a REBUILD_DAYS tail rather than
just the newest hour, then let the upsert push only what changed.

Failure policy
--------------
The feature store PUSH is non-fatal. It is idempotent and re-sends a 72h window
every run, so a transient failure self-heals on the next hour - blocking the
forecast for it would turn a recoverable blip into a visible outage. The READ
is fatal, because without history there is nothing to build; it retries inside
src.data.feature_store instead.

Run:
    python -m pipelines.hourly
    python -m pipelines.hourly --skip-predict
    python -m pipelines.hourly --skip-push
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, timedelta

import pandas as pd

from src import config as cfg
from src.data import clean as C
from src.data.feature_store import do_incremental, get_fs, read_raw
from src.data.fetch_openmeteo import fetch_air_quality, fetch_weather
from src.features.build_features import assert_no_leakage, build_supervised

# how far back to re-fetch from the API each run. Generous on purpose: CAMS
# revises recently published hours, and upsert makes overlap free.
FETCH_DAYS = 10

# how much tail to rebuild features over. Must comfortably exceed the longest
# origin window (168h) so the first rebuilt row has full history.
REBUILD_DAYS = 30

CALENDAR_COLS = ["hour_local", "dayofweek", "month", "dayofyear"]


def fetch_recent() -> pd.DataFrame:
    end = date.today()
    start = (end - timedelta(days=FETCH_DAYS)).isoformat()
    print(f"fetching {start} -> {end} from Open-Meteo")
    aq = fetch_air_quality(start, end.isoformat())
    wx = fetch_weather(start, end.isoformat())
    df = aq.merge(wx, on="time", how="inner", validate="one_to_one")
    return df.drop(columns=["boundary_layer_height"], errors="ignore")


def merge_history(stored: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine stored history with freshly fetched hours; fresh wins on overlap."""
    stored = stored.drop(columns=CALENDAR_COLS, errors="ignore").reset_index()
    stored["time"] = pd.to_datetime(stored["time"], utc=True)
    fresh["time"] = pd.to_datetime(fresh["time"], utc=True)

    cols = [c for c in stored.columns if c in fresh.columns]
    combined = pd.concat([stored[cols], fresh[cols]], ignore_index=True)
    combined = (combined.sort_values("time")
                        .drop_duplicates(subset="time", keep="last")
                        .reset_index(drop=True))
    print(f"merged: {len(stored)} stored + {len(fresh)} fresh -> {len(combined)} rows")
    return combined


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning steps as src.data.clean, applied to an in-memory frame."""
    df = C.trim_leading_gap(df)
    df = C.enforce_hourly_grid(df)
    df = C.clip_physical(df)
    df = C.fill_short_gaps(df)
    return C.add_calendar(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--skip-push", action="store_true")
    args = ap.parse_args()

    fs = get_fs()

    # ---------------------------------------------------------- 1. assemble
    # fatal if this fails: without history there is nothing to build. Retries
    # for transient Query Service errors live in src.data.feature_store.
    stored = read_raw(fs)
    print(f"feature store: {len(stored)} rows through {stored.index.max()}")
    combined = merge_history(stored, fetch_recent())

    # ---------------------------------------------------------- 2. clean
    cleaned = clean_frame(combined)
    cfg.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    cleaned.reset_index().to_parquet(cfg.DATA_INTERIM / "clean.parquet", index=False)
    print(f"clean: {cleaned.shape}, through {cleaned.index.max()}")

    # ---------------------------------------------------------- 3. features
    tail_start = cleaned.index.max() - pd.Timedelta(days=REBUILD_DAYS)
    for h in cfg.HORIZONS_H:
        data = build_supervised(cleaned, h)
        assert_no_leakage(cleaned, data, h)
        data = data[data.index >= tail_start]
        out = cfg.DATA_PROCESSED / f"dataset_h{h}.parquet"
        data.reset_index().to_parquet(out, index=False)
        print(f"  h={h}: {len(data)} tail rows -> {out.name}")

    # ---------------------------------------------------------- 4. push
    push_failed = False
    if not args.skip_push:
        try:
            do_incremental(fs)
        except Exception as exc:                               # noqa: BLE001
            # Non-fatal by design. The push is idempotent and re-sends a 72h
            # window each run, so this self-heals next hour. Failing here would
            # also skip the forecast, turning a blip into a visible outage.
            push_failed = True
            print(f"WARNING: feature store push failed ({type(exc).__name__}): {exc}")
            print("         continuing to forecast; the next run re-sends this window")

    # ---------------------------------------------------------- 5. forecast
    if not args.skip_predict:
        from src.models.predict import predict, render

        render(predict())

    if push_failed:
        print("\nhourly pipeline complete (with a non-fatal push failure)")
    else:
        print("\nhourly pipeline complete")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)