"""
Hourly pipeline - the CI entrypoint run by .github/workflows/hourly.yml.

    Open-Meteo (last FETCH_DAYS)  ->  clean  ->  features  ->  push  ->  forecast

Where history comes from, and why it changed
--------------------------------------------
A CI runner starts with an empty filesystem, so data/interim/clean.parquet does
not exist and history has to come from somewhere each run.

This job originally read history from the Hopsworks feature store. That worked,
but the read time grew steadily - 2.6s at first, 56s after a few days, 136s
after a week - because the offline store accumulates one Delta commit per
insert and every read merges them all. Three consecutive hourly runs eventually
failed when the read outran the Query Service's own timeout.

Open-Meteo is the actual source of truth, this job already calls it, and it
serves 60 days as quickly as 10. So history now comes from the API and the
feature store is a WRITE SINK on this path. That removes a growing, timeout-
prone dependency from a job that runs 24 times a day. The daily training job
still reads the store, where a slow read once a day is fine.

FETCH_DAYS must comfortably exceed the longest origin window (168h) plus
REBUILD_DAYS, so the first rebuilt row has full history behind it.

Target maturity
---------------
The air-quality API is a CAMS FORECAST product: for the current day it returns
provisional values for hours that have not happened yet. Those must never
become training labels, or the model learns to imitate CAMS rather than to
predict air quality. build_supervised() is given the current hour as a cap so
only matured targets are written.

Failure policy
--------------
The feature store PUSH is non-fatal. It is idempotent and re-sends a lookback
window every run, so a transient failure self-heals on the next hour - blocking
the forecast for it would turn a recoverable blip into a visible outage.

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
from src.data.clean import clean_frame
from src.data.feature_store import do_incremental, get_fs
from src.data.fetch_openmeteo import fetch_air_quality, fetch_weather
from src.features.build_features import assert_no_leakage, build_supervised

# History fetched from Open-Meteo each run. Must exceed REBUILD_DAYS plus the
# longest origin window (168h = 7d) with margin.
FETCH_DAYS = 60

# how much tail to rebuild features over
REBUILD_DAYS = 30

MIN_HISTORY_HOURS = 24 * 14      # refuse to build features on a short series


def fetch_recent() -> pd.DataFrame:
    """Pull the recent window from both Open-Meteo endpoints and merge."""
    end = date.today()
    start = (end - timedelta(days=FETCH_DAYS)).isoformat()
    print(f"fetching {start} -> {end} from Open-Meteo ({FETCH_DAYS}d)")
    aq = fetch_air_quality(start, end.isoformat())
    wx = fetch_weather(start, end.isoformat())
    df = aq.merge(wx, on="time", how="inner", validate="one_to_one")
    return df.drop(columns=["boundary_layer_height"], errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--skip-push", action="store_true")
    args = ap.parse_args()

    # ---------------------------------------------------------- 1. fetch
    combined = fetch_recent()
    print(f"fetched {len(combined)} hours "
          f"({combined['time'].min()} -> {combined['time'].max()})")

    if len(combined) < MIN_HISTORY_HOURS:
        # Fail loudly: too little history silently produces NaN rolling
        # features, which the serving null check would then reject anyway.
        raise RuntimeError(
            f"only {len(combined)} hours fetched, need at least "
            f"{MIN_HISTORY_HOURS} for the 168h origin windows"
        )

    # ---------------------------------------------------------- 2. clean
    # clean_frame is imported from src.data.clean rather than reimplemented -
    # two copies of the cleaning sequence would drift apart and become
    # training/serving skew.
    cleaned = clean_frame(combined)
    cfg.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    cleaned.reset_index().to_parquet(cfg.DATA_INTERIM / "clean.parquet", index=False)
    print(f"clean: {cleaned.shape}, through {cleaned.index.max()}")

    # ---------------------------------------------------------- 3. features
    now = pd.Timestamp.now(tz="UTC").floor("h")
    tail_start = cleaned.index.max() - pd.Timedelta(days=REBUILD_DAYS)
    print(f"building features (targets matured as of {now})")

    for h in cfg.HORIZONS_H:
        data = build_supervised(cleaned, h, max_target_time=now)
        assert_no_leakage(cleaned, data, h)
        data = data[data.index >= tail_start]
        out = cfg.DATA_PROCESSED / f"dataset_h{h}.parquet"
        data.reset_index().to_parquet(out, index=False)
        print(f"  h={h}: {len(data)} tail rows -> {out.name}")

    # ---------------------------------------------------------- 4. push
    push_failed = False
    if not args.skip_push:
        try:
            do_incremental(get_fs())
        except Exception as exc:                               # noqa: BLE001
            # Non-fatal by design - see the module docstring.
            push_failed = True
            print(f"WARNING: feature store push failed ({type(exc).__name__}): {exc}")
            print("         continuing to forecast; the next run re-sends this window")

    # ---------------------------------------------------------- 5. forecast
    if not args.skip_predict:
        from src.models.predict import predict, render

        render(predict())

    suffix = " (with a non-fatal push failure)" if push_failed else ""
    print(f"\nhourly pipeline complete{suffix}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)