"""
Hopsworks integration - feature store + model registry.

Layout in the project `aqi_proj`:

    aqi_raw_hourly       v1   cleaned hourly observations (source of truth)
    aqi_features_h24     v1   model-ready features, horizon 24h
    aqi_features_h48     v1   model-ready features, horizon 48h
    aqi_features_h72     v1   model-ready features, horizon 72h

Both raw and computed features are stored. The raw group is the audit trail and
lets you rebuild features after a logic change; the feature groups guarantee
training and serving read identical columns, which is the drift problem a
feature store exists to solve.

Primary key is `unix_ts` (int64 seconds) rather than the timestamp itself,
because Hopsworks primary keys must be a simple scalar. `time` is the event
time, which powers point-in-time joins. With Delta time travel, inserting a row
whose `unix_ts` already exists UPSERTS it, so re-running the hourly job over
overlapping hours is safe and idempotent.

Reliability
-----------
Both directions are retried, for different reasons:

  writes  delta-rs opens a direct HDFS RPC connection, which drops
          intermittently from outside the Hopsworks network. Chunking keeps
          each session short; upsert makes every retry idempotent.
  reads   the Arrow Flight Query Service returns transient gRPC UNAVAILABLE
          ("Socket closed") under load. Reads are pure, so retrying is free.

Neither retry hides a real error: schema mismatches and duplicate keys still
fail immediately and loudly.

Setup
-----
    pip install hopsworks deltalake
    export HOPSWORKS_API_KEY="..."      # never commit this

Usage
-----
    python -m src.data.feature_store --check
    python -m src.data.feature_store --backfill
    python -m src.data.feature_store --backfill --only 72     # resume one group
    python -m src.data.feature_store --incremental            # hourly job
    python -m src.data.feature_store --register-models
"""

from __future__ import annotations

import argparse
import json
import os
import time as _time

import pandas as pd

from src import config as cfg

PROJECT = "aqi_proj"
RAW_FG = "aqi_raw_hourly"
FEAT_FG = "aqi_features_h{h}"
FG_VERSION = 1

# Delta gives upsert-on-primary-key. Override to "NONE" only if the delta
# library cannot be installed - see the note in do_incremental().
TIME_TRAVEL_FORMAT = os.environ.get("HOPSWORKS_TT_FORMAT", "DELTA")

# write tuning
CHUNK_ROWS = 8000
MAX_RETRIES = 4
RETRY_BASE_S = 5

# read tuning
READ_RETRIES = 3
READ_BACKOFF_S = 20

# how many recent hours the incremental job re-sends. Overlap absorbs late
# CAMS corrections; upsert makes the duplication harmless.
INCREMENTAL_LOOKBACK_H = 72


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def login():
    """Connect using HOPSWORKS_API_KEY. Identical locally and in CI."""
    import hopsworks

    key = os.environ.get("HOPSWORKS_API_KEY")
    if not key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is not set.\n"
            "  local :  export HOPSWORKS_API_KEY='...'\n"
            "  CI    :  add it as a GitHub repository secret"
        )
    return hopsworks.login(project=PROJECT, api_key_value=key)


def get_fs():
    return login().get_feature_store()


# --------------------------------------------------------------------------- #
# frame preparation
# --------------------------------------------------------------------------- #
def to_hopsworks_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a time-indexed or time-columned frame for Hopsworks.

    - exactly one `time` column, tz-naive UTC (tz-aware types round-trip
      unreliably through the offline store)
    - `unix_ts` int64 seconds added as the primary key
    - column names lowercased; Hopsworks does this anyway, and doing it here
      keeps local and remote schemas identical
    - no stray index column, whatever the caller passed in
    """
    out = df.copy()

    # If `time` is the index, promote it to a column. If it is ALREADY a
    # column, discard the index entirely. A boolean-filtered frame carries a
    # plain Index rather than a RangeIndex under pandas 2.x, and calling
    # reset_index() on it injects a spurious `index` column that Hopsworks
    # rejects with "index (type: 'bigint') does not exist in feature group".
    if "time" in out.columns:
        out = out.reset_index(drop=True)
    else:
        out = out.reset_index()

    if "time" not in out.columns:
        raise ValueError("expected a 'time' column or a time-named index")

    ts = pd.to_datetime(out["time"], utc=True)
    out["unix_ts"] = (ts.astype("int64") // 10**9).astype("int64")
    out["time"] = ts.dt.tz_localize(None)
    out.columns = [c.lower() for c in out.columns]

    for c in out.select_dtypes("float32").columns:
        out[c] = out[c].astype("float64")

    if "index" in out.columns:
        # belt and braces: nothing above should produce this, but a silent
        # schema mismatch costs a full CI cycle to diagnose
        raise ValueError("stray 'index' column - would break the feature group schema")

    dupes = int(out["unix_ts"].duplicated().sum())
    if dupes:
        raise ValueError(f"{dupes} duplicate timestamps - refusing to write")
    return out


def _fg(fs, name: str, description: str):
    return fs.get_or_create_feature_group(
        name=name,
        version=FG_VERSION,
        description=description,
        primary_key=["unix_ts"],
        event_time="time",
        online_enabled=False,
        time_travel_format=TIME_TRAVEL_FORMAT,
    )


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #
def _insert_chunked(fg, frame: pd.DataFrame, label: str) -> None:
    """Write in chunks, retrying each with exponential backoff.

    Safe to retry because the primary key is deterministic: re-sending a chunk
    overwrites those rows rather than duplicating them.
    """
    n = len(frame)
    if n == 0:
        print(f"    {label}: nothing to write")
        return

    for start in range(0, n, CHUNK_ROWS):
        part = frame.iloc[start:start + CHUNK_ROWS]
        end = start + len(part)
        for attempt in range(MAX_RETRIES):
            try:
                fg.insert(part, write_options={"wait_for_job": True})
                print(f"    {label}: rows {start}-{end} of {n} ok")
                break
            except Exception as exc:                          # noqa: BLE001
                wait = RETRY_BASE_S * 2 ** attempt
                print(f"    {label}: chunk at {start} failed "
                      f"({type(exc).__name__}: {exc}); retry in {wait}s "
                      f"[{attempt + 1}/{MAX_RETRIES}]")
                _time.sleep(wait)
        else:
            raise RuntimeError(
                f"{label}: chunk at row {start} failed after {MAX_RETRIES} attempts. "
                f"Rerun with --backfill --only to resume just this group."
            )


def write_raw(fs, df: pd.DataFrame) -> None:
    frame = to_hopsworks_frame(df)
    fg = _fg(fs, RAW_FG, f"Cleaned hourly air-quality and weather for {cfg.CITY}")
    print(f"  {RAW_FG}: {len(frame)} rows x {frame.shape[1]} cols")
    _insert_chunked(fg, frame, RAW_FG)


def write_features(fs, h: int, df: pd.DataFrame) -> None:
    frame = to_hopsworks_frame(df)
    name = FEAT_FG.format(h=h)
    fg = _fg(fs, name, f"Model-ready features and target for a {h}h-ahead forecast")
    print(f"  {name}: {len(frame)} rows x {frame.shape[1]} cols")
    _insert_chunked(fg, frame, name)


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #
def _read_with_retry(fg, label: str) -> pd.DataFrame:
    """Read a feature group, retrying transient Query Service failures.

    The Arrow Flight service intermittently returns gRPC UNAVAILABLE
    ("Socket closed") when it is under load - a server-side condition, not a
    client bug. Reads are pure, so retrying is always safe.
    """
    for attempt in range(READ_RETRIES):
        try:
            return fg.read()
        except Exception as exc:                              # noqa: BLE001
            if attempt == READ_RETRIES - 1:
                raise
            wait = READ_BACKOFF_S * (attempt + 1)
            print(f"  {label}: read failed ({type(exc).__name__}), "
                  f"retry in {wait}s [{attempt + 1}/{READ_RETRIES}]")
            _time.sleep(wait)
    raise RuntimeError(f"{label}: unreachable")               # pragma: no cover


def _restore_index(df: pd.DataFrame) -> pd.DataFrame:
    """Offline reads come back unordered - always re-sort.

    If the feature groups were created with time_travel_format="NONE" (no
    upsert), an `ingested_at` column would be needed here to deduplicate.
    With Delta the store already holds one row per key.
    """
    df = df.drop(columns=["unix_ts", "index"], errors="ignore")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").set_index("time")


def read_raw(fs) -> pd.DataFrame:
    return _restore_index(_read_with_retry(_fg(fs, RAW_FG, ""), RAW_FG))


def read_features(fs, h: int) -> pd.DataFrame:
    name = FEAT_FG.format(h=h)
    return _restore_index(_read_with_retry(_fg(fs, name, ""), name))


def latest_timestamp(fs, name: str):
    """Most recent event time already stored, or None if empty/absent."""
    try:
        fg = fs.get_feature_group(name, version=FG_VERSION)
    except Exception:                                         # noqa: BLE001
        return None
    try:
        mx = fg.select(["unix_ts"]).read()["unix_ts"].max()
        return pd.Timestamp(int(mx), unit="s", tz="UTC") if pd.notna(mx) else None
    except Exception:                                         # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #
def register_models(project) -> None:
    """Push the six artifacts (3 point heads, 3 alert heads) with their metrics.

    The metric names written here are what src/models/predict.py selects on:
    `RMSE` for point heads, `alert_f1` for alert heads.
    """
    import shutil
    import tempfile

    mr = project.get_model_registry()

    for h in cfg.HORIZONS_H:
        meta_path = cfg.MODELS_DIR / f"model_h{h}_meta.json"
        if not meta_path.exists():
            print(f"  h={h}: no metadata, skipping (run training first)")
            continue
        meta = json.loads(meta_path.read_text())

        for kind, fname in [("point", f"model_h{h}.joblib"),
                            ("alert", f"alert_h{h}.joblib")]:
            src = cfg.MODELS_DIR / fname
            if not src.exists():
                print(f"  h={h} {kind}: {fname} missing, skipping")
                continue

            if kind == "point":
                metrics = {k: float(v) for k, v in meta["holdout"].items()
                           if isinstance(v, (int, float)) and v == v}
            else:
                metrics = {f"alert_{k}": float(v)
                           for k, v in meta["alert_quantile_head"].items()
                           if isinstance(v, (int, float))}

            with tempfile.TemporaryDirectory() as tmp:
                shutil.copy(src, tmp)
                shutil.copy(meta_path, tmp)
                model = mr.python.create_model(
                    name=f"aqi_{kind}_h{h}",
                    metrics=metrics,
                    description=(
                        f"{meta['point_model'] if kind == 'point' else meta['alert_model']} "
                        f"| {h}h-ahead {'point forecast' if kind == 'point' else 'alert head'} "
                        f"for {cfg.CITY}"
                    ),
                )
                model.save(tmp)
            print(f"  registered aqi_{kind}_h{h}  {metrics}")


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #
def do_backfill(fs, only: list[str] | None = None) -> None:
    """Full history. `only` lets you resume a subset: raw 24 48 72."""
    print("BACKFILL - full history" + (f"  (only: {' '.join(only)})" if only else ""))

    if not only or "raw" in only:
        write_raw(fs, pd.read_parquet(cfg.DATA_INTERIM / "clean.parquet"))

    for h in cfg.HORIZONS_H:
        if only and str(h) not in only:
            continue
        write_features(fs, h, pd.read_parquet(cfg.DATA_PROCESSED / f"dataset_h{h}.parquet"))


def do_incremental(fs) -> None:
    """Send recent rows only.

    We deliberately re-send a lookback window rather than only strictly-new
    rows: CAMS revises recently published hours, and upsert lets those
    corrections propagate at no cost. Under time_travel_format="NONE" this
    would append duplicates instead, and the cutoff would have to be `last`.

    Note the .copy() on each filtered frame - a boolean-filtered slice is a
    view, and to_hopsworks_frame mutates its input's columns.
    """
    print(f"INCREMENTAL - last {INCREMENTAL_LOOKBACK_H}h plus any new rows")
    last = latest_timestamp(fs, RAW_FG)
    print(f"  latest stored: {last}")

    if last is None:
        print("  feature group empty - falling back to full backfill")
        do_backfill(fs)
        return

    cutoff = last - pd.Timedelta(hours=INCREMENTAL_LOOKBACK_H)

    clean = pd.read_parquet(cfg.DATA_INTERIM / "clean.parquet")
    clean["time"] = pd.to_datetime(clean["time"], utc=True)
    write_raw(fs, clean[clean["time"] > cutoff].copy())

    for h in cfg.HORIZONS_H:
        d = pd.read_parquet(cfg.DATA_PROCESSED / f"dataset_h{h}.parquet")
        d["time"] = pd.to_datetime(d["time"], utc=True)
        write_features(fs, h, d[d["time"] > cutoff].copy())


def do_check(fs) -> None:
    print(f"connected to project '{PROJECT}'  (time travel: {TIME_TRAVEL_FORMAT})")
    for name in [RAW_FG] + [FEAT_FG.format(h=h) for h in cfg.HORIZONS_H]:
        print(f"  {name:<22} latest event time: {latest_timestamp(fs, name)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--register-models", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="GROUP",
                    help="subset for --backfill: raw 24 48 72")
    args = ap.parse_args()

    if not (args.backfill or args.incremental or args.register_models or args.check):
        ap.error("pick one of --backfill / --incremental / --register-models / --check")

    project = login()
    fs = project.get_feature_store()

    if args.check:
        do_check(fs)
    if args.backfill:
        do_backfill(fs, args.only)
    if args.incremental:
        do_incremental(fs)
    if args.register_models:
        register_models(project)

    print("done")


if __name__ == "__main__":
    main()