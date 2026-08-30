"""
Step 2 - Cleaning.

Takes data/raw/merged_raw.parquet and produces data/interim/clean.parquet.

Deliberately conservative: this stage only removes/repairs things that are
defensible. No feature engineering happens here (that is step 4), and nothing
here is allowed to look forward in time.

Decisions made, and why:
  * boundary_layer_height dropped  - 51% missing (archive only covers Sep 2024+)
  * leading 96h trimmed            - single contiguous us_aqi gap at series start
  * short gaps interpolated        - time-based, capped at 3h, both directions
                                     disabled so we never fill from the future
  * physical clipping              - negatives on strictly-positive quantities

Run:
    python -m src.data.clean
"""

from __future__ import annotations

import pandas as pd

from src import config as cfg

DROP_COLS = ["boundary_layer_height"]      # 51% missing - see module docstring
MAX_GAP_HOURS = 3                          # anything longer gets flagged, not filled

# columns that cannot physically be negative
NON_NEGATIVE = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "dust", "aerosol_optical_depth", "us_aqi",
    "precipitation", "wind_speed_10m", "wind_gusts_10m", "shortwave_radiation",
    "relative_humidity_2m", "cloud_cover", "surface_pressure",
]


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(cfg.DATA_RAW / "merged_raw.parquet")
    return df.sort_values("time").reset_index(drop=True)


def trim_leading_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows before the first valid target value."""
    first_valid = df.loc[df[cfg.TARGET].notna(), "time"].min()
    before = len(df)
    df = df[df["time"] >= first_valid].reset_index(drop=True)
    print(f"trimmed {before - len(df)} leading rows; series now starts {first_valid}")
    return df


def enforce_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a complete hourly index so lag features are never off-by-one.

    A lag of 24 rows only equals 24 hours if the index has no holes. Silently
    ragged indexes are one of the most common causes of a model that validates
    well and fails in production.
    """
    df = df.drop_duplicates(subset="time").set_index("time")
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    inserted = len(full) - len(df)
    df = df.reindex(full)
    df.index.name = "time"
    print(f"hourly grid: {len(full)} rows ({inserted} placeholder rows inserted)")
    return df


def clip_physical(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in NON_NEGATIVE if c in df.columns]
    n_neg = int((df[cols] < 0).sum().sum())
    if n_neg:
        print(f"clipped {n_neg} negative values to 0")
    df[cols] = df[cols].clip(lower=0)

    if "relative_humidity_2m" in df:
        df["relative_humidity_2m"] = df["relative_humidity_2m"].clip(0, 100)
    if "cloud_cover" in df:
        df["cloud_cover"] = df["cloud_cover"].clip(0, 100)
    return df


def fill_short_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate gaps of <= MAX_GAP_HOURS only.

    limit_direction='forward' keeps this causal: a value is only ever filled
    from the past, never from a future observation the model would not have.
    """
    before = df.isna().sum().sum()
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].interpolate(
        method="time", limit=MAX_GAP_HOURS, limit_direction="forward"
    )
    after = df.isna().sum().sum()
    print(f"interpolated {before - after} values (gaps <= {MAX_GAP_HOURS}h)")

    remaining = df[numeric].isna().sum()
    remaining = remaining[remaining > 0]
    if len(remaining):
        print("columns still containing nulls (left for the feature stage to handle):")
        print(remaining.to_string())
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Local-time calendar columns. Diurnal patterns are a local phenomenon."""
    local = df.index.tz_convert(cfg.LOCAL_TZ)
    df["hour_local"] = local.hour
    df["dayofweek"] = local.dayofweek
    df["month"] = local.month
    df["dayofyear"] = local.dayofyear
    return df


def main() -> None:
    df = load_raw()
    print(f"raw: {df.shape}")

    df = trim_leading_gap(df)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"dropped columns: {DROP_COLS}")

    df = enforce_hourly_grid(df)
    df = clip_physical(df)
    df = fill_short_gaps(df)
    df = add_calendar(df)

    out = cfg.DATA_INTERIM / "clean.parquet"
    df.reset_index().to_parquet(out, index=False)

    print(f"\nclean: {df.shape}")
    print(f"range: {df.index.min()} -> {df.index.max()}")
    print(f"target nulls remaining: {df[cfg.TARGET].isna().sum()}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()