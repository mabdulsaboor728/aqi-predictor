"""
Step 2 - Cleaning.

Takes data/raw/merged_raw.parquet and produces data/interim/clean.parquet.

Deliberately conservative: this stage only removes or repairs what is
defensible. No feature engineering happens here (that is step 4), and nothing
here is allowed to look forward in time.

Decisions made, and why:
  * boundary_layer_height dropped  - 51% missing (archive only covers Sep 2024+)
                                     and its correlation with the target was
                                     0.003, so keeping it would have cost half
                                     the training data for nothing
  * leading rows trimmed           - one contiguous us_aqi gap at series start
  * short gaps forward-filled      - strictly past-only, capped at MAX_GAP_HOURS
  * physical clipping              - negatives on strictly-positive quantities

The functions here are pure and take a frame, so pipelines/hourly.py can reuse
the identical cleaning steps on an in-memory frame in CI. Any divergence
between the two paths would be training/serving skew.

Run:
    python -m src.data.clean
"""

from __future__ import annotations

import pandas as pd

from src import config as cfg

DROP_COLS = ["boundary_layer_height"]      # 51% missing - see module docstring
MAX_GAP_HOURS = 3                          # anything longer is left as NaN

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
    """Clip values that cannot physically occur.

    CAMS produces small negative concentrations near zero as a numerical
    artefact, so this is correcting the model output rather than the physics.
    """
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
    """Forward-fill gaps of at most MAX_GAP_HOURS. Strictly past-only.

    This is NOT interpolation, deliberately.

    pandas' interpolate(method="time") always reads the observations on BOTH
    sides of a gap. `limit_direction` only chooses which NaN positions get
    written - it does not stop the future endpoint from participating in the
    arithmetic. So interpolate(limit_direction="forward") on [0, NaN, 2]
    returns [0, 1, 2]: the middle value was computed from the one after it.

    That is a leak. A 3pm gap filled from the 5pm reading puts a future
    observation into every lag and rolling feature computed at 3pm, and no
    downstream check would catch it - the value looks perfectly plausible.

    ffill carries the last known value forward and reads nothing after t. The
    result is a step rather than a smooth ramp, which is less accurate for a
    continuous series. That is the correct trade: an accurate value built from
    the future is worse than a slightly stale value built only from the past,
    because only the second one is available at serving time.

    Gaps longer than MAX_GAP_HOURS are left as NaN for the feature stage to
    handle, rather than propagating a stale value indefinitely.
    """
    before = int(df.isna().sum().sum())
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].ffill(limit=MAX_GAP_HOURS)
    after = int(df.isna().sum().sum())
    print(f"forward-filled {before - after} values (gaps <= {MAX_GAP_HOURS}h)")

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


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The full cleaning sequence, applied to an in-memory frame.

    Exposed so pipelines/hourly.py runs the identical steps in CI instead of
    reimplementing them.
    """
    df = trim_leading_gap(df)
    df = enforce_hourly_grid(df)
    df = clip_physical(df)
    df = fill_short_gaps(df)
    return add_calendar(df)


def main() -> None:
    df = load_raw()
    print(f"raw: {df.shape}")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"dropped columns: {DROP_COLS}")

    df = clean_frame(df)

    out = cfg.DATA_INTERIM / "clean.parquet"
    df.reset_index().to_parquet(out, index=False)

    print(f"\nclean: {df.shape}")
    print(f"range: {df.index.min()} -> {df.index.max()}")
    print(f"target nulls remaining: {int(df[cfg.TARGET].isna().sum())}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()