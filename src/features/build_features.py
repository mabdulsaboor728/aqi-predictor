

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as cfg


POLLUTANTS = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide",
              "sulphur_dioxide", "carbon_monoxide", "dust",
              "aerosol_optical_depth"]

AQI_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
POLLUTANT_LAGS = [1, 24]


SUBINDEX_WINDOWS = {"pm2_5": 24, "pm10": 24, "ozone": 8,
                    "nitrogen_dioxide": 1, "sulphur_dioxide": 24,
                    "carbon_monoxide": 8}


WEATHER_POINT = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                 "precipitation", "surface_pressure", "cloud_cover",
                 "wind_speed_10m", "wind_gusts_10m", "shortwave_radiation"]

WEATHER_WINDOWS = [24, 48, 72]



def origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything knowable at time t from pollutant + AQI history."""
    out = pd.DataFrame(index=df.index)
    t = cfg.TARGET

    
    out[f"{t}_t"] = df[t]
    for lag in AQI_LAGS:
        out[f"{t}_lag{lag}_t"] = df[t].shift(lag)

    for w in [3, 6, 24, 72, 168]:
        out[f"{t}_rmean{w}_t"] = df[t].rolling(w, min_periods=max(2, w // 2)).mean()
    for w in [24, 72]:
        out[f"{t}_rstd{w}_t"] = df[t].rolling(w, min_periods=w // 2).std()
        out[f"{t}_rmax{w}_t"] = df[t].rolling(w, min_periods=w // 2).max()
        out[f"{t}_rmin{w}_t"] = df[t].rolling(w, min_periods=w // 2).min()

    
    for w in [3, 6, 24, 72]:
        out[f"{t}_delta{w}_t"] = df[t] - df[t].shift(w)
        out[f"{t}_rate{w}_t"] = out[f"{t}_delta{w}_t"] / w

    
    out[f"{t}_same_hour_yesterday_t"] = df[t].shift(24)
    out[f"{t}_anomaly_vs_72h_t"] = df[t] - out[f"{t}_rmean72_t"]

    
    for col, w in SUBINDEX_WINDOWS.items():
        if col not in df:
            continue
        if w == 1:
            out[f"{col}_t"] = df[col]
        else:
            out[f"{col}_r{w}h_t"] = df[col].rolling(w, min_periods=max(2, w // 2)).mean()

    for col in POLLUTANTS:
        if col not in df:
            continue
        for lag in POLLUTANT_LAGS:
            out[f"{col}_lag{lag}_t"] = df[col].shift(lag)
        out[f"{col}_delta24_t"] = df[col] - df[col].shift(24)

    
    if {"pm2_5", "pm10"}.issubset(df.columns):
        out["pm25_pm10_ratio_t"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)

    return out


def forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything knowable for an arbitrary future hour from the weather forecast.

    Indexed by the hour it describes. build_supervised() later shifts this
    backwards by h so it lands on the origin row.
    """
    out = pd.DataFrame(index=df.index)

    for col in WEATHER_POINT:
        if col in df:
            out[f"{col}_f"] = df[col]

    
    if "wind_direction_10m" in df:
        rad = np.deg2rad(df["wind_direction_10m"].astype(float))
        out["wind_dir_sin_f"] = np.sin(rad)
        out["wind_dir_cos_f"] = np.cos(rad)

    
    for w in WEATHER_WINDOWS:
        out[f"wind_rmean{w}_f"] = df["wind_speed_10m"].rolling(w, min_periods=w // 2).mean()
        out[f"precip_rsum{w}_f"] = df["precipitation"].rolling(w, min_periods=w // 2).sum()
        out[f"temp_rmean{w}_f"] = df["temperature_2m"].rolling(w, min_periods=w // 2).mean()

    
    if {"temperature_2m", "dew_point_2m"}.issubset(df.columns):
        out["dew_spread_f"] = df["temperature_2m"] - df["dew_point_2m"]

    
    local = df.index.tz_convert(cfg.LOCAL_TZ)
    hour = np.asarray(local.hour)
    doy = np.asarray(local.dayofyear)
    dow = np.asarray(local.dayofweek)

    out["hour_sin_f"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos_f"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin_f"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos_f"] = np.cos(2 * np.pi * doy / 365.25)
    out["hour_f"] = hour
    out["month_f"] = np.asarray(local.month)
    out["dayofweek_f"] = dow
    out["is_weekend_f"] = (dow >= 5).astype(int)

    return out


def build_supervised(df: pd.DataFrame, horizon: int,
                     max_target_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """Assemble the (X, y) frame for one horizon, indexed by origin time t.

    max_target_time drops origins whose target hour t+h has not yet MATURED.
    The air-quality API is a CAMS forecast product and returns provisional
    values for future hours; a row built from those would train the model to
    reproduce another model's forecast rather than an observation. Pass the
    current hour when building from live data.
    """
    origin = origin_features(df)
    fcst = forecast_features(df)

    
    fcst_at_target = fcst.shift(-horizon)
    fcst_at_target.columns = [f"{c}_h{horizon}" for c in fcst_at_target.columns]

    y = df[cfg.TARGET].shift(-horizon).rename("target")

    data = pd.concat([origin, fcst_at_target, y], axis=1)
    data["horizon"] = horizon
    data = data.dropna(subset=["target"])

    if max_target_time is not None:
        mature = (data.index + pd.Timedelta(hours=horizon)) <= max_target_time
        dropped = int((~mature).sum())
        if dropped:
            print(f"  h={horizon}: dropped {dropped} rows with unmatured targets "
                  f"(target after {max_target_time})")
        data = data[mature]

    return data



def assert_no_leakage(df: pd.DataFrame, data: pd.DataFrame, horizon: int) -> None:
    """Cheap but real checks. Run them every time; they cost milliseconds."""
    t = cfg.TARGET
    if data.empty:
        print(f"  h={horizon}: no rows to check")
        return

    
    probe = data.index[len(data) // 2]
    expected = df.loc[probe + pd.Timedelta(hours=horizon), t]
    actual = data.loc[probe, "target"]
    assert np.isclose(expected, actual), f"target misaligned at {probe}"

    
    origin_cols = [c for c in data.columns if c.endswith("_t")]
    aligned = data[origin_cols].join(data["target"])
    for c in origin_cols:
        r = aligned[c].corr(aligned["target"])
        assert not (r > 0.999), f"{c} is perfectly correlated with target - leak"

    
    persistence_r = data[f"{t}_t"].corr(data["target"])
    assert persistence_r < 0.99, "origin AQI too close to target - check shift sign"

    print(f"  leakage checks passed (persistence corr at h={horizon}: {persistence_r:.3f})")


def main() -> None:
    df = pd.read_parquet(cfg.DATA_INTERIM / "clean.parquet").set_index("time")
    print(f"clean: {df.shape}")

   
    for h in cfg.HORIZONS_H:
        data = build_supervised(df, h)
        assert_no_leakage(df, data, h)

        out = cfg.DATA_PROCESSED / f"dataset_h{h}.parquet"
        data.reset_index().to_parquet(out, index=False)

        n_feat = data.shape[1] - 2  
        print(f"h={h:>2}: {data.shape[0]:>6} rows, {n_feat} features -> {out.name}")

    print(f"\nsaved to {cfg.DATA_PROCESSED}")


if __name__ == "__main__":
    main()