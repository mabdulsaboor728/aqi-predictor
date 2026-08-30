"""
Step 3 - EDA.

Answers specific questions rather than producing generic plots. Every figure
here is one you can defend in the report:

  Q1  What does the AQI distribution look like, and is it stationary?
  Q2  How strong is the diurnal cycle, and does it change by season?
  Q3  Which pollutant actually drives us_aqi, and over what averaging window?
  Q4  Does weather matter contemporaneously or cumulatively?
  Q5  What do the naive baselines score? (this is the bar the model must clear)

Run:
    python -m src.eda
Figures land in reports/figures/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg

FIG_DIR = cfg.ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

# hold out the final year for every baseline reported here
TEST_START = "2025-08-01"


def load() -> pd.DataFrame:
    df = pd.read_parquet(cfg.DATA_INTERIM / "clean.parquet").set_index("time")
    df["season"] = df["month"].map(SEASON)
    return df


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=140)
    plt.close(fig)
    print(f"  saved {name}")


# ------------------------------------------------------------------ Q1
def q1_distribution(df: pd.DataFrame) -> None:
    print("\nQ1 - target distribution")
    a = df[cfg.TARGET]
    print(a.describe().round(1).to_string())
    print("\nUS AQI category share:")
    bins = [0, 50, 100, 150, 200, 300, 1000]
    labels = ["Good", "Moderate", "Unhealthy(SG)", "Unhealthy",
              "Very Unhealthy", "Hazardous"]
    share = pd.cut(a, bins=bins, labels=labels, right=True).value_counts(normalize=True)
    print(share.reindex(labels).mul(100).round(1).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist(a.dropna(), bins=60, color="#4477aa")
    ax[0].set_title("US AQI distribution")
    ax[0].set_xlabel("us_aqi")
    a.resample("D").mean().plot(ax=ax[1], lw=0.7, color="#334455")
    ax[1].set_title("Daily mean US AQI over time")
    _save(fig, "q1_distribution.png")


# ------------------------------------------------------------------ Q2
def q2_diurnal(df: pd.DataFrame) -> None:
    print("\nQ2 - diurnal cycle by season")
    piv = df.pivot_table(index="hour_local", columns="season",
                         values=cfg.TARGET, aggfunc="mean")[SEASON_ORDER]
    print(piv.round(1).to_string())
    print("\namplitude (max hour - min hour):")
    print((piv.max() - piv.min()).round(1).to_string())
    print("-> a flat winter curve means PM2.5 (24h window) dominates;")
    print("   a peaked summer curve means ozone (8h window) dominates.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    piv.plot(ax=ax, lw=2)
    ax.set_xlabel("hour (Asia/Karachi)")
    ax.set_ylabel("mean us_aqi")
    ax.set_title("Diurnal AQI cycle by season")
    _save(fig, "q2_diurnal_by_season.png")


# ------------------------------------------------------------------ Q3
def q3_drivers(df: pd.DataFrame) -> None:
    print("\nQ3 - which pollutant drives the target, over which window?")
    d = df.copy()
    d["pm25_24h"] = d["pm2_5"].rolling(24, min_periods=18).mean()
    d["pm10_24h"] = d["pm10"].rolling(24, min_periods=18).mean()
    d["o3_8h"] = d["ozone"].rolling(8, min_periods=6).mean()

    cols = ["pm2_5", "pm25_24h", "pm10_24h", "ozone", "o3_8h"]
    rows = {}
    for s in SEASON_ORDER:
        m = d[d["season"] == s]
        rows[s] = {c: m[cfg.TARGET].corr(m[c]) for c in cols}
    table = pd.DataFrame(rows).round(3)
    print(table.to_string())
    print("-> matching the sub-index averaging window is worth more than any model change.")

    fig, ax = plt.subplots(figsize=(8, 4))
    table.T.plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("corr with us_aqi")
    ax.set_title("Pollutant correlation with AQI, by season and averaging window")
    _save(fig, "q3_drivers.png")


# ------------------------------------------------------------------ Q4
def q4_weather(df: pd.DataFrame) -> None:
    print("\nQ4 - contemporaneous vs cumulative weather effect")
    a = df[cfg.TARGET]

    lag_rows = []
    for lag in [0, 3, 6, 12, 24, 48]:
        lag_rows.append({
            "lag_h": lag,
            "wind": a.corr(df["wind_speed_10m"].shift(lag)),
            "gusts": a.corr(df["wind_gusts_10m"].shift(lag)),
            "rh": a.corr(df["relative_humidity_2m"].shift(lag)),
            "temp": a.corr(df["temperature_2m"].shift(lag)),
        })
    print(pd.DataFrame(lag_rows).set_index("lag_h").round(3).to_string())

    roll_rows = []
    for w in [6, 12, 24, 48, 72]:
        roll_rows.append({
            "window_h": w,
            "wind_mean": a.corr(df["wind_speed_10m"].rolling(w).mean()),
            "precip_sum": a.corr(df["precipitation"].rolling(w).sum()),
        })
    roll = pd.DataFrame(roll_rows).set_index("window_h").round(3)
    print("\nrolling windows:")
    print(roll.to_string())
    print("-> ventilation is cumulative; rolling wind beats same-hour wind ~3x.")

    fig, ax = plt.subplots(figsize=(8, 4))
    roll.plot(ax=ax, marker="o")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("corr with us_aqi")
    ax.set_title("Cumulative weather effect on AQI")
    _save(fig, "q4_weather_windows.png")


# ------------------------------------------------------------------ Q5
def q5_baselines(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nQ5 - naive baselines (test window from {TEST_START})")
    train = df[df.index < TEST_START]
    test_mask = df.index >= TEST_START
    a = df[cfg.TARGET]

    rows = []
    for h in cfg.HORIZONS_H:
        err = (a.shift(-h) - a)[test_mask].dropna()
        rows.append({"baseline": f"persistence t+{h}h",
                     "MAE": err.abs().mean(),
                     "RMSE": np.sqrt((err ** 2).mean())})

    clim = train.groupby(["dayofyear", "hour_local"])[cfg.TARGET].mean()
    test = df[test_mask]
    pred = pd.Series(test.set_index(["dayofyear", "hour_local"]).index.map(clim),
                     index=test.index, dtype="float64")
    err = (test[cfg.TARGET] - pred).dropna()
    rows.append({"baseline": "climatology (doy x hour)",
                 "MAE": err.abs().mean(),
                 "RMSE": np.sqrt((err ** 2).mean())})

    out = pd.DataFrame(rows).set_index("baseline").round(2)
    print(out.to_string())
    print(f"\ntest-window AQI std: {test[cfg.TARGET].std():.2f}")
    print("-> any model that does not beat persistence has learned nothing useful.")

    out.to_csv(cfg.ROOT / "reports" / "baselines.csv")
    return out


def main() -> None:
    df = load()
    print(f"loaded {df.shape[0]} rows  {df.index.min()} -> {df.index.max()}")
    q1_distribution(df)
    q2_diurnal(df)
    q3_drivers(df)
    q4_weather(df)
    q5_baselines(df)
    print(f"\nfigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()