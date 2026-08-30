"""
Step 7 - Inference.

Produces a 3-day AQI forecast for the configured city, right now.

The availability contract, made concrete
----------------------------------------
Training built each row from two sources. Serving reproduces that split exactly,
from live data:

    origin features (_t)      pollutant + AQI history up to the origin hour
                              -> feature store (aqi_raw_hourly)

    forecast features (_f_h)  weather + calendar at origin + h
                              -> Open-Meteo forecast API

The subtle part is the rolling weather windows. `wind_rmean72_f` evaluated at
t+24 spans t-48 .. t+24: half past actuals, half forecast. So recent ACTUAL
weather and FORECAST weather are concatenated into one continuous hourly series
BEFORE any rolling feature is computed. Computing rolling stats on the forecast
alone silently produces wrong values that nothing downstream detects.

We rebuild the full frame and hand it to the same build_features functions used
in training. Reusing that code is the point: a reimplementation here is exactly
how training/serving skew starts. Verified offline - the assembled serving row
matches the corresponding training row to ~4e-12 across all 91 features.

Model selection
---------------
Heads are chosen by REGISTERED METRIC, not by version number. A retrain can be
worse than what it replaced, so best-by-metric means the daily job can push
freely while inference keeps serving the best model ever trained.

  point head  lowest RMSE
  alert head  highest F1  (its job is the recall/precision balance, so RMSE
                           would pick the wrong one)

CAVEAT: cross-version metric comparison is only valid if every version was
scored on the SAME evaluation window. train.py uses a fixed HOLDOUT_START with
no end date, so the window grows and metrics slowly drift out of comparability.
Either freeze the holdout end date, or register a rolling-90-day metric and
select on that - SELECTION already prefers it if present.

Run:
    python -m src.models.predict                 # registry, best version
    python -m src.models.predict --local         # local joblib files
    python -m src.models.predict --version 3     # pin an exact version
    python -m src.models.predict --json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import requests

from src import config as cfg
from src.features.build_features import forecast_features, origin_features

# history needed before the origin for the longest window (lag168 / rmean168)
HISTORY_HOURS = 24 * 14
FORECAST_HOURS = 24 * 5

ALERT_THRESHOLD = 150

# Metric preference per head, in order. The first one actually present on a
# version is used. Listing the rolling metric first means this upgrades
# automatically once train.py starts registering it.
SELECTION = {
    "point": [("rmse_rolling90", "min"), ("RMSE", "min")],
    "alert": [("alert_f1_rolling90", "max"), ("alert_f1", "max")],
}

# operational staleness thresholds - warn, never fail
MAX_DATA_AGE_H = 6
MAX_MODEL_AGE_DAYS = 14

AQI_BANDS = [
    (50, "Good", "#00e400"),
    (100, "Moderate", "#ffff00"),
    (150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (200, "Unhealthy", "#ff0000"),
    (300, "Very Unhealthy", "#8f3f97"),
    (10**6, "Hazardous", "#7e0023"),
]

_MR = None          # cached model registry handle


def band(aqi: float) -> tuple[str, str]:
    for limit, name, colour in AQI_BANDS:
        if aqi <= limit:
            return name, colour
    return AQI_BANDS[-1][1], AQI_BANDS[-1][2]


# --------------------------------------------------------------------------- #
# live data
# --------------------------------------------------------------------------- #
def fetch_recent_history() -> pd.DataFrame:
    """Recent pollutant + weather actuals. Feature store first, API as fallback."""
    try:
        from src.data.feature_store import get_fs, read_raw

        df = read_raw(get_fs())
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=HISTORY_HOURS)
        df = df[df.index >= cutoff]
        if len(df) >= HISTORY_HOURS * 0.8:
            print(f"  history: {len(df)} rows from feature store "
                  f"({df.index.min()} -> {df.index.max()})")
            return df
        print(f"  history: feature store had only {len(df)} rows, falling back to API")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  history: feature store unavailable ({type(exc).__name__}), using API")

    from src.data.fetch_openmeteo import fetch_air_quality, fetch_weather

    end = datetime.now(timezone.utc).date()
    start = (end - timedelta(days=HISTORY_HOURS // 24 + 1)).isoformat()
    aq = fetch_air_quality(start, end.isoformat())
    wx = fetch_weather(start, end.isoformat())
    df = aq.merge(wx, on="time", how="inner").set_index("time").sort_index()
    return df.drop(columns=["boundary_layer_height"], errors="ignore")


def fetch_weather_forecast() -> pd.DataFrame:
    """Future weather. This is what makes a 72h horizon possible."""
    params = {
        "latitude": cfg.LATITUDE,
        "longitude": cfg.LONGITUDE,
        "hourly": ",".join(cfg.WX_VARS),
        "forecast_days": min(FORECAST_HOURS // 24 + 1, 16),
        "timezone": "UTC",
    }
    r = requests.get(cfg.WX_FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df.drop(columns=["boundary_layer_height"], errors="ignore")
    print(f"  forecast: {len(df)} hours ({df.index.min()} -> {df.index.max()})")
    return df


def assemble(history: pd.DataFrame, forecast: pd.DataFrame):
    """Splice actuals and forecast into one continuous hourly frame.

    Origin `t` is the last hour with an AQI value AT OR BEFORE the current
    hour. The cap matters: Open-Meteo's air-quality endpoint returns CAMS
    FORECAST values for future hours, not only observations. Training only ever
    saw archived values, so letting a forecast hour become the origin would
    build every lag and rolling feature from a different distribution than the
    model was fitted on - and would silently report a negative data age.

    Rows after t carry weather only; their pollutant columns stay NaN, which is
    correct - we do not have them and must not invent them.
    """
    now = pd.Timestamp.now(tz="UTC").floor("h")

    observed = history[history[cfg.TARGET].notna()]
    observed = observed[observed.index <= now]
    if observed.empty:
        raise RuntimeError("no AQI observations at or before the current hour")
    origin = observed.index.max()

    hist = history.loc[:origin]
    fut = forecast[forecast.index > origin]

    combined = pd.concat([hist, fut], axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    full = pd.date_range(combined.index.min(), combined.index.max(), freq="h", tz="UTC")
    if len(full) != len(combined):
        print(f"  warning: {len(full) - len(combined)} gaps in spliced series, reindexing")
        combined = combined.reindex(full)
    combined.index.name = "time"

    # weather must be complete across the whole span or rolling features break
    wx_cols = [c for c in cfg.WX_VARS if c in combined.columns]
    holes = combined[wx_cols].isna().sum()
    if (holes > 0).any():
        raise RuntimeError(
            "weather has gaps after splicing - rolling features would be wrong:\n"
            f"{holes[holes > 0].to_string()}"
        )

    print(f"  spliced: {len(combined)} hours, origin = {origin}  (now = {now})")
    return combined, origin


# --------------------------------------------------------------------------- #
# feature assembly - reuses the training code path
# --------------------------------------------------------------------------- #
def build_row(combined: pd.DataFrame, origin: pd.Timestamp, horizon: int,
              expected: list[str]) -> pd.DataFrame:
    target_time = origin + pd.Timedelta(hours=horizon)
    if target_time not in combined.index:
        raise RuntimeError(f"forecast does not reach {target_time}")

    row_origin = origin_features(combined).loc[[origin]]

    fcst = forecast_features(combined).loc[[target_time]]
    fcst.columns = [f"{c}_h{horizon}" for c in fcst.columns]
    fcst.index = [origin]

    row = pd.concat([row_origin, fcst], axis=1)

    missing = [c for c in expected if c not in row.columns]
    if missing:
        raise RuntimeError(f"h={horizon}: {len(missing)} features missing, "
                           f"e.g. {missing[:5]}")
    row = row[expected]

    nulls = row.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        # fail loudly - a silently imputed feature degrades every prediction
        # with nothing in the logs, the worst failure mode for a cron job
        raise RuntimeError(f"h={horizon}: null features at serving time:\n{nulls.to_string()}")
    return row


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def _registry():
    """One login, reused across all six loads. Six separate logins cost ~25s."""
    global _MR
    if _MR is None:
        from src.data.feature_store import login
        _MR = login().get_model_registry()
    return _MR


def _has_metric(model, metric: str) -> bool:
    return metric in (model.training_metrics or {})


def _pick_version(mr, name: str, kind: str, pin: int | None):
    """Best registered version by metric, with graceful degradation.

    Order of preference:
      1. explicit --version pin
      2. get_best_model on the first metric the returned model actually
         carries (rolling window if registered, else the frozen holdout)
      3. highest version number
    """
    if pin is not None:
        return mr.get_model(name, version=pin), f"pinned v{pin}", None

    for metric, direction in SELECTION[kind]:
        try:
            m = mr.get_best_model(name, metric, direction)
        except Exception:                                     # noqa: BLE001
            continue
        # get_best_model can return a model even when the metric is absent,
        # which would make the log claim a selection that never happened
        if m is not None and _has_metric(m, metric):
            return m, f"best by {metric}", metric

    models = mr.get_models(name)
    if not models:
        raise RuntimeError(f"no versions registered for {name}")
    m = max(models, key=lambda v: v.version)
    return m, "latest version (no comparable metric)", None


def _model_age_days(model) -> float | None:
    """Age of a registered model, or None if the timestamp is unreadable."""
    created = getattr(model, "created", None)
    if created is None:
        return None
    try:
        if isinstance(created, (int, float)):
            # Hopsworks returns epoch MILLISECONDS. pd.to_datetime on a bare
            # number assumes nanoseconds and lands in 1970, which is what
            # produced the "20696 days old" warning.
            ts = pd.to_datetime(created, unit="ms", utc=True)
        else:
            ts = pd.to_datetime(created, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 86400
        # 10 years is far beyond any plausible model age; anything larger
        # means the epoch unit was misread, so report unknown rather than
        # emitting a nonsense staleness warning
        return age if -1 < age < 3650 else None
    except Exception:                                          # noqa: BLE001
        return None


def load_models(horizon: int, local: bool, pin: int | None, warnings: list[str]):
    """Return (point_model, alert_model, provenance dict)."""
    if not local:
        try:
            mr = _registry()
            loaded, prov = {}, {}
            for kind in ("point", "alert"):
                name = f"aqi_{kind}_h{horizon}"
                m, how, metric = _pick_version(mr, name, kind, pin)
                path = m.download()
                fname = f"{'model' if kind == 'point' else 'alert'}_h{horizon}.joblib"
                loaded[kind] = joblib.load(f"{path}/{fname}")

                score = (m.training_metrics or {}).get(metric) if metric else None
                age = _model_age_days(m)
                prov[kind] = {"version": m.version, "selected_by": how,
                              "metric": metric, "score": score,
                              "age_days": round(age, 1) if age is not None else None}

                detail = f", {metric}={float(score):.3f}" if score is not None else ""
                aged = f", {age:.0f}d old" if age is not None else ""
                print(f"    {name}: v{m.version} ({how}{detail}{aged})")

                if age is not None and age > MAX_MODEL_AGE_DAYS:
                    warnings.append(
                        f"{name} v{m.version} is {age:.0f} days old "
                        f"(threshold {MAX_MODEL_AGE_DAYS}) - retraining may be stalled"
                    )
            return loaded["point"], loaded["alert"], prov
        except Exception as exc:                              # noqa: BLE001
            print(f"  h={horizon}: registry unavailable ({type(exc).__name__}), using local")
            warnings.append(f"h={horizon}: fell back to local models")

    return (joblib.load(cfg.MODELS_DIR / f"model_h{horizon}.joblib"),
            joblib.load(cfg.MODELS_DIR / f"alert_h{horizon}.joblib"),
            {"point": {"version": "local"}, "alert": {"version": "local"}})


def expected_features(horizon: int) -> list[str]:
    meta = cfg.MODELS_DIR / f"model_h{horizon}_meta.json"
    if meta.exists():
        return json.loads(meta.read_text())["features"]
    d = pd.read_parquet(cfg.DATA_PROCESSED / f"dataset_h{horizon}.parquet")
    return [c for c in d.columns if c not in ("time", "target", "horizon")]


# --------------------------------------------------------------------------- #
def predict(local: bool = False, pin: int | None = None) -> dict:
    warnings: list[str] = []

    print("assembling live inputs")
    combined, origin = assemble(fetch_recent_history(), fetch_weather_forecast())
    current = float(combined.loc[origin, cfg.TARGET])

    issued = pd.Timestamp.now(tz="UTC")
    data_age_h = (issued - origin).total_seconds() / 3600
    if data_age_h > MAX_DATA_AGE_H:
        warnings.append(
            f"observations are {data_age_h:.0f}h old (threshold {MAX_DATA_AGE_H}h) - "
            f"the hourly ingestion job may not be running"
        )

    out = []
    for h in cfg.HORIZONS_H:
        row = build_row(combined, origin, h, expected_features(h))
        point_m, alert_m, prov = load_models(h, local, pin, warnings)

        aqi = float(point_m.predict(row)[0])
        upper = float(alert_m.predict(row)[0])
        name, colour = band(aqi)

        out.append({
            "horizon_h": h,
            "valid_at": (origin + pd.Timedelta(hours=h)).isoformat(),
            "aqi": round(aqi, 1),
            "aqi_upper_q90": round(upper, 1),
            "category": name,
            "colour": colour,
            "alert": bool(upper > ALERT_THRESHOLD),
            "alert_reason": (
                f"90th-percentile forecast {upper:.0f} exceeds {ALERT_THRESHOLD}"
                if upper > ALERT_THRESHOLD else None
            ),
            "models": prov,
        })

    result = {
        "city": cfg.CITY,
        "issued_at": issued.isoformat(),
        "data_through": origin.isoformat(),
        "data_age_hours": round(data_age_h, 1),
        "current_aqi": round(current, 1),
        "current_category": band(current)[0],
        "forecast": out,
        "warnings": warnings,
    }

    path = cfg.ROOT / "reports" / "latest_forecast.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def render(r: dict) -> None:
    issued = pd.Timestamp(r["issued_at"]).tz_convert(cfg.LOCAL_TZ)
    through = pd.Timestamp(r["data_through"]).tz_convert(cfg.LOCAL_TZ)

    print(f"\n{'=' * 64}")
    print(f"{r['city']} air quality forecast")
    print(f"  issued        {issued:%Y-%m-%d %H:%M} PKT")
    print(f"  data through  {through:%Y-%m-%d %H:%M} PKT  ({r['data_age_hours']:.0f}h old)")
    print(f"  now           AQI {r['current_aqi']:.0f}  ({r['current_category']})")
    print(f"{'-' * 64}")
    print(f"{'when':<24}{'AQI':>6}{'q90':>7}   category")
    for f in r["forecast"]:
        when = pd.Timestamp(f["valid_at"]).tz_convert(cfg.LOCAL_TZ)
        flag = "  << ALERT" if f["alert"] else ""
        print(f"+{f['horizon_h']:>2}h  {when:%a %d %b %H:%M}    "
              f"{f['aqi']:>5.0f}{f['aqi_upper_q90']:>7.0f}   {f['category']}{flag}")
    print("=" * 64)

    alerts = [f for f in r["forecast"] if f["alert"]]
    if alerts:
        print(f"\n{len(alerts)} hazardous-level alert(s):")
        for a in alerts:
            print(f"  +{a['horizon_h']}h - {a['alert_reason']}")

    if r["warnings"]:
        print("\noperational warnings:")
        for w in r["warnings"]:
            print(f"  ! {w}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="skip the registry")
    ap.add_argument("--version", type=int, default=None,
                    help="pin an exact registry version instead of best-by-metric")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = predict(local=args.local, pin=args.version)
    print(json.dumps(r, indent=2)) if args.json else render(r)


if __name__ == "__main__":
    main()