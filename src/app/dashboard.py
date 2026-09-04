"""
Streamlit dashboard - the public face of the forecasting system.

    streamlit run src/app/dashboard.py

Data sources, in order of preference:
  reports/latest_forecast.json   written hourly by pipelines/hourly.py
  reports/history/*.json         one snapshot per hourly run, for the backtest
  Open-Meteo air-quality API     observed AQI, to score past forecasts against

Deliberately has NO Hopsworks dependency. Everything it needs is either
committed to the repo by the hourly workflow or fetchable without credentials,
so the app deploys to Streamlit Cloud with no secrets configured.

The colour scale is the US AQI category scale. It is not decoration - it is the
standard public-health encoding people already know how to read, so the page
leans on it to carry the headline rather than inventing its own palette.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
FORECAST = ROOT / "reports" / "latest_forecast.json"
HISTORY = ROOT / "reports" / "history"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

LAT, LON = 33.6844, 73.0479
LOCAL_TZ = "Asia/Karachi"
ALERT_THRESHOLD = 150

# US AQI breakpoints: (upper bound, label, colour, what to actually do)
BANDS = [
    (50, "Good", "#00e400", "Air quality is fine. No precautions needed."),
    (100, "Moderate", "#ffcc00", "Unusually sensitive people may want to limit long outdoor exertion."),
    (150, "Unhealthy for sensitive groups", "#ff7e00",
     "Children, older adults and people with heart or lung conditions should limit prolonged outdoor exertion."),
    (200, "Unhealthy", "#ff0000",
     "Everyone should limit prolonged outdoor exertion. Sensitive groups should stay indoors."),
    (300, "Very unhealthy", "#8f3f97",
     "Avoid outdoor exertion. Keep windows closed and run air filtration if you have it."),
    (10**6, "Hazardous", "#7e0023", "Stay indoors. Avoid all outdoor activity."),
]


def band(aqi: float):
    for limit, label, colour, advice in BANDS:
        if aqi <= limit:
            return label, colour, advice
    return BANDS[-1][1:]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300)
def load_forecast() -> dict | None:
    if not FORECAST.exists():
        return None
    return json.loads(FORECAST.read_text())


@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    """Every past forecast, flattened to one row per (issued, horizon)."""
    if not HISTORY.exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(HISTORY.glob("forecast_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:                                     # noqa: BLE001
            continue
        for f in d.get("forecast", []):
            rows.append({
                "issued_at": pd.to_datetime(d["issued_at"], utc=True),
                "data_through": pd.to_datetime(d.get("data_through", d["issued_at"]), utc=True),
                "horizon_h": f["horizon_h"],
                "valid_at": pd.to_datetime(f["valid_at"], utc=True),
                "predicted": f["aqi"],
                "upper_q90": f.get("aqi_upper_q90"),
                "alerted": bool(f.get("alert")),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def fetch_observed(days: int = 14) -> pd.DataFrame:
    """Observed AQI, for scoring past forecasts. No credentials needed."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={"latitude": LAT, "longitude": LON, "hourly": "us_aqi",
                "start_date": start.isoformat(), "end_date": end.isoformat(),
                "domains": "cams_global", "timezone": "UTC"},
        timeout=30,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"valid_at": pd.to_datetime(h["time"], utc=True),
                       "observed": h["us_aqi"]})
    return df.dropna()


@st.cache_data(ttl=3600)
def load_model_meta() -> list[dict]:
    out = []
    for p in sorted(MODELS.glob("model_h*_meta.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:                                     # noqa: BLE001
            continue
    return out


def local(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(LOCAL_TZ)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Islamabad air quality forecast",
                   page_icon="🌫", layout="wide")

st.markdown("""
<style>
  .headline { padding: 1.75rem 2rem; border-radius: 4px; color: #111;
              margin-bottom: 0.5rem; }
  .headline .num { font-size: 4.5rem; font-weight: 700; line-height: 1;
                   letter-spacing: -0.02em; }
  .headline .cat { font-size: 1.35rem; font-weight: 600; margin-top: 0.35rem; }
  .headline .advice { font-size: 0.95rem; margin-top: 0.75rem; max-width: 60ch;
                      opacity: 0.85; }
  .stamp { color: #666; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

fc = load_forecast()

if fc is None:
    st.title("Islamabad air quality forecast")
    st.warning(
        "No forecast file yet. Run `python -m pipelines.hourly` to generate one, "
        "or wait for the hourly workflow to publish."
    )
    st.stop()

label, colour, advice = band(fc["current_aqi"])
dark = label in ("Unhealthy", "Very unhealthy", "Hazardous")
text = "#fff" if dark else "#111"

st.markdown(f"""
<div class="headline" style="background:{colour}; color:{text};">
  <div class="num">{fc['current_aqi']:.0f}</div>
  <div class="cat">{label}</div>
  <div class="advice">{advice}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"<div class='stamp'>{fc['city']} &nbsp;·&nbsp; "
    f"observations through {local(fc['data_through']):%a %d %b, %H:%M} "
    f"({fc['data_age_hours']:.0f}h ago) &nbsp;·&nbsp; "
    f"forecast issued {local(fc['issued_at']):%H:%M}</div>",
    unsafe_allow_html=True,
)

for w in fc.get("warnings", []):
    st.warning(w, icon="⚠")

alerts = [f for f in fc["forecast"] if f["alert"]]
if alerts:
    worst = max(alerts, key=lambda a: a["aqi_upper_q90"])
    when = local(worst["valid_at"])
    st.error(
        f"**Unhealthy air likely on {when:%A}.** The 90th-percentile forecast for "
        f"{when:%a %H:%M} reaches {worst['aqi_upper_q90']:.0f}, above the "
        f"{ALERT_THRESHOLD} threshold. Sensitive groups should plan to stay indoors.",
        icon="🚨",
    )

st.divider()

# --------------------------------------------------------------- next 3 days
st.subheader("Next three days")

obs_err = None
try:
    observed = fetch_observed(5)
except Exception as exc:                                      # noqa: BLE001
    observed, obs_err = pd.DataFrame(), exc

fig = go.Figure()

if not observed.empty:
    recent = observed[observed["valid_at"] >= pd.Timestamp(fc["data_through"]) - pd.Timedelta(days=3)]
    recent = recent[recent["valid_at"] <= pd.Timestamp(fc["data_through"])]
    fig.add_trace(go.Scatter(
        x=recent["valid_at"].dt.tz_convert(LOCAL_TZ), y=recent["observed"],
        name="Observed", mode="lines", line=dict(color="#444", width=2),
    ))

fx = [pd.Timestamp(fc["data_through"])] + [pd.Timestamp(f["valid_at"]) for f in fc["forecast"]]
fy = [fc["current_aqi"]] + [f["aqi"] for f in fc["forecast"]]
fu = [fc["current_aqi"]] + [f["aqi_upper_q90"] for f in fc["forecast"]]
fx_local = [pd.Timestamp(x).tz_convert(LOCAL_TZ) for x in fx]

fig.add_trace(go.Scatter(
    x=fx_local, y=fu, name="90th percentile", mode="lines",
    line=dict(color="#c44", width=1, dash="dot"),
    fill=None, hovertemplate="%{y:.0f}<extra>90th pct</extra>",
))
fig.add_trace(go.Scatter(
    x=fx_local, y=fy, name="Forecast", mode="lines+markers",
    line=dict(color="#c44", width=3), marker=dict(size=9),
    fill="tonexty", fillcolor="rgba(204,68,68,0.10)",
    hovertemplate="%{y:.0f}<extra>Forecast</extra>",
))

for limit, lab, col, _ in BANDS[:4]:
    fig.add_hline(y=limit, line=dict(color=col, width=1, dash="dash"),
                  annotation_text=lab, annotation_position="right",
                  annotation_font=dict(size=10, color="#888"))

fig.update_layout(
    height=380, margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified", plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    yaxis=dict(title="US AQI", gridcolor="#eee"),
    xaxis=dict(gridcolor="#f5f5f5"),
)
st.plotly_chart(fig, use_container_width=True)

cols = st.columns(3)
for c, f in zip(cols, fc["forecast"]):
    lab, col, _ = band(f["aqi"])
    when = local(f["valid_at"])
    with c:
        st.markdown(
            f"**{when:%A}** &nbsp;<span class='stamp'>{when:%d %b, %H:%M}</span>",
            unsafe_allow_html=True)
        st.markdown(
            f"<span style='font-size:2.2rem;font-weight:700'>{f['aqi']:.0f}</span> "
            f"<span style='color:{col};font-weight:600'>{lab}</span>",
            unsafe_allow_html=True)
        st.caption(f"Could reach {f['aqi_upper_q90']:.0f} in the worst case")

if obs_err is not None:
    st.caption(f"Recent observations unavailable ({type(obs_err).__name__}); "
               f"showing the forecast only.")

st.divider()

# --------------------------------------------------------- forecast accuracy
st.subheader("How accurate has this been?")

hist = load_history()
if hist.empty:
    st.info("No forecast history yet. The hourly job saves a snapshot each run, "
            "and accuracy appears here once forecasts start maturing.")
elif observed.empty:
    st.info("Observations could not be fetched, so past forecasts cannot be scored.")
else:
    scored = hist.merge(observed, on="valid_at", how="inner")
    scored["error"] = scored["predicted"] - scored["observed"]

    if scored.empty:
        oldest = hist["issued_at"].min()
        st.info(
            f"{len(hist)} forecasts saved since {local(oldest):%d %b %H:%M}, but none "
            f"have matured yet. The first 24-hour forecast can be scored a day after "
            f"it was issued."
        )
    else:
        summary = (scored.groupby("horizon_h")
                   .agg(n=("error", "size"),
                        MAE=("error", lambda e: e.abs().mean()),
                        bias=("error", "mean"))
                   .round(1))
        summary.index = [f"+{h}h" for h in summary.index]
        summary.columns = ["Forecasts scored", "Average error (AQI)", "Bias"]

        left, right = st.columns([1, 2])
        with left:
            st.dataframe(summary, use_container_width=True)
            st.caption(
                "Bias shows whether the model runs high or low. Negative means it "
                "under-predicted, which matters most on bad-air days."
            )
        with right:
            sc = go.Figure()
            for h, grp in scored.groupby("horizon_h"):
                grp = grp.sort_values("valid_at")
                sc.add_trace(go.Scatter(
                    x=grp["valid_at"].dt.tz_convert(LOCAL_TZ), y=grp["predicted"],
                    mode="markers", name=f"+{h}h forecast", marker=dict(size=7),
                ))
            obs_win = observed[observed["valid_at"] >= scored["valid_at"].min()]
            sc.add_trace(go.Scatter(
                x=obs_win["valid_at"].dt.tz_convert(LOCAL_TZ), y=obs_win["observed"],
                mode="lines", name="Observed", line=dict(color="#444", width=2),
            ))
            sc.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                             plot_bgcolor="white", hovermode="x unified",
                             yaxis=dict(title="US AQI", gridcolor="#eee"),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(sc, use_container_width=True)

    st.caption(f"{len(hist)} forecasts saved, "
               f"{local(hist['issued_at'].min()):%d %b} to "
               f"{local(hist['issued_at'].max()):%d %b}.")

st.divider()

# -------------------------------------------------------------- how it works
st.subheader("How the forecast is made")

meta = load_model_meta()
if meta:
    tbl = pd.DataFrame([{
        "Horizon": f"+{m['horizon']}h",
        "Model": m["point_model"],
        "Typical error (MAE)": round(m["holdout"]["MAE"], 1),
        "RMSE": round(m["holdout"]["RMSE"], 1),
        "vs. assuming no change": f"{(1 - m['holdout']['RMSE'] / m['holdout_persistence']['RMSE']) * 100:.0f}% better",
        "Catches bad-air days": f"{m['alert_quantile_head']['recall'] * 100:.0f}%",
    } for m in meta])
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.caption(
        "Scored on a held-out year the models never trained on. The baseline is "
        "persistence — assuming tomorrow matches today — which is a surprisingly "
        "hard bar at 24 hours and a weak one at 72."
    )

with st.expander("What drives the prediction"):
    st.write(
        "Two models run for each horizon. One predicts the most likely AQI; a "
        "separate model predicts a 90th-percentile upper bound, used only for the "
        "warning. Optimising a single model for both jobs makes it hedge toward "
        "the average and miss the days that matter."
    )
    st.write(
        "Inputs split into what is known now — recent pollutant readings, their "
        "rolling averages, and how fast AQI is changing — and what is known about "
        "the future: the weather forecast for the target hour, plus season and "
        "time of day."
    )
    shap = FIGURES / "shap_top20_h72.png"
    if shap.exists():
        st.image(str(shap), caption="Feature importance at the three-day horizon")

st.caption(
    "Air quality data from the Copernicus CAMS global model via Open-Meteo, at "
    "roughly 40 km resolution. Regional estimates rather than street-level "
    "readings, so local peaks near traffic will be higher than shown."
)