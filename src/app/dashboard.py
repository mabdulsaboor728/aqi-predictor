"""
Streamlit dashboard - the public face of the forecasting system.

    streamlit run src/app/dashboard.py

Data sources:
  reports/latest_forecast.json   written hourly by pipelines/hourly.py
  reports/history/*.json         one snapshot per hourly run, for the backtest
  Open-Meteo air-quality API     live pollutants + observed AQI
  Open-Meteo forecast API        live weather conditions

No Hopsworks dependency: everything is either committed by the hourly workflow
or fetchable without credentials, so the app deploys with no secrets.

Design notes
------------
The layout is a bulletin rather than a monitoring console. The audience is a
person deciding whether to go outside, not an operator watching a system.

Two live themes carry state before any number is read:

  SKY     the masthead glyph, condition word and wash behind the conditions
          strip are derived from cloud cover, rain, wind and local hour, so
          the page looks like the weather outside. Smog gets its own state
          when the air is bad and the sky is clear.
  AIR     the AQI block carries a particulate texture whose density scales
          with the reading. A clean day is nearly bare; a hazardous one is
          dense. The texture is information, not decoration.

Category colours follow the US AQI standard's semantics, but each has a light
tint for fills and a deeper ink for text. The official values (#00e400,
#ffff00) are specified for small badges and glare across large areas. Keeping
the hue keeps the recognition; adjusting the value makes it readable.

Charts strip Plotly's defaults deliberately: no gridlines except faint
horizontals, no axis spines, no legend box, direct labelling on the series
instead of a key. Nothing should read as a notebook plot.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
FORECAST = ROOT / "reports" / "latest_forecast.json"
HISTORY = ROOT / "reports" / "history"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

LAT, LON = 33.6844, 73.0479
LOCAL_TZ = "Asia/Karachi"
ALERT_THRESHOLD = 150

# ---------------------------------------------------------------- palette
PAPER = "#F8F9F8"
INK = "#22252B"
MUTED = "#6E7580"
RULE = "#E3E6E3"
FAINT = "#EFF1EF"

# US AQI categories: (upper bound, label, tint for fills, ink for text, advice)
BANDS = [
    (50, "Good", "#E9F2EC", "#2F7D52",
     "Air quality is fine. No precautions needed."),
    (100, "Moderate", "#FBF4DE", "#A8791A",
     "Unusually sensitive people may want to shorten long outdoor sessions."),
    (150, "Unhealthy for sensitive groups", "#FBEBDC", "#BE5F1D",
     "Children, older adults and anyone with heart or lung conditions should "
     "keep outdoor exertion short."),
    (200, "Unhealthy", "#F9E4E4", "#B93A2E",
     "Everyone should limit prolonged outdoor exertion. Sensitive groups are "
     "better off indoors."),
    (300, "Very unhealthy", "#EFE5F2", "#77497F",
     "Avoid outdoor exertion. Keep windows shut and run filtration if you have it."),
    (10**6, "Hazardous", "#EEDDDD", "#7E2323",
     "Stay indoors. Avoid all outdoor activity."),
]


def band(aqi: float):
    for limit, label, tint, ink, advice in BANDS:
        if aqi <= limit:
            return label, tint, ink, advice
    return BANDS[-1][1:]


# Concentration at which each pollutant starts to matter for health, in the
# units Open-Meteo returns. Used only to scale the load bars and identify which
# pollutant is currently binding. This is NOT a reproduction of the EPA
# sub-index arithmetic and is labelled as a relative load, not an AQI.
POLLUTANTS = [
    ("pm2_5", "PM2.5", "µg/m³", 35.4),
    ("pm10", "PM10", "µg/m³", 154.0),
    ("ozone", "Ozone", "µg/m³", 100.0),
    ("nitrogen_dioxide", "NO₂", "µg/m³", 100.0),
    ("sulphur_dioxide", "SO₂", "µg/m³", 75.0),
    ("carbon_monoxide", "CO", "µg/m³", 4000.0),
]

WEATHER_FIELDS = [
    ("temperature_2m", "Temperature", "°C", "{:.0f}"),
    ("relative_humidity_2m", "Humidity", "%", "{:.0f}"),
    ("wind_speed_10m", "Wind", "km/h", "{:.0f}"),
    ("precipitation", "Rain", "mm", "{:.1f}"),
    ("surface_pressure", "Pressure", "hPa", "{:.0f}"),
    ("cloud_cover", "Cloud", "%", "{:.0f}"),
]

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

# Plain-language labels for raw feature names.
#
# Per-prediction labels arrive already humanised in the forecast JSON, written
# by src/models/predict.py. This local copy is only for the global-importance
# chart, which reads a raw CSV. It is duplicated rather than imported because
# the dashboard's slim requirements exclude scikit-learn, so importing the
# prediction module fails on Streamlit Cloud.
_SHORT = {
    "us_aqi": "AQI now",
    "us_aqi_same_hour_yesterday": "AQI at this hour yesterday",
    "us_aqi_anomaly_vs_72h": "AQI vs its 3-day average",
    "pm25_pm10_ratio": "PM2.5 to PM10 ratio",
    "dew_spread": "Dew point spread",
    "hour_sin": "Time of day", "hour_cos": "Time of day", "hour": "Time of day",
    "doy_sin": "Season", "doy_cos": "Season", "month": "Month",
    "dayofweek": "Day of week", "is_weekend": "Weekend",
    "wind_dir_sin": "Wind direction", "wind_dir_cos": "Wind direction",
    "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
    "dew_point_2m": "Dew point", "precipitation": "Rain",
    "surface_pressure": "Pressure", "cloud_cover": "Cloud cover",
    "wind_speed_10m": "Wind speed", "wind_gusts_10m": "Wind gusts",
    "shortwave_radiation": "Sunlight",
    "pm2_5": "PM2.5", "pm10": "PM10", "ozone": "Ozone",
    "nitrogen_dioxide": "NO₂", "sulphur_dioxide": "SO₂",
    "carbon_monoxide": "CO", "dust": "Dust",
    "aerosol_optical_depth": "Aerosol depth",
}
_PATTERNS = [
    (r"^us_aqi_lag(\d+)$", "AQI {0}h ago"),
    (r"^us_aqi_rmean(\d+)$", "AQI, {0}h average"),
    (r"^us_aqi_rstd(\d+)$", "AQI swing over {0}h"),
    (r"^us_aqi_rmax(\d+)$", "AQI peak in last {0}h"),
    (r"^us_aqi_rmin(\d+)$", "AQI low in last {0}h"),
    (r"^us_aqi_delta(\d+)$", "AQI change over {0}h"),
    (r"^us_aqi_rate(\d+)$", "AQI trend over {0}h"),
    (r"^wind_rmean(\d+)$", "Wind, {0}h average"),
    (r"^precip_rsum(\d+)$", "Rain, {0}h total"),
    (r"^temp_rmean(\d+)$", "Temperature, {0}h average"),
    (r"^(.+?)_r(\d+)h$", "{0}, {1}h average"),
    (r"^(.+?)_lag(\d+)$", "{0} {1}h ago"),
    (r"^(.+?)_delta(\d+)$", "{0} change over {1}h"),
]


def humanise(feat: str) -> str:
    f = re.sub(r"_h\d+$", "", feat)
    forecast = f.endswith("_f")
    f = re.sub(r"_[tf]$", "", f)
    suffix = " (forecast)" if forecast else ""

    if f in _SHORT:
        return _SHORT[f] + suffix
    for pat, tmpl in _PATTERNS:
        m = re.match(pat, f)
        if m:
            parts = [_SHORT.get(g, g.replace("_", " ")) for g in m.groups()]
            return tmpl.format(*parts) + suffix
    return f.replace("_", " ") + suffix


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300)
def load_forecast() -> dict | None:
    return json.loads(FORECAST.read_text()) if FORECAST.exists() else None


@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
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
                "horizon_h": f["horizon_h"],
                "valid_at": pd.to_datetime(f["valid_at"], utc=True),
                "predicted": f["aqi"],
                "upper_q90": f.get("upper_q90", f.get("aqi_upper_q90")),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=900)
def fetch_air_quality(days: int = 14) -> pd.DataFrame:
    """Hourly pollutants and AQI. Also supplies observations for the backtest."""
    end = datetime.now(timezone.utc).date()
    variables = ["us_aqi"] + [p[0] for p in POLLUTANTS]
    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={"latitude": LAT, "longitude": LON,
                "hourly": ",".join(variables),
                "start_date": (end - timedelta(days=days)).isoformat(),
                "end_date": end.isoformat(),
                "domains": "cams_global", "timezone": "UTC"},
        timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


@st.cache_data(ttl=900)
def fetch_weather() -> pd.DataFrame:
    variables = [w[0] for w in WEATHER_FIELDS] + ["wind_direction_10m"]
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": LAT, "longitude": LON,
                "hourly": ",".join(variables),
                "past_days": 2, "forecast_days": 1, "timezone": "UTC"},
        timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


@st.cache_data(ttl=3600)
def load_model_meta() -> list[dict]:
    out = []
    for p in sorted(MODELS.glob("model_h*_meta.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:                                     # noqa: BLE001
            continue
    return out


def latest_row(df: pd.DataFrame):
    """Most recent populated row at or before now."""
    if df.empty:
        return None
    sub = df[df.index <= pd.Timestamp.now(tz="UTC")].dropna(how="all")
    return sub.iloc[-1] if len(sub) else None


def local(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(LOCAL_TZ)


# --------------------------------------------------------------------------- #
# live sky theme
# --------------------------------------------------------------------------- #
def sky_state(wrow, aqi: float, hour: int) -> dict:
    """Classify the sky from live weather, and pick its visual treatment.

    Order matters: rain overrides cloud, and heavy smog under a clear sky gets
    its own state because "sunny" would be actively misleading on a day when
    the air is dangerous.
    """
    night = hour < 6 or hour >= 19
    cloud = float(wrow["cloud_cover"]) if wrow is not None and "cloud_cover" in wrow \
        and pd.notna(wrow["cloud_cover"]) else None
    rain = float(wrow["precipitation"]) if wrow is not None and "precipitation" in wrow \
        and pd.notna(wrow["precipitation"]) else 0.0
    wind = float(wrow["wind_speed_10m"]) if wrow is not None and "wind_speed_10m" in wrow \
        and pd.notna(wrow["wind_speed_10m"]) else 0.0

    if wrow is None or cloud is None:
        key, word = "unknown", "Conditions unavailable"
    elif rain >= 0.2:
        key, word = "rain", "Raining"
    elif aqi >= 150 and cloud < 55:
        key, word = "smog", "Hazy with smog"
    elif cloud >= 80:
        key, word = "overcast", "Overcast"
    elif wind >= 25:
        key, word = "wind", "Blustery"
    elif cloud >= 35:
        key, word = "partly", "Partly cloudy"
    else:
        key, word = "clear", "Clear"

    if key == "clear" and night:
        key, word = "night", "Clear night"
    elif key == "partly" and night:
        word = "Cloudy night"

    themes = {
        # accent, page background (top -> bottom), text tone on the page
        "clear":    ("#E0A21C", "#DCEBF7", "#EFF4F1", "#F8F9F8"),
        "night":    ("#7286AD", "#D3D9E9", "#E7EAF1", "#F5F6F8"),
        "partly":   ("#6D8DA6", "#DFE9F0", "#EDF2F2", "#F8F9F8"),
        "overcast": ("#78818A", "#DCDFE2", "#EAECEC", "#F7F8F8"),
        "rain":     ("#3F7398", "#D2E0EA", "#E4EDF1", "#F5F8F9"),
        "wind":     ("#4E8A76", "#DAEAE3", "#E9F2EC", "#F6F9F7"),
        "smog":     ("#A6742F", "#EDE1CB", "#F3EDE0", "#F9F7F2"),
        "unknown":  (MUTED, "#EFF1EF", "#F4F5F4", PAPER),
    }
    accent, top, mid, bottom = themes[key]
    return {
        "key": key, "word": word, "accent": accent, "night": night,
        # the whole page, not a strip
        "page": f"linear-gradient(180deg,{top} 0%,{mid} 34%,{bottom} 72%,{PAPER} 100%)",
        "wash": f"linear-gradient(160deg,{top} 0%,{mid} 62%,{bottom} 100%)",
        "panel": bottom,
    }


def sky_glyph(key: str, accent: str, size: int = 44) -> str:
    """Hand-drawn SVG rather than an emoji or icon font, so the line weight and
    colour sit with the rest of the page instead of importing another style."""
    s, a, w = size, accent, 1.7
    common = (f'<svg width="{s}" height="{s}" viewBox="0 0 44 44" fill="none" '
              f'xmlns="http://www.w3.org/2000/svg" '
              f'stroke="{a}" stroke-width="{w}" stroke-linecap="round" '
              f'stroke-linejoin="round">')
    sun_rays = "".join(
        f'<line x1="{22 + 11 * math.cos(math.radians(d)):.1f}" '
        f'y1="{22 + 11 * math.sin(math.radians(d)):.1f}" '
        f'x2="{22 + 15.5 * math.cos(math.radians(d)):.1f}" '
        f'y2="{22 + 15.5 * math.sin(math.radians(d)):.1f}"/>'
        for d in range(0, 360, 45))
    cloud = ('<path d="M13 30h17a6 6 0 0 0 .6-11.97A9 9 0 0 0 13.2 20 '
             'A5.5 5.5 0 0 0 13 30Z"/>')

    if key == "clear":
        return (common + f'<circle cx="22" cy="22" r="7.5" fill="{a}" '
                f'fill-opacity=".18"/>' + sun_rays + "</svg>")
    if key == "night":
        return (common + '<path d="M27.5 10.5A11 11 0 1 0 33 27.8'
                'a8.6 8.6 0 0 1-5.5-17.3Z" fill="' + a + '" fill-opacity=".16"/>'
                '<circle cx="32" cy="13" r="1"/><circle cx="36" cy="19" r="1"/>'
                '</svg>')
    if key == "partly":
        return (common + f'<circle cx="16" cy="16" r="5.5" fill="{a}" '
                f'fill-opacity=".18"/>'
                '<line x1="16" y1="7" x2="16" y2="4.5"/>'
                '<line x1="8.5" y1="16" x2="6" y2="16"/>'
                '<line x1="10.4" y1="10.4" x2="8.6" y2="8.6"/>'
                + cloud.replace('d="M13', 'd="M15') + "</svg>")
    if key == "overcast":
        return (common + f'<path d="M11 32h18a6.3 6.3 0 0 0 .6-12.6'
                f'A9.4 9.4 0 0 0 11.2 21.5 A5.7 5.7 0 0 0 11 32Z" '
                f'fill="{a}" fill-opacity=".14"/>'
                '<path d="M17 16.5a8 8 0 0 1 12.5 1.5" opacity=".55"/></svg>')
    if key == "rain":
        return (common + f'<path d="M12 26h18a6 6 0 0 0 .6-12A9 9 0 0 0 12.2 16'
                f'A5.5 5.5 0 0 0 12 26Z" fill="{a}" fill-opacity=".14"/>'
                '<line x1="16" y1="31" x2="14.5" y2="35.5"/>'
                '<line x1="22" y1="31" x2="20.5" y2="36.5"/>'
                '<line x1="28" y1="31" x2="26.5" y2="35.5"/></svg>')
    if key == "wind":
        return (common + '<path d="M6 17h16a4.5 4.5 0 1 0-4.5-4.5"/>'
                '<path d="M6 24h22a5 5 0 1 1-5 5"/>'
                '<path d="M6 31h11a3.5 3.5 0 1 1-3.5 3.5" opacity=".6"/></svg>')
    if key == "smog":
        return (common + f'<circle cx="22" cy="17" r="6.5" fill="{a}" '
                f'fill-opacity=".16"/>'
                '<line x1="9" y1="27" x2="35" y2="27"/>'
                '<line x1="12" y1="31.5" x2="32" y2="31.5" opacity=".7"/>'
                '<line x1="15" y1="36" x2="29" y2="36" opacity=".45"/></svg>')
    return common + '<circle cx="22" cy="22" r="9" opacity=".4"/></svg>'


def ambient(key: str, accent: str) -> str:
    """Full-page atmospheric layer behind the content.

    This is what makes the weather legible at a glance before any text is read:
    the page background shifts with the sky, and a fixed layer carries slow
    motion appropriate to the condition - drifting cloud, falling rain, blowing
    streaks, a breathing sun, twinkling stars, or rolling haze.

    Pure CSS keyframes on absolutely positioned elements. No JavaScript, so it
    cannot interfere with the pearl cursor component, and everything is
    suppressed under prefers-reduced-motion.
    """
    def blob(cls, x, y, w, h, colour, opacity, dur, delay=0.0, blur=38):
        return (f'<div class="{cls}" style="left:{x}%;top:{y}%;width:{w}px;'
                f'height:{h}px;background:{colour};opacity:{opacity};'
                f'filter:blur({blur}px);animation-duration:{dur}s;'
                f'animation-delay:{delay}s"></div>')

    parts = []

    if key == "clear":
        # Deliberately strong: a weak glow reads as a rendering artefact rather
        # than as sunlight. This should be unmistakable at a glance.
        parts.append(f'<div class="sunglow" style="background:radial-gradient('
                     f'circle,{accent}FF 0%,{accent}CC 14%,{accent}77 32%,'
                     f'{accent}33 52%,transparent 74%)"></div>')
        parts.append(f'<div class="sundisc" style="background:radial-gradient('
                     f'circle,#FFF8E0 0%,{accent} 55%,{accent}00 100%)"></div>')
        for i, (x, y, w, h, o, d) in enumerate(
                [(8, 16, 340, 100, .70, 150), (58, 34, 420, 125, .58, 190),
                 (28, 64, 300, 92, .48, 170)]):
            parts.append(blob("drift", x, y, w, h, "#FFFFFF", o, d, -i * 30))

    elif key == "night":
        parts.append('<div class="sunglow" style="background:radial-gradient('
                     'circle,#DCE4FAEE 0%,#B9C6EA99 26%,#8FA0CC44 50%,'
                     'transparent 74%)"></div>')
        parts.append('<div class="sundisc" style="background:radial-gradient('
                     'circle,#FDFCF4 0%,#E4E9F7 60%,#E4E9F700 100%)"></div>')
        star = 1103515245
        for i in range(46):
            star = (star * 1103515245 + 12345) & 0x7FFFFFFF
            x = (star >> 8) % 100
            star = (star * 1103515245 + 12345) & 0x7FFFFFFF
            y = (star >> 8) % 78
            star = (star * 1103515245 + 12345) & 0x7FFFFFFF
            d = 2.4 + ((star >> 6) % 40) / 10
            star = (star * 1103515245 + 12345) & 0x7FFFFFFF
            sz = 2 + ((star >> 6) % 3)
            parts.append(f'<div class="star" style="left:{x}%;top:{y}%;'
                         f'width:{sz}px;height:{sz}px;'
                         f'animation-duration:{d:.1f}s;'
                         f'animation-delay:-{d / 2:.1f}s"></div>')

    elif key in ("partly", "overcast"):
        heavy = key == "overcast"
        base = "#B7C0C8" if heavy else "#FFFFFF"
        if not heavy:
            parts.append(f'<div class="sunglow" style="background:radial-gradient('
                         f'circle,{accent}AA 0%,{accent}55 30%,{accent}1A 52%,'
                         f'transparent 72%)"></div>')
        specs = ([(-18, 6, 520, 150, .85, 120), (32, 24, 600, 175, .78, 155),
                  (2, 50, 480, 140, .68, 135), (56, 68, 540, 158, .58, 175)]
                 if heavy else
                 [(-12, 12, 380, 112, .82, 140), (46, 32, 450, 130, .70, 175),
                  (18, 66, 340, 104, .58, 155)])
        for i, (x, y, w, h, o, d) in enumerate(specs):
            parts.append(blob("drift", x, y, w, h, base, o, d, -i * 28))

    elif key == "rain":
        for i, (x, y, w, h, o, d) in enumerate(
                [(-14, 4, 560, 158, .88, 110), (36, 22, 620, 180, .78, 140),
                 (6, 46, 520, 148, .66, 125)]):
            parts.append(blob("drift", x, y, w, h, "#93A7B8", o, d, -i * 25))
        seed = 7919
        for i in range(130):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            x = (seed >> 8) % 100
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            dur = 0.62 + ((seed >> 7) % 60) / 100
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            delay = ((seed >> 7) % 200) / 100
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            ln = 18 + ((seed >> 7) % 30)
            parts.append(f'<div class="drop" style="left:{x}%;height:{ln}px;'
                         f'animation-duration:{dur:.2f}s;'
                         f'animation-delay:-{delay:.2f}s"></div>')

    elif key == "wind":
        for i, (x, y, w, h, o, d) in enumerate(
                [(-12, 16, 440, 128, .70, 90), (42, 44, 500, 142, .56, 110)]):
            parts.append(blob("drift", x, y, w, h, "#FFFFFF", o, d, -i * 20))
        seed = 4211
        for i in range(30):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            y = 4 + ((seed >> 8) % 88)
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            wdt = 120 + ((seed >> 7) % 300)
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            dur = 2.8 + ((seed >> 7) % 36) / 10
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            delay = ((seed >> 7) % 500) / 100
            parts.append(f'<div class="gust" style="top:{y}%;width:{wdt}px;'
                         f'background:linear-gradient(90deg,transparent,'
                         f'{accent}CC,transparent);'
                         f'animation-duration:{dur:.1f}s;'
                         f'animation-delay:-{delay:.2f}s"></div>')

    elif key == "smog":
        parts.append(f'<div class="sunglow" style="background:radial-gradient('
                     f'circle,{accent}BB 0%,{accent}66 28%,{accent}22 52%,'
                     f'transparent 74%)"></div>')
        for i, (y, o, d) in enumerate([(10, .80, 90), (28, .72, 115),
                                       (46, .62, 135), (64, .52, 105),
                                       (82, .42, 125)]):
            parts.append(f'<div class="haze" style="top:{y}%;opacity:{o};'
                         f'background:linear-gradient(90deg,transparent,'
                         f'#C09B62 0%,#B98F4E 50%,transparent);'
                         f'animation-duration:{d}s;'
                         f'animation-delay:-{i * 22}s"></div>')

    return f'<div class="ambient" aria-hidden="true">{"".join(parts)}</div>'


def particulates(aqi: float, colour: str, w: int = 300, h: int = 210) -> str:
    """Texture whose density scales with the reading.

    Deterministic from the AQI value, so it is stable between reruns within a
    single reading and visibly changes when the air does. A Good day shows
    roughly 18 specks; a Hazardous one several hundred.
    """
    # Curve tuned to this city's observed range (14 to 218, median 106): a Good
    # day is almost bare, a typical day is visibly speckled, and the worst
    # readings on record are dense without going solid black.
    n = int(min(420, 6 + (max(aqi, 0) / 100.0) ** 2.0 * 105))
    seed = int(aqi * 977) or 1
    dots = []
    for i in range(n):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        x = (seed >> 7) % w
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        y = (seed >> 7) % h
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        r = 0.6 + ((seed >> 9) % 22) / 14
        o = 0.10 + ((seed >> 5) % 26) / 100
        dots.append(f'<circle cx="{x}" cy="{y}" r="{r:.2f}" opacity="{o:.2f}"/>')
    return (f'<svg class="grit" width="100%" height="100%" viewBox="0 0 {w} {h}" '
            f'preserveAspectRatio="xMidYMid slice" fill="{colour}">'
            f'{"".join(dots)}</svg>')


def html_table(df: pd.DataFrame, index_header: str = "",
               align_right_from: int = 1) -> str:
    """Render a table as themed markup.

    st.dataframe brings its own component styling, which ignores the page theme
    and renders as a dark slab against a light bulletin. These tables are small
    and static, so plain markup gives full control and costs nothing.
    """
    heads = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for idx, r in df.iterrows():
        cells = "".join(
            f'<td class="{"num" if i >= align_right_from - 1 else ""}">{v}</td>'
            for i, v in enumerate(r))
        first = f"<th scope='row'>{idx}</th>" if index_header != "__none__" else ""
        rows.append(f"<tr>{first}{cells}</tr>")
    idx_head = (f"<th>{index_header}</th>" if index_header != "__none__" else "")
    return (f'<div class="tbl"><table><thead><tr>{idx_head}{heads}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def sparkline(series: pd.Series, colour: str, w: int = 168, h: int = 34) -> str:
    """Tiny inline SVG trend - calmer and cheaper than a chart component."""
    v = series.dropna().tolist()
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1
    step = w / (len(v) - 1)
    pts = " ".join(f"{i * step:.1f},{h - 3 - (x - lo) / rng * (h - 6):.1f}"
                   for i, x in enumerate(v))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'preserveAspectRatio="none" style="display:block">'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="1.6" stroke-linejoin="round" opacity="0.75"/></svg>')


def style_chart(fig: go.Figure, height: int = 340) -> go.Figure:
    """Strip Plotly's defaults: faint horizontal rules only, no spines, no box."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=MUTED),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=RULE, font_size=12,
                        font_family="IBM Plex Sans, sans-serif"),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False,
                     ticks="outside", tickcolor=RULE, ticklen=4)
    fig.update_yaxes(showgrid=True, gridcolor=FAINT, gridwidth=1,
                     zeroline=False, showline=False, ticks="")
    return fig


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Islamabad air quality", page_icon="◐",
                   layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------- pearl cursor
# Keeps the dashboard's original visual design intact. The normal system cursor
# is hidden and replaced with a small, multi-colour cluster of pearl-like beads
# that smoothly follows mouse movement.
def inject_pearl_cluster_cursor():
    components.html(
        """
        <script>
        (function () {
          const doc = (window.parent && window.parent.document)
            ? window.parent.document
            : document;

          const layerId = "pearl-cluster-layer-v1";
          const styleId = "pearl-cluster-style-v1";

          // Streamlit reruns this component. Remove the old pearl layer first so
          // repeated reruns never create duplicate cursors.
          const oldLayer = doc.getElementById(layerId);
          if (oldLayer) oldLayer.remove();

          let style = doc.getElementById(styleId);
          if (!style) {
            style = doc.createElement("style");
            style.id = styleId;
            style.textContent = `
              html, body, body *, .stApp, .stApp * {
                cursor: none !important;
              }

              #${layerId} {
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 2147483646;
              }

              #${layerId} .pearl {
                position: fixed;
                left: 0;
                top: 0;
                border-radius: 999px;
                pointer-events: none;
                will-change: transform, opacity;
                box-shadow:
                  0 6px 14px rgba(15, 23, 42, .14),
                  inset 1px 1px 2px rgba(255, 255, 255, .95),
                  inset -2px -2px 4px rgba(0, 0, 0, .10);
              }
            `;
            doc.head.appendChild(style);
          }

          const layer = doc.createElement("div");
          layer.id = layerId;
          doc.body.appendChild(layer);

          // Pearl cluster. Colours sweep the full spectrum so the trail reads as
          // an iridescent strand rather than a single-hue tail.
          const specs = [
            {size: 17, c1: "#ffffff", c2: "#f4d7ff", c3: "#be7cff"},
            {size: 15, c1: "#ffffff", c2: "#dcd6ff", c3: "#7c6bff"},
            {size: 14, c1: "#ffffff", c2: "#c7f9ff", c3: "#34d3ff"},
            {size: 13, c1: "#ffffff", c2: "#c9fff4", c3: "#19c6b6"},
            {size: 12, c1: "#ffffff", c2: "#d7ffe8", c3: "#45d483"},
            {size: 11, c1: "#ffffff", c2: "#ecffd0", c3: "#9fd93a"},
            {size: 11, c1: "#ffffff", c2: "#ffe8b6", c3: "#ffbf47"},
            {size: 10, c1: "#ffffff", c2: "#ffdcc0", c3: "#ff9440"},
            {size:  9, c1: "#ffffff", c2: "#ffd6cf", c3: "#ff6a52"},
            {size:  9, c1: "#ffffff", c2: "#ffd8e5", c3: "#ff72a0"},
            {size:  8, c1: "#ffffff", c2: "#ffd4f4", c3: "#f45fd0"},
            {size:  7, c1: "#ffffff", c2: "#e6ddff", c3: "#8e7dff"},
            {size:  7, c1: "#ffffff", c2: "#d8fff8", c3: "#33d7c7"},
            {size:  6, c1: "#ffffff", c2: "#fff0d4", c3: "#ffb341"},
            {size:  6, c1: "#ffffff", c2: "#d5ecff", c3: "#4aa8ff"},
            {size:  5, c1: "#ffffff", c2: "#ffe1ec", c3: "#ff5f8f"}
          ];

          const pearls = specs.map((s, i) => {
            const el = doc.createElement("div");
            el.className = "pearl";
            el.style.width = `${s.size}px`;
            el.style.height = `${s.size}px`;
            el.style.opacity = "0";
            el.style.background =
              `radial-gradient(circle at 30% 28%, ${s.c1} 0 16%, ${s.c2} 38%, ${s.c3} 100%)`;
            layer.appendChild(el);

            return {
              el,
              size: s.size,
              x: -100,
              y: -100,
              tx: -100,
              ty: -100,
              lag: 0.19 - i * 0.0085,
              angle: i * 0.85,
              radius: 1.5 + i * 1.15
            };
          });

          let pointerX = -100;
          let pointerY = -100;
          let lastX = -100;
          let lastY = -100;
          let speed = 0;
          let raf = null;

          function onMove(e) {
            pointerX = e.clientX;
            pointerY = e.clientY;

            const dx = pointerX - lastX;
            const dy = pointerY - lastY;
            speed = Math.min(26, Math.hypot(dx, dy));

            lastX = pointerX;
            lastY = pointerY;

            pearls.forEach((p, i) => {
              p.el.style.opacity = String(Math.max(0.60, 0.98 - i * 0.022));
            });
          }

          function onLeave() {
            pearls.forEach(p => p.el.style.opacity = "0");
          }

          function animate() {
            let leadX = pointerX;
            let leadY = pointerY;

            pearls.forEach((p, i) => {
              p.angle += 0.02 + i * 0.003;

              // A tiny organic orbit keeps it feeling like a cluster instead of
              // a straight dotted trail.
              const radiusBoost = 1 + speed * 0.015;
              const swirlX = Math.cos(p.angle) * p.radius * radiusBoost;
              const swirlY = Math.sin(p.angle) * p.radius * radiusBoost;

              p.tx = leadX + swirlX;
              p.ty = leadY + swirlY;

              p.x += (p.tx - p.x) * Math.max(0.05, p.lag);
              p.y += (p.ty - p.y) * Math.max(0.05, p.lag);

              p.el.style.transform =
                `translate(${p.x - p.size / 2}px, ${p.y - p.size / 2}px)`;

              leadX = p.x;
              leadY = p.y;
            });

            speed *= 0.90;
            raf = requestAnimationFrame(animate);
          }

          doc.addEventListener("mousemove", onMove, {passive: true});
          doc.addEventListener("mouseenter", onMove, {passive: true});
          doc.addEventListener("mouseleave", onLeave, {passive: true});

          if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            doc.addEventListener("mousemove", function (e) {
              pearls.forEach((p, i) => {
                p.x = e.clientX + i * 1.7;
                p.y = e.clientY + i * 1.7;
                p.el.style.opacity = "0.95";
                p.el.style.transform =
                  `translate(${p.x - p.size / 2}px, ${p.y - p.size / 2}px)`;
              });
            }, {passive: true});
          } else {
            animate();
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


inject_pearl_cluster_cursor()

fc = load_forecast()

aq_err = wx_err = None
try:
    aq = fetch_air_quality()
except Exception as exc:                                      # noqa: BLE001
    aq, aq_err = pd.DataFrame(), exc
try:
    wx = fetch_weather()
except Exception as exc:                                      # noqa: BLE001
    wx, wx_err = pd.DataFrame(), exc

wrow = latest_row(wx)
now_local = pd.Timestamp.now(tz=LOCAL_TZ)
current_aqi = fc["current_aqi"] if fc else 0.0
sky = sky_state(wrow, current_aqi, now_local.hour)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;500;600&display=swap');

.stApp {{ background: {sky['page']}; background-attachment: fixed; }}
header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding: 2rem 3rem 4rem; max-width: 1180px;
                   position: relative; z-index: 1; }}

html, body, [class*="css"], .stMarkdown, p, div, span, li, td, th {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif; color: {INK};
}}

/* ---- live sky: full-page ambient layer ---- */
.ambient {{ position:fixed; inset:0; overflow:hidden; pointer-events:none;
           z-index:0; }}
.ambient > div {{ position:absolute; }}
.ambient .sunglow {{ top:-300px; right:-220px; width:1000px; height:1000px;
                    border-radius:50%; animation:breathe 9s ease-in-out infinite; }}
.ambient .sundisc {{ top:22px; right:78px; width:132px; height:132px;
                    border-radius:50%; filter:blur(3px);
                    animation:breathe 9s ease-in-out infinite; }}
.ambient .drift {{ border-radius:50%; animation-name:drift;
                  animation-timing-function:linear;
                  animation-iteration-count:infinite; }}
.ambient .star {{ border-radius:50%; background:#7185B5;
                 box-shadow:0 0 5px #9CADD8;
                 animation-name:twinkle; animation-iteration-count:infinite;
                 animation-timing-function:ease-in-out; }}
.ambient .drop {{ top:-60px; width:1.8px; border-radius:1px;
                 background:linear-gradient(180deg,transparent,#5E819C);
                 animation-name:fall; animation-timing-function:linear;
                 animation-iteration-count:infinite; }}
.ambient .gust {{ height:2.2px; border-radius:2px; animation-name:blow;
                 animation-timing-function:ease-in-out;
                 animation-iteration-count:infinite; }}
.ambient .haze {{ left:-30%; width:160%; height:150px; filter:blur(30px);
                 animation-name:roll; animation-timing-function:ease-in-out;
                 animation-direction:alternate;
                 animation-iteration-count:infinite; }}

@keyframes breathe {{ 0%,100% {{ transform:scale(1); opacity:.9; }}
                      50%     {{ transform:scale(1.07); opacity:1; }} }}
@keyframes drift  {{ from {{ transform:translateX(-18vw); }}
                     to   {{ transform:translateX(118vw); }} }}
@keyframes twinkle {{ 0%,100% {{ opacity:.2; }} 50% {{ opacity:1; }} }}
@keyframes fall   {{ from {{ transform:translateY(-12vh); opacity:0; }}
                     8%   {{ opacity:1; }}
                     to   {{ transform:translateY(115vh); opacity:.6; }} }}
@keyframes blow   {{ 0%   {{ transform:translateX(-35vw); opacity:0; }}
                     18%  {{ opacity:1; }}
                     100% {{ transform:translateX(125vw); opacity:0; }} }}
@keyframes roll   {{ from {{ transform:translateX(-10%); }}
                     to   {{ transform:translateX(10%); }} }}

@keyframes rise {{ from {{ opacity:0; transform:translateY(9px); }}
                   to   {{ opacity:1; transform:none; }} }}
.rise {{ animation: rise .5s cubic-bezier(.22,.7,.3,1) both; }}

@media (prefers-reduced-motion: reduce) {{
  .rise, .ambient * {{ animation:none !important; }}
  .ambient .drop, .ambient .gust {{ display:none; }}
}}

/* masthead with the live sky state */
.mast {{ display:flex; justify-content:space-between; align-items:flex-end;
        flex-wrap:wrap; gap:1rem 1.6rem;
        border-bottom:1.5px solid {INK}; padding-bottom:.7rem; margin-bottom:1.7rem; }}
.mast h1 {{ font-family:'IBM Plex Serif',serif; font-size:1.5rem; font-weight:600;
           margin:0; letter-spacing:-.015em; line-height:1.1; }}
.mast .place {{ font-size:.78rem; color:{MUTED}; margin-top:.15rem; }}
.sky {{ display:flex; align-items:center; gap:.75rem; margin-left:auto; }}
.sky .glyph {{ line-height:0; }}
.sky .txt .word {{ font-size:.95rem; font-weight:600; color:{sky['accent']};
                  line-height:1.15; }}
.sky .txt .temp {{ font-family:'IBM Plex Serif',serif; font-size:1.15rem;
                  font-weight:500; font-variant-numeric:tabular-nums; }}
.mast .when {{ font-size:.8rem; color:{MUTED}; text-align:right; line-height:1.45; }}

/* reading */
.reading {{ display:grid; grid-template-columns:308px 1fr; gap:2.6rem;
           align-items:start; }}
.figure {{ position:relative; overflow:hidden; padding:1.5rem 1.6rem 1.3rem;
          border-radius:4px; min-height:196px; }}
.figure .grit {{ position:absolute; inset:0; pointer-events:none; }}
.figure > *:not(.grit) {{ position:relative; z-index:1; }}
.figure .n {{ font-family:'IBM Plex Serif',serif; font-size:5.6rem; font-weight:500;
             line-height:.86; letter-spacing:-.038em; font-variant-numeric:tabular-nums; }}
.figure .cat {{ font-size:.95rem; font-weight:600; margin-top:.75rem; line-height:1.3; }}
.figure .spark {{ margin-top:1rem; }}
.figure .sparklab {{ font-size:.7rem; color:{MUTED}; margin-top:.15rem; }}
.advice {{ font-size:1.04rem; line-height:1.62; max-width:52ch; margin-bottom:1.6rem; }}

.pol {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.2rem 2rem; }}
.pol .row {{ padding:.44rem 0; border-bottom:1px solid {FAINT}; }}
.pol .top {{ display:flex; justify-content:space-between; align-items:baseline; }}
.pol .name {{ font-size:.82rem; font-weight:500; }}
.pol .val {{ font-size:.82rem; font-variant-numeric:tabular-nums; color:{MUTED}; }}
.pol .track {{ height:3px; background:{FAINT}; margin-top:.36rem; border-radius:2px; }}
.pol .fill {{ height:3px; border-radius:2px; }}
.pol .lead {{ font-weight:600; }}

.sec {{ font-family:'IBM Plex Serif',serif; font-size:1.14rem; font-weight:600;
       border-top:1px solid {RULE}; padding-top:1.15rem; margin:2.5rem 0 .35rem; }}
.subtle {{ font-size:.84rem; color:{MUTED}; line-height:1.58; }}

.days {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.6rem; margin-top:.45rem; }}
.day {{ border-top:2px solid; padding-top:.75rem; }}
.day .lab {{ font-size:.83rem; font-weight:600; }}
.day .sub {{ font-size:.74rem; color:{MUTED}; margin-top:.08rem; }}
.day .n {{ font-family:'IBM Plex Serif',serif; font-size:2.6rem; font-weight:500;
          line-height:1.04; margin-top:.38rem; font-variant-numeric:tabular-nums; }}
.day .cat {{ font-size:.78rem; font-weight:600; margin-top:.1rem; }}
.day .rng {{ font-size:.74rem; color:{MUTED}; margin-top:.32rem; }}

/* conditions, washed with the live sky */
.condwrap {{ background:{sky['wash']}; border-radius:5px;
            padding:1.15rem 1.3rem 1.25rem; margin-top:.55rem;
            border:1px solid rgba(255,255,255,.65);
            box-shadow:0 1px 0 rgba(255,255,255,.5) inset; }}
.cond {{ display:grid; grid-template-columns:repeat(7,1fr); gap:1.2rem; }}
.cond .c {{ border-left:2px solid rgba(255,255,255,.9); padding-left:.75rem; }}
.cond .k {{ font-size:.72rem; color:{MUTED}; }}
.cond .v {{ font-family:'IBM Plex Serif',serif; font-size:1.5rem; font-weight:500;
           font-variant-numeric:tabular-nums; line-height:1.2; }}
.cond .u {{ font-size:.72rem; color:{MUTED}; margin-left:.12rem; }}

.alert {{ border-left:3px solid #B93A2E; background:#F9E4E4; padding:.95rem 1.15rem;
         margin:1.4rem 0 0; font-size:.92rem; line-height:1.55; border-radius:0 3px 3px 0; }}
.alert b {{ color:#B93A2E; }}
.note {{ border-left:3px solid {MUTED}; background:#F1F2F1; padding:.78rem 1rem;
        margin:.85rem 0; font-size:.85rem; color:{MUTED}; border-radius:0 3px 3px 0; }}

.foot {{ border-top:1px solid {RULE}; margin-top:2.8rem; padding-top:1.05rem;
        font-size:.78rem; color:{MUTED}; line-height:1.62; }}

/* assistant, styled so the chat widgets read as part of the bulletin */
.stChatMessage {{ background:transparent !important; padding:.15rem 0 !important;
                 border:none !important; }}
.stChatMessage [data-testid="stChatMessageContent"] p {{ font-size:.93rem;
                 line-height:1.6; margin-bottom:.4rem; }}
[data-testid="stChatInput"] {{ background:rgba(255,255,255,.75);
                 border:1px solid {RULE}; border-radius:4px; }}
[data-testid="stChatInput"] textarea {{ font-family:'IBM Plex Sans',sans-serif;
                 font-size:.93rem; }}
.stButton button {{ background:rgba(255,255,255,.7); border:1px solid {RULE};
                   color:{INK}; font-size:.8rem; font-weight:500;
                   border-radius:3px; padding:.36rem .7rem; cursor:none;
                   white-space:normal; text-align:left; line-height:1.35;
                   min-height:2.4rem; }}
.stButton button:hover {{ border-color:{INK}; background:#FFFFFF; color:{INK}; }}
.chatnote {{ font-size:.78rem; color:{MUTED}; line-height:1.55;
            border-left:2px solid {RULE}; padding-left:.7rem; margin:.2rem 0 1rem; }}
.tbl {{ margin-top:.55rem; overflow-x:auto; }}
.tbl table {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
.tbl thead th {{ font-weight:600; font-size:.74rem; color:{MUTED};
                text-align:left; padding:0 .85rem .5rem 0;
                border-bottom:1.5px solid {INK}; white-space:nowrap; }}
.tbl thead th:not(:first-child) {{ text-align:right; }}
.tbl tbody th {{ font-weight:600; text-align:left; padding:.55rem .85rem .55rem 0;
                border-bottom:1px solid {RULE}; white-space:nowrap; }}
.tbl tbody td {{ padding:.55rem .85rem .55rem 0; border-bottom:1px solid {RULE};
                font-variant-numeric:tabular-nums; }}
.tbl tbody td.num {{ text-align:right; }}
.tbl tbody tr:last-child th, .tbl tbody tr:last-child td {{ border-bottom:none; }}
.tbl tbody tr:hover th, .tbl tbody tr:hover td {{ background:rgba(255,255,255,.55); }}

/* tabs, flattened to sit with the rules rather than float above them */
.stTabs [data-baseweb="tab-list"] {{ gap:1.7rem; background:transparent;
                                    border-bottom:1px solid {RULE}; padding:0; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; padding:.45rem 0;
                               font-size:.86rem; font-weight:500; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:{INK}; font-weight:600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{INK}; height:2px; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:.9rem; }}

/* details block, styled to read as a section rather than a widget */
.stApp details {{ border-top:1px solid {RULE}; margin-top:1.4rem;
                 padding-top:.2rem; }}
.stApp details summary {{ cursor:none; list-style:none; padding:.85rem 0 .2rem;
                         font-family:'IBM Plex Serif',serif; font-size:1rem;
                         font-weight:600; color:{INK}; }}
.stApp details summary::-webkit-details-marker {{ display:none; }}
.stApp details summary::after {{ content:"  +"; color:{MUTED}; font-weight:400; }}
.stApp details[open] summary::after {{ content:"  −"; }}

@media (max-width: 900px) {{
  .block-container {{ padding: 1.4rem 1.2rem 3rem; }}
  .reading {{ grid-template-columns:1fr; gap:1.5rem; }}
  .pol, .days {{ grid-template-columns:1fr 1fr; }}
  .cond {{ grid-template-columns:repeat(3,1fr); gap:1rem; }}
  .mast .when {{ text-align:left; }}
}}
</style>
""", unsafe_allow_html=True)

# The atmospheric layer sits behind everything, fixed to the viewport.
st.markdown(ambient(sky["key"], sky["accent"]), unsafe_allow_html=True)

if fc is None:
    st.markdown("<div class='mast'><h1>Islamabad air quality</h1></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='note'>No forecast published yet. Run "
                "<code>python -m pipelines.hourly</code>, or wait for the hourly "
                "job to publish one.</div>", unsafe_allow_html=True)
    st.stop()

label, tint, ink, advice = band(fc["current_aqi"])
issued, through = local(fc["issued_at"]), local(fc["data_through"])

temp_html = ""
if wrow is not None and "temperature_2m" in wrow and pd.notna(wrow["temperature_2m"]):
    temp_html = f'<div class="temp">{wrow["temperature_2m"]:.0f}°C</div>'

st.markdown(f"""
<div class="mast rise">
  <div>
    <h1>Islamabad air quality</h1>
    <div class="place">Capital Territory, Pakistan &nbsp;·&nbsp; 33.68°N 73.05°E</div>
  </div>
  <div class="sky">
    <div class="glyph">{sky_glyph(sky['key'], sky['accent'])}</div>
    <div class="txt"><div class="word">{sky['word']}</div>{temp_html}</div>
  </div>
  <div class="when">Readings to {through:%a %d %B, %H:%M}<br>
       forecast issued {issued:%H:%M} PKT</div>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------- reading
spark = ""
if not aq.empty and "us_aqi" in aq:
    recent = aq["us_aqi"].dropna().tail(48)
    if len(recent) > 2:
        spark = (f'<div class="spark">{sparkline(recent, ink)}</div>'
                 f'<div class="sparklab">past 48 hours</div>')

pol_html = ""
arow = latest_row(aq)
if arow is not None:
    loads = [(k, n, u, float(arow[k]), min(float(arow[k]) / ref, 1.0))
             for k, n, u, ref in POLLUTANTS if k in arow and pd.notna(arow[k])]
    if loads:
        driver = max(loads, key=lambda x: x[4])[0]
        cells = []
        for key, name, unit, val, load in loads:
            lead = " lead" if key == driver else ""
            bar = ink if key == driver else "#B9C0B9"
            cells.append(
                f'<div class="row"><div class="top">'
                f'<span class="name{lead}">{name}</span>'
                f'<span class="val">{val:,.0f} {unit}</span></div>'
                f'<div class="track"><div class="fill" '
                f'style="width:{load * 100:.0f}%;background:{bar}"></div></div></div>')
        pol_html = f'<div class="pol">{"".join(cells)}</div>'

st.markdown(f"""
<div class="reading rise">
  <div class="figure" style="background:{tint}">
    {particulates(fc['current_aqi'], ink)}
    <div class="n" style="color:{ink}">{fc['current_aqi']:.0f}</div>
    <div class="cat" style="color:{ink}">{label}</div>
    {spark}
  </div>
  <div>
    <div class="advice">{advice}</div>
    {pol_html}
  </div>
</div>""", unsafe_allow_html=True)

if pol_html:
    st.markdown("<div class='subtle' style='margin-top:.65rem'>Bars show each "
                "pollutant against the level where it begins to affect health. "
                "The one in bold carries the highest relative load right now. "
                "The speckling behind the reading thickens as the air does.</div>",
                unsafe_allow_html=True)

alerts = [f for f in fc["forecast"] if f["alert"]]
if alerts:
    worst = max(alerts, key=lambda a: a["aqi_upper_q90"])
    when = local(worst["valid_at"])
    st.markdown(
        f"<div class='alert'><b>Unhealthy air likely on {when:%A}.</b> "
        f"The 90th-percentile forecast for {when:%a %H:%M} reaches "
        f"{worst['aqi_upper_q90']:.0f}, above the {ALERT_THRESHOLD} threshold. "
        f"Sensitive groups should plan indoor alternatives.</div>",
        unsafe_allow_html=True)

for w in fc.get("warnings", []):
    st.markdown(f"<div class='note'>{w}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------- outlook
st.markdown("<div class='sec'>Three-day outlook</div>", unsafe_allow_html=True)

hi_y = max([f["aqi_upper_q90"] for f in fc["forecast"]] + [fc["current_aqi"]]) + 35
fig = go.Figure()

prev = 0
for limit, lab, bt, bi, _ in BANDS:
    if prev >= hi_y:
        break
    top = min(limit, hi_y)
    fig.add_hrect(y0=prev, y1=top, fillcolor=bt, opacity=0.55,
                  line_width=0, layer="below")
    if limit < hi_y:
        fig.add_annotation(
            xref="paper", x=1, y=(prev + top) / 2, xanchor="left", xshift=8,
            text=lab.replace("Unhealthy for sensitive groups", "Sensitive"),
            showarrow=False, font=dict(size=10, color=bi))
    prev = limit

if not aq.empty and "us_aqi" in aq:
    obs = aq["us_aqi"].dropna()
    obs = obs[(obs.index >= pd.Timestamp(fc["data_through"]) - pd.Timedelta(days=3))
              & (obs.index <= pd.Timestamp(fc["data_through"]))]
    if len(obs):
        fig.add_trace(go.Scatter(
            x=obs.index.tz_convert(LOCAL_TZ), y=obs.values, mode="lines",
            line=dict(color=INK, width=2.2, shape="spline", smoothing=0.6),
            hovertemplate="%{y:.0f}<extra>observed</extra>"))

fx = [local(fc["data_through"])] + [local(f["valid_at"]) for f in fc["forecast"]]
fy = [fc["current_aqi"]] + [f["aqi"] for f in fc["forecast"]]
fu = [fc["current_aqi"]] + [f["aqi_upper_q90"] for f in fc["forecast"]]

fig.add_trace(go.Scatter(x=fx, y=fu, mode="lines", line=dict(width=0),
                         hovertemplate="%{y:.0f}<extra>worst case</extra>"))
fig.add_trace(go.Scatter(
    x=fx, y=fy, mode="lines+markers",
    line=dict(color="#B93A2E", width=2.6, dash="3px,3px",
              shape="spline", smoothing=0.6),
    marker=dict(size=8, color="#B93A2E", line=dict(width=2, color=PAPER)),
    fill="tonexty", fillcolor="rgba(185,58,46,0.10)",
    hovertemplate="%{y:.0f}<extra>forecast</extra>"))

fig.add_annotation(x=local(fc["data_through"]), y=hi_y, text="now",
                   showarrow=False, yanchor="top", xanchor="left", xshift=5,
                   font=dict(size=10, color=MUTED))

style_chart(fig, 360)
fig.update_yaxes(range=[0, hi_y])
fig.update_layout(margin=dict(l=8, r=90, t=8, b=8))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

cards = []
for f in fc["forecast"]:
    lab, bt, bi, _ = band(f["aqi"])
    when = local(f["valid_at"])
    cards.append(
        f'<div class="day" style="border-color:{bi}">'
        f'<div class="lab">{when:%A}</div>'
        f'<div class="sub">{when:%d %B, %H:%M} · +{f["horizon_h"]}h</div>'
        f'<div class="n" style="color:{bi}">{f["aqi"]:.0f}</div>'
        f'<div class="cat" style="color:{bi}">{lab}</div>'
        f'<div class="rng">up to {f["aqi_upper_q90"]:.0f} in the worst case</div>'
        f'</div>')
st.markdown(f'<div class="days">{"".join(cards)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- conditions
st.markdown("<div class='sec'>Conditions now</div>", unsafe_allow_html=True)

if wrow is None:
    st.markdown(f"<div class='note'>Weather unavailable"
                f"{f' ({type(wx_err).__name__})' if wx_err else ''}.</div>",
                unsafe_allow_html=True)
else:
    cells = []
    for key, name, unit, fmt in WEATHER_FIELDS:
        if key in wrow and pd.notna(wrow[key]):
            cells.append(f'<div class="c"><div class="k">{name}</div>'
                         f'<div class="v">{fmt.format(wrow[key])}'
                         f'<span class="u">{unit}</span></div></div>')
    if "wind_direction_10m" in wrow and pd.notna(wrow["wind_direction_10m"]):
        d = COMPASS[int((float(wrow["wind_direction_10m"]) % 360) / 22.5 + 0.5) % 16]
        cells.append(f'<div class="c"><div class="k">Wind from</div>'
                     f'<div class="v">{d}</div></div>')
    st.markdown(f'<div class="condwrap"><div class="cond">{"".join(cells)}</div></div>',
                unsafe_allow_html=True)
    st.markdown("<div class='subtle' style='margin-top:.9rem'>Wind and rain matter "
                "most cumulatively: two or three days of ventilation clears the air "
                "far more than one windy hour.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ accuracy
st.markdown("<div class='sec'>How accurate has this been?</div>",
            unsafe_allow_html=True)

hist = load_history()
if hist.empty:
    st.markdown("<div class='note'>No forecast history yet. Each hourly run saves "
                "a snapshot, and accuracy appears here as forecasts mature.</div>",
                unsafe_allow_html=True)
elif aq.empty or "us_aqi" not in aq:
    st.markdown("<div class='note'>Observations unavailable, so past forecasts "
                "cannot be scored.</div>", unsafe_allow_html=True)
else:
    obs_df = aq["us_aqi"].dropna().rename("observed").reset_index()
    obs_df.columns = ["valid_at", "observed"]
    scored = hist.merge(obs_df, on="valid_at", how="inner")
    scored["error"] = scored["predicted"] - scored["observed"]

    if scored.empty:
        st.markdown(
            f"<div class='note'>{len(hist)} forecasts saved since "
            f"{local(hist['issued_at'].min()):%d %B, %H:%M}, none matured yet. "
            f"A 24-hour forecast can first be scored a day after it is issued.</div>",
            unsafe_allow_html=True)
    else:
        left, right = st.columns([1, 1.7], gap="large")
        with left:
            s = (scored.groupby("horizon_h")
                 .agg(n=("error", "size"),
                      mae=("error", lambda e: e.abs().mean()),
                      bias=("error", "mean")).round(1))
            s.index = [f"+{h}h" for h in s.index]
            s.columns = ["Scored", "Average error", "Bias"]
            st.markdown(html_table(s, "Horizon"), unsafe_allow_html=True)
            st.markdown("<div class='subtle'>Bias shows whether the forecast runs "
                        "high or low. Negative means it under-predicted, which "
                        "matters most on the bad days.</div>",
                        unsafe_allow_html=True)
        with right:
            sc = go.Figure()
            obs_win = obs_df[obs_df["valid_at"] >= scored["valid_at"].min()]
            sc.add_trace(go.Scatter(
                x=obs_win["valid_at"].dt.tz_convert(LOCAL_TZ), y=obs_win["observed"],
                mode="lines",
                line=dict(color=INK, width=2, shape="spline", smoothing=0.6),
                hovertemplate="%{y:.0f}<extra>observed</extra>"))
            for (h, grp), c in zip(scored.groupby("horizon_h"),
                                   ["#D98C82", "#C4655A", "#B93A2E"]):
                grp = grp.sort_values("valid_at")
                sc.add_trace(go.Scatter(
                    x=grp["valid_at"].dt.tz_convert(LOCAL_TZ), y=grp["predicted"],
                    mode="markers", marker=dict(size=6, color=c, opacity=0.85),
                    hovertemplate=f"%{{y:.0f}}<extra>+{h}h forecast</extra>"))
            style_chart(sc, 300)
            st.plotly_chart(sc, use_container_width=True,
                            config={"displayModeBar": False})

    st.markdown(f"<div class='subtle'>{len(hist)} forecasts saved, "
                f"{local(hist['issued_at'].min()):%d %B} to "
                f"{local(hist['issued_at'].max()):%d %B}.</div>",
                unsafe_allow_html=True)

# --------------------------------------------------------- why this forecast
explained = [f for f in fc["forecast"] if (f.get("explanation") or {}).get("drivers")]

if explained:
    st.markdown("<div class='sec'>Why this forecast</div>", unsafe_allow_html=True)

    tabs = st.tabs([f"{local(f['valid_at']):%A}  ·  +{f['horizon_h']}h"
                    for f in explained])
    for tab, f in zip(tabs, explained):
        with tab:
            ex = f["explanation"]
            drivers = ex["drivers"][::-1]          # largest at the top of the chart
            base = ex["baseline"]

            bars = go.Figure(go.Bar(
                x=[d["effect"] for d in drivers],
                y=[d["label"] for d in drivers],
                orientation="h",
                marker=dict(color=["#B93A2E" if d["effect"] > 0 else "#2F7D52"
                                   for d in drivers]),
                hovertemplate="%{x:+.1f} AQI<extra>%{y}</extra>",
            ))
            style_chart(bars, max(210, 34 * len(drivers)))
            bars.update_xaxes(showgrid=True, gridcolor=FAINT, zeroline=True,
                              zerolinecolor=RULE, zerolinewidth=1,
                              title=dict(text="effect on the forecast (AQI points)",
                                         font=dict(size=11, color=MUTED)))
            bars.update_yaxes(showgrid=False, tickfont=dict(size=11.5, color=INK))
            bars.update_layout(margin=dict(l=8, r=20, t=8, b=32), hovermode="closest")
            st.plotly_chart(bars, use_container_width=True,
                            config={"displayModeBar": False})

            up = sum(d["effect"] for d in drivers if d["effect"] > 0)
            down = -sum(d["effect"] for d in drivers if d["effect"] < 0)
            st.markdown(
                f"<div class='subtle'>Starting from a typical {base:.0f}, these "
                f"factors push the {local(f['valid_at']):%A} forecast up by "
                f"{up:.0f} and down by {down:.0f}, landing at "
                f"<b>{f['aqi']:.0f}</b>. Red raises the forecast, green lowers "
                f"it.</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='subtle' style='margin-top:1rem'>Contributions are SHAP "
        "values computed for this specific prediction, not averages — they are "
        "recalculated every hour alongside the forecast itself.</div>",
        unsafe_allow_html=True)

# --------------------------------------------------------------- how it works
st.markdown("<div class='sec'>How the forecast is made</div>",
            unsafe_allow_html=True)

meta = load_model_meta()
if meta:
    tbl = pd.DataFrame([{
        "Horizon": f"+{m['horizon']}h",
        "Model": m["point_model"],
        "Typical error": round(m["holdout"]["MAE"], 1),
        "RMSE": round(m["holdout"]["RMSE"], 1),
        "Beats no-change by":
            f"{(1 - m['holdout']['RMSE'] / m['holdout_persistence']['RMSE']) * 100:.0f}%",
        "Catches bad-air days":
            f"{m['alert_quantile_head']['recall'] * 100:.0f}%",
    } for m in meta])
    tbl = tbl.set_index("Horizon")
    st.markdown(html_table(tbl, "Horizon"), unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Scored on a held-out year the models never "
                "trained on. The comparison is against persistence — assuming "
                "tomorrow matches today — which is a hard bar at 24 hours and a "
                "weak one at 72.</div>", unsafe_allow_html=True)

st.markdown("<div class='sec'>What the model looks at overall</div>",
            unsafe_allow_html=True)
if True:
    st.markdown(
        "<div class='advice' style='font-size:.96rem'>"
        "Two models run for each horizon. One predicts the most likely AQI. A "
        "separate model predicts a 90th-percentile upper bound, used only for the "
        "warning — a single model optimised for both jobs hedges toward the "
        "average and misses the days that matter most.\n\n"
        "Inputs divide into what is known now — recent pollutant readings, their "
        "rolling averages, and how fast AQI is changing — and what is known about "
        "the future: the weather forecast for the target hour, plus season and "
        "time of day. At one day out the recent readings dominate. At three days "
        "they have largely decayed, and season and weather carry the prediction."
        "</div>", unsafe_allow_html=True)
    imp_path = ROOT / "reports" / "shap_importance_h72.csv"
    if imp_path.exists():
        try:
            imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0].head(14)[::-1]
            gf = go.Figure(go.Bar(
                x=imp.values, y=[humanise(i) for i in imp.index],
                orientation="h", marker=dict(color="#8A9299"),
                hovertemplate="%{x:.2f}<extra>%{y}</extra>"))
            style_chart(gf, 380)
            gf.update_xaxes(showgrid=True, gridcolor=FAINT,
                            title=dict(text="average influence across all hours",
                                       font=dict(size=11, color=MUTED)))
            gf.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK))
            gf.update_layout(margin=dict(l=8, r=20, t=8, b=32), hovermode="closest")
            st.plotly_chart(gf, use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown("<div class='subtle'>Averaged over the whole test period "
                        "at the three-day horizon.</div>", unsafe_allow_html=True)
        except Exception:                                     # noqa: BLE001
            pass

# ------------------------------------------------------------------ assistant
st.markdown("<div class='sec'>Ask about going outside</div>",
            unsafe_allow_html=True)

try:
    from src.app import assistant as bot
except Exception:                                             # noqa: BLE001
    import importlib.util as _il
    _spec = _il.spec_from_file_location(
        "assistant", Path(__file__).parent / "assistant.py")
    bot = _il.module_from_spec(_spec)
    _spec.loader.exec_module(bot)

_key = bot.api_key()

if not _key:
    st.markdown(
        "<div class='note'>The assistant needs an OpenAI key. Add "
        "<code>OPENAI_API_KEY</code> to <code>.streamlit/secrets.toml</code> "
        "locally, or to the app's Secrets panel on Streamlit Cloud.</div>",
        unsafe_allow_html=True)
else:
    st.markdown(
        "<div class='chatnote'>Ask whether it's a good time to be outside. "
        "The assistant reads the live readings and forecast above — it isn't "
        "guessing from general knowledge. It gives everyday guidance, not "
        "medical advice; see a doctor for symptoms that persist or worsen, and "
        "seek emergency care for severe breathing difficulty or chest pain."
        "</div>", unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    if not st.session_state.chat:
        cols = st.columns(len(bot.SUGGESTIONS))
        for col, s in zip(cols, bot.SUGGESTIONS):
            with col:
                if st.button(s, key=f"sug_{hash(s)}", use_container_width=True):
                    st.session_state.pending = s

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed = st.chat_input("Can I go for a walk right now?")
    question = typed or st.session_state.pop("pending", None)

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Context is rebuilt on every turn from the objects this page already
        # rendered, so the assistant cannot drift from what the user is seeing.
        hist_series = (aq["us_aqi"] if not aq.empty and "us_aqi" in aq
                       else None)
        context = bot.build_context(fc, arow, wrow, sky, hist_series)

        with st.chat_message("assistant"):
            try:
                with st.spinner(""):
                    reply = bot.ask(question, st.session_state.chat[:-1],
                                    context, _key)
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant",
                                              "content": reply})
            except Exception as exc:                          # noqa: BLE001
                st.markdown(f"<div class='note'>Couldn't reach the assistant "
                            f"({type(exc).__name__}). The readings and forecast "
                            f"above are unaffected.</div>",
                            unsafe_allow_html=True)

    if st.session_state.chat:
        if st.button("Clear conversation", key="clearchat"):
            st.session_state.chat = []
            st.rerun()

errs = [e for e in (aq_err, wx_err) if e is not None]
st.markdown(
    "<div class='foot'>Air quality from the Copernicus CAMS global model via "
    "Open-Meteo, at roughly 40 km resolution — regional estimates rather than "
    "street-level readings, so peaks near traffic run higher than shown. "
    "Forecasts update hourly; models retrain daily."
    + (f"<br>Some live data could not be fetched this load "
       f"({', '.join(type(e).__name__ for e in errs)})." if errs else "")
    + "</div>", unsafe_allow_html=True)