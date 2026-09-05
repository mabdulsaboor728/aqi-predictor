"""
Air-quality health assistant.

A narrow-scope chat helper that answers one kind of question: given the air
quality and weather *right now in Islamabad*, and what the person tells you
about themselves, is it a good idea to go outside?

Live grounding
--------------
Every request carries a freshly built context block containing the current AQI
and category, the measured pollutant concentrations, current weather, and this
system's own three-day forecast including the alert-head upper bound. The model
is told to treat that block as the only source of truth for current conditions
and never to substitute remembered or general knowledge about Islamabad's air.
This is why the assistant can answer "can I run this evening" correctly rather
than reciting averages.

Safety posture
--------------
This gives general public-health guidance of the kind an air quality bulletin
carries. It is not a clinician. Three rules are enforced in the prompt:

  * severe symptoms are routed straight to emergency care, with no air-quality
    discussion attached
  * it never names a condition the person has not named, and never advises on
    medication or dosing
  * persistent or worsening symptoms get a recommendation to see a doctor

Setup
-----
Add the key to Streamlit secrets (.streamlit/secrets.toml locally, or the app
Settings -> Secrets panel on Streamlit Cloud):

    OPENAI_API_KEY = "sk-..."

Never commit it. .streamlit/secrets.toml belongs in .gitignore.
"""

from __future__ import annotations

import os

MODEL = "gpt-4o-mini"
MAX_TOKENS = 500
TEMPERATURE = 0.3
HISTORY_TURNS = 8          # how much conversation to send back

AQI_BANDS = [
    (50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"),
    (200, "Unhealthy"), (300, "Very unhealthy"), (10**6, "Hazardous"),
]

# Concentration at which each pollutant starts to matter, for context only.
POLLUTANT_REFS = {
    "pm2_5": ("PM2.5", "µg/m³", 35.4),
    "pm10": ("PM10", "µg/m³", 154.0),
    "ozone": ("ozone", "µg/m³", 100.0),
    "nitrogen_dioxide": ("nitrogen dioxide", "µg/m³", 100.0),
    "sulphur_dioxide": ("sulphur dioxide", "µg/m³", 75.0),
    "carbon_monoxide": ("carbon monoxide", "µg/m³", 4000.0),
}

WEATHER_LABELS = {
    "temperature_2m": ("temperature", "°C"),
    "relative_humidity_2m": ("relative humidity", "%"),
    "wind_speed_10m": ("wind speed", "km/h"),
    "precipitation": ("precipitation", "mm in the last hour"),
    "surface_pressure": ("surface pressure", "hPa"),
    "cloud_cover": ("cloud cover", "%"),
}


def api_key() -> str | None:
    """Streamlit secrets first, environment second, so the same code runs both
    locally and on Streamlit Cloud."""
    try:
        import streamlit as st
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:                                         # noqa: BLE001
        pass
    return os.environ.get("OPENAI_API_KEY")


def _category(aqi: float) -> str:
    for limit, label in AQI_BANDS:
        if aqi <= limit:
            return label
    return AQI_BANDS[-1][1]


def build_context(fc: dict, arow=None, wrow=None, sky: dict | None = None,
                  aqi_history=None) -> str:
    """Assemble the live facts block from this system's own data.

    Everything here comes from the same objects the page renders, so the
    assistant and the dashboard can never disagree about current conditions.

    The current wall-clock time is stated explicitly and separately from the
    observation timestamp. Without it the model has no idea what "now" is and
    will happily recommend a time that has already passed.
    """
    import pandas as pd

    now = pd.Timestamp.now(tz="Asia/Karachi")
    through = pd.Timestamp(fc["data_through"]).tz_convert("Asia/Karachi")

    lines = ["CURRENT CONDITIONS IN ISLAMABAD, PAKISTAN",
             "(measured and forecast by this system - authoritative)", ""]

    lines.append(f"RIGHT NOW it is {now:%A %d %B %Y, %H:%M} Pakistan time. "
                 f"Any time earlier than this has already passed.")
    lines.append(f"Latest observations are from {through:%A %H:%M} "
                 f"({fc.get('data_age_hours', 0):.0f}h ago).")
    lines.append(f"Current US AQI: {fc['current_aqi']:.0f} "
                 f"({_category(fc['current_aqi'])})")

    if sky:
        lines.append(f"Sky and weather right now: {sky['word']}")

    if wrow is not None:
        bits = []
        for key, (name, unit) in WEATHER_LABELS.items():
            if key in wrow and pd.notna(wrow[key]):
                bits.append(f"{name} {float(wrow[key]):.0f} {unit}")
        if bits:
            lines.append("Weather: " + ", ".join(bits))

    if arow is not None:
        lines.append("")
        lines.append("Measured pollutant concentrations:")
        loads = []
        for key, (name, unit, ref) in POLLUTANT_REFS.items():
            if key in arow and pd.notna(arow[key]):
                val = float(arow[key])
                pct = val / ref * 100
                loads.append((name, val, unit, pct))
                lines.append(f"  - {name}: {val:,.1f} {unit} "
                             f"({pct:.0f}% of the level where it starts to "
                             f"affect health)")
        if loads:
            worst = max(loads, key=lambda x: x[3])
            lines.append(f"  The pollutant carrying the highest relative load "
                         f"right now is {worst[0]}.")

    # --- forecast, with past points explicitly marked so they cannot be
    #     recommended by mistake
    lines.append("")
    lines.append("Forecast from this system's models. These are the ONLY future "
                 "times available - there are no other forecast points:")
    future = 0
    for f in fc.get("forecast", []):
        valid = pd.Timestamp(f["valid_at"]).tz_convert("Asia/Karachi")
        if valid <= now:
            lines.append(f"  - {valid:%A %H:%M}: ALREADY PASSED, ignore")
            continue
        future += 1
        hours_away = (valid - now).total_seconds() / 3600
        flag = "  ** ALERT: unhealthy air likely **" if f.get("alert") else ""
        lines.append(
            f"  - {valid:%A %d %b, %H:%M} (in about {hours_away:.0f} hours): "
            f"AQI {f['aqi']:.0f} ({_category(f['aqi'])}), could reach "
            f"{f['aqi_upper_q90']:.0f} in the worst case{flag}")
    if future == 0:
        lines.append("  (no future forecast points available - the forecast is "
                     "stale; say so rather than guessing)")

    # --- typical daily rhythm, so "when is it usually cleanest" can be answered
    #     from data instead of invented
    if aqi_history is not None and len(aqi_history) > 48:
        try:
            h = aqi_history.dropna().tail(24 * 7)
            local_hours = h.index.tz_convert("Asia/Karachi").hour
            by_hour = h.groupby(local_hours).mean()
            best = by_hour.idxmin()
            worst = by_hour.idxmax()
            lines.append("")
            lines.append(
                f"Typical daily pattern over the past week: AQI is usually "
                f"lowest around {best:02d}:00 (about {by_hour.min():.0f}) and "
                f"highest around {worst:02d}:00 (about {by_hour.max():.0f}) "
                f"Pakistan time. Use this for questions about the best time of "
                f"day, but say it is a typical pattern rather than a forecast.")
        except Exception:                                     # noqa: BLE001
            pass

    lines.append("")
    lines.append("US AQI bands: 0-50 Good, 51-100 Moderate, 101-150 Unhealthy "
                 "for sensitive groups, 151-200 Unhealthy, 201-300 Very "
                 "unhealthy, 301+ Hazardous.")
    lines.append("Note: these readings come from the Copernicus CAMS model at "
                 "roughly 40 km resolution, so they describe the region rather "
                 "than a specific street. Air beside heavy traffic will be "
                 "worse than the number shown.")
    return "\n".join(lines)


SYSTEM = """You are the air quality assistant for a public forecasting service \
covering Islamabad, Pakistan. You help people decide whether outdoor activity \
is sensible given the air quality right now and over the next three days.

GROUNDING
The block below contains live measurements and forecasts from this service. \
Treat it as the only authoritative source for current conditions. Never \
substitute general or remembered facts about Islamabad's air quality, and \
never invent a number. If someone asks about something the block does not \
cover, say you do not have that reading rather than guessing.

{context}

TIME
The block states the current time. Never recommend a time that has already \
passed today, and never invent a forecast time. You have exactly three \
forecast points and nothing between them - if someone asks when the air will \
improve and none of those three is better, say so rather than inventing an \
hour. For questions about the best time of day, use the typical daily pattern \
if it is given, and make clear it is a pattern rather than a prediction.

SCOPE
Only discuss air quality, pollution, weather, and how they affect health and \
outdoor activity. If asked about anything else, say briefly that you only \
cover air quality here, and offer to help with that instead. Do not answer \
general knowledge, coding, or personal questions.

SAFETY - these override everything else
1. If someone describes severe symptoms - struggling to breathe at rest, chest \
pain or tightness, blue or grey lips or face, confusion, fainting, unable to \
speak in full sentences, or an asthma attack not responding to their inhaler - \
tell them to seek emergency medical care immediately. Give no air quality \
advice in that reply; it is not relevant and delays them.
2. Never diagnose. Never name a medical condition the person has not told you \
they have. Never suggest, adjust or comment on medication or dosing.
3. If symptoms are persistent, worsening, or new to them, recommend they see a \
doctor. Say plainly that you are not a medical professional.

HOW TO ANSWER
Answer the actual question directly - usually "yes, go", "yes but shorten it", \
or "better to wait". Give the reason in one or two sentences, citing the \
specific numbers from the block.

Take their situation seriously. Someone who mentions asthma, a heart \
condition, pregnancy, being a young child or older adult, or current symptoms \
should get more cautious advice than a healthy adult at the same AQI.

Use the forecast when timing matters. If air is bad now but the model expects \
it to clear, say when.

Be brief and conversational - two to four sentences for most questions. No \
headings, no bullet lists unless they genuinely help. Plain language, not \
clinical language. Do not repeat the full readings back unless asked."""


def ask(question: str, history: list[dict], context: str,
        key: str, model: str = MODEL) -> str:
    """Send one turn. Raises on API failure so the caller can show the error."""
    from openai import OpenAI

    client = OpenAI(api_key=key)
    messages = [{"role": "system", "content": SYSTEM.format(context=context)}]
    messages += history[-HISTORY_TURNS * 2:]
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    return resp.choices[0].message.content.strip()


SUGGESTIONS = [
    "Is it safe to go for a run this evening?",
    "I have asthma — should I avoid going out today?",
    "Is it okay to take my toddler to the park?",
    "When is the air expected to be cleanest in the next three days?",
]