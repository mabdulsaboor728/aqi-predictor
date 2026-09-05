# Three-Day Air Quality Forecasting for Islamabad

**An end-to-end, automated AQI prediction system**

Muhammad Abdul Saboor · 10Pearls Shine Internship · September 2026

Repository: `github.com/mabdulsaboor728/aqi-predictor`
Dashboard: `aqi-predictor-by-mas728.streamlit.app`

---

## 1. Executive summary

This project forecasts the US Air Quality Index for Islamabad at 24, 48 and 72
hours ahead. It runs continuously and unattended: an hourly job ingests data,
rebuilds features and publishes a forecast; a daily job retrains and registers
models; a public dashboard displays the result.

**Headline results**, measured on a rolling twelve-month holdout the models
never trained on:

| Horizon | Model | RMSE | MAE | R² | Persistence RMSE | Improvement |
|---|---|---|---|---|---|---|
| +24h | HistGradientBoosting | 13.56 | 10.20 | 0.807 | 16.32 | 16.9% |
| +48h | XGBoost | 17.49 | 13.55 | 0.679 | 21.66 | 19.2% |
| +72h | XGBoost | 19.01 | 14.86 | 0.620 | 24.51 | 22.4% |

The improvement over the persistence baseline *grows* with horizon, which is
the desired shape: at 24 hours persistence is already strong, while at 72 hours
it has no skill left (R² −0.148) and the model's seasonal and weather-forecast
inputs carry the prediction alone.

A second model per horizon handles hazardous-air warnings. The RMSE-optimal
forecast detects only 48% of Unhealthy hours at 72 hours ahead; a
90th-percentile quantile model detects 92%. These are different objectives and
one model cannot serve both.

**Three findings shaped the work more than any modelling choice:**

1. The US AQI target is constructed from *rolling averages* of its component
   pollutants. Matching those averaging windows in the features raised the
   correlation between PM2.5 and winter AQI from 0.573 to **0.966**.
2. Wind clears pollution cumulatively, not instantly. Same-hour wind speed
   correlates −0.109 with AQI; a 72-hour rolling mean correlates **−0.356**.
3. The original cross-validation design silently froze the training set, so
   "daily retraining" refit identical data every day. Fixing it also reversed
   the model ranking.

---

## 2. Problem definition and framing

### 2.1 The task

Predict `us_aqi` for Islamabad at three future horizons, updating hourly, with
an automated pipeline and a dashboard.

### 2.2 Choice of city

Lahore is the city Pakistan cares most about, but its air quality is dominated
by crop-residue burning and winter inversions — regime switches that
meteorological features cannot observe. Islamabad's AQI moves with variables
the model can actually see: wind, humidity, boundary layer conditions, and
season.

The city is a configuration parameter (`src/config.py`), so the same pipeline
runs anywhere Open-Meteo has coverage by changing two coordinates. Islamabad
was chosen as the primary target because it yields a model whose behaviour can
be explained rather than one whose errors are unattributable.

### 2.3 Direct versus recursive multi-horizon

Two ways to forecast 72 hours ahead:

- **Recursive**: predict t+1, feed the prediction back as input, repeat 72
  times. Errors compound across steps.
- **Direct**: train a separate model for each horizon, each mapping directly
  from information at t to the target at t+h.

Direct was chosen, and the results vindicate it. Different model families win
at different horizons (gradient boosting variants at all three, but Ridge is
competitive at 24h and badly beaten at 72h with five times the fold-to-fold
variance), and the balance of feature importance shifts substantially with
horizon. The problem genuinely changes shape as the horizon extends; one model
cannot be optimal for all three.

The cost is six model artifacts instead of one. A single model with horizon as
an input feature is a legitimate alternative that was not tested — noted
honestly rather than dismissed.

### 2.4 The availability contract

This is the central design constraint of the whole system.

At origin time `t`, two kinds of information exist, with different
availability:

| Kind | Available at | Source at serving time |
|---|---|---|
| Pollutant and AQI history | up to `t` only | feature store |
| Weather | through `t+h` | weather forecast API |

Weather at `t+h` is legitimate because a national meteorological service
forecasts it independently — the model is consuming an available input, not
cheating. Pollutant data at `t+h` does not exist and using it would be leakage.

Every feature name encodes which side it belongs to: `_t` for origin-side,
`_f_h{horizon}` for forecast-side. The suffix makes availability auditable with
a single grep, which is a stronger guarantee than a claim of care.

---

## 3. Data

### 3.1 Sources

| Source | Endpoint | Variables |
|---|---|---|
| Air quality | Open-Meteo, CAMS global | 9 (8 pollutants + `us_aqi`) |
| Weather | Open-Meteo Historical Forecast | 12 |

Coverage: 5 August 2022 to present, hourly, at 33.6844°N 73.0479°E.

**Why the Historical Forecast archive rather than ERA5 reanalysis.** ERA5 is
more accurate about what the weather actually was. But at prediction time only
a *forecast* is available. Training on reanalysis and serving on forecasts
means the model sees a different input distribution in production than in
validation. The Historical Forecast archive stores what the forecast said at
the time, in the same variables and units as the live forecast API, so training
and serving stay consistent. (Section 11.1 explains why this is necessary but
not sufficient.)

**`domains=cams_global` is pinned explicitly.** Open-Meteo defaults to `auto`,
which can route to the European CAMS domain — which has no coverage over
Pakistan and would return silent nulls.

### 3.2 Cleaning

| Step | Detail | Effect |
|---|---|---|
| Drop `boundary_layer_height` | 51% missing (archive covers Sep 2024+); correlation with target 0.003 | Keeping it would cost half the training data for no signal |
| Trim leading rows | one contiguous 96-hour `us_aqi` gap at series start | 96 rows removed |
| Enforce hourly grid | reindex to a complete hourly UTC index | Guarantees lag-24 means 24 hours |
| Clip physical impossibilities | negatives on strictly-positive quantities | 51 values clipped |
| Fill short gaps | forward-fill, capped at 3 hours | 0 values (no interior gaps present) |

Result: 35,664 rows × 20 variables, zero target nulls.

**On the gap fill.** The first implementation used
`interpolate(method="time", limit_direction="forward")`, believing the argument
made the fill causal. It does not. `limit_direction` controls which NaN
positions get *written*, not which endpoints participate in the arithmetic:
`[0, NaN, 2]` becomes `[0, 1, 2]`, with the middle value computed from the
observation that comes after it. A gap at 3pm would be filled using the 5pm
reading, injecting a future observation into every lag and rolling feature at
3pm — with no downstream check able to catch it, because the value looks
entirely plausible.

Replaced with `ffill(limit=3)`, which reads nothing after `t`. The result is a
step rather than a smooth ramp, which is less accurate for a continuous series.
That is the correct trade: an accurate value built from the future is worse
than a slightly stale value built only from the past, because only the second
is available at serving time. The bug never fired on this dataset — there are
no interior gaps — but it would have on the first CAMS outage.

### 3.3 Enforcing the hourly grid

Reindexing onto a complete hourly index looks like a formality. It is not.
A lag of 24 *rows* equals 24 *hours* only if the index has no holes. A silently
ragged index is one of the most common causes of a model that validates well
and fails in production, because the lag features mean something different in
each regime.

---

## 4. Exploratory analysis

Each analysis answered a specific question rather than producing a generic plot
gallery. All figures are in `reports/figures/`.

### 4.1 Target distribution

| Statistic | Value |
|---|---|
| Mean | 112.3 |
| Std | 31.2 |
| Median | 106 |
| Min / Max | 14 / 218 |

| Category | Share |
|---|---|
| Good | 0.1% |
| Moderate | 42.9% |
| Unhealthy for sensitive groups | 40.0% |
| Unhealthy | 16.8% |
| Very unhealthy | 0.2% |

Islamabad spends the large majority of hours in the Moderate to
Unhealthy-for-Sensitive-Groups range. Genuinely hazardous readings are rare in
this dataset, which has consequences for how confidently the model can be
evaluated at the extreme (Section 6.6).

### 4.2 Seasonality

Monthly mean AQI:

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 143.6 | 108.0 | 91.3 | 83.9 | 101.7 | 114.9 | 122.6 | 115.6 | 116.3 | 100.9 | 117.4 | 129.7 |

A winter smog peak in January, a spring minimum in April, and a secondary
summer rise driven by dust and ozone. Month is therefore a genuine predictor,
not decoration.

### 4.3 The diurnal cycle changes completely by season

Amplitude (highest hour minus lowest hour) of mean AQI:

| Season | Amplitude |
|---|---|
| Winter (DJF) | **0.5** |
| Spring (MAM) | 36.8 |
| Summer (JJA) | 46.4 |
| Autumn (SON) | 19.8 |

Winter AQI is essentially flat across the day. This is not a data error, and
explaining it reframed the entire feature design.

**US AQI is the maximum of per-pollutant sub-indices, and each sub-index uses a
different averaging window** — PM2.5 over 24 hours, ozone over 8 hours. In
winter, PM2.5 is the binding constraint, and a 24-hour rolling mean barely
moves hour to hour, so the AQI is flat. In summer, ozone binds instead, and its
8-hour window produces a sharp peak at 18:00 local.

Correlation of `us_aqi` with candidate features, by season:

| Feature | DJF | MAM | JJA | SON |
|---|---|---|---|---|
| `pm2_5` (raw hourly) | 0.573 | 0.379 | 0.437 | 0.580 |
| `pm2_5` rolling 24h | **0.966** | 0.595 | 0.679 | 0.881 |
| `pm10` rolling 24h | 0.828 | 0.523 | 0.602 | 0.799 |
| `ozone` (raw hourly) | −0.066 | 0.200 | 0.281 | 0.092 |
| `ozone` rolling 8h | −0.100 | 0.501 | **0.588** | 0.260 |

**Simply matching the averaging window to the sub-index definition raised
winter correlation from 0.573 to 0.966.** No amount of hyperparameter tuning
would have recovered that. This single insight did more for model quality than
every algorithm choice combined.

It also means hour-of-day means something completely different in January than
in June, which a linear model cannot express — an early indication that tree
models would win at longer horizons.

### 4.4 Weather acts cumulatively, not instantly

Correlation of `us_aqi` at time t with weather at t − lag:

| Lag | Wind speed | Gusts | Humidity | Temperature |
|---|---|---|---|---|
| 0h | −0.109 | −0.116 | 0.101 | −0.051 |
| 6h | −0.138 | −0.062 | 0.054 | −0.007 |
| 12h | −0.224 | −0.261 | 0.205 | −0.161 |
| 24h | −0.153 | −0.159 | 0.065 | −0.045 |

Rolling windows:

| Window | Wind mean | Precipitation sum |
|---|---|---|
| 6h | −0.131 | −0.057 |
| 24h | −0.304 | −0.108 |
| 48h | −0.349 | −0.139 |
| 72h | **−0.356** | −0.143 |

Same-hour wind looks nearly useless. Accumulated ventilation over two to three
days is more than three times stronger. Ventilation is a cumulative process,
and the window matters more than the variable.

### 4.5 Baselines

Two naive baselines, evaluated on a held-out period (test std 30.45):

| Baseline | MAE | RMSE |
|---|---|---|
| Persistence +24h | 12.53 | 17.13 |
| Persistence +48h | 16.43 | 22.16 |
| Persistence +72h | 18.48 | 24.60 |
| Climatology (day-of-year × hour) | 21.07 | 27.39 |

Two things follow. First, persistence at 72 hours already captures most of the
variance, so it is a genuine bar rather than a formality — a model scoring RMSE
26 at 72 hours would be an expensive copy of `y[t]`. Second, **climatology
loses to persistence at every horizon**, which says recent state matters more
than seasonal average, and that lag features would carry the model with
calendar features as a modifier.

---

## 5. Feature engineering

### 5.1 Inventory

91 features from 20 cleaned variables, split by availability.

**Origin side — 62 features, suffix `_t`.** Computed from history at or before
`t`.

| Group | Count | Contents |
|---|---|---|
| AQI lags | 9 | 1, 2, 3, 6, 12, 24, 48, 72, 168 hours |
| AQI rolling statistics | 11 | mean over 3/6/24/72/168h; std, max, min over 24/72h |
| AQI deltas and rates | 8 | change and change-per-hour over 3/6/24/72h |
| AQI level and anchors | 3 | current value, same hour yesterday, anomaly vs 72h mean |
| Pollutant sub-index rolls | 5 | PM2.5 24h, PM10 24h, ozone 8h, SO₂ 24h, CO 8h |
| Pollutant lags | 16 | 8 pollutants × lags 1h and 24h |
| Pollutant 24h deltas | 8 | one per pollutant |
| Level and ratio | 2 | NO₂ current, PM2.5/PM10 ratio |

**Forecast side — 29 features, suffix `_f_h{horizon}`.** Evaluated at the
target hour.

| Group | Count | Contents |
|---|---|---|
| Point weather | 9 | temperature, humidity, dew point, precipitation, pressure, cloud, wind speed, gusts, radiation |
| Wind direction | 2 | sin/cos encoding |
| Rolling weather | 9 | wind mean, precipitation sum, temperature mean over 24/48/72h |
| Inversion proxy | 1 | dew-point spread |
| Calendar | 8 | hour sin/cos, day-of-year sin/cos, hour, month, weekday, weekend flag |

### 5.2 Features derived from the EDA

Four groups exist because the analysis found something, not because a template
suggested them:

**Sub-index-matched rolling windows.** PM2.5 and PM10 over 24 hours, ozone and
CO over 8 hours, matching the US AQI definition (Section 4.3).

**Cumulative weather.** Rolling wind means and precipitation sums over 24, 48
and 72 hours, because ventilation accumulates (Section 4.4).

**Cyclical encodings.** Hour and day-of-year as sin/cos pairs, so 23:00 sits
next to 00:00 and 31 December next to 1 January. Wind direction likewise, since
359° and 1° are adjacent.

**Anomaly and momentum.** `anomaly_vs_72h` and the delta/rate family tell the
model whether current AQI is high *relative to its own recent level*, which is
more informative than the absolute value during a seasonal transition.

### 5.3 Rolling weather windows that span the origin

A subtlety worth stating precisely. `wind_rmean72_f` evaluated at `t+24` covers
`t−48` through `t+24`: partly past actuals, partly forecast. Both are available
when the forecast is issued, so the feature is legitimate — but the serving
code must concatenate recent actual weather with the forecast into one
continuous hourly series *before* computing any rolling statistic. Computing it
on the forecast alone produces silently wrong values that nothing downstream
detects.

### 5.4 Target maturity

The Open-Meteo air-quality endpoint is a CAMS *forecast* product: for the
current day it returns provisional values for hours that have not happened yet.
Without a cap, the pipeline would emit training rows whose "target" is another
model's forecast rather than an observation — teaching the model to imitate
CAMS instead of to predict air quality.

`build_supervised()` therefore takes a `max_target_time` argument. The hourly
pipeline passes the current hour; a static historical backfill leaves it unset.
In routine operation this drops five or six rows per run.

---

## 6. Modelling

### 6.1 Evaluation protocol

Three mechanisms guard against optimistic evaluation.

**Purged walk-forward cross-validation.** A training row at time T carries a
target at T + horizon. If validation begins at T + 1, that target lies inside
the validation window and the model has effectively seen the answer. Standard
`TimeSeriesSplit` does not handle this. A gap of `horizon + 24` hours separates
every training fold from its validation fold, and the same gap separates the
development set from the holdout.

**Minimum training size.** Folds begin only after two full years of data. This
was added after discovering it changed the answer (Section 6.3).

**Rolling holdout.** The final 365 days of available data, scored exactly once
per training run.

### 6.2 The frozen-training-set bug

The first implementation used a fixed holdout start date and no end date:

```python
HOLDOUT_START = "2025-09-01"
dev = X.index < HOLDOUT_START
```

This silently froze the training set. New data only ever enlarged the holdout,
so every daily retrain refit **identical rows** and could not adapt to drift.
The system's automated retraining was performing no learning.

| Simulated data end | Frozen dev rows | Rolling dev rows |
|---|---|---|
| Aug 2026 | 26,952 | 26,735 |
| Nov 2026 | 26,952 | 28,895 |
| Feb 2027 | 26,952 | 31,055 |
| Aug 2027 | 26,952 | 35,495 |

Replaced with a holdout of fixed *length* anchored to the end of the data. The
training set now grows; the evaluation window stays approximately comparable
across versions, differing by a day rather than by months. Each registered
model records its exact holdout window in metadata so any cross-version
comparison can be audited rather than assumed valid.

### 6.3 The cross-validation fix that reversed the ranking

The original walk-forward split began its first fold after roughly six months
of data. A model that has never seen a full seasonal cycle cannot use
`doy_sin`, `month`, or any seasonal feature — so early folds were unfairly bad,
and they penalised the models that rely most on seasonal structure.

Before the fix, Ridge appeared to win at 24 hours. After imposing a two-year
minimum training size:

| h=24 | CV RMSE | vs persistence |
|---|---|---|
| Persistence | 21.11 | — |
| Ridge | 17.63 (±1.38) | +16.5% |
| HistGBM | **17.06 (±1.17)** | **+19.2%** |

| h=72 | CV RMSE | vs persistence |
|---|---|---|
| Persistence | 31.04 | — |
| Ridge | 27.86 (±4.37) | +10.3% |
| HistGBM | **22.80 (±1.30)** | **+26.6%** |

Gradient boosting now wins at every horizon, with a far tighter fold spread
(±1.30 versus Ridge's ±4.37 at 72 hours). Ridge's apparent win was a protocol
artifact, not a finding. This is the clearest illustration in the project of
why the evaluation design deserves as much scrutiny as the model.

### 6.4 Model comparison

Five families were compared under an identical protocol. Cross-validated RMSE
on the development set:

| Model | h=24 | h=48 | h=72 |
|---|---|---|---|
| Persistence | 20.90 | 27.83 | 31.02 |
| Ridge | 17.60 (±1.48) | 24.23 (±1.85) | 28.10 (±5.18) |
| Random Forest | 17.41 (±1.76) | 21.95 (±1.40) | 23.66 (±1.88) |
| HistGradientBoosting | **16.87 (±1.40)** | 21.38 (±1.26) | 22.54 (±1.72) |
| XGBoost | 16.94 (±1.32) | **21.03 (±1.23)** | **22.47 (±1.68)** |

Held-out year, selected model per horizon:

| Horizon | Model | RMSE | MAE | R² | Persistence RMSE | Gain |
|---|---|---|---|---|---|---|
| +24h | HistGBM | 13.56 | 10.20 | 0.807 | 16.32 | 16.9% |
| +48h | XGBoost | 17.49 | 13.55 | 0.679 | 21.66 | 19.2% |
| +72h | XGBoost | 19.01 | 14.86 | 0.620 | 24.51 | 22.4% |

**A caution on model selection.** At 24 and 72 hours, HistGBM and XGBoost are
separated by less than a fifth of the fold-to-fold noise. The defensible claim
is that gradient boosting clearly beats linear and bagged models; the choice
between the two boosting implementations is within noise. Ridge's collapse at
72 hours (±5.18 spread, R² −0.013) is a real result: a linear model cannot
handle the seasonal regime shifts once persistence stops carrying it.

### 6.5 Deep learning comparison

The brief asks for models spanning statistical to deep learning. GRU and LSTM
sequence models were built and evaluated under the identical protocol.

Architecture: two encoders — one reading 168 hours of raw pollutant and weather
history ending at `t`, one reading the weather forecast sequence from `t+1` to
`t+h` — concatenated with calendar features and passed to a dense head. The
network predicts the residual from persistence rather than the level, since
neural networks have no cheap way to express `y ≈ x`. Huber loss, because dust
events are genuine outliers that would dominate an MSE gradient.

| Horizon | GRU | LSTM | Best tree model |
|---|---|---|---|
| +24h | 13.98 | 14.56 | **13.56** |
| +48h | 17.75 | 19.63 | **17.49** |
| +72h | 20.97 | 20.07 | **19.01** |

Both recurrent models sit between persistence and gradient boosting.
Importantly, their advantage over persistence stays flat with horizon (13.9% →
14.1% for GRU) while boosting's grows (16.9% → 22.4%) — the sequence models are
weakest at exactly the horizon where seasonal and forecast-weather structure
must carry the prediction, which the hand-built features encode explicitly and
the networks would have had to discover.

Training loss fell from 0.29 to 0.03 while validation stalled, with early
stopping firing at epochs 10–16. With ~27,000 parameters on ~23,000 training
samples this is expected; the limit is data volume, not regularisation.

**Where the sequence models are better**, and it is a real finding:

| Horizon | GRU bias on AQI>150 | Boosting bias on AQI>150 |
|---|---|---|
| +24h | −6.73 | −16.63 |
| +48h | −6.29 | −25.31 |
| +72h | −5.53 | −27.60 |

The recurrent models mean-revert roughly four times less. Two causes, both
about the loss rather than the architecture: Huber loss does not punish large
errors quadratically, so there is much less incentive to hedge toward the mean;
and the residual formulation anchors every prediction to persistence, which
preserves extremes structurally.

**Conclusion: gradient boosting is the production model.** Deep learning was
tested under an identical protocol and rejected on evidence. Its lower
mean-reversion bias is recorded as support for the same insight that motivated
the quantile alert head.

The GRU-versus-LSTM difference (0.6 to 1.9 RMSE, single seeds) is within
seed-to-seed noise and no claim is made about it.

### 6.6 Mean reversion, and the two-head design

Every RMSE-optimised model in this project under-predicts high-AQI hours, and
the effect worsens with horizon. CV bias on hours above AQI 150:

| Horizon | Bias |
|---|---|
| +24h | −16.63 |
| +48h | −25.31 |
| +72h | −27.60 |

These figures are *conditional* on the target exceeding AQI 150. Overall bias
across all holdout hours is near zero at every horizon — the model is not
systematically low, it is systematically low **in the tail**. That distinction
matters: an unconditional bias of −28 would be a calibration failure, whereas a
tail-conditional one is the expected cost of minimising squared error.

Detection performance at threshold AQI > 150, h=72:

| Model | Recall | Precision | RMSE |
|---|---|---|---|
| Point forecast | 0.27 | 0.82 | 22.59 |
| Quantile 0.75 | 0.50 | 0.71 | 24.63 |
| Quantile 0.90 | 0.62 | 0.58 | 30.09 |
| Quantile 0.95 | 0.72 | 0.46 | 35.93 |

The RMSE-optimal model is not broken — it is correctly optimised for the wrong
objective. **Two models per horizon** are therefore trained and deployed:

- **Point head** — RMSE-optimal, produces the number shown on the dashboard.
- **Alert head** — HistGradientBoosting with quantile loss at α = 0.90, used
  only for the threshold warning.

Final holdout alert performance:

| Horizon | Point head recall | q90 recall | q90 precision |
|---|---|---|---|
| +24h | 0.72 | 0.94 | 0.63 |
| +48h | 0.60 | 0.94 | 0.58 |
| +72h | 0.48 | 0.92 | 0.60 |

**The trade-off is explicit and deliberate.** Precision falls to roughly 0.60,
so about 40% of alerts are false alarms. For a public health warning, a missed
Unhealthy day costs far more than an unnecessary mask. Both numbers are
reported side by side rather than only the favourable one, and the quantile is
a single configuration constant if a different balance is preferred.

### 6.7 Approaches tested and rejected

Recorded because a measured rejection is more informative than an untested
assumption:

| Approach | Result | Decision |
|---|---|---|
| Predicting delta from persistence (trees) | identical CV RMSE | rejected — `us_aqi_t` already handles it |
| MAE / absolute-error loss | worse RMSE, no bias improvement | rejected |
| Ridge + GBM blend | −1% at h=24, worse at h=72 | rejected |
| HistGBM + XGBoost average | 22.51 vs 22.41 — inside noise | rejected |
| Sample weighting toward high AQI | RMSE 22.51 vs 22.59, bias −25.0 vs −28.0 | not adopted; quantile head is a cleaner fix |
| Deeper boosting (31–63 leaves) | worse at every horizon | rejected — 7 leaves chosen |

The tuning sweep found that *shallower and more regularised* beat the default
capacity at every horizon, which says the signal lives in the feature
construction rather than in model complexity.

### 6.8 Error analysis

Error by month, h=72 (MAE):

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12.2 | 15.1 | 10.2 | 12.2 | **25.4** | 17.4 | 19.1 | 15.8 | 14.8 | 13.1 | 9.6 | 13.0 |

May is the worst month at every horizon, and November the best. Pre-monsoon May
brings dust storms and the onset of ozone season, both of which arrive faster
than a 72-hour forecast can track and neither of which the weather features
observe directly.

Error by AQI band shows the mean-reversion pattern of Section 6.6: the model
over-predicts clean air and under-predicts dirty air, with the effect growing
across horizons. The Very Unhealthy band contains only 17 holdout hours, so no
conclusion is drawn from it.

---

## 7. Leakage verification

Claims of care are worth less than tests. Three were implemented and run on
every feature build.

**Test A — causality by truncation.** Delete every row after time `t`,
recompute all 62 origin features, and compare with the values computed on the
full series. Any feature that peeks forward will change.

> Result: **0 of 62 features changed.**

This is the definitive test, because it makes no assumption about how a leak
might occur.

**Test B — target alignment.** The target on row `t` must equal the raw series
at `t+h`, and forecast-side features must describe the target hour.

> Result: pass at all horizons. At h=72, the row for 2024-04-21 00:00 carries
> target 78.0, which is `us_aqi` at 2024-04-24 00:00; its temperature feature
> is the temperature at that same hour.

**Test C — correlation ceiling.** No origin feature may correlate with the
target more strongly than the raw autocorrelation allows.

> Result: top origin correlation is `us_aqi_t` at 0.8046 (h=24) and 0.6029
> (h=72), matching the raw AQI autocorrelation at those lags to four decimals.

**Training/serving parity.** Separately, the assembled serving feature row was
compared against the corresponding training row across all 91 features:

> Maximum absolute difference: **4 × 10⁻¹²** at all three horizons.

This holds only because the serving code calls the same `origin_features()` and
`forecast_features()` functions used in training rather than reimplementing
them. A reimplementation is precisely how training/serving skew begins.

---

## 8. Explainability

SHAP values were computed on holdout rows using `TreeExplainer`. Full outputs
in `reports/shap_*.csv` and `reports/figures/shap_*.png`.

### 8.1 Importance shifts with horizon — a falsifiable prediction

Section 5 argued that the day-1 to day-3 degradation is gentle because the
model has two information sources decaying at different rates: recent state,
which decays fast, and seasonal plus forecast structure, which does not. If
true, SHAP should show importance shifting from the first to the second as the
horizon grows.

Share of total mean |SHAP|:

| Feature group | h=24 | h=72 |
|---|---|---|
| AQI history | 48.6% | 36.7% |
| Weather forecast | 24.2% | **36.8%** |
| Pollutant history | 21.4% | 13.8% |
| Calendar | 5.7% | **12.8%** |

Origin-side features fall from 70.0% to 50.5% of total importance;
forecast-side features rise from 29.9% to 49.6%. At three days out the model
leans about equally on what it can observe now and on what the weather forecast
tells it.

This was a genuine test. Had the split stayed flat, the explanation given
throughout this report would have been wrong and would have required rewriting.

### 8.2 The model independently rediscovered the seasonal chemistry

Section 4.3 established from correlations that winter AQI is PM2.5-driven and
summer shifts toward ozone. SHAP at h=72, computed from the fitted model with
no knowledge of that analysis:

| Feature family | Winter mean abs SHAP | Summer mean abs SHAP |
|---|---|---|
| PM2.5 features | 3.02 | 1.57 |
| Ozone features | 0.62 | 1.15 |

Two independent methods agreeing is the strongest form of evidence available
here: the model learned the physics rather than memorising the target.

### 8.3 Individual features

At h=24, `us_aqi_t` dominates (mean |SHAP| 15.25), followed by
`wind_rmean24_f` at 4.16 — notably high for a variable whose raw correlation
with AQI is only −0.30, and consistent with the cumulative-ventilation finding.
A per-prediction waterfall plot is generated for the highest-AQI holdout hour
and doubles as the dashboard's explanation panel.

---

## 9. System architecture

```
Open-Meteo API
      │
      ├─ air quality (CAMS global) ─┐
      └─ weather forecast ──────────┤
                                    ▼
                        pipelines/hourly.py  (every hour, GitHub Actions)
                                    │
                    clean ─ features ─ leakage checks
                                    │
                  ┌─────────────────┴──────────────────┐
                  ▼                                    ▼
         Hopsworks feature store              src/models/predict.py
         (4 feature groups)                            │
                  │                        model registry (6 models,
                  │                        best version by metric)
                  ▼                                    │
       pipelines/daily.py (daily)                      ▼
       train ─ evaluate ─ register        reports/latest_forecast.json
                  │                                    │
                  └────────────────────────────────────┤
                                                       ▼
                                          Streamlit dashboard
```

### 9.1 Feature store

Hopsworks project `aqi_proj`, four feature groups:

| Group | Contents |
|---|---|
| `aqi_raw_hourly` | cleaned hourly observations — the audit trail |
| `aqi_features_h24/48/72` | model-ready features and target per horizon |

Both raw and computed features are stored. The raw group allows features to be
rebuilt after a logic change; the feature groups guarantee that training and
serving read identical columns, which is the drift problem a feature store
exists to solve.

Primary key is `unix_ts` (int64 seconds) rather than the timestamp, because
Hopsworks primary keys must be scalar; `time` is registered as the event time.
With Delta time travel, re-inserting an existing key upserts rather than
duplicates, which makes the hourly job idempotent and safe to retry.

### 9.2 Model registry

Six artifacts — three point heads and three alert heads — each registered with
its holdout metrics and metadata including the exact evaluation window, the
feature list, and the training environment versions.

Inference selects the **best registered version by metric**, not the latest:
lowest RMSE for point heads, highest F1 for alert heads. A retrain that scores
worse is recorded but never served. This means the daily job can push freely
without a deployment gate, and the registry retains the full history for drift
diagnosis.

### 9.3 Model serving

There is no Hopsworks deployment endpoint, deliberately. Predictions are
computed hourly in batch and written to a JSON artifact. A real-time endpoint
would add infrastructure without reducing any latency a user experiences,
because the forecast only changes once an hour. Batch inference plus a static
artifact is the appropriate pattern for this cadence.

### 9.4 Automation

| Workflow | Schedule | Work |
|---|---|---|
| `hourly.yml` | every hour, :07 | fetch → clean → features → push → forecast → commit |
| `daily.yml` | 02:40 UTC | pull features → train 3 horizons × 5 models → register → commit |

The runner filesystem is ephemeral, so nothing depends on cached state between
runs.

### 9.5 Dashboard

Streamlit, deployed on Community Cloud, reading only committed JSON and the
public Open-Meteo API — no credentials required. Sections: current AQI with the
standard category colour and corresponding health guidance; three-day forecast
with the 90th-percentile band; hazardous-air alert banner; **forecast-versus-
actual accuracy** scored from the accumulating history log; and a plain-language
explanation of how the forecast is produced.

The accuracy panel is the part that distinguishes this from a dashboard that
only asserts its own quality: it scores every past forecast against what
actually happened.

---

## 10. Engineering and reliability

Six failure modes were encountered in live operation and fixed. They are
documented because the fixes describe the system's actual robustness better
than a clean description would.

### 10.1 A failure taxonomy

The system distinguishes three categories, each with a different response:

| Category | Example | Response |
|---|---|---|
| Transient external | Query Service gRPC UNAVAILABLE | retry with backoff |
| Idempotent and self-healing | feature store push fails | log, continue, next run re-sends |
| Correctness | schema mismatch, null feature at serving | fail immediately and loudly |

Treating all three identically — crashing on everything or swallowing
everything — is what separates a script from a pipeline.

### 10.2 Incidents

**Delta library missing.** Feature group creation failed because recent
Hopsworks versions default to Delta format, which requires the `deltalake`
package client-side. Delta is desirable here because it provides
upsert-on-primary-key, so the dependency was added rather than the format
downgraded.

**Spurious `index` column.** Boolean-filtering a DataFrame under pandas 2.x
produces a plain `Index` rather than a `RangeIndex`, so `reset_index()` injected
a column named `index` that broke the feature group schema. The bug appeared
only in the incremental path — which local testing had never exercised, because
every local run used `--skip-push`. A guard now raises locally with a clear
message rather than failing server-side.

**Schema drift from dtype inference.** After narrowing the hourly fetch to 60
days, a column with no NaN in that window inferred `int64` where the schema
expected `double`, and the write was rejected. Two attempts to fix this by
guessing dtypes from column names were both wrong, in opposite directions —
`relative_humidity_2m` is stored as `bigint` while `ozone` is `double`, which
no naming convention predicts. The correct fix reads `fg.features` from
Hopsworks and casts to whatever the group actually declares. **The server is
the only reliable source of truth for its own schema.**

**A transient read triggering a destructive backfill.** `latest_timestamp()`
caught every exception and returned `None`. A transient Query Service failure
therefore made an existing 35,000-row group look empty, and the incremental job
responded by launching a full backfill from a 60-day local window. Now a failed
read raises, `None` means only "the group does not exist", and a size check
refuses to backfill from insufficient local data. *This exact failure mode had
been flagged in a code review two days before it occurred.*

**Half-registered models.** The Hopsworks metadata cluster intermittently
returns HTTP 500 mid-upload, leaving four of six models registered and failing
the job. Registration is now retried with `create_model` inside the retry loop
— a half-created version cannot be saved into again — and any model still
unregistered after retries fails the job explicitly rather than exiting green
with an inconsistent registry.

**Concurrent pushes.** Both workflows commit to `main`. A daily run lasting the
best part of an hour would find the remote moved by the hourly job and its push
rejected — failing a job whose training and registration had already succeeded.
Both now stash uncommitted pipeline artifacts, then rebase and retry.

### 10.3 Read-latency growth

Feature store read time grew measurably over the first week of operation:

| Point in time | `read_raw` duration |
|---|---|
| Initial backfill | 2.6 s |
| After ~3 days | 56 s |
| After ~7 days | 136 s |

The cause is Delta commit accumulation: each hourly insert creates a new file,
and every read merges them all. At 136 s the read began exceeding the Query
Service's own timeout, causing three consecutive hourly failures.

The hourly job was restructured to source history from Open-Meteo instead —
the actual system of record, already being called, and equally fast for 60 days
as for 10. The feature store became a write sink on that path, and it was
verified that a 60-day window produces features **identical to the full
four-year series** (maximum difference 1.2 × 10⁻¹¹ across the rebuild tail).
The daily training job still reads the store in full, where a slow read once a
day is acceptable.

### 10.4 Environment reproducibility

A CI run failed with `ModuleNotFoundError: No module named '_loss'` — a
scikit-learn internal. Models pickled on macOS could not be unpickled by the
CI runner's different scikit-learn version. Pickled artifacts are bound to the
exact library versions that created them.

Fixed by pinning `scikit-learn`, `xgboost`, `numpy` and `joblib` exactly, and by
recording the training environment in each model's metadata so a future
mismatch is diagnosable rather than mysterious. Requirements are also split:
the dashboard runs on a slim set, because the full development set pulls in
`confluent-kafka` (via Hopsworks), which needs a C library absent from the
Streamlit runner.

---

## 11. Limitations

Stated plainly, in order of importance.

### 11.1 Forecast lead-time skew — the most significant limitation

Weather features are drawn from Open-Meteo's Historical Forecast archive, which
stitches together the *early* hours of successive model runs. The archived
value for a given past hour therefore came from a run initialised a few hours
earlier — not 72 hours earlier.

In production, the `t+72` weather feature will come from a genuine 72-hour-lead
forecast, which is meaningfully less accurate. **The model is trained on better
weather than it will ever see live**, so the reported holdout scores are
optimistic to an unquantified degree.

This is train/serve skew, not target leakage: no future AQI touches the model,
as Section 7 verifies. But it is real, and SHAP shows weather-forecast features
carry ~37% of importance at 72 hours, so it is not negligible.

**Two remedies.** The rigorous fix is Open-Meteo's `previous_runs` archive,
which serves forecasts at a specified lead time, allowing each horizon's
features to be built from a genuinely matched forecast. This requires rebuilding
every weather feature and a full re-backfill. The measurement approach — already
underway — logs live forecasts hourly and will allow live error to be compared
against holdout error directly within a few weeks, quantifying the skew rather
than assuming it is small.

### 11.2 CAMS resolution

The target is a CAMS global model estimate at roughly 40 km resolution, not a
ground station reading. Over four years the maximum observed value is 218, which
is lower than Islamabad ground sensors report. **The system forecasts CAMS, not
measured air quality.** Spatial detail — traffic corridors, industrial plumes —
is smoothed away. Ground-station data would both improve realism and provide a
higher-variance target that is harder but more useful to predict.

### 11.3 Metric comparability across registry versions

Model selection compares registered metrics across versions, which is strictly
valid only if all versions were scored on identical rows. The rolling holdout
keeps windows the same *length*, so consecutive retrains differ by one day — but
over months they diverge. Each version records its exact window so a comparison
can be audited. A fully rigorous scheme would rescore every candidate on one
frozen benchmark before selection.

A visible consequence: point heads currently remain at version 1 because later
retrains scored marginally worse on slightly shifted windows. The mechanism is
working as designed — a worse model is never served — but it means no retrained
point model has yet reached production.

### 11.4 Missing-data policy differs between training and serving

Training drops only null targets; Ridge and Random Forest median-impute
features while the boosting models handle nulls natively. Serving rejects any
null before predicting. Model selection therefore includes rows that production
would refuse to serve. In practice the feature set is complete after the
warm-up period, so the effect is small, but the policies should be unified.

### 11.5 Extreme events are under-represented

The Very Unhealthy band contains 17 holdout hours across four years. Alert
performance above AQI 200 cannot be meaningfully assessed from this data, and
the reported recall figures apply to the AQI > 150 threshold where sample size
is adequate.

### 11.6 Single location

All results are for one grid cell. The pipeline is coordinate-parameterised and
would run elsewhere, but the feature design — particularly the sub-index window
matching — was validated against Islamabad's seasonal chemistry and would need
revalidation for a city with different dominant pollutants.

---

## 12. Future work

Ordered by expected value per unit of effort.

1. **Lead-time-matched weather features** via `previous_runs`. Removes the
   headline limitation and would give an honest estimate of live performance.
2. **Ground-station data** from the Pakistan EPA or Punjab AQI network,
   fused with or replacing CAMS as the target. The single largest realism gain.
3. **A frozen benchmark window** scored alongside the rolling holdout, making
   cross-version model selection rigorous and enabling true drift detection —
   a degrading rolling metric against a flat frozen one means the world changed,
   not the code.
4. **Fire and crop-burning proxies**, such as MODIS/VIIRS active fire counts
   upwind. The residual error at 72 hours is largely events that weather
   features cannot observe.
5. **Feature-store compaction**, scheduled, to bound the read-latency growth
   documented in Section 10.3.
6. **Feature pruning.** 91 features from 20 variables is a high expansion
   ratio with substantial redundancy. SHAP suggests a much smaller set carries
   the signal; a leaner model would retrain faster and be easier to defend.
7. **Multi-city rollout**, which the configuration design already supports.

---

## 13. Reproduction

```
aqi-predictor/
├── .github/workflows/     hourly.yml, daily.yml
├── pipelines/             hourly.py, daily.py     CI entrypoints
├── src/
│   ├── config.py          location, horizons, variables, paths
│   ├── data/              fetch_openmeteo.py, clean.py, feature_store.py
│   ├── features/          build_features.py
│   ├── models/            train.py, predict.py, explain.py, lstm.py
│   └── app/               dashboard.py
├── models/                6 artifacts + metadata
└── reports/               metrics, figures, forecast history
```

```bash
pip install -r requirements-dev.txt
export HOPSWORKS_API_KEY="..."

python -m src.data.fetch_openmeteo        # backfill from Aug 2022
python -m src.data.clean
python -m src.eda                         # figures and baselines
python -m src.features.build_features
python -m src.models.train                # 3 horizons x 5 models
python -m src.models.explain              # SHAP
python -m src.data.feature_store --backfill --register-models
python -m src.models.predict              # live forecast
streamlit run src/app/dashboard.py
```

Deep-learning comparison (requires `torch`):

```bash
python -m src.models.lstm --arch gru
python -m src.models.lstm --arch lstm
```

---

## 14. Closing note

The most valuable outcomes of this project were not the accuracy figures.

The single largest quality improvement came from reading the US AQI definition
carefully enough to notice it is built from rolling averages — a
domain-understanding gain, not a modelling one. The most consequential bug fix
was to the cross-validation protocol, which had been quietly freezing the
training set and had also produced the wrong model ranking. And the alert
system exists in its current form because a measurement showed the accurate
model was missing three-quarters of the days it most needed to catch.

Each of those came from asking what the numbers meant rather than whether they
had improved.
