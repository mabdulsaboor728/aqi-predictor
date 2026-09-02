"""
Step 5 - Training pipeline.

Trains and compares several models per horizon using PURGED walk-forward
cross-validation, then refits the winner on all development data and scores it
once on a rolling holdout.

Evaluation protocol
-------------------
  holdout   : the final HOLDOUT_DAYS of available data, scored once
  CV        : expanding-window walk-forward over everything before the holdout,
              with a minimum training size and a purge gap
  baseline  : persistence, reported alongside every model so the comparison is
              always against the bar rather than against zero

Why the holdout window ROLLS
----------------------------
An earlier version used a fixed start date with no end. That silently froze the
training set: new data only ever enlarged the holdout, so every "daily retrain"
refit the identical rows and could not adapt to drift. A window of constant
LENGTH anchored to the end of the data keeps the training set growing while
holding the evaluation window approximately comparable across versions -
consecutive retrains differ by a day of holdout, not by months.

Over long periods the windows do still diverge, so a "best ever registered
version" comparison is only approximately valid. Each model records its exact
holdout window in metadata so the comparison can be audited. A fully rigorous
scheme would rescore every candidate on one frozen benchmark before selection;
that is noted as future work rather than implemented here.

Two heads per horizon
---------------------
  point head -> the number shown on the dashboard, RMSE-optimal
  alert head -> a high quantile, used only for the threshold warning

The RMSE-optimal forecast recalls only about half of AQI>150 hours at h=72:
squared error rewards hedging toward the mean, which is the wrong behaviour for
a warning system. These are different objectives and one model cannot serve
both.

Purging
-------
A training row at time T carries a target at T + horizon. If validation begins
at T + 1, that target lies inside the validation window and the model has
effectively seen the answer. Standard TimeSeriesSplit does not handle this, so
a gap of (horizon + 24) hours separates every train fold from its validation
fold, and the same gap separates the development set from the holdout.

Run:
    python -m src.models.train
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as cfg

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# rolling holdout: always the final HOLDOUT_DAYS of available data
HOLDOUT_DAYS = 365
MIN_TRAIN_HOURS = 24 * 365 * 2      # two full seasonal cycles before fold 1
N_SPLITS = 4
SEED = 42

ALERT_THRESHOLD = 150               # US AQI "Unhealthy"
ALERT_QUANTILE = 0.90

REPORTS = cfg.ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# tuned by CV sweep - shallow and regularised beat deep
GBM_KW = dict(
    max_iter=800, learning_rate=0.03, max_leaf_nodes=7,
    min_samples_leaf=100, l2_regularization=3.0,
    early_stopping=False, random_state=SEED,
)


# --------------------------------------------------------------------------- #
def point_models() -> dict:
    zoo = {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=8,
                                  max_features=0.4, n_jobs=-1, random_state=SEED),
        ),
        "hist_gbm": HistGradientBoostingRegressor(**GBM_KW),
    }
    if HAS_XGB:
        zoo["xgboost"] = XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=4,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=3.0, objective="reg:squarederror",
            tree_method="hist", n_jobs=-1, random_state=SEED,
        )
    return zoo


def alert_model():
    """High-quantile head. Deliberately biased upward: for a health warning,
    a false alarm costs far less than a missed Unhealthy day."""
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=ALERT_QUANTILE, **GBM_KW
    )


# --------------------------------------------------------------------------- #
def purged_folds(n: int, horizon: int, n_splits: int = N_SPLITS,
                 min_train: int = MIN_TRAIN_HOURS, gap_extra: int = 24):
    """Expanding window with a purge gap AND a minimum training size.

    The minimum training size stops early folds from scoring models that have
    never seen a full seasonal cycle - without it, boosted models were
    penalised hardest and the model ranking came out wrong.
    """
    gap = horizon + gap_extra
    usable = n - min_train - gap
    if usable <= 0:
        raise ValueError(
            f"not enough development data for min_train={min_train}h "
            f"(have {n} rows). Reduce MIN_TRAIN_HOURS or HOLDOUT_DAYS."
        )
    size = usable // n_splits
    out = []
    for i in range(n_splits):
        train_end = min_train + size * i
        v0 = train_end + gap
        v1 = min(v0 + size, n)
        if v0 < n:
            out.append((np.arange(train_end), np.arange(v0, v1)))
    return out


def score(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    high = y_true > ALERT_THRESHOLD
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(r2_score(y_true, y_pred)),
        "bias": float(err.mean()),
        "bias_high": float(err[high].mean()) if high.sum() > 10 else float("nan"),
    }


def alert_score(y_true, y_pred) -> dict:
    act = np.asarray(y_true) > ALERT_THRESHOLD
    pred = np.asarray(y_pred) > ALERT_THRESHOLD
    tp, fp, fn = int((pred & act).sum()), int((pred & ~act).sum()), int((~pred & act).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "recall": float(rec),
        "precision": float(prec),
        "f1": float(2 * rec * prec / max(rec + prec, 1e-9)),
        "n_events": int(act.sum()),
    }


def load_horizon(h: int):
    d = pd.read_parquet(cfg.DATA_PROCESSED / f"dataset_h{h}.parquet").set_index("time")
    y = d.pop("target")
    return d.drop(columns=["horizon"], errors="ignore"), y


# --------------------------------------------------------------------------- #
def run_horizon(h: int):
    X, y = load_horizon(h)

    # Rolling split. The purge on the dev side matters as much as it does
    # between CV folds: a dev row at t carries its target at t+h, which must
    # not fall inside the holdout.
    data_end = X.index.max()
    holdout_start = data_end - pd.Timedelta(days=HOLDOUT_DAYS)
    dev_mask = X.index < (holdout_start - pd.Timedelta(hours=h + 24))
    hold_mask = X.index >= holdout_start

    Xd, yd = X[dev_mask], y[dev_mask]
    Xh, yh = X[hold_mask], y[hold_mask]

    if len(Xh) == 0:
        raise ValueError(f"h={h}: holdout is empty - is HOLDOUT_DAYS longer than the data?")

    folds = purged_folds(len(Xd), h)

    print(f"\n{'=' * 70}\nhorizon {h}h   dev={len(Xd)}  holdout={len(Xh)}")
    print(f"holdout window: {holdout_start:%Y-%m-%d} -> {data_end:%Y-%m-%d} "
          f"({HOLDOUT_DAYS}d rolling)")
    print(f"{len(folds)} folds | min train {MIN_TRAIN_HOURS}h | purge {h + 24}h")

    persist = [score(yd.iloc[v], Xd["us_aqi_t"].iloc[v]) for _, v in folds]
    rows = [{"model": "persistence",
             **{k: float(np.nanmean([p[k] for p in persist])) for k in persist[0]}}]
    print(f"  {'persistence':<14} RMSE {rows[0]['RMSE']:6.2f}"
          f"{'':<12}MAE {rows[0]['MAE']:6.2f}  R2 {rows[0]['R2']:6.3f}")

    for name, proto in point_models().items():
        t0 = time.time()
        s = [score(yd.iloc[v], clone(proto).fit(Xd.iloc[t], yd.iloc[t]).predict(Xd.iloc[v]))
             for t, v in folds]
        agg = {k: float(np.nanmean([x[k] for x in s])) for k in s[0]}
        agg["RMSE_std"] = float(np.std([x["RMSE"] for x in s]))
        rows.append({"model": name, **agg, "fit_s": round(time.time() - t0, 1)})
        print(f"  {name:<14} RMSE {agg['RMSE']:6.2f} (+-{agg['RMSE_std']:.2f})  "
              f"MAE {agg['MAE']:6.2f}  R2 {agg['R2']:6.3f}  bias>150 {agg['bias_high']:+6.2f}")

    cv = pd.DataFrame(rows).set_index("model")
    best_name = cv.drop(index="persistence")["RMSE"].idxmin()

    # --- refit both heads on all dev data, then touch the holdout once
    point = clone(point_models()[best_name]).fit(Xd, yd)
    alert = clone(alert_model()).fit(Xd, yd)

    p_point, p_alert = point.predict(Xh), alert.predict(Xh)
    hold, base = score(yh, p_point), score(yh, Xh["us_aqi_t"])
    a_point, a_alert = alert_score(yh, p_point), alert_score(yh, p_alert)

    print(f"  -> point head: {best_name}")
    print(f"     holdout  RMSE {hold['RMSE']:6.2f}  MAE {hold['MAE']:6.2f}  R2 {hold['R2']:.3f}"
          f"   ({(1 - hold['RMSE'] / base['RMSE']) * 100:.1f}% vs persistence)")
    print(f"     alerts   point head : recall {a_point['recall']:.2f}  "
          f"precision {a_point['precision']:.2f}")
    print(f"              q{ALERT_QUANTILE} head : recall {a_alert['recall']:.2f}  "
          f"precision {a_alert['precision']:.2f}   (n={a_alert['n_events']} events)")

    joblib.dump(point, cfg.MODELS_DIR / f"model_h{h}.joblib")
    joblib.dump(alert, cfg.MODELS_DIR / f"alert_h{h}.joblib")

    import sys as _sys
    meta = {
        "horizon": h, "point_model": best_name,
        "alert_model": f"hist_gbm_quantile_{ALERT_QUANTILE}",
        "alert_threshold": ALERT_THRESHOLD,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_train": int(len(Xd)), "n_features": int(X.shape[1]),
        "cv": {"min_train_hours": MIN_TRAIN_HOURS, "n_folds": len(folds),
               "purge_hours": h + 24},
        # the exact rows this version was scored on, so a cross-version
        # registry comparison can be audited rather than assumed valid
        "holdout_window": {
            "start": holdout_start.isoformat(),
            "end": data_end.isoformat(),
            "days": HOLDOUT_DAYS,
            "n_rows": int(len(Xh)),
        },
        "holdout": hold, "holdout_persistence": base,
        "alert_point_head": a_point, "alert_quantile_head": a_alert,
        # pickles are tied to the versions that wrote them; recording the
        # environment makes a load failure diagnosable instead of mysterious
        "env": {
            "python": ".".join(map(str, _sys.version_info[:3])),
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__,
            "xgboost": __import__("xgboost").__version__ if HAS_XGB else None,
        },
        "features": list(X.columns),
    }
    (cfg.MODELS_DIR / f"model_h{h}_meta.json").write_text(json.dumps(meta, indent=2))

    cv["horizon"] = h
    return cv.reset_index(), meta


def main() -> None:
    if not HAS_XGB:
        print("note: xgboost not installed - skipping it (pip install xgboost)")

    cvs, metas = [], []
    for h in cfg.HORIZONS_H:
        c, m = run_horizon(h)
        cvs.append(c)
        metas.append(m)

    pd.concat(cvs, ignore_index=True).to_csv(REPORTS / "model_comparison.csv", index=False)

    summary = pd.DataFrame([{
        "horizon": m["horizon"], "point_model": m["point_model"],
        "RMSE": round(m["holdout"]["RMSE"], 2),
        "MAE": round(m["holdout"]["MAE"], 2),
        "R2": round(m["holdout"]["R2"], 3),
        "persist_RMSE": round(m["holdout_persistence"]["RMSE"], 2),
        "gain_%": round((1 - m["holdout"]["RMSE"] / m["holdout_persistence"]["RMSE"]) * 100, 1),
        "recall_point": round(m["alert_point_head"]["recall"], 2),
        "recall_q90": round(m["alert_quantile_head"]["recall"], 2),
        "prec_q90": round(m["alert_quantile_head"]["precision"], 2),
        "n_train": m["n_train"],
        "holdout_start": m["holdout_window"]["start"][:10],
    } for m in metas])

    print(f"\n{'=' * 70}\nholdout summary")
    print(summary.to_string(index=False))
    summary.to_csv(REPORTS / "holdout_summary.csv", index=False)
    print(f"\nmodels -> {cfg.MODELS_DIR}\nreports -> {REPORTS}")


if __name__ == "__main__":
    main()