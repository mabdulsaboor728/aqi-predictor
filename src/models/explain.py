"""
Step 6 - Explainability with SHAP.

Produces four things, in order of how much they matter for the report:

  1. Feature-GROUP importance by horizon.
     Does the model shift from recent-state features toward calendar and
     forecast-weather features as the horizon grows? The EDA predicted it
     should. If it does not, the explanation we have been giving for the gentle
     day1 -> day3 degradation is wrong and needs rewriting.

  2. Global importance + beeswarm per horizon.
     Which individual features carry the model, and in which direction.

  3. Seasonal contrast (winter vs summer).
     EDA Q3 found winter AQI is PM2.5-driven and summer is ozone-driven. SHAP
     should show the same split. This is a genuine falsifiable check, not a
     decoration.

  4. A worked single-prediction waterfall for a high-AQI hour, which doubles as
     the "why this forecast" panel in the dashboard.

Run:
    pip install shap
    python -m src.models.explain
    python -m src.models.explain --horizons 72 --sample 3000
"""

from __future__ import annotations

import argparse
import json
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config as cfg

HOLDOUT_START = "2025-09-01"
FIG_DIR = cfg.ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS = cfg.ROOT / "reports"


# --------------------------------------------------------------------------- #
def feature_group(col: str) -> str:
    """Map a feature name to one of four availability/semantic families."""
    if col.endswith("_t"):
        return "aqi_history" if col.startswith("us_aqi") else "pollutant_history"
    if re.search(r"(hour_|doy_|month_|dayofweek_|is_weekend)", col):
        return "calendar"
    return "weather_forecast"


def load_model_and_data(h: int):
    model = joblib.load(cfg.MODELS_DIR / f"model_h{h}.joblib")
    d = pd.read_parquet(cfg.DATA_PROCESSED / f"dataset_h{h}.parquet").set_index("time")
    y = d.pop("target")
    X = d.drop(columns=["horizon"], errors="ignore")
    mask = X.index >= HOLDOUT_START
    return model, X[mask], y[mask]


def get_explainer(model, X_bg: pd.DataFrame):
    """TreeExplainer where possible; fall back to the model-agnostic path.

    Pipelines (ridge, random_forest) wrap the estimator, so unwrap first.
    XGBoost and HistGradientBoosting are both handled by TreeExplainer in
    current shap versions, but the fallback keeps this from being brittle.
    """
    est = model
    if hasattr(model, "steps"):                 # sklearn Pipeline
        est = model.steps[-1][1]
        X_bg = pd.DataFrame(
            model[:-1].transform(X_bg), columns=X_bg.columns, index=X_bg.index
        )
    try:
        return shap.TreeExplainer(est), est, X_bg, "tree"
    except Exception as exc:                    # noqa: BLE001
        print(f"    TreeExplainer unavailable ({type(exc).__name__}); using Permutation")
        return (shap.PermutationExplainer(est.predict, X_bg.iloc[:200]),
                est, X_bg, "permutation")


def transform_for(model, X: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "steps"):
        return pd.DataFrame(model[:-1].transform(X), columns=X.columns, index=X.index)
    return X


# --------------------------------------------------------------------------- #
def run_horizon(h: int, n_sample: int, rng: np.random.Generator) -> dict:
    print(f"\n{'=' * 70}\nhorizon {h}h")
    model, X, y = load_model_and_data(h)

    take = min(n_sample, len(X))
    pos = np.sort(rng.choice(len(X), take, replace=False))
    Xs, ys = X.iloc[pos], y.iloc[pos]
    print(f"  explaining {take} holdout rows, {X.shape[1]} features")

    explainer, est, _, kind = get_explainer(model, Xs)
    Xt = transform_for(model, Xs)
    sv = explainer(Xt) if kind != "tree" else explainer(Xt, check_additivity=False)
    vals = sv.values if hasattr(sv, "values") else np.asarray(sv)

    # ---------------------------------------------------------------- global
    imp = pd.Series(np.abs(vals).mean(0), index=X.columns).sort_values(ascending=False)
    print("\n  top 15 features (mean |SHAP|):")
    for name, v in imp.head(15).items():
        print(f"    {name:<38} {v:6.3f}   [{feature_group(name)}]")

    # ---------------------------------------------------------------- groups
    groups = pd.Series({c: feature_group(c) for c in X.columns})
    grp = imp.groupby(groups).sum()
    grp_pct = (grp / grp.sum() * 100).sort_values(ascending=False)
    print("\n  importance by group (% of total |SHAP|):")
    for name, v in grp_pct.items():
        print(f"    {name:<20} {v:5.1f}%   ({int((groups == name).sum())} features)")

    # ---------------------------------------------------------------- plots
    shap.summary_plot(vals, Xt, max_display=20, show=False)
    plt.title(f"SHAP summary - horizon {h}h")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_beeswarm_h{h}.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    imp.head(20)[::-1].plot.barh(ax=ax, color="#4477aa")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"Top 20 features - horizon {h}h")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"shap_top20_h{h}.png", dpi=140)
    plt.close(fig)

    # dependence plots for the three strongest features
    for feat in imp.head(3).index:
        shap.dependence_plot(feat, vals, Xt, show=False)
        plt.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_]", "", feat)
        plt.savefig(FIG_DIR / f"shap_dep_h{h}_{safe}.png", dpi=140)
        plt.close()

    # ---------------------------------------------------------------- seasonal
    month = Xs.index.tz_convert(cfg.LOCAL_TZ).month
    winter = np.isin(month, [12, 1, 2])
    summer = np.isin(month, [6, 7, 8])
    seasonal = None
    if winter.sum() > 50 and summer.sum() > 50:
        seasonal = pd.DataFrame({
            "winter": np.abs(vals[winter]).mean(0),
            "summer": np.abs(vals[summer]).mean(0),
        }, index=X.columns)
        pm = [c for c in X.columns if "pm2_5" in c or "pm25" in c]
        o3 = [c for c in X.columns if "ozone" in c or "o3" in c]
        print(f"\n  seasonal check (n_winter={winter.sum()}, n_summer={summer.sum()}):")
        print(f"    PM2.5 features  winter {seasonal.loc[pm, 'winter'].sum():6.3f}  "
              f"summer {seasonal.loc[pm, 'summer'].sum():6.3f}")
        print(f"    ozone features  winter {seasonal.loc[o3, 'winter'].sum():6.3f}  "
              f"summer {seasonal.loc[o3, 'summer'].sum():6.3f}")
        print("    (EDA predicted PM2.5 dominant in winter, ozone rising in summer)")

    # ---------------------------------------------------------------- waterfall
    worst = ys.idxmax()
    wi = list(Xs.index).index(worst)
    base = float(np.ravel(sv.base_values)[wi]) if hasattr(sv, "base_values") else float(ys.mean())
    exp = shap.Explanation(values=vals[wi], base_values=base,
                           data=Xt.iloc[wi].values, feature_names=list(X.columns))
    shap.plots.waterfall(exp, max_display=14, show=False)
    plt.title(f"Why AQI {ys.loc[worst]:.0f} was forecast for {worst:%Y-%m-%d %H:%M} (+{h}h)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_waterfall_h{h}.png", dpi=140)
    plt.close()
    print(f"\n  waterfall saved for peak hour {worst} (actual AQI {ys.loc[worst]:.0f})")

    imp.to_csv(REPORTS / f"shap_importance_h{h}.csv", header=["mean_abs_shap"])
    if seasonal is not None:
        seasonal.to_csv(REPORTS / f"shap_seasonal_h{h}.csv")

    return {"horizon": h, "explainer": kind, "n_explained": int(take),
            "group_pct": grp_pct.round(2).to_dict(),
            "top10": imp.head(10).round(4).to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=cfg.HORIZONS_H)
    ap.add_argument("--sample", type=int, default=4000,
                    help="holdout rows to explain (SHAP cost scales with this)")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    results = [run_horizon(h, args.sample, rng) for h in args.horizons]

    shift = pd.DataFrame({r["horizon"]: r["group_pct"] for r in results}).fillna(0)
    print(f"\n{'=' * 70}\nGROUP IMPORTANCE BY HORIZON (% of total |SHAP|)")
    print(shift.round(1).to_string())
    print("\nExpected if the EDA story holds: aqi_history share FALLS with horizon,")
    print("calendar + weather_forecast share RISES.")

    shift.to_csv(REPORTS / "shap_group_shift.csv")
    (REPORTS / "shap_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nfigures -> {FIG_DIR}\ntables  -> {REPORTS}")


if __name__ == "__main__":
    main()