"""
Daily pipeline - the CI entrypoint run by .github/workflows/daily.yml.

    feature store -> full feature datasets -> train -> evaluate -> register

Unlike the hourly job, training needs the FULL history, so this pulls every row
from the feature groups rather than a tail.

On not gating the retrain
-------------------------
There is no "only register if better" check here, deliberately. Inference
selects the best registered version by metric (see src/models/predict.py), so a
worse retrain is registered but never served. Adding a gate would only hide the
degradation; registering it and letting selection ignore it keeps the full
history visible in the registry, which is what you want when diagnosing drift.

The job still prints a comparison against the previously registered best so a
regression is obvious in the CI log.

Run:
    python -m pipelines.daily
    python -m pipelines.daily --skip-register
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

import pandas as pd

from src import config as cfg
from src.data.feature_store import get_fs, login, read_features, register_models


def pull_datasets(fs) -> None:
    """Materialise the full feature groups to local parquet for training."""
    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for h in cfg.HORIZONS_H:
        d = read_features(fs, h)
        out = cfg.DATA_PROCESSED / f"dataset_h{h}.parquet"
        d.reset_index().to_parquet(out, index=False)
        print(f"  h={h}: {len(d)} rows -> {out.name}  "
              f"({d.index.min():%Y-%m-%d} -> {d.index.max():%Y-%m-%d})")


def previous_best(project) -> dict:
    """Registered RMSE per horizon before this run, for the comparison log."""
    out = {}
    try:
        mr = project.get_model_registry()
        for h in cfg.HORIZONS_H:
            try:
                m = mr.get_best_model(f"aqi_point_h{h}", "RMSE", "min")
                if m is not None:
                    out[h] = (m.version, (m.training_metrics or {}).get("RMSE"))
            except Exception:                                  # noqa: BLE001
                continue
    except Exception:                                          # noqa: BLE001
        pass
    return out


def compare(before: dict) -> None:
    print("\nretrain comparison (holdout RMSE)")
    print(f"  {'horizon':<10}{'previous':>12}{'new':>10}{'change':>10}")
    for h in cfg.HORIZONS_H:
        meta_path = cfg.MODELS_DIR / f"model_h{h}_meta.json"
        if not meta_path.exists():
            continue
        new = json.loads(meta_path.read_text())["holdout"]["RMSE"]
        prev = before.get(h, (None, None))[1]
        if prev is None:
            print(f"  {h:<10}{'-':>12}{new:>10.2f}{'first':>10}")
        else:
            delta = new - prev
            mark = "" if delta <= 0.5 else "   <-- worse"
            print(f"  {h:<10}{prev:>12.2f}{new:>10.2f}{delta:>+10.2f}{mark}")
    print("  (a worse model is still registered; selection by metric will not serve it)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-register", action="store_true")
    args = ap.parse_args()

    project = login()
    fs = project.get_feature_store()

    print("pulling full feature datasets")
    pull_datasets(fs)

    before = previous_best(project)

    print("\ntraining")
    from src.models import train

    train.main()

    compare(before)

    if not args.skip_register:
        print("\nregistering models")
        register_models(project)

    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    (cfg.ROOT / "reports" / "last_train.json").write_text(
        json.dumps({"trained_at": stamp}, indent=2)
    )
    print(f"\ndaily pipeline complete ({stamp})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)