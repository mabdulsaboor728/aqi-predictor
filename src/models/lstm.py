"""
Deep-learning comparison arm: GRU / LSTM sequence models.

Architecture
------------
Two encoders, matching the same availability contract as the tabular pipeline:

    history encoder   past SEQ_LEN hours of pollutants + weather + AQI,
                      ending AT t. Never extends past t.
    future encoder    weather forecast for t+1 .. t+h. Legitimately available
                      when the forecast is issued.

    [h_hist ; h_future ; calendar(t+h)] -> MLP -> prediction

Residual mode (default ON)
--------------------------
The network predicts (aqi[t+h] - aqi[t]) rather than the level. Neural nets,
unlike Ridge, have no free way to express y ~= x, so making persistence the
anchor and learning only the correction usually converges much faster. Use
--no-residual to test the level formulation.

Protocol
--------
Identical splits to src/models/train.py so numbers are directly comparable:
  holdout   : from 2025-09-01, scored once
  validation: last VAL_FRAC of dev, separated from train by a purge gap of
              (horizon + 24) hours, used for early stopping only
  scaling   : StandardScaler fit on the TRAIN slice only

Usage
-----
    pip install torch
    python -m src.models.train_seq                      # GRU, all horizons
    python -m src.models.train_seq --arch lstm
    python -m src.models.train_seq --horizons 72 --epochs 60
    python -m src.models.train_seq --no-residual
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config as cfg

# --------------------------------------------------------------------------- #
SEQ_LEN = 168          # one week of history
HOLDOUT_START = "2025-09-01"
VAL_FRAC = 0.15
ALERT_THRESHOLD = 150
SEED = 42

# history channels: everything observable at t
HIST_VARS = [
    "us_aqi", "pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide",
    "carbon_monoxide", "dust", "aerosol_optical_depth",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
    "shortwave_radiation",
]

# future channels: weather only - no pollutant forecast exists
FUT_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
    "shortwave_radiation",
]

REPORTS = cfg.ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():          # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def build_windows(df: pd.DataFrame, horizon: int):
    """Return (hist, fut, cal, y, origin_times) aligned on the ORIGIN index.

    For origin position i:
        hist = rows [i-SEQ_LEN+1 .. i]      inclusive of t
        fut  = rows [i+1 .. i+horizon]      the forecast window
        y    = us_aqi at i+horizon
    """
    H = df[HIST_VARS].to_numpy(np.float32)
    F = df[FUT_VARS].to_numpy(np.float32)
    aqi = df[cfg.TARGET].to_numpy(np.float32)

    local = df.index.tz_convert(cfg.LOCAL_TZ)
    hour = np.asarray(local.hour, dtype=np.float32)
    doy = np.asarray(local.dayofyear, dtype=np.float32)
    CAL = np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        (np.asarray(local.dayofweek) >= 5).astype(np.float32),
    ], axis=1)

    n = len(df)
    first = SEQ_LEN - 1
    last = n - horizon - 1
    idx = np.arange(first, last + 1)

    # sliding_window_view is a view, not a copy - cheap for 27k x 168 x 18
    hist = np.lib.stride_tricks.sliding_window_view(H, SEQ_LEN, axis=0)
    hist = hist.transpose(0, 2, 1)[idx - first]                   # (N, SEQ_LEN, C)

    fut = np.lib.stride_tricks.sliding_window_view(F, horizon, axis=0)
    fut = fut.transpose(0, 2, 1)[idx + 1]                         # (N, horizon, C)

    return (
        np.ascontiguousarray(hist),
        np.ascontiguousarray(fut),
        CAL[idx + horizon],                                       # calendar AT t+h
        aqi[idx + horizon],                                       # target
        aqi[idx],                                                 # persistence anchor
        df.index[idx],
    )


def assert_alignment(df, hist, fut, cal, y, anchor, times, horizon):
    """Cheap sanity checks - the sequence equivalent of the tabular leakage tests."""
    k = len(times) // 2
    t = times[k]
    tgt_t = t + pd.Timedelta(hours=horizon)

    assert np.isclose(y[k], df.loc[tgt_t, cfg.TARGET]), "target misaligned"
    assert np.isclose(anchor[k], df.loc[t, cfg.TARGET]), "anchor misaligned"
    # last history step must be time t itself, not t+1
    assert np.isclose(hist[k, -1, 0], df.loc[t, cfg.TARGET]), "history overruns t"
    # first future step must be t+1
    assert np.isclose(fut[k, 0, 0], df.loc[t + pd.Timedelta(hours=1), FUT_VARS[0]]), \
        "future window misaligned"
    # last future step must be t+h
    assert np.isclose(fut[k, -1, 0], df.loc[tgt_t, FUT_VARS[0]]), "future end misaligned"
    print(f"  alignment checks passed at {t}")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class SeqForecaster(nn.Module):
    def __init__(self, n_hist: int, n_fut: int, n_cal: int,
                 arch: str = "gru", hidden: int = 64, layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        rnn = nn.GRU if arch == "gru" else nn.LSTM
        self.arch = arch
        self.enc_hist = rnn(n_hist, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.enc_fut = rnn(n_fut, hidden // 2, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + hidden // 2 + n_cal, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _last(out, state):
        return state[0][-1] if isinstance(state, tuple) else state[-1]

    def forward(self, hist, fut, cal):
        oh, sh = self.enc_hist(hist)
        of, sf = self.enc_fut(fut)
        z = torch.cat([self._last(oh, sh), self._last(of, sf), cal], dim=1)
        return self.head(z).squeeze(-1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    hi = y > ALERT_THRESHOLD
    pr = p > ALERT_THRESHOLD
    tp, fp, fn = (pr & hi).sum(), (pr & ~hi).sum(), (~pr & hi).sum()
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "R2": float(1 - (e ** 2).sum() / ss),
        "bias": float(e.mean()),
        "bias_high": float(e[hi].mean()) if hi.sum() > 10 else float("nan"),
        "alert_recall": float(rec),
        "alert_precision": float(prec),
    }


# --------------------------------------------------------------------------- #
def run(horizon: int, args) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev_ = device()

    df = pd.read_parquet(cfg.DATA_INTERIM / "clean.parquet").set_index("time")
    df = df.ffill().bfill()          # residual NaNs would poison the tensors

    hist, fut, cal, y, anchor, times = build_windows(df, horizon)
    print(f"\n{'=' * 70}\nhorizon {horizon}h | {args.arch.upper()} | device {dev_}")
    print(f"  windows: hist {hist.shape}  fut {fut.shape}  cal {cal.shape}")
    assert_alignment(df, hist, fut, cal, y, anchor, times, horizon)

    is_hold = times >= HOLDOUT_START
    dev_idx = np.where(~is_hold)[0]
    hold_idx = np.where(is_hold)[0]

    # purge: a training origin at i has its target at i+horizon, which must not
    # fall inside validation
    gap = horizon + 24
    n_val = int(len(dev_idx) * VAL_FRAC)
    val_idx = dev_idx[-n_val:]
    tr_idx = dev_idx[: len(dev_idx) - n_val - gap]
    print(f"  train {len(tr_idx)} | purge {gap} | val {len(val_idx)} | holdout {len(hold_idx)}")

    # --- scale on train only
    def fit_scaler(a):
        flat = a.reshape(-1, a.shape[-1])
        return flat.mean(0), flat.std(0) + 1e-6

    mh, sh = fit_scaler(hist[tr_idx])
    mf, sf = fit_scaler(fut[tr_idx])
    HS = ((hist - mh) / sh).astype(np.float32)
    FS = ((fut - mf) / sf).astype(np.float32)

    # --- target: residual from persistence, or raw level
    if args.residual:
        target = (y - anchor).astype(np.float32)
    else:
        target = y.astype(np.float32)
    ty_m, ty_s = target[tr_idx].mean(), target[tr_idx].std() + 1e-6
    TS = ((target - ty_m) / ty_s).astype(np.float32)

    def tensors(idx):
        return TensorDataset(
            torch.from_numpy(HS[idx]), torch.from_numpy(FS[idx]),
            torch.from_numpy(cal[idx].astype(np.float32)),
            torch.from_numpy(TS[idx]),
        )

    tr_dl = DataLoader(tensors(tr_idx), batch_size=args.batch, shuffle=True, drop_last=True)
    va_dl = DataLoader(tensors(val_idx), batch_size=512)
    ho_dl = DataLoader(tensors(hold_idx), batch_size=512)

    model = SeqForecaster(hist.shape[-1], fut.shape[-1], cal.shape[-1],
                          arch=args.arch, hidden=args.hidden,
                          layers=args.layers, dropout=args.dropout).to(dev_)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = nn.HuberLoss(delta=1.0)     # robust to the dust-event outliers

    def predict(dl, idx):
        model.eval()
        out = []
        with torch.no_grad():
            for h_, f_, c_, _ in dl:
                out.append(model(h_.to(dev_), f_.to(dev_), c_.to(dev_)).cpu().numpy())
        p = np.concatenate(out) * ty_s + ty_m
        return p + anchor[idx] if args.residual else p

    best, best_state, patience = np.inf, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for h_, f_, c_, t_ in tr_dl:
            opt.zero_grad()
            loss = lossf(model(h_.to(dev_), f_.to(dev_), c_.to(dev_)), t_.to(dev_))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(t_)
        vm = metrics(y[val_idx], predict(va_dl, val_idx))
        sched.step(vm["RMSE"])
        flag = ""
        if vm["RMSE"] < best - 1e-4:
            best, patience = vm["RMSE"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            patience += 1
        if ep % args.log_every == 0 or flag:
            print(f"    ep {ep:>3}  train {tot / len(tr_idx):.4f}  "
                  f"val RMSE {vm['RMSE']:6.2f}  MAE {vm['MAE']:6.2f}{flag}")
        if patience >= args.patience:
            print(f"    early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    hm = metrics(y[hold_idx], predict(ho_dl, hold_idx))
    pm = metrics(y[hold_idx], anchor[hold_idx])

    print(f"  trained in {time.time() - t0:.0f}s")
    print(f"  HOLDOUT  RMSE {hm['RMSE']:6.2f}  MAE {hm['MAE']:6.2f}  R2 {hm['R2']:.3f}  "
          f"bias>150 {hm['bias_high']:+.2f}")
    print(f"  persist  RMSE {pm['RMSE']:6.2f}  MAE {pm['MAE']:6.2f}   "
          f"-> {(1 - hm['RMSE'] / pm['RMSE']) * 100:.1f}% better")
    print(f"  alerts   recall {hm['alert_recall']:.2f}  precision {hm['alert_precision']:.2f}")

    torch.save({"state_dict": best_state, "arch": args.arch, "horizon": horizon,
                "seq_len": SEQ_LEN, "residual": args.residual,
                "scalers": {"hist": (mh.tolist(), sh.tolist()),
                            "fut": (mf.tolist(), sf.tolist()),
                            "target": (float(ty_m), float(ty_s))},
                "hist_vars": HIST_VARS, "fut_vars": FUT_VARS},
               cfg.MODELS_DIR / f"seq_{args.arch}_h{horizon}.pt")

    return {"horizon": horizon, "arch": args.arch, "residual": args.residual,
            "holdout": hm, "persistence": pm,
            "n_params": sum(p.numel() for p in model.parameters())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--horizons", type=int, nargs="+", default=cfg.HORIZONS_H)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--no-residual", dest="residual", action="store_false")
    ap.set_defaults(residual=True)
    args = ap.parse_args()

    print(f"target mode: {'residual from persistence' if args.residual else 'raw level'}")
    results = [run(h, args) for h in args.horizons]

    tab = pd.DataFrame([{
        "horizon": r["horizon"], "arch": r["arch"],
        "RMSE": round(r["holdout"]["RMSE"], 2),
        "MAE": round(r["holdout"]["MAE"], 2),
        "R2": round(r["holdout"]["R2"], 3),
        "persist_RMSE": round(r["persistence"]["RMSE"], 2),
        "gain_%": round((1 - r["holdout"]["RMSE"] / r["persistence"]["RMSE"]) * 100, 1),
        "recall": round(r["holdout"]["alert_recall"], 2),
        "precision": round(r["holdout"]["alert_precision"], 2),
        "params": r["n_params"],
    } for r in results])

    print(f"\n{'=' * 70}\n{args.arch.upper()} holdout summary")
    print(tab.to_string(index=False))

    suffix = args.arch + ("" if args.residual else "_level")
    tab.to_csv(REPORTS / f"seq_{suffix}_summary.csv", index=False)
    (REPORTS / f"seq_{suffix}_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {REPORTS}")


if __name__ == "__main__":
    main()