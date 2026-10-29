"""Build the leakage-safe machine-learning dataset from the EPANET raw time series.

Task: from the last `window_steps` steps of sensor/control history predict the pressure at J2, J3 and J4
`horizon_steps` steps ahead.

Leakage rules enforced here (each is also covered by tests/test_dataset.py):
  1. Split by whole scenario ID (never by random rows or windows); the manifest is saved.
  2. Windows are cut inside one scenario only, so a window can never cross a scenario boundary.
  3. Scalers are fitted on training scenarios only.
  4. Hidden simulation truth (leak_flag, leak_severity, scenario_type, ...) is never an input feature.
     The `meta_*` arrays in the .npz files are for analysis and splitting only, never for model input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.feature_engineering import add_time_features, resolve_input_features
from src.preprocessing import (DataError, apply_input_scaler, assert_disjoint, fit_scalers, load_raw_timeseries,
                               save_scalers, scenario_table, split_scenarios, write_json)
from src.utils import resolve_path

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- windows (one scenario)
def build_scenario_windows(features: np.ndarray, targets: np.ndarray, window: int, horizon: int):
    """Sliding windows over ONE scenario.

    features: (T, F) rows in time order; targets: (T, K) target-node pressures per row.
    Window i covers rows i .. i+window-1 (the last row is "now"); its target is row i+window-1+horizon.
    Returns x (n, window, F), y (n, K), end_idx (n,) = index of the "now" row. n = T - window - horizon + 1.
    """
    t = len(features)
    n = t - window - horizon + 1
    if window < 1 or horizon < 1:
        raise DataError("window_steps and horizon_steps must be at least 1")
    if n < 1:
        raise DataError(f"a scenario with {t} steps is too short for window={window} and horizon={horizon}")
    x = sliding_window_view(features, window, axis=0)[:n]          # (n, F, window)
    x = np.ascontiguousarray(np.moveaxis(x, 2, 1))                  # (n, window, F)
    end_idx = np.arange(n) + window - 1
    y = targets[end_idx + horizon]
    return x, y, end_idx


# --------------------------------------------------------------------------- whole dataset
def build_dataset(cfg: dict, paths: dict | None = None, seed: int | None = None, window: int | None = None,
                  horizon: int | None = None) -> dict:
    """Run the full Phase 3 data pipeline and write every output file. Returns a summary dictionary."""
    ml = cfg["ml"]
    P = {k: resolve_path(v) for k, v in {**cfg["paths"], **(paths or {})}.items()}
    window = int(ml["window_steps"] if window is None else window)
    horizon = int(ml["horizon_steps"] if horizon is None else horizon)
    seed = int(ml["split"].get("seed") if ml["split"].get("seed") is not None else cfg["project"]["seed"]) \
        if seed is None else int(seed)
    targets_nodes = list(ml["target_nodes"])
    target_cols = [f"{n}_pressure" for n in targets_nodes]

    raw = load_raw_timeseries(P["raw_timeseries"], steps=int(cfg["scenarios"]["steps"]))
    raw = add_time_features(raw, bool(ml.get("hour_fractional", True)))
    features, dropped = resolve_input_features(raw, ml)
    missing_t = [c for c in target_cols if c not in raw.columns]
    if missing_t:
        raise DataError(f"target columns missing from the raw data: {missing_t}")

    # 1. split by whole scenario ----------------------------------------------------------------
    table = scenario_table(raw)
    sp = ml["split"]
    manifest = split_scenarios(table, {s: float(sp[s]) for s in SPLITS}, seed, bool(sp.get("stratify_by_type", True)))
    assert_disjoint(manifest, expected_ids=set(table["scenario_id"]))
    split_of = dict(zip(manifest["scenario_id"], manifest["split"]))

    # 2. windows inside each scenario ------------------------------------------------------------
    parts: dict[str, dict[str, list]] = {s: {k: [] for k in ("x", "y", "sid", "stype", "end", "onset", "active")} for s in SPLITS}
    train_rows = []
    for sid, g in raw.groupby("scenario_id", sort=True):
        g = g.sort_values("timestamp")
        x, y, end_idx = build_scenario_windows(g[features].to_numpy("float64"), g[target_cols].to_numpy("float64"),
                                               window, horizon)
        leak = g["leak_flag"].to_numpy()                          # ANALYSIS ONLY (never an input feature)
        target_idx = end_idx + horizon
        s = split_of[sid]
        d = parts[s]
        d["x"].append(x); d["y"].append(y)
        d["sid"].append(np.full(len(x), sid)); d["stype"].append(np.full(len(x), g["scenario_type"].iloc[0]))
        d["end"].append(end_idx)
        d["onset"].append((leak[target_idx] == 1) & (leak[end_idx] == 0)); d["active"].append(leak[target_idx] == 1)
        if s == "train":
            train_rows.append(g[features].to_numpy("float64"))

    data = {s: {k: np.concatenate(v) for k, v in parts[s].items()} for s in SPLITS}
    for s in SPLITS:
        if len(data[s]["x"]) == 0:
            raise DataError(f"the {s} split has no windows")

    # 3. scalers: TRAINING data only ----------------------------------------------------------------
    train_rows = np.concatenate(train_rows)
    input_scaler, target_scaler = fit_scalers(train_rows, data["train"]["y"])
    save_scalers(input_scaler, target_scaler, P["input_scaler"], P["target_scaler"])

    # 4. write the .npz files ---------------------------------------------------------------------------
    keys = {"train": "train_npz", "val": "val_npz", "test": "test_npz"}
    for s in SPLITS:
        d = data[s]
        x_raw = d["x"].astype("float32")
        arrays = {
            "X": apply_input_scaler(input_scaler, d["x"]),                       # scaled inputs  (N, W, F)
            "y": target_scaler.transform(d["y"]).astype("float32"),              # scaled targets (N, 3)
            "X_raw": x_raw,                                                      # unscaled inputs
            "y_raw": d["y"].astype("float32"),                                   # future pressures in metres
            "meta_scenario_id": d["sid"].astype(str), "meta_scenario_type": d["stype"].astype(str),
            "meta_end_step": d["end"].astype("int16"),
            "meta_leak_onset": d["onset"], "meta_leak_active": d["active"],     # analysis only
        }
        if not all(np.isfinite(arrays[k]).all() for k in ("X", "y", "X_raw", "y_raw")):
            raise DataError(f"non-finite values in the {s} arrays")
        P[keys[s]].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(P[keys[s]], **arrays)

    manifest_out = manifest.copy()
    manifest_out.to_csv(P["split_manifest"], index=False)
    counts = {s: {"scenarios": int((manifest["split"] == s).sum()), "windows": int(len(data[s]["x"]))} for s in SPLITS}
    write_json({
        "input_features": features,
        "dropped_features": dropped,
        "target_names": target_cols,
        "window_steps": window, "horizon_steps": horizon,
        "window_hours": window * float(cfg["simulation"]["report_timestep_min"]) / 60.0,
        "horizon_minutes": horizon * float(cfg["simulation"]["report_timestep_min"]),
        "x_shape_per_window": [window, len(features)],
        "excluded_hidden_columns": list(ml.get("forbidden_inputs", [])),
        "split_seed": seed, "split_ratios": {s: float(sp[s]) for s in SPLITS},
        "stratified_by_scenario_type": bool(sp.get("stratify_by_type", True)),
        "counts": counts,
        "scalers": {"input": str(Path(cfg["paths"]["input_scaler"])), "target": str(Path(cfg["paths"]["target_scaler"])),
                    "fit_on": "training scenarios only"},
        "npz_arrays": {"X": "scaled inputs", "y": "scaled targets", "X_raw": "unscaled inputs",
                       "y_raw": "future pressures (m)", "meta_*": "analysis only, never model inputs"},
    }, P["feature_names"])

    return {"features": features, "dropped": dropped, "counts": counts, "manifest": manifest, "seed": seed,
            "window": window, "horizon": horizon, "paths": P,
            "shapes": {s: {"X": data[s]["x"].shape, "y": data[s]["y"].shape} for s in SPLITS},
            "n_scenarios": int(len(table))}


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    """Read one split .npz into a dictionary of arrays."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"{path} not found. Run: python scripts/build_ml_dataset.py")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
