"""Feature engineering for the Phase 3 forecasting task.

Two kinds of features live here:

1. The model INPUT features per time step (sensor and control readings plus hour_sin/hour_cos).
2. The hand-made features for the random-forest baseline (lags, differences, rolling mean/std, flow and
   demand changes). They are computed ONLY from the input window, so they can never see the future.

Hidden simulation truth (leak_flag, leak_severity, scenario_type, ...) is never allowed as an input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureError(ValueError):
    """Raised when the configured input features are invalid or unsafe."""


TIME_FEATURES = ("hour_sin", "hour_cos")


# --------------------------------------------------------------------------- time features
def hour_of_day(timestamps: pd.Series, fractional: bool = True) -> np.ndarray:
    """Hour of day of each timestamp. Fractional: 06:45 -> 6.75 (15-minute steps stay distinguishable)."""
    ts = pd.to_datetime(timestamps)
    hour = ts.dt.hour.to_numpy(dtype="float64")
    if fractional:
        hour = hour + ts.dt.minute.to_numpy(dtype="float64") / 60.0 + ts.dt.second.to_numpy(dtype="float64") / 3600.0
    return hour


def add_time_features(df: pd.DataFrame, fractional: bool = True) -> pd.DataFrame:
    """Add hour_sin = sin(2*pi*hour/24) and hour_cos = cos(2*pi*hour/24), computed from each row's timestamp."""
    out = df.copy()
    hour = hour_of_day(out["timestamp"], fractional)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    return out


# --------------------------------------------------------------------------- input feature selection
def check_forbidden(features: list[str], forbidden: list[str]) -> None:
    bad = sorted(set(features) & set(forbidden))
    if bad:
        raise FeatureError(
            f"These columns are hidden simulation truth and may not be model inputs: {bad}. "
            "Remove them from ml.input_features in config/config.yaml.")


def resolve_input_features(df: pd.DataFrame, ml_cfg: dict) -> tuple[list[str], dict[str, str]]:
    """Return (features actually used, {dropped feature: reason}).

    A listed feature whose column is completely empty (tank_level: the network has no tank) is dropped and
    recorded. A feature that is only PARTLY empty is an error: values are never silently filled in.
    """
    wanted = list(ml_cfg["input_features"])
    check_forbidden(wanted, ml_cfg.get("forbidden_inputs", []))
    if len(set(wanted)) != len(wanted):
        raise FeatureError("ml.input_features contains duplicates.")
    policy = ml_cfg.get("unavailable_feature_policy", "drop")
    used: list[str] = []
    dropped: dict[str, str] = {}
    for f in wanted:
        if f in TIME_FEATURES:
            used.append(f)
            continue
        if f not in df.columns:
            raise FeatureError(f"Input feature '{f}' is not a column of the raw time series. "
                               f"Available columns: {sorted(df.columns)}")
        if df[f].isna().all():
            if policy == "error":
                raise FeatureError(f"Input feature '{f}' is completely empty in the data.")
            dropped[f] = "column is completely empty in the data (the network has no such sensor, e.g. no tank)"
            continue
        if df[f].isna().any():
            raise FeatureError(f"Input feature '{f}' has {int(df[f].isna().sum())} empty values. "
                               "Missing values are never filled in silently; fix the data or remove the feature.")
        used.append(f)
    if not used:
        raise FeatureError("No input features left after dropping empty columns.")
    return used, dropped


# --------------------------------------------------------------------------- random-forest features
def _idx(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def rf_feature_names(feature_names: list[str], rf_cfg: dict) -> list[str]:
    """Names of the hand-made features, in the exact column order produced by make_rf_features."""
    return _build_rf(None, feature_names, rf_cfg)[1]


def make_rf_features(x_raw: np.ndarray, feature_names: list[str], rf_cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Hand-made features for the random forest, from RAW (unscaled) windows of shape (N, W, F).

    Pure function of the input window: it has no access to targets, so it cannot leak the future.
    """
    if x_raw.ndim != 3 or x_raw.shape[2] != len(feature_names):
        raise FeatureError(f"expected windows of shape (N, W, {len(feature_names)}), got {x_raw.shape}")
    return _build_rf(x_raw, feature_names, rf_cfg)


def _build_rf(x: np.ndarray | None, feature_names: list[str], rf_cfg: dict) -> tuple[np.ndarray | None, list[str]]:
    ix = _idx(feature_names)
    window = x.shape[1] if x is not None else None
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, values) -> None:
        names.append(name)
        if x is not None:
            cols.append(np.asarray(values, dtype="float32"))

    def at(step_back: int, feat: str):          # value `step_back` steps before the current step (0 = now)
        return x[:, -1 - step_back, ix[feat]] if x is not None else None

    def check_step(k: int, what: str) -> None:
        if window is not None and k > window - 1:
            raise FeatureError(f"{what} {k} needs more history than the window ({window} steps).")

    pressures = [f for f in feature_names if f.endswith("_pressure")]
    flows = [f for f in feature_names if f.endswith("_flow")]
    demands = [f for f in feature_names if f.endswith("_demand")]
    controls = [f for f in ("reservoir_head", "pump_speed", "valve_setting", "tank_level") if f in ix]
    lags, diffs, rolls = rf_cfg["lags"], rf_cfg["diff_steps"], rf_cfg["rolling_windows"]

    for f in feature_names:                                     # 1. everything as measured now
        add(f"{f}_now", at(0, f) if x is not None else None)
    for k in lags:                                              # 2. last pressure values (lags)
        check_step(k, "lag")
        for f in pressures:
            add(f"{f}_lag{k}", at(k, f) if x is not None else None)
    for k in diffs:                                             # 3. pressure differences over time
        check_step(k, "diff step")
        for f in pressures:
            add(f"{f}_diff{k}", (at(0, f) - at(k, f)) if x is not None else None)
    junction_p = [f for f in pressures]                         # 4. pressure differences between junctions
    for a in range(len(junction_p)):
        for b in range(a + 1, len(junction_p)):
            fa, fb = junction_p[a], junction_p[b]
            add(f"{fa[:2]}_minus_{fb[:2]}_pressure", (at(0, fa) - at(0, fb)) if x is not None else None)
    branch = [f for f in pressures if f[:2] in ("J2", "J3", "J4")]
    if len(branch) >= 2:
        add("branch_spread_now", (np.max([at(0, f) for f in branch], axis=0) - np.min([at(0, f) for f in branch], axis=0))
            if x is not None else None)
    for w in rolls:                                             # 5. rolling mean / std of pressure
        check_step(w - 1, "rolling window - 1")
        for f in pressures:
            seg = x[:, -w:, ix[f]] if x is not None else None
            add(f"{f}_rollmean{w}", seg.mean(axis=1) if x is not None else None)
            add(f"{f}_rollstd{w}", seg.std(axis=1) if x is not None else None)
    for kind, group in (("flow", flows), ("demand", demands), ("control", controls)):   # 6. recent changes
        for k in diffs:
            check_step(k, "diff step")
            for f in group:
                add(f"{f}_change{k}", (at(0, f) - at(k, f)) if x is not None else None)
    matrix = np.stack(cols, axis=1) if x is not None else None
    return matrix, names
