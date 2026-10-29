"""Loading raw data, splitting by scenario, and fitting scalers (training data only).

The rules that keep the dataset leakage-safe:
  * the split is made on WHOLE scenarios (never on rows or windows);
  * scalers are fitted on training scenarios only and then merely APPLIED to validation/test.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.simulation import RAW_COLUMNS


class DataError(Exception):
    """Raised with a readable message when the raw data is missing or malformed."""


# --------------------------------------------------------------------------- loading
def load_raw_timeseries(path: str | Path, steps: int | None = None) -> pd.DataFrame:
    """Read data/synthetic/raw_timeseries.csv, sorted by scenario and time, with structural checks."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"{path} not found. Create it first: python scripts/generate_synthetic_data.py --mode standard")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{path} is missing columns {missing}. Regenerate it with scripts/generate_synthetic_data.py.")
    if df.empty:
        raise DataError(f"{path} contains no rows.")
    df = df.sort_values(["scenario_id", "timestamp"], kind="stable").reset_index(drop=True)
    if df.duplicated(["scenario_id", "timestamp"]).any():
        raise DataError("duplicate (scenario_id, timestamp) rows in the raw time series")
    sizes = df.groupby("scenario_id").size()
    if steps is not None and not (sizes == steps).all():
        raise DataError(f"every scenario must have exactly {steps} rows; found sizes {sorted(sizes.unique())}")
    if sizes.nunique() != 1:
        raise DataError(f"scenarios have different numbers of rows: {sorted(sizes.unique())}")
    return df


def scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario: scenario_id and scenario_type (type is used for stratifying the split only)."""
    types = df.groupby("scenario_id")["scenario_type"].nunique()
    if (types > 1).any():
        raise DataError(f"scenarios with more than one scenario_type: {list(types[types > 1].index)}")
    return df.groupby("scenario_id", as_index=False)["scenario_type"].first()


# --------------------------------------------------------------------------- splitting
def _targets(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Whole-scenario counts per split that add up to n (largest remainder)."""
    raw = {s: ratios[s] * n for s in ("train", "val", "test")}
    counts = {s: int(np.floor(v)) for s, v in raw.items()}
    for s in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True):
        if sum(counts.values()) >= n:
            break
        counts[s] += 1
    return counts


def split_scenarios(table: pd.DataFrame, ratios: dict[str, float], seed: int, stratify: bool = True) -> pd.DataFrame:
    """Assign every scenario ID to train / val / test. Deterministic for a given seed.

    With stratify=True each scenario_type is spread over the splits in proportion (so rare types such as
    blockage appear in validation and test as well). The unit that is split is always the whole scenario.
    """
    total = sum(ratios[s] for s in ("train", "val", "test"))
    if abs(total - 1.0) > 1e-6:
        raise DataError(f"split ratios must sum to 1.0 (got {total:.4f})")
    if len(table) < 3:
        raise DataError("need at least 3 scenarios to make train/val/test splits")
    rng = np.random.default_rng(int(seed))
    ids_by_type = {t: list(g["scenario_id"]) for t, g in table.groupby("scenario_type" if stratify else lambda _: "all")}
    if not stratify:
        ids_by_type = {"all": list(table["scenario_id"])}
    for t in ids_by_type:                                         # reproducible shuffle inside each stratum
        ids_by_type[t] = [ids_by_type[t][i] for i in rng.permutation(len(ids_by_type[t]))]

    goal = _targets(len(table), ratios)
    quota = {(t, s): len(ids) * ratios[s] for t, ids in ids_by_type.items() for s in ("train", "val", "test")}
    take = {k: int(np.floor(v)) for k, v in quota.items()}
    left_split = {s: goal[s] - sum(take[(t, s)] for t in ids_by_type) for s in goal}
    left_type = {t: len(ids) - sum(take[(t, s)] for s in goal) for t, ids in ids_by_type.items()}
    for (t, s) in sorted(quota, key=lambda k: (quota[k] - take[k], k[1] != "train"), reverse=True):
        if left_split[s] > 0 and left_type[t] > 0:
            take[(t, s)] += 1; left_split[s] -= 1; left_type[t] -= 1
    for t in ids_by_type:                                         # safety net (cannot normally trigger)
        for s in goal:
            while left_type[t] > 0 and left_split[s] > 0:
                take[(t, s)] += 1; left_split[s] -= 1; left_type[t] -= 1

    rows = []
    for t, ids in ids_by_type.items():
        pos = 0
        for s in ("train", "val", "test"):
            for sid in ids[pos:pos + take[(t, s)]]:
                rows.append((sid, s))
            pos += take[(t, s)]
    manifest = pd.DataFrame(rows, columns=["scenario_id", "split"])
    manifest = table.merge(manifest, on="scenario_id").sort_values("scenario_id").reset_index(drop=True)
    assert_disjoint(manifest, expected_ids=set(table["scenario_id"]))
    return manifest[["scenario_id", "split", "scenario_type"]]


def assert_disjoint(manifest: pd.DataFrame, expected_ids: set[str] | None = None) -> None:
    """Hard check: every scenario is in exactly one split; no overlap; nothing lost."""
    if manifest["scenario_id"].duplicated().any():
        dup = manifest.loc[manifest["scenario_id"].duplicated(), "scenario_id"].tolist()
        raise DataError(f"scenario IDs assigned more than once: {dup[:5]}")
    groups = {s: set(g["scenario_id"]) for s, g in manifest.groupby("split")}
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = groups[names[i]] & groups[names[j]]
            if both:
                raise DataError(f"scenario overlap between {names[i]} and {names[j]}: {sorted(both)[:5]}")
    if expected_ids is not None and set(manifest["scenario_id"]) != expected_ids:
        raise DataError("split manifest does not cover exactly the scenarios in the data")
    if not {"train", "val", "test"} <= set(groups):
        raise DataError(f"a split is empty: found {sorted(groups)}")


# --------------------------------------------------------------------------- scalers
def fit_scalers(train_rows: np.ndarray, train_targets: np.ndarray) -> tuple[StandardScaler, StandardScaler]:
    """Fit input and target scalers on TRAINING data only.

    train_rows: (n_rows, F) feature values of the training scenarios (one row per time step).
    train_targets: (n_windows, 3) future pressures of the training windows.
    """
    input_scaler = StandardScaler().fit(train_rows)
    target_scaler = StandardScaler().fit(train_targets)
    return input_scaler, target_scaler


def apply_input_scaler(scaler: StandardScaler, x_raw: np.ndarray) -> np.ndarray:
    n, w, f = x_raw.shape
    return scaler.transform(x_raw.reshape(-1, f)).reshape(n, w, f).astype("float32")


def save_scalers(input_scaler, target_scaler, input_path: str | Path, target_path: str | Path) -> None:
    for obj, p in ((input_scaler, input_path), (target_scaler, target_path)):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, p)


def load_scalers(input_path: str | Path, target_path: str | Path):
    for p in (input_path, target_path):
        if not Path(p).exists():
            raise DataError(f"{p} not found. Run: python scripts/build_ml_dataset.py")
    return joblib.load(input_path), joblib.load(target_path)


def write_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
