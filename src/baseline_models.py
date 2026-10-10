"""Baseline models for the 15-minute-ahead pressure forecast (no deep learning).

1. PersistenceBaseline   - next pressure = current pressure (J2, J3, J4).
2. RuleBasedBaseline     - REACTIVE engineering rule: judges the pressures measured NOW against the thresholds.
                           It does not forecast anything, so it cannot warn before a violation happens.
3. RandomForestBaseline  - scikit-learn random forest on hand-made lag / difference / rolling features.

Every model takes RAW (unscaled) windows of shape (N, W, F). Targets are pressures in metres, shape (N, 3).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.feature_engineering import make_rf_features
from src.labels import PRESSURE_RISK_CLASSES, pressure_risk_classes, pressure_risk_flags


def _pressure_columns(feature_names: list[str], target_nodes: list[str]) -> list[int]:
    cols = []
    for n in target_nodes:
        name = f"{n}_pressure"
        if name not in feature_names:
            raise ValueError(f"'{name}' must be an input feature to use it as the 'current pressure'.")
        cols.append(feature_names.index(name))
    return cols


class PersistenceBaseline:
    """Next pressure equals the pressure measured now. The simplest possible forecast."""

    name = "persistence"

    def __init__(self, feature_names: list[str], target_nodes: list[str]):
        self.target_nodes = list(target_nodes)
        self.cols = _pressure_columns(feature_names, target_nodes)

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        return np.asarray(x_raw[:, -1, self.cols], dtype="float64")


class RuleBasedBaseline:
    """Reactive threshold rules on the CURRENT pressures (low / high / branch imbalance).

    This is a status check, not a forecast: it labels the situation as it is now and assumes the next step looks the
    same. By construction its risk class equals the risk class of the persistence forecast.
    """

    name = "rule_based_reactive"
    description = "reactive: judges current pressures against thresholds; does not predict"

    def __init__(self, feature_names: list[str], target_nodes: list[str], settings: dict):
        self.target_nodes = list(target_nodes)
        self.settings = settings
        self.cols = _pressure_columns(feature_names, target_nodes)

    def current_pressures(self, x_raw: np.ndarray) -> np.ndarray:
        return np.asarray(x_raw[:, -1, self.cols], dtype="float64")

    def flags(self, x_raw: np.ndarray):
        return pressure_risk_flags(self.current_pressures(x_raw), self.target_nodes, self.settings)

    def predict_risk_class(self, x_raw: np.ndarray) -> np.ndarray:
        return pressure_risk_classes(self.current_pressures(x_raw), self.target_nodes, self.settings)


class RandomForestBaseline:
    """Multi-output random forest on engineered window features.

    With predict_delta=True the forest learns the CHANGE from the current pressure, and the output is added back to the
    current pressure, so it is still a forecast of the pressure itself. Trees cannot extrapolate, so predicting the
    change is much more robust than predicting absolute pressure for scenarios whose pressure level is unusual.
    """

    name = "random_forest"

    def __init__(self, feature_names: list[str], target_nodes: list[str], rf_cfg: dict, seed: int,
                 n_estimators: int | None = None):
        self.feature_names = list(feature_names)
        self.target_nodes = list(target_nodes)
        self.rf_cfg = dict(rf_cfg)
        self.seed = int(seed)
        self.predict_delta = bool(rf_cfg.get("predict_delta", True))
        self.cols = _pressure_columns(feature_names, target_nodes)
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators if n_estimators is not None else rf_cfg["n_estimators"]),
            min_samples_leaf=int(rf_cfg["min_samples_leaf"]), max_features=rf_cfg["max_features"],
            n_jobs=rf_cfg.get("n_jobs", -1), random_state=self.seed)
        self.rf_feature_names: list[str] = []

    def _features(self, x_raw: np.ndarray) -> np.ndarray:
        feats, names = make_rf_features(x_raw, self.feature_names, self.rf_cfg)
        self.rf_feature_names = names
        return feats

    def _current(self, x_raw: np.ndarray) -> np.ndarray:
        return np.asarray(x_raw[:, -1, self.cols], dtype="float64")

    def fit(self, x_raw: np.ndarray, y_raw: np.ndarray) -> "RandomForestBaseline":
        target = y_raw - self._current(x_raw) if self.predict_delta else y_raw
        self.model.fit(self._features(x_raw), target)
        return self

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        out = self.model.predict(self._features(x_raw))
        return out + self._current(x_raw) if self.predict_delta else out

    def feature_importances(self) -> dict[str, float]:
        return dict(sorted(zip(self.rf_feature_names, self.model.feature_importances_), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------ saving / loading
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names, "target_nodes": self.target_nodes,
                     "rf_cfg": self.rf_cfg, "seed": self.seed, "predict_delta": self.predict_delta,
                     "rf_feature_names": self.rf_feature_names}, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestBaseline":
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/train_baselines.py")
        d = joblib.load(path)
        obj = cls(d["feature_names"], d["target_nodes"], d["rf_cfg"], d["seed"])
        obj.model, obj.predict_delta, obj.rf_feature_names = d["model"], d["predict_delta"], d["rf_feature_names"]
        return obj


__all__ = ["PersistenceBaseline", "RuleBasedBaseline", "RandomForestBaseline", "PRESSURE_RISK_CLASSES"]
