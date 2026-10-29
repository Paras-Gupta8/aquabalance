"""Multi-output GRU for 15-minute-ahead pressure forecasting at J2, J3, J4 (decision support, not control).

GRUForecaster  - the torch network. Input (batch, history_window, num_features) of SCALED sensor/control features,
                 output (batch, 3) in "model units" (standardised change from the current pressure, or standardised
                 pressure, depending on predict_delta).
GRUPredictor   - wraps a trained network with everything needed to forecast from RAW windows: input scaling, output
                 de-standardisation and, if predict_delta, adding back the current pressure. It is rebuilt from the
                 single checkpoint file, so inference needs no other file.

The network only ever sees the sensor/control history windows. Scenario type, leak flags and leak severity are not
columns of those windows (see src/feature_engineering.py) and are never passed to the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.labels import pressure_risk_classes


class ModelError(RuntimeError):
    """A readable problem with the model, its inputs or a checkpoint."""


class GRUForecaster(nn.Module):
    """GRU(64) -> dropout -> GRU(32) -> last hidden state -> Dense(32)+ReLU -> Linear(3)."""

    def __init__(self, num_features: int, num_outputs: int = 3, hidden_1: int = 64, hidden_2: int = 32,
                 dropout: float = 0.20, dense: int = 32):
        super().__init__()
        if num_features < 1 or num_outputs < 1:
            raise ModelError("num_features and num_outputs must be at least 1")
        self.num_features, self.num_outputs = int(num_features), int(num_outputs)
        self.gru1 = nn.GRU(num_features, hidden_1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden_1, hidden_2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_2, dense), nn.ReLU(), nn.Linear(dense, num_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.num_features:
            raise ModelError(f"expected input (batch, history_window, {self.num_features}); got {tuple(x.shape)}")
        seq, _ = self.gru1(x)                  # (batch, window, hidden_1)
        seq = self.drop(seq)
        _, h_last = self.gru2(seq)             # h_last: (1, batch, hidden_2) = final hidden state of the second GRU
        return self.head(h_last[-1])           # (batch, 3)


def architecture_from_config(gru_cfg: dict) -> dict:
    return {"hidden_1": int(gru_cfg["hidden_1"]), "hidden_2": int(gru_cfg["hidden_2"]),
            "dropout": float(gru_cfg["dropout"]), "dense": int(gru_cfg["dense"])}


def pick_device(name: str = "auto") -> torch.device:
    """CPU unless the config asks for cuda, or 'auto' finds a GPU."""
    name = str(name).lower()
    if name == "cpu":
        return torch.device("cpu")
    if name in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if name == "cuda":
            raise ModelError("device 'cuda' requested but no GPU is available; use device: auto or cpu")
        return torch.device("cpu")
    raise ModelError(f"unknown device '{name}' (use auto, cpu or cuda)")


class GRUPredictor:
    """Forecast pressure (metres) from raw, unscaled windows of shape (N, W, F)."""

    def __init__(self, model: nn.Module, feature_names: list[str], target_nodes: list[str],
                 input_mean: np.ndarray, input_scale: np.ndarray, out_mean: np.ndarray, out_std: np.ndarray,
                 predict_delta: bool, device: torch.device | str = "cpu", batch_size: int = 2048, name: str = "gru"):
        self.name = name                       # "gru" by default; "lstm" / "tcn" for the Phase 8 comparison models
        self.model = model.to(device).eval()
        self.feature_names, self.target_nodes = list(feature_names), list(target_nodes)
        self.input_mean = np.asarray(input_mean, "float64")
        self.input_scale = np.asarray(input_scale, "float64")
        self.out_mean, self.out_std = np.asarray(out_mean, "float64"), np.asarray(out_std, "float64")
        self.predict_delta = bool(predict_delta)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.cols = []
        for n in self.target_nodes:
            if f"{n}_pressure" not in self.feature_names:
                raise ModelError(f"'{n}_pressure' must be an input feature (it is the 'current pressure')")
            self.cols.append(self.feature_names.index(f"{n}_pressure"))

    # ------------------------------------------------------------------ helpers
    def scale_inputs(self, x_raw: np.ndarray) -> np.ndarray:
        x_raw = np.asarray(x_raw)
        if x_raw.ndim != 3 or x_raw.shape[2] != len(self.feature_names):
            raise ModelError(f"expected raw windows (N, W, {len(self.feature_names)}); got {x_raw.shape}")
        if not np.isfinite(x_raw).all():
            raise ModelError("the input windows contain NaN or infinite values")
        return ((x_raw.astype("float64") - self.input_mean) / self.input_scale).astype("float32")

    def _to_metres(self, z: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
        out = z.astype("float64") * self.out_std + self.out_mean
        return out + np.asarray(x_raw[:, -1, self.cols], "float64") if self.predict_delta else out

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict_scaled(self, x_scaled: np.ndarray) -> np.ndarray:
        """Raw network output for already-scaled windows (N, W, F) -> (N, 3) model units."""
        self.model.eval()
        parts = []
        for i in range(0, len(x_scaled), self.batch_size):
            xb = torch.as_tensor(np.ascontiguousarray(x_scaled[i:i + self.batch_size]), dtype=torch.float32,
                                 device=self.device)
            parts.append(self.model(xb).cpu().numpy())
        return np.concatenate(parts) if parts else np.zeros((0, len(self.target_nodes)), "float32")

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """Forecast pressure in metres, shape (N, 3), from raw windows."""
        z = self.predict_scaled(self.scale_inputs(x_raw))
        if not np.isfinite(z).all():
            raise ModelError("the network produced NaN/infinite output")
        return self._to_metres(z, np.asarray(x_raw))

    def predict_risk_class(self, x_raw: np.ndarray, settings: dict) -> np.ndarray:
        """Risk class of the PREDICTED future pressure. Uses the forecast only, never any ground truth."""
        return pressure_risk_classes(self.predict(x_raw), self.target_nodes, settings)

    # ------------------------------------------------------------------ checkpoint
    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "GRUPredictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/train_gru.py")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        arch = ck["architecture"]
        mtype = ck.get("model_type", "gru")                # checkpoints written before Phase 8 are GRUs
        if mtype == "gru":
            model = GRUForecaster(ck["num_features"], len(ck["target_nodes"]), **arch)
        else:
            from src.sequence_models import build_model
            model = build_model(mtype, ck["num_features"], len(ck["target_nodes"]), arch)
        model.load_state_dict(ck["model_state"])
        return cls(model, ck["feature_names"], ck["target_nodes"], np.array(ck["input_mean"]),
                   np.array(ck["input_scale"]), np.array(ck["out_mean"]), np.array(ck["out_std"]),
                   ck["predict_delta"], device, name=mtype)
