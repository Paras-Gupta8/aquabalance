"""Phase 8: LSTM and TCN forecasters, built to be compared fairly with the Phase 4 GRU.

All three networks take the same input (batch, history_window, num_features) of SCALED sensor/control features and give the
same output (batch, 3): the standardised change in pressure at J2, J3, J4 over the next 15 minutes. They are trained with the
same loop (src/training.py), the same splits, loss, optimiser, seed and early stopping, and scored with the same metrics.
Their sizes are chosen so the parameter counts are close (GRU 26,691; LSTM about 26.7k; TCN about 26k).

LSTMForecaster - same shape as the GRU: LSTM(56) -> dropout -> LSTM(26) -> last hidden state -> Dense(32)+ReLU -> Linear(3).
TCNForecaster  - temporal convolutional network: residual blocks of two causal, dilated Conv1d layers (dilations 1, 2, 4),
                 so each output only uses the past (causal) and the receptive field (29 steps) covers the 12-step window.
                 The last time step's features feed the same dense head.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.gru_model import GRUForecaster, ModelError, architecture_from_config as gru_architecture

MODEL_TYPES = ("gru", "lstm", "tcn")


class LSTMForecaster(nn.Module):
    """LSTM(hidden_1) -> dropout -> LSTM(hidden_2) -> last hidden state -> Dense+ReLU -> Linear(outputs)."""

    def __init__(self, num_features: int, num_outputs: int = 3, hidden_1: int = 56, hidden_2: int = 26,
                 dropout: float = 0.20, dense: int = 32):
        super().__init__()
        if num_features < 1 or num_outputs < 1:
            raise ModelError("num_features and num_outputs must be at least 1")
        self.num_features, self.num_outputs = int(num_features), int(num_outputs)
        self.lstm1 = nn.LSTM(num_features, hidden_1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_2, dense), nn.ReLU(), nn.Linear(dense, num_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.num_features:
            raise ModelError(f"expected input (batch, history_window, {self.num_features}); got {tuple(x.shape)}")
        seq, _ = self.lstm1(x)
        seq = self.drop(seq)
        _, (h_last, _c) = self.lstm2(seq)
        return self.head(h_last[-1])


class CausalConv1d(nn.Module):
    """Conv1d that only looks at the past: the input is padded on the LEFT by (kernel-1)*dilation, never on the right."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (batch, channels, time)
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """Residual block: causal dilated conv -> ReLU -> dropout -> causal dilated conv -> ReLU -> dropout, plus a skip path."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.c1 = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.c2 = CausalConv1d(c_out, c_out, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(F.relu(self.c1(x)))
        y = self.drop(F.relu(self.c2(y)))
        return F.relu(y + self.skip(x))


class TCNForecaster(nn.Module):
    """Stack of residual TCN blocks, then the same Dense+ReLU -> Linear head applied to the LAST time step."""

    def __init__(self, num_features: int, num_outputs: int = 3, channels: int = 38, kernel_size: int = 3,
                 dilations: tuple = (1, 2, 4), dropout: float = 0.10, dense: int = 32):
        super().__init__()
        if num_features < 1 or num_outputs < 1:
            raise ModelError("num_features and num_outputs must be at least 1")
        self.num_features, self.num_outputs = int(num_features), int(num_outputs)
        blocks, c_in = [], num_features
        for d in dilations:
            blocks.append(TCNBlock(c_in, channels, int(kernel_size), int(d), dropout))
            c_in = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(channels, dense), nn.ReLU(), nn.Linear(dense, num_outputs))
        self.receptive_field = 1 + 2 * (int(kernel_size) - 1) * sum(int(d) for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.num_features:
            raise ModelError(f"expected input (batch, history_window, {self.num_features}); got {tuple(x.shape)}")
        h = self.blocks(x.transpose(1, 2))          # (batch, channels, time)
        return self.head(h[:, :, -1])               # features at the newest time step


def architecture_from_config(model_type: str, model_cfg: dict) -> dict:
    """Architecture arguments for `model_type` from its config section (gru / lstm / tcn)."""
    if model_type == "gru":
        return gru_architecture(model_cfg)
    if model_type == "lstm":
        return {"hidden_1": int(model_cfg["hidden_1"]), "hidden_2": int(model_cfg["hidden_2"]),
                "dropout": float(model_cfg["dropout"]), "dense": int(model_cfg["dense"])}
    if model_type == "tcn":
        return {"channels": int(model_cfg["channels"]), "kernel_size": int(model_cfg["kernel_size"]),
                "dilations": [int(d) for d in model_cfg["dilations"]], "dropout": float(model_cfg["dropout"]),
                "dense": int(model_cfg["dense"])}
    raise ModelError(f"unknown model type '{model_type}' (use one of {MODEL_TYPES})")


def build_model(model_type: str, num_features: int, num_outputs: int, arch: dict) -> nn.Module:
    if model_type == "gru":
        return GRUForecaster(num_features, num_outputs, **arch)
    if model_type == "lstm":
        return LSTMForecaster(num_features, num_outputs, **arch)
    if model_type == "tcn":
        return TCNForecaster(num_features, num_outputs, **{**arch, "dilations": tuple(arch["dilations"])})
    raise ModelError(f"unknown model type '{model_type}' (use one of {MODEL_TYPES})")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
