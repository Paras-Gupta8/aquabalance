"""Training loop for the GRU forecaster.

Adam + Huber (or MSE) loss, ReduceLROnPlateau, early stopping, best-checkpoint saving and restoring, deterministic seeds.

Rules this file keeps:
  * training uses the TRAIN split only; the validation split only chooses the checkpoint / stops training;
    the test split is never touched here;
  * the output normalisation (mean/std of the quantity the network predicts) comes from TRAIN rows only;
  * a NaN/infinite loss or input stops training with a clear error. It is never silently accepted.
"""
from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.gru_model import GRUForecaster, GRUPredictor, ModelError, architecture_from_config, pick_device


class TrainingError(RuntimeError):
    """Training cannot continue (non-finite loss, bad data, bad settings)."""


# --------------------------------------------------------------------------- reproducibility
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch. With deterministic=True torch also avoids non-deterministic kernels where it can."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- loss
def make_loss(name: str, huber_delta: float = 1.0) -> nn.Module:
    name = str(name).lower()
    if name == "huber":
        return nn.HuberLoss(delta=float(huber_delta))
    if name == "mse":
        return nn.MSELoss()
    raise TrainingError(f"unknown loss '{name}' (use 'huber' or 'mse')")


# --------------------------------------------------------------------------- targets
def pressure_columns(feature_names: list[str], nodes: list[str]) -> list[int]:
    try:
        return [feature_names.index(f"{n}_pressure") for n in nodes]
    except ValueError as exc:
        raise TrainingError(f"every target node needs its '<node>_pressure' input feature: {exc}") from None


def model_targets(x_raw: np.ndarray, y_raw: np.ndarray, cols: list[int], predict_delta: bool) -> np.ndarray:
    """What the network is trained to output (before standardisation): the change from now, or the pressure itself."""
    y = np.asarray(y_raw, "float64")
    return y - np.asarray(x_raw[:, -1, cols], "float64") if predict_delta else y


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise TrainingError(f"{name} contains NaN or infinite values; fix the data before training")


# --------------------------------------------------------------------------- one pass helpers
def _run_epoch(model, loader, loss_fn, optimizer, device, grad_clip: float) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        if not torch.isfinite(loss):
            raise TrainingError(f"training loss became {loss.item()} - stopping instead of accepting it. "
                                f"Check the inputs, lower the learning rate or the gradient clip.")
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


@torch.no_grad()
def _evaluate(model, x, y_z, loss_fn, device, batch: int = 2048):
    """Validation loss (standardised units) and the raw network output for all rows."""
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        outs.append(model(x[i:i + batch].to(device)).cpu())
    z = torch.cat(outs)
    loss = loss_fn(z, y_z).item()
    return loss, z.numpy()


# --------------------------------------------------------------------------- checkpoint
def save_checkpoint(path: str | Path, model: GRUForecaster, meta: dict) -> None:
    """Save weights + everything needed to rebuild a GRUPredictor (plain lists/numbers, loadable with weights_only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **meta}, tmp)
    os.replace(tmp, path)                               # never leave a half-written checkpoint behind


# --------------------------------------------------------------------------- main entry
def train_gru(gru_cfg: dict, data: dict, feature_names: list[str], nodes: list[str], input_mean, input_scale,
              checkpoint_path: str | Path, seed: int, log=print, max_epochs: int | None = None) -> dict:
    """Train on data['train'], pick the best checkpoint on data['val'], restore it and return a GRUPredictor.

    data[split] needs X (scaled windows), X_raw, y_raw. Returns {"predictor", "history", "best_epoch", ...}.
    """
    t_start = time.perf_counter()
    set_seed(seed, bool(gru_cfg.get("deterministic", True)))
    device = pick_device(gru_cfg.get("device", "auto"))
    predict_delta = bool(gru_cfg["predict_delta"])
    metric_name = str(gru_cfg.get("checkpoint_metric", "val_mae")).lower()
    if metric_name not in ("val_mae", "val_loss"):
        raise TrainingError("checkpoint_metric must be 'val_mae' or 'val_loss'")
    batch_size = int(gru_cfg["batch_size"])
    max_epochs = int(max_epochs if max_epochs is not None else gru_cfg["max_epochs"])
    patience = int(gru_cfg["early_stopping_patience"])
    min_delta = float(gru_cfg.get("min_delta", 0.0))
    cols = pressure_columns(feature_names, nodes)

    for split in ("train", "val"):
        for key in ("X", "X_raw", "y_raw"):
            _check_finite(f"{split}.{key}", data[split][key])
        if data[split]["X"].shape[2] != len(feature_names) or data[split]["y_raw"].shape[1] != len(nodes):
            raise TrainingError(f"{split} data does not match the feature list / target nodes; rebuild the dataset")

    # output normalisation from TRAIN rows only
    t_train = model_targets(data["train"]["X_raw"], data["train"]["y_raw"], cols, predict_delta)
    out_mean = t_train.mean(axis=0)
    out_std = np.where(t_train.std(axis=0) < 1e-9, 1.0, t_train.std(axis=0))

    def to_z(split):
        t = model_targets(data[split]["X_raw"], data[split]["y_raw"], cols, predict_delta)
        return torch.as_tensor(((t - out_mean) / out_std).astype("float32"))

    x_tr = torch.as_tensor(np.ascontiguousarray(data["train"]["X"]), dtype=torch.float32)
    x_va = torch.as_tensor(np.ascontiguousarray(data["val"]["X"]), dtype=torch.float32)
    z_tr, z_va = to_z("train"), to_z("val")
    gen = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(TensorDataset(x_tr, z_tr), batch_size=batch_size, shuffle=True, generator=gen)

    arch = architecture_from_config(gru_cfg)
    model = GRUForecaster(len(feature_names), len(nodes), **arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = make_loss(gru_cfg.get("loss", "huber"), gru_cfg.get("huber_delta", 1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(gru_cfg["learning_rate"]))
    sch = gru_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(sch.get("factor", 0.5)), patience=int(sch.get("patience", 4)),
        min_lr=float(sch.get("min_lr", 1e-5)))

    cur_va = np.asarray(data["val"]["X_raw"][:, -1, cols], "float64")
    y_va = np.asarray(data["val"]["y_raw"], "float64")
    meta = {"architecture": arch, "num_features": len(feature_names), "feature_names": list(feature_names),
            "target_nodes": list(nodes), "input_mean": [float(v) for v in np.asarray(input_mean).ravel()],
            "input_scale": [float(v) for v in np.asarray(input_scale).ravel()],
            "out_mean": [float(v) for v in out_mean], "out_std": [float(v) for v in out_std],
            "predict_delta": predict_delta, "window": int(data["train"]["X"].shape[1]), "seed": int(seed)}

    log(f"GRU on {device}: {n_params:,} parameters, {len(x_tr):,} train / {len(x_va):,} validation windows, "
        f"batch {batch_size}, loss {gru_cfg.get('loss', 'huber')}, lr {gru_cfg['learning_rate']}, "
        f"output = {'change from current pressure' if predict_delta else 'absolute pressure'}")
    log(f"{'epoch':>5} {'train_loss':>11} {'val_loss':>10} {'val_MAE_m':>10} {'lr':>9} {'sec':>6}")

    history, best_value, best_epoch, best_state, bad = [], np.inf, 0, None, 0
    stopped_early = False
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        tr_loss = _run_epoch(model, loader, loss_fn, optimizer, device, float(gru_cfg.get("grad_clip", 0.0)))
        va_loss, z_pred = _evaluate(model, x_va, z_va, loss_fn, device)
        if not np.isfinite(va_loss) or not np.isfinite(z_pred).all():
            raise TrainingError(f"validation loss/output became non-finite at epoch {epoch}; stopping")
        pred_m = z_pred.astype("float64") * out_std + out_mean
        pred_m = pred_m + cur_va if predict_delta else pred_m
        va_mae = float(np.abs(pred_m - y_va).mean())
        value = va_mae if metric_name == "val_mae" else va_loss
        lr = optimizer.param_groups[0]["lr"]
        improved = value < best_value - min_delta
        if improved:
            best_value, best_epoch, bad = value, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            save_checkpoint(checkpoint_path, model, {**meta, "best_epoch": epoch, "checkpoint_metric": metric_name,
                                                     "best_value": float(value), "val_mae_m": va_mae})
        else:
            bad += 1
        scheduler.step(value)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_mae_m": va_mae, "lr": lr,
                        "seconds": time.perf_counter() - t0, "is_best": bool(improved)})
        log(f"{epoch:>5} {tr_loss:>11.5f} {va_loss:>10.5f} {va_mae:>10.4f} {lr:>9.2e} "
            f"{history[-1]['seconds']:>6.1f}{'  <- best, checkpoint saved' if improved else ''}")
        if bad >= patience:
            stopped_early = True
            log(f"Early stopping: no improvement in {metric_name} for {patience} epochs "
                f"(best epoch {best_epoch}, {metric_name} {best_value:.5f}).")
            break
    if best_state is None:
        raise TrainingError("no epoch produced a usable checkpoint")

    model.load_state_dict(best_state)                       # restore the best weights before any test evaluation
    log(f"Restored best checkpoint from epoch {best_epoch} ({metric_name} {best_value:.5f}).")
    predictor = GRUPredictor(model, feature_names, nodes, np.asarray(input_mean).ravel(), np.asarray(input_scale).ravel(),
                             out_mean, out_std, predict_delta, device)
    return {"predictor": predictor, "history": history, "best_epoch": best_epoch, "best_value": float(best_value),
            "checkpoint_metric": metric_name, "stopped_early": stopped_early, "epochs_run": len(history),
            "n_parameters": int(n_params), "device": str(device), "seed": int(seed),
            "train_seconds": time.perf_counter() - t_start, "meta": meta}
