"""Evaluation of the forecasting baselines: metrics, risk metrics, error analysis, figures, experiment runner.

Two kinds of numbers are reported for every model and split:

* pressure-forecast quality (MAE, RMSE, R2, per-node MAE) for models that predict pressure;
* risk quality: the predicted (or, for the rule-based baseline, current) pressures are turned into a risk class with
  the SAME configured thresholds as the true future pressures, then compared with macro precision / recall / F1 and
  a confusion matrix. Classes: normal, low_pressure_risk, high_pressure_risk, pressure_imbalance. Leak/burst events are
  not included because a pressure forecast cannot reveal a known simulation event.

R2 is high for every model because pressure levels differ a lot BETWEEN scenarios; judge the models by MAE and by how
much they improve on persistence (mae_skill_vs_persistence).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, r2_score  # noqa: E402

from src.baseline_models import PersistenceBaseline, RandomForestBaseline, RuleBasedBaseline  # noqa: E402
from src.dataset_builder import load_split  # noqa: E402
from src.labels import PRESSURE_RISK_CLASSES, label_settings, pressure_risk_classes  # noqa: E402
from src.preprocessing import DataError  # noqa: E402
from src.utils import resolve_path  # noqa: E402

# Chart colours: categorical slots 1-2 of the validated reference palette (blue, orange) + neutral ink.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LEAK_BAND = "#bdbab0"          # neutral band: marks the hidden leak period without competing with the series colours
MIN_Y_SPAN_M = 2.0             # smallest y-axis range drawn, so harmless sub-0.1 m wiggles are not magnified
SHORT = {"normal": "normal", "low_pressure_risk": "low pressure", "high_pressure_risk": "high pressure",
         "pressure_imbalance": "imbalance"}


# =============================================================================== metrics
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, nodes: list[str]) -> dict:
    """MAE / RMSE / R2 over all outputs plus per-node values. Shapes (N, K); errors are in metres."""
    y_true, y_pred = np.asarray(y_true, "float64"), np.asarray(y_pred, "float64")
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != len(nodes):
        raise ValueError(f"expected two arrays of shape (N, {len(nodes)}); got {y_true.shape} and {y_pred.shape}")
    err = y_pred - y_true
    out = {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
           "R2": float(r2_score(y_true, y_pred))}
    for i, n in enumerate(nodes):
        out[f"MAE_{n}"] = float(np.abs(err[:, i]).mean())
        out[f"RMSE_{n}"] = float(np.sqrt((err[:, i] ** 2).mean()))
        out[f"R2_{n}"] = float(r2_score(y_true[:, i], y_pred[:, i]))
    return out


def risk_metrics(true_cls, pred_cls, classes=PRESSURE_RISK_CLASSES) -> tuple[dict, np.ndarray]:
    """Macro precision / recall / F1 (over the classes that occur in truth or prediction), accuracy, confusion matrix.

    The confusion matrix always has all `classes` as rows (actual) and columns (predicted), in that order.
    """
    true_cls, pred_cls = np.asarray(true_cls), np.asarray(pred_cls)
    present = [c for c in classes if c in set(true_cls) | set(pred_cls)]
    p, r, f, _ = precision_recall_fscore_support(true_cls, pred_cls, labels=present, average="macro", zero_division=0)
    per_f1 = precision_recall_fscore_support(true_cls, pred_cls, labels=list(classes), average=None, zero_division=0)[2]
    out = {"risk_macro_precision": float(p), "risk_macro_recall": float(r), "risk_macro_f1": float(f),
           "risk_accuracy": float(accuracy_score(true_cls, pred_cls)), "risk_classes_in_macro": len(present)}
    for c, v in zip(classes, per_f1):
        out[f"risk_f1_{c}"] = float(v)
    return out, confusion_matrix(true_cls, pred_cls, labels=list(classes))


# =============================================================================== evaluation of all models
def _timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def evaluate_models(models: dict, data: dict, settings: dict, nodes: list[str], splits: list[str]) -> dict:
    """Score persistence, the rule-based baseline and the random forest on each split."""
    rows, preds, cms, classes_pred, classes_true = [], {}, {}, {}, {}
    pers, rule, rf = models["persistence"], models["rule_based_reactive"], models["random_forest"]
    for split in splits:
        x, y = data[split]["X_raw"], data[split]["y_raw"].astype("float64")
        true_cls = pressure_risk_classes(y, nodes, settings)
        classes_true[split] = true_cls
        for model in (pers, rule, rf):
            row = {"model": model.name, "split": split, "n_windows": len(x)}
            if model is rule:
                pred_cls, dt = _timed(rule.predict_risk_class, x)
                row["notes"] = "reactive status check on current pressures; does not forecast pressure"
                preds[(model.name, split)] = None
            else:
                yp, dt = _timed(model.predict, x)
                preds[(model.name, split)] = yp
                row.update(regression_metrics(y, yp, nodes))
                pred_cls = pressure_risk_classes(yp, nodes, settings)
                row["notes"] = ("next pressure = current pressure" if model is pers else
                                "multi-output forest on lag/difference/rolling features" +
                                (" predicting the change from current pressure" if rf.predict_delta else ""))
            rm, cm = risk_metrics(true_cls, pred_cls)
            row.update(rm)
            row["inference_ms_per_1000_windows"] = 1000.0 * dt / max(len(x), 1) * 1000.0
            cms[(model.name, split)] = cm
            classes_pred[(model.name, split)] = pred_cls
            rows.append(row)
    metrics = pd.DataFrame(rows)
    base = metrics[metrics["model"] == "persistence"].set_index("split")["MAE"]
    metrics["mae_skill_vs_persistence"] = [
        np.nan if pd.isna(r["MAE"]) else 1.0 - r["MAE"] / base[r["split"]] for _, r in metrics.iterrows()]
    first = ["model", "split", "n_windows", "MAE", "RMSE", "R2"] + [f"MAE_{n}" for n in nodes] + [
        "mae_skill_vs_persistence", "risk_macro_precision", "risk_macro_recall", "risk_macro_f1", "risk_accuracy",
        "inference_ms_per_1000_windows"]
    metrics = metrics[first + [c for c in metrics.columns if c not in first]]
    return {"metrics": metrics, "preds": preds, "cms": cms, "classes_pred": classes_pred, "classes_true": classes_true}


# =============================================================================== error analysis
def _err_row(model, split, kind, group, err_abs, risk_acc=np.nan):
    return {"model": model, "split": split, "group_kind": kind, "group": group, "n_windows": int(err_abs.size) if
            err_abs is not None else np.nan,
            "MAE": float(err_abs.mean()) if err_abs is not None else np.nan,
            "RMSE": float(np.sqrt((err_abs ** 2).mean())) if err_abs is not None else np.nan,
            "p95_abs_error": float(np.quantile(err_abs, 0.95)) if err_abs is not None else np.nan,
            "max_abs_error": float(err_abs.max()) if err_abs is not None else np.nan, "risk_accuracy": risk_acc}


def error_analysis(result: dict, data: dict, nodes: list[str], splits: list[str]) -> pd.DataFrame:
    """Where do the models fail? Errors by scenario family, by node, and at leak onset (hidden truth, analysis only)."""
    rows = []
    for split in splits:
        d = data[split]
        y = d["y_raw"].astype("float64")
        stype, onset, active = d["meta_scenario_type"], d["meta_leak_onset"], d["meta_leak_active"]
        for (name, sp), yp in result["preds"].items():
            if sp != split:
                continue
            correct = result["classes_pred"][(name, split)] == result["classes_true"][split]
            if yp is None:                                   # rule-based: risk accuracy only
                rows.append(_err_row(name, split, "all", "all", None, float(correct.mean())))
                for t in sorted(set(stype)):
                    rows.append(_err_row(name, split, "scenario_type", t, None, float(correct[stype == t].mean())))
                continue
            err = np.abs(yp - y)
            rows.append(_err_row(name, split, "all", "all", err.ravel(), float(correct.mean())))
            for i, n in enumerate(nodes):
                rows.append(_err_row(name, split, "node", n, err[:, i]))
            for t in sorted(set(stype)):
                m = stype == t
                rows.append(_err_row(name, split, "scenario_type", t, err[m].ravel(), float(correct[m].mean())))
            for label, m in (("window whose target step is the first leak step", onset),
                             ("window with leak active at target step", active),
                             ("all other windows", ~active)):
                if m.any():
                    rows.append(_err_row(name, split, "leak_context", label, err[m].ravel(), float(correct[m].mean())))
    return pd.DataFrame(rows)


# =============================================================================== figures
def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def pick_comparison_scenarios(test: dict, n: int, prefer=("burst_leak", "reduced_supply", "normal", "valve_misconfiguration")):
    """Deterministically choose n test scenarios, preferring interesting families (first ID of each)."""
    ids, types = test["meta_scenario_id"], test["meta_scenario_type"]
    chosen = []
    for fam in prefer:
        match = sorted(set(ids[types == fam]))
        if match and len(chosen) < n:
            chosen.append(match[0])
    for sid in sorted(set(ids)):
        if len(chosen) >= n:
            break
        if sid not in chosen:
            chosen.append(sid)
    return chosen[:n]


def plot_prediction_comparison(test: dict, preds: dict, nodes: list[str], step_minutes: float, horizon: int,
                               path: str | Path, n_scenarios: int = 3) -> Path:
    """Actual vs persistence vs random-forest pressure for a few test scenarios (rows) and nodes (columns)."""
    sids = pick_comparison_scenarios(test, n_scenarios)
    shown_leak = False
    fig, axes = plt.subplots(len(sids), len(nodes), figsize=(4.3 * len(nodes), 2.7 * len(sids) + 0.9),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, sid in enumerate(sids):
        m = test["meta_scenario_id"] == sid
        hours = (test["meta_end_step"][m].astype(float) + horizon) * step_minutes / 60.0
        stype = str(test["meta_scenario_type"][m][0])
        leak_active = test["meta_leak_active"][m].astype(bool)
        for c, node in enumerate(nodes):
            ax = axes[r, c]
            _style_axes(ax)
            ax.plot(hours, test["y_raw"][m][:, c], color=INK, lw=2.2, label="actual (EPANET)")
            ax.plot(hours, preds["persistence"][m][:, c], color=ORANGE, lw=1.6, ls=(0, (4, 2)), label="persistence")
            ax.plot(hours, preds["random_forest"][m][:, c], color=BLUE, lw=1.6, label="random forest")
            lo, hi = ax.get_ylim()                         # a flat line must not look like a big swing
            if hi - lo < MIN_Y_SPAN_M:
                mid = (hi + lo) / 2.0
                ax.set_ylim(mid - MIN_Y_SPAN_M / 2.0, mid + MIN_Y_SPAN_M / 2.0)
            if leak_active.any():                          # simulation truth, drawn for context only
                ax.fill_between(hours, 0, 1, where=leak_active, transform=ax.get_xaxis_transform(),
                                color=LEAK_BAND, alpha=0.55, lw=0, zorder=0)
                shown_leak = True
            if r == 0:
                ax.set_title(f"{node} pressure", fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(f"{sid}\n{stype.replace('_', ' ')}\npressure (m)", fontsize=8, color=INK)
            if r == len(sids) - 1:
                ax.set_xlabel("hour of day of the predicted step", fontsize=8, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if shown_leak:
        handles.append(Patch(facecolor=LEAK_BAND, alpha=0.55, label="leak active (simulation truth, not a model input)"))
        labels.append("leak active (simulation truth, not a model input)")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_confusion_matrices(cms: dict[str, np.ndarray], classes, path: str | Path, subtitle: str = "") -> Path:
    """Row-normalised confusion matrices (counts and % of each actual class) side by side."""
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    fig, axes = plt.subplots(1, len(cms), figsize=(5.6 * len(cms), 5.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    labels = [SHORT.get(c, c) for c in classes]
    for ax, (title, cm) in zip(axes[0], cms.items()):
        rows = cm.sum(axis=1, keepdims=True)
        frac = np.divide(cm, rows, out=np.zeros(cm.shape, dtype=float), where=rows > 0)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
        ax.set_facecolor(SURFACE)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{cm[i, j]}\n{frac[i, j]:.0%}" if rows[i, 0] > 0 else f"{cm[i, j]}\n–"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if frac[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(classes)), labels, fontsize=8, color=MUTED, rotation=20)
        ax.set_yticks(range(len(classes)), [f"{l}\n(n={int(n)})" for l, n in zip(labels, rows[:, 0])], fontsize=8, color=MUTED)
        ax.set_xlabel("risk class predicted for the next step", fontsize=9, color=INK)
        ax.set_ylabel("actual risk class of the next step", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    if subtitle:
        fig.text(0.01, 0.01, subtitle, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# =============================================================================== the whole experiment
def run_baseline_experiment(cfg: dict, paths: dict | None = None, n_estimators: int | None = None,
                            make_figures: bool = True) -> dict:
    """Train the random forest on the TRAIN split, evaluate all baselines, write metrics, analysis and figures."""
    ml = cfg["ml"]
    P = {k: resolve_path(v) for k, v in {**cfg["paths"], **(paths or {})}.items()}
    if not Path(P["feature_names"]).exists():
        raise DataError(f"{P['feature_names']} not found. Run: python scripts/build_ml_dataset.py")
    meta = json.loads(Path(P["feature_names"]).read_text(encoding="utf-8"))
    features, nodes = meta["input_features"], list(ml["target_nodes"])
    splits = list(ml["evaluation"]["splits_reported"])
    data = {s: load_split(P[f"{s}_npz"]) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        if data[s]["y_raw"].shape[1] != len(nodes) or data[s]["X_raw"].shape[2] != len(features):
            raise DataError(f"{s}.npz does not match feature_names.json / ml.target_nodes; rebuild the dataset")

    settings = label_settings(cfg)
    seed = int(cfg["project"]["seed"])
    pers = PersistenceBaseline(features, nodes)
    rule = RuleBasedBaseline(features, nodes, settings)
    rf = RandomForestBaseline(features, nodes, ml["random_forest"], seed, n_estimators)
    t0 = time.perf_counter()
    rf.fit(data["train"]["X_raw"], data["train"]["y_raw"].astype("float64"))      # TRAIN split only
    fit_seconds = time.perf_counter() - t0
    rf.save(P["random_forest"])

    result = evaluate_models({"persistence": pers, "rule_based_reactive": rule, "random_forest": rf},
                             data, settings, nodes, splits)
    metrics = result["metrics"]
    metrics["rf_fit_seconds"] = np.where(metrics["model"] == "random_forest", fit_seconds, np.nan)
    errors = error_analysis(result, data, nodes, splits)
    for key, frame in (("baseline_metrics", metrics), ("baseline_error_analysis", errors)):
        Path(P[key]).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(P[key], index=False, float_format="%.6g")

    if make_figures:
        step_min = float(cfg["simulation"]["report_timestep_min"])
        horizon = int(meta["horizon_steps"])
        plot_prediction_comparison(
            data["test"], {"persistence": result["preds"][("persistence", "test")],
                           "random_forest": result["preds"][("random_forest", "test")]},
            nodes, step_min, horizon, P["fig_prediction_comparison"], int(ml["evaluation"]["comparison_scenarios"]))
        pers_cm = result["cms"][("persistence", "test")]
        rule_cm = result["cms"][("rule_based_reactive", "test")]
        same = bool(np.array_equal(pers_cm, rule_cm))
        plot_confusion_matrices(
            {("Persistence = rule-based reactive (identical by construction)" if same else "Persistence"): pers_cm,
             "Random forest": result["cms"][("random_forest", "test")]},
            PRESSURE_RISK_CLASSES, P["fig_confusion_matrix"],
            "Test split, risk class from the 3 consumer junctions J2-J4 with the configured lab-prototype thresholds.")
    return {"metrics": metrics, "errors": errors, "model": rf, "paths": P, "fit_seconds": fit_seconds,
            "result": result, "data": data}


# =============================================================================================================
# Phase 4: GRU evaluation, comparison with persistence and random forest, figures, experiment runner
# =============================================================================================================
AQUA = "#1baf7a"                                   # categorical slot 3 (random forest in the 3-model charts)
MODEL_COLORS = {"persistence": ORANGE, "random_forest": AQUA, "gru": BLUE}
MODEL_LABELS = {"persistence": "Persistence", "random_forest": "Random forest", "gru": "GRU"}
# metric -> True when LOWER is better
METRIC_DIRECTION = {"MAE": True, "RMSE": True, "R2": False, "MAE_J2": True, "MAE_J3": True, "MAE_J4": True,
                    "risk_macro_precision": False, "risk_macro_recall": False, "risk_macro_f1": False}


def evaluate_forecasters(models: dict, data: dict, settings: dict, nodes: list[str], splits: list[str]) -> dict:
    """Score any pressure forecasters on each split. `models` maps name -> object with .predict(x_raw) -> (N, K) metres.

    The risk class of the PREDICTED pressure is compared with the risk class of the TRUE future EPANET pressure.
    The forecasters themselves never see the true pressure or any scenario label.
    """
    rows, preds, cms, classes_pred, classes_true = [], {}, {}, {}, {}
    for split in splits:
        x, y = data[split]["X_raw"], data[split]["y_raw"].astype("float64")
        true_cls = pressure_risk_classes(y, nodes, settings)
        classes_true[split] = true_cls
        for name, model in models.items():
            yp, dt = _timed(model.predict, x)
            yp = np.asarray(yp, "float64")
            if not np.isfinite(yp).all():
                raise ValueError(f"model '{name}' produced NaN/infinite predictions on the {split} split")
            row = {"model": name, "split": split, "n_windows": len(x)}
            row.update(regression_metrics(y, yp, nodes))
            pred_cls = pressure_risk_classes(yp, nodes, settings)
            rm, cm = risk_metrics(true_cls, pred_cls)
            row.update(rm)
            row["inference_ms_per_1000_windows"] = 1000.0 * dt / max(len(x), 1) * 1000.0
            rows.append(row)
            preds[(name, split)], cms[(name, split)], classes_pred[(name, split)] = yp, cm, pred_cls
    metrics = pd.DataFrame(rows)
    base = metrics[metrics["model"] == "persistence"].set_index("split")["MAE"] if "persistence" in models else None
    if base is not None:
        metrics["mae_skill_vs_persistence"] = [1.0 - r["MAE"] / base[r["split"]] for _, r in metrics.iterrows()]
    return {"metrics": metrics, "preds": preds, "cms": cms, "classes_pred": classes_pred, "classes_true": classes_true}


def metric_verdicts(metrics: pd.DataFrame, split: str = "test", model: str = "gru",
                    others=("persistence", "random_forest"), tol: float = 1e-9) -> dict:
    """Honest per-metric comparison of `model` against each other model on one split (no cherry-picking).

    Returns {other: {"better": [...], "worse": [...], "tied": [...], "improves_every_metric": bool, "detail": {...}}}.
    """
    m = metrics[metrics["split"] == split].set_index("model")
    out = {}
    for other in others:
        better, worse, tied, detail = [], [], [], {}
        for metric, lower_better in METRIC_DIRECTION.items():
            a, b = float(m.loc[model, metric]), float(m.loc[other, metric])
            diff = (b - a) if lower_better else (a - b)           # positive = `model` is better
            detail[metric] = {model: a, other: b, "model_better_by": diff}
            (better if diff > tol else worse if diff < -tol else tied).append(metric)
        out[other] = {"better": better, "worse": worse, "tied": tied,
                      "improves_every_metric": not worse and not tied and bool(better), "detail": detail}
    return out


def model_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per model and split with the headline columns, plus GRU-vs-model MAE change on each split."""
    cols = ["model", "split", "n_windows", "MAE", "RMSE", "R2", "MAE_J2", "MAE_J3", "MAE_J4",
            "mae_skill_vs_persistence", "risk_macro_precision", "risk_macro_recall", "risk_macro_f1", "risk_accuracy",
            "inference_ms_per_1000_windows"]
    order = {"persistence": 0, "random_forest": 1, "gru": 2}
    table = metrics[[c for c in cols if c in metrics.columns]].copy()
    table["_o"] = table["model"].map(order)
    table["_s"] = table["split"].map({"train": 0, "val": 1, "test": 2})
    return table.sort_values(["_s", "_o"]).drop(columns=["_o", "_s"]).reset_index(drop=True)


def gru_error_by_scenario_type(result: dict, data: dict, nodes: list[str], split: str = "test") -> pd.DataFrame:
    """MAE / RMSE / risk accuracy per scenario family (the family is used for ANALYSIS only, never as an input)."""
    d, y = data[split], data[split]["y_raw"].astype("float64")
    stype = d["meta_scenario_type"]
    rows = []
    for t in ["all"] + sorted(set(stype)):
        mask = np.ones(len(y), bool) if t == "all" else stype == t
        row = {"split": split, "scenario_type": t, "n_windows": int(mask.sum()),
               "n_scenarios": int(len(set(d["meta_scenario_id"][mask])))}
        for name in ("persistence", "random_forest", "gru"):
            if (name, split) not in result["preds"]:
                continue
            err = np.abs(result["preds"][(name, split)][mask] - y[mask])
            row[f"{name}_MAE"] = float(err.mean())
            row[f"{name}_RMSE"] = float(np.sqrt((err ** 2).mean()))
            row[f"{name}_risk_accuracy"] = float((result["classes_pred"][(name, split)][mask] ==
                                                   result["classes_true"][split][mask]).mean())
        if "gru_MAE" in row:
            row["gru_better_than_persistence"] = bool(row["gru_MAE"] < row["persistence_MAE"])
            row["gru_better_than_random_forest"] = bool(row["gru_MAE"] < row["random_forest_MAE"])
        rows.append(row)
    return pd.DataFrame(rows)


def single_window_latency_ms(predictor, x_raw: np.ndarray, repeats: int = 200) -> float:
    """Median time to forecast ONE window (what a live dashboard call would cost), in milliseconds."""
    one = x_raw[:1]
    predictor.predict(one)                                    # warm-up
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        predictor.predict(x_raw[i % len(x_raw):i % len(x_raw) + 1])
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)


# --------------------------------------------------------------------------- Phase 4 figures
def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_training_history(history: list[dict], best_epoch: int, path: str | Path, metric_name: str = "val_mae") -> Path:
    """Left: training and validation loss. Right: validation MAE in metres. Separate panels (units differ)."""
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        _style_axes(ax)
        ax.set_xlabel("epoch", fontsize=9, color=INK)
        ax.axvline(best_epoch, color=MUTED, lw=1.0, ls=":")
    a1.plot(ep, [h["train_loss"] for h in history], color=ORANGE, lw=2, label="training loss")
    a1.plot(ep, [h["val_loss"] for h in history], color=BLUE, lw=2, label="validation loss")
    a1.set_yscale("log")
    a1.set_ylabel("loss (standardised units, log scale)", fontsize=9, color=INK)
    a1.set_title("Loss per epoch", fontsize=10, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=9)
    a2.plot(ep, [h["val_mae_m"] for h in history], color=BLUE, lw=2, label="validation MAE")
    a2.scatter([best_epoch], [history[best_epoch - 1]["val_mae_m"]], color=BLUE, s=55, zorder=5,
               edgecolor=SURFACE, linewidth=2)
    a2.annotate(f"best epoch {best_epoch}\n{history[best_epoch - 1]['val_mae_m']:.3f} m", (best_epoch, history[best_epoch - 1]["val_mae_m"]),
                textcoords="offset points", xytext=(12, -34), fontsize=9, color=INK)
    a2.set_ylabel("validation MAE (metres)", fontsize=9, color=INK)
    a2.set_title("Validation error per epoch", fontsize=10, color=INK, loc="left")
    fig.text(0.01, 0.01, f"Dotted line = epoch of the saved checkpoint (chosen by {metric_name}).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, path)


def plot_gru_prediction_vs_actual(test: dict, preds: dict, nodes: list[str], step_minutes: float, horizon: int,
                                  path: str | Path, n_scenarios: int = 3) -> Path:
    """Actual EPANET pressure vs persistence vs GRU for a few test scenarios (rows) and nodes (columns)."""
    sids = pick_comparison_scenarios(test, n_scenarios)
    fig, axes = plt.subplots(len(sids), len(nodes), figsize=(4.3 * len(nodes), 2.7 * len(sids) + 0.9),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    shown_leak = False
    for r, sid in enumerate(sids):
        m = test["meta_scenario_id"] == sid
        hours = (test["meta_end_step"][m].astype(float) + horizon) * step_minutes / 60.0
        leak_active = test["meta_leak_active"][m].astype(bool)
        stype = str(test["meta_scenario_type"][m][0])
        for c, node in enumerate(nodes):
            ax = axes[r, c]
            _style_axes(ax)
            ax.plot(hours, test["y_raw"][m][:, c], color=INK, lw=2.2, label="actual (EPANET)")
            ax.plot(hours, preds["persistence"][m][:, c], color=ORANGE, lw=1.5, ls=(0, (4, 2)), label="persistence")
            ax.plot(hours, preds["gru"][m][:, c], color=BLUE, lw=1.7, label="GRU")
            lo, hi = ax.get_ylim()
            if hi - lo < MIN_Y_SPAN_M:
                mid = (hi + lo) / 2.0
                ax.set_ylim(mid - MIN_Y_SPAN_M / 2.0, mid + MIN_Y_SPAN_M / 2.0)
            if leak_active.any():
                ax.fill_between(hours, 0, 1, where=leak_active, transform=ax.get_xaxis_transform(),
                                color=LEAK_BAND, alpha=0.55, lw=0, zorder=0)
                shown_leak = True
            if r == 0:
                ax.set_title(f"{node} pressure, 15 min ahead", fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(f"{sid}\n{stype.replace('_', ' ')}\npressure (m)", fontsize=8, color=INK)
            if r == len(sids) - 1:
                ax.set_xlabel("hour of day of the predicted step", fontsize=8, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if shown_leak:
        handles.append(Patch(facecolor=LEAK_BAND, alpha=0.55))
        labels.append("leak active (simulation truth, not a model input)")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def plot_per_node_error(metrics: pd.DataFrame, nodes: list[str], path: str | Path, split: str = "test") -> Path:
    """Grouped bars: MAE per node for each model. Bars carry value labels (relief for the light aqua)."""
    m = metrics[metrics["split"] == split].set_index("model")
    names = [n for n in ("persistence", "random_forest", "gru") if n in m.index]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    w = 0.8 / len(names)
    x = np.arange(len(nodes))
    for i, n in enumerate(names):
        vals = [float(m.loc[n, f"MAE_{node}"]) for node in nodes]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w * 0.88, color=MODEL_COLORS[n], label=MODEL_LABELS[n])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{n}" for n in nodes], fontsize=10, color=INK)
    ax.set_ylabel("mean absolute error (metres)", fontsize=9, color=INK)
    ax.set_title(f"Per-node forecast error, {split} split (lower is better)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- the whole Phase 4 experiment
def run_gru_experiment(cfg: dict, paths: dict | None = None, max_epochs: int | None = None, make_figures: bool = True,
                       log=print, seed: int | None = None, overrides: dict | None = None) -> dict:
    """Train the GRU on the TRAIN split, restore the best checkpoint, evaluate everything, write every Phase 4 file."""
    from src.gru_model import GRUPredictor
    from src.preprocessing import load_scalers, write_json
    from src.training import train_gru

    ml = cfg["ml"]
    gru_cfg = {**cfg["gru"], **(overrides or {})}
    P = {k: resolve_path(v) for k, v in {**cfg["paths"], **(paths or {})}.items()}
    if not Path(P["feature_names"]).exists():
        raise DataError(f"{P['feature_names']} not found. Run: python scripts/build_ml_dataset.py")
    if not Path(P["random_forest"]).exists():
        raise DataError(f"{P['random_forest']} not found. Run: python scripts/train_baselines.py (the GRU is compared with it)")
    meta = json.loads(Path(P["feature_names"]).read_text(encoding="utf-8"))
    features, nodes = meta["input_features"], list(ml["target_nodes"])
    data = {s: load_split(P[f"{s}_npz"]) for s in ("train", "val", "test")}
    for s in data:
        if data[s]["y_raw"].shape[1] != len(nodes) or data[s]["X_raw"].shape[2] != len(features):
            raise DataError(f"{s}.npz does not match feature_names.json / ml.target_nodes; rebuild the dataset")
    in_scaler, _ = load_scalers(P["input_scaler"], P["target_scaler"])
    seed = int(gru_cfg["seed"] if gru_cfg.get("seed") is not None else cfg["project"]["seed"]) if seed is None else int(seed)

    fit = train_gru(gru_cfg, data, features, nodes, in_scaler.mean_, in_scaler.scale_, P["gru_checkpoint"], seed,
                    log=log, max_epochs=max_epochs)
    # the model used for every number below is RELOADED from the saved checkpoint file (proves the file works)
    predictor = GRUPredictor.from_checkpoint(P["gru_checkpoint"], device="cpu")
    pers = PersistenceBaseline(features, nodes)
    rf = RandomForestBaseline.load(P["random_forest"])
    settings = label_settings(cfg)
    splits = ["train", "val", "test"]
    result = evaluate_forecasters({"persistence": pers, "random_forest": rf, "gru": predictor}, data, settings, nodes, splits)
    metrics = result["metrics"]
    table = model_comparison_table(metrics)
    verdict = metric_verdicts(metrics, "test")
    by_type = gru_error_by_scenario_type(result, data, nodes, "test")
    latency = single_window_latency_ms(predictor, data["test"]["X_raw"])

    config_out = {"task": "pressure forecasting and decision support: previous window of sensor/control history -> "
                          "pressure at the target nodes `horizon` step(s) ahead (simulation-based, not autonomous control)",
                  "window_steps": meta["window_steps"], "horizon_steps": meta["horizon_steps"],
                  "target_nodes": nodes, "input_features": features, "architecture": fit["meta"]["architecture"],
                  "predict_delta": fit["meta"]["predict_delta"],
                  "output_normalisation": {"mean": fit["meta"]["out_mean"], "std": fit["meta"]["out_std"]},
                  "training": {k: gru_cfg[k] for k in ("loss", "huber_delta", "learning_rate", "batch_size", "max_epochs",
                                                       "early_stopping_patience", "checkpoint_metric", "grad_clip",
                                                       "scheduler", "deterministic") if k in gru_cfg},
                  "seed": seed, "device": fit["device"], "n_parameters": fit["n_parameters"],
                  "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
                  "stopped_early": fit["stopped_early"], "checkpoint": str(P["gru_checkpoint"]),
                  "forbidden_inputs_not_used": ml["forbidden_inputs"]}
    import torch
    config_out["torch_version"] = torch.__version__
    write_json(config_out, P["gru_config"])

    def block(split):
        row = metrics[(metrics["model"] == "gru") & (metrics["split"] == split)].iloc[0].to_dict()
        return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in row.items()
                if k not in ("model",)}
    gm = {"metrics": {s: block(s) for s in splits},
          "risk_confusion_matrix_test": {"classes": list(PRESSURE_RISK_CLASSES),
                                         "rows_actual_cols_predicted": result["cms"][("gru", "test")].tolist()},
          "training": {k: fit[k] for k in ("best_epoch", "best_value", "checkpoint_metric", "stopped_early",
                                           "epochs_run", "n_parameters", "device", "seed", "train_seconds")},
          "inference": {"ms_per_1000_windows_batch": float(block("test")["inference_ms_per_1000_windows"]),
                        "ms_single_window_median": latency},
          "history": fit["history"], "verdict_test_split": verdict,
          "note": "risk labels are derived from the (predicted) future pressure with the configured thresholds; "
                  "true labels come from EPANET future pressure and are used for scoring only."}
    write_json(gm, P["gru_metrics"])
    for path, frame in ((P["model_comparison"], table), (P["gru_error_by_scenario_type"], by_type)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.6g")

    if make_figures:
        step_min, horizon = float(cfg["simulation"]["report_timestep_min"]), int(meta["horizon_steps"])
        plot_training_history(fit["history"], fit["best_epoch"], P["fig_gru_history"], fit["checkpoint_metric"])
        plot_gru_prediction_vs_actual(
            data["test"], {"persistence": result["preds"][("persistence", "test")], "gru": result["preds"][("gru", "test")]},
            nodes, step_min, horizon, P["fig_gru_prediction"], int(ml["evaluation"]["comparison_scenarios"]))
        plot_per_node_error(metrics, nodes, P["fig_gru_per_node"], "test")
        plot_confusion_matrices({"GRU (risk class of the predicted pressure)": result["cms"][("gru", "test")]},
                                PRESSURE_RISK_CLASSES, P["fig_gru_confusion"],
                                "Test split, risk class from J2-J4 with the configured lab-prototype thresholds.")
    return {"fit": fit, "predictor": predictor, "metrics": metrics, "comparison": table, "verdict": verdict,
            "by_type": by_type, "paths": P, "result": result, "data": data, "gru_metrics": gm, "latency_ms": latency}
