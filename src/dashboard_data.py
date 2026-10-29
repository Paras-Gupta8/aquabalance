"""Data and model layer of the AquaBalance dashboard (no Streamlit code in here, so it can be tested).

Two modes:
  * full  - the user's own pipeline outputs exist (data/synthetic/*, data/processed/*, models/trained/*): use them.
  * demo  - they do not: use the small saved EPANET-generated example in data/sample/ (built by
            scripts/build_demo_sample.py from the real pipeline). Labelled "Synthetic EPANET simulation example."

All pressures and flows shown come from EPANET (via the saved time series) or from a model forecast of them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import add_time_features, resolve_input_features
from src.risk_engine import RiskAssessment, RiskSettings, assess_windows
from src.utils import resolve_path

SYNTHETIC_BANNER = ("Synthetic EPANET simulation. All values on this page come from simulated scenarios, "
                    "not from a real water utility.")
DEMO_LABEL = "Synthetic EPANET simulation example."
PROTOTYPE_STATEMENT = ("This is a simulation-based decision-support prototype, not autonomous real-world utility control.")
MODEL_ORDER = ["persistence", "random_forest", "gru", "lstm", "tcn"]
MODEL_NAMES = {"persistence": "Persistence", "random_forest": "Random Forest", "gru": "GRU", "lstm": "LSTM", "tcn": "TCN"}
RISK_TEXT = {"normal": "Normal", "high_pressure_risk": "High pressure", "low_pressure_risk": "Low pressure",
             "pressure_imbalance": "Pressure imbalance", "leak_like_pattern": "Leak-like pattern",
             "burst_like_pattern": "Burst-like pattern"}
SAMPLE_DIR = "data/sample"
MODEL_NOTES = {"persistence": "next pressure = current pressure; no learning",
               "random_forest": "forest on lag/rolling features; predicts the change from the current pressure",
               "gru": "recurrent neural network on the last 3 hours of sensor values",
               "lstm": "recurrent network (LSTM) with a parameter count close to the GRU",
               "tcn": "causal dilated 1-D convolution network with residual blocks, parameter count close to the GRU",
               "rule_based_reactive": "threshold check on current pressures; does not forecast"}


def dd_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


class DashboardError(RuntimeError):
    """A readable problem the dashboard can show to the user."""


# =============================================================================================== data source
@dataclass
class DataSource:
    mode: str                         # "full" | "demo"
    label: str
    timeseries: pd.DataFrame
    metadata: pd.DataFrame
    specs_path: Path
    comparison_path: Path | None
    forecasts: pd.DataFrame | None = None     # stored forecasts (demo mode only)
    notes: list = field(default_factory=list)


def _exists(*paths) -> bool:
    return all(Path(p).exists() for p in paths)


def load_source(cfg: dict, force_demo: bool = False, sample_dir: str | Path | None = None) -> DataSource:
    """Use the full pipeline outputs when they exist, otherwise the saved demo sample."""
    P = {k: resolve_path(v) for k, v in cfg["paths"].items()}
    sd = resolve_path(sample_dir or SAMPLE_DIR)
    if not force_demo and _exists(P["raw_timeseries"], P["scenario_specs"], P["scenario_metadata"]):
        ts = pd.read_csv(P["raw_timeseries"], parse_dates=["timestamp"])
        meta = pd.read_csv(P["scenario_metadata"])
        comp = P["model_comparison"] if Path(P["model_comparison"]).exists() else sd / "sample_model_comparison.csv"
        return DataSource("full", "Synthetic EPANET simulation results from your pipeline run.", _sorted(ts), meta,
                          P["scenario_specs"], comp if Path(comp).exists() else None)
    files = {k: sd / f for k, f in {"ts": "sample_timeseries.csv", "meta": "sample_scenario_metadata.csv",
                                    "specs": "sample_scenario_specs.jsonl", "fc": "sample_forecasts.csv",
                                    "cmp": "sample_model_comparison.csv"}.items()}
    missing = [f.name for k, f in files.items() if k != "cmp" and not f.exists()]
    if missing:
        raise DashboardError(
            "No simulation data found. Neither the pipeline outputs (data/synthetic/) nor the saved demo sample "
            f"(data/sample/: missing {', '.join(missing)}) exist. Run `git pull`, or run Phase 2 "
            "(python scripts/generate_synthetic_data.py).")
    ts = pd.read_csv(files["ts"], parse_dates=["timestamp"])
    return DataSource("demo", DEMO_LABEL, _sorted(ts), pd.read_csv(files["meta"]), files["specs"],
                      files["cmp"] if files["cmp"].exists() else None, pd.read_csv(files["fc"]),
                      ["Demo mode: showing the saved example in data/sample/ (trained models and the full dataset were not found)."])


def _sorted(ts: pd.DataFrame) -> pd.DataFrame:
    return ts.sort_values(["scenario_id", "timestamp"]).reset_index(drop=True)


def scenario_types(src: DataSource) -> list[str]:
    return sorted(src.metadata["scenario_type"].unique())


def scenario_ids(src: DataSource, scenario_type: str | None = None) -> list[str]:
    m = src.metadata
    if scenario_type and scenario_type != "All":
        m = m[m["scenario_type"] == scenario_type]
    return sorted(m["scenario_id"].astype(str))


def scenario_series(src: DataSource, sid: str) -> pd.DataFrame:
    df = src.timeseries[src.timeseries["scenario_id"] == sid].reset_index(drop=True)
    if df.empty:
        raise DashboardError(f"Scenario {sid} is not in the data.")
    return df


def scenario_type_of(src: DataSource, sid: str) -> str:
    return str(src.metadata.loc[src.metadata["scenario_id"] == sid, "scenario_type"].iloc[0])


def decision_steps(cfg: dict, n_rows: int) -> list[int]:
    """Valid 'now' rows: a full history window behind it and a next step to forecast."""
    w = int(cfg["ml"]["window_steps"])
    return list(range(w - 1, n_rows - int(cfg["ml"]["horizon_steps"])))


# =============================================================================================== windows
def feature_table(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Scenario rows with the hour features added, and the model input features actually used (tank dropped: no tank)."""
    framed = add_time_features(df, bool(cfg["ml"].get("hour_fractional", True)))
    feats, _ = resolve_input_features(framed, cfg["ml"])
    return framed, feats


def windows(framed: pd.DataFrame, feats: list[str], cfg: dict, steps: list[int]) -> np.ndarray:
    """Raw (unscaled) windows (N, W, F); window i ends at the 'now' row steps[i]."""
    w = int(cfg["ml"]["window_steps"])
    arr = framed[feats].to_numpy("float64")
    return np.stack([arr[s - w + 1:s + 1] for s in steps]) if steps else np.zeros((0, w, len(feats)))


# =============================================================================================== models
class ModelHub:
    """Forecast providers. 'live' = trained artifact loaded here; 'stored' = pre-computed forecast saved in the demo
    sample (real model output for test scenarios); 'unavailable' = neither. Persistence needs no artifact."""

    def __init__(self, cfg: dict, src: DataSource):
        self.cfg, self.src = cfg, src
        self.nodes = list(cfg["ml"]["target_nodes"])
        self._live: dict = {}
        self.info: dict = {}
        P = {k: resolve_path(v) for k, v in cfg["paths"].items()}
        self.info["persistence"] = {"mode": "live", "note": "next pressure = current pressure (no trained model needed)"}
        self._load_rf(P)
        for name in ("gru", "lstm", "tcn"):
            self._load_net(P, name)
        self._stored = set()
        if src.forecasts is not None:
            self._stored = set(src.forecasts["model"].unique())
            for name in self._stored:
                if self.info.get(name, {}).get("mode") != "live":
                    self.info[name] = {"mode": "stored", "note": "stored forecast from the trained model (demo sample)"}

    # ---- loading (never crash the dashboard because an optional artifact or library is missing)
    def _load_rf(self, P) -> None:
        try:
            from src.baseline_models import RandomForestBaseline
            if self.src.mode == "full" and Path(P["random_forest"]).exists():
                self._live["random_forest"] = RandomForestBaseline.load(P["random_forest"])
                self.info["random_forest"] = {"mode": "live", "note": "trained random forest loaded"}
                return
        except Exception as exc:                                                    # noqa: BLE001
            self.info["random_forest"] = {"mode": "unavailable", "note": f"could not load: {exc}"}
            return
        self.info.setdefault("random_forest", {"mode": "unavailable", "note": "models/trained/random_forest.joblib not found"})

    def _load_net(self, P, name: str) -> None:
        """GRU / LSTM / TCN checkpoint -> live predictor. Needs PyTorch; any failure just makes the model unavailable."""
        key = f"{name}_checkpoint"
        if not (self.src.mode == "full" and key in P and Path(P[key]).exists()):
            self.info[name] = {"mode": "unavailable", "note": f"models/trained/{name}_best.pt not found (train it first)"}
            return
        try:
            from src.gru_model import GRUPredictor            # needs PyTorch
            self._live[name] = GRUPredictor.from_checkpoint(P[key])
            self.info[name] = {"mode": "live", "note": f"trained {dd_name(name)} loaded"}
        except Exception as exc:                                                    # noqa: BLE001
            self.info[name] = {"mode": "unavailable", "note": f"could not load {dd_name(name)} ({type(exc).__name__}: {exc})"}

    # ---- queries
    def available(self) -> list[str]:
        return [m for m in MODEL_ORDER if self.info.get(m, {}).get("mode") in ("live", "stored")]

    def unavailable(self) -> dict:
        return {m: self.info[m]["note"] for m in MODEL_ORDER if self.info.get(m, {}).get("mode") == "unavailable"}

    def can_forecast(self, model: str, sid: str) -> bool:
        mode = self.info.get(model, {}).get("mode")
        if mode == "live":
            return True
        return mode == "stored" and bool(((self.src.forecasts["model"] == model) & (self.src.forecasts["scenario_id"] == sid)).any())

    def predict(self, model: str, sid: str, steps: list[int], x_raw: np.ndarray, feats: list[str]) -> np.ndarray:
        """Forecast (N, 3) in metres for the next 15-minute step, one row per decision step."""
        mode = self.info.get(model, {}).get("mode")
        if model == "persistence":
            cols = [feats.index(f"{n}_pressure") for n in self.nodes]
            return x_raw[:, -1, cols]
        if mode == "live":
            m = self._live[model]
            if getattr(m, "feature_names", feats) != feats:
                raise DashboardError(f"The saved {MODEL_NAMES[model]} expects different input features than the data has.")
            return np.asarray(m.predict(x_raw), "float64")
        if mode == "stored":
            f = self.src.forecasts
            f = f[(f["model"] == model) & (f["scenario_id"] == sid)].set_index("step")
            miss = [s for s in steps if s not in f.index]
            if miss:
                raise DashboardError(f"No stored {MODEL_NAMES[model]} forecast for {sid} at step {miss[0]}.")
            return f.loc[steps, [f"{n}_pressure" for n in self.nodes]].to_numpy("float64")
        raise DashboardError(f"{MODEL_NAMES.get(model, model)} is not available: {self.info.get(model, {}).get('note', '')}")


# =============================================================================================== assessment
@dataclass
class StepView:
    """Everything the dashboard shows for one scenario / time step / model."""
    sid: str
    step: int
    model: str
    now_ts: pd.Timestamp
    next_ts: pd.Timestamp
    forecast: dict
    actual_next: dict
    current_pressure: dict
    assessment: RiskAssessment          # predicted next-step risk
    current_assessment: RiskAssessment  # risk of the present state (forecast = current pressure)
    window: np.ndarray
    feats: list
    node_status: dict


def assess_step(cfg: dict, df: pd.DataFrame, step: int, forecast: np.ndarray, feats: list[str], x_window: np.ndarray):
    ks = RiskSettings.from_config(cfg)
    nodes = list(cfg["ml"]["target_nodes"])
    a = assess_windows(x_window[None], np.asarray(forecast)[None], feats, nodes, ks)[0]
    cols = [feats.index(f"{n}_pressure") for n in nodes]
    cur = assess_windows(x_window[None], x_window[None, -1, cols], feats, nodes, ks)[0]
    return a, cur


def node_status(cfg: dict, a: RiskAssessment, cur: RiskAssessment) -> dict:
    """green/amber/red per node J2-J4 (J1 is the hub: shown, not scored). Worst of the current and predicted state."""
    ks = RiskSettings.from_config(cfg)
    out = {}
    order = {"safe": 0, "warning": 1, "risk": 2}
    for n in ks.nodes:
        lo = ks.min_for(n)
        worst, why = "safe", "inside the configured safe range"
        for tag, asm in (("now", cur), ("next 15 min", a)):
            p = asm.predicted_pressure[n]
            s, r = "safe", None
            if p < lo:
                s, r = "risk", f"{p:.1f} m is below the {lo:.0f} m minimum ({tag})"
            elif p > ks.p_max_m:
                s, r = "risk", f"{p:.1f} m is above the {ks.p_max_m:.0f} m maximum ({tag})"
            elif p < lo + ks.near_limit_margin_m or p > ks.p_max_m - ks.near_limit_margin_m:
                s, r = "warning", f"{p:.1f} m is close to a limit ({tag})"
            if n == ks.critical_node and s != "risk" and p < lo + ks.critical_warning_margin_m:
                s, r = "warning", f"{p:.1f} m is within {ks.critical_warning_margin_m:.0f} m of the critical minimum ({tag})"
            if order[s] > order[worst]:
                worst, why = s, r
        out[n] = (worst, why)
    if a.pressure_imbalance_flag:
        pr = {n: a.predicted_pressure[n] for n in ks.nodes}
        for n in (max(pr, key=pr.get), min(pr, key=pr.get)):
            if out[n][0] == "safe":
                out[n] = ("warning", f"part of a pressure imbalance (spread {a.pressure_spread:.1f} m)")
    return out


def build_step_view(cfg: dict, src: DataSource, hub: ModelHub, sid: str, step: int, model: str) -> StepView:
    df = scenario_series(src, sid)
    framed, feats = feature_table(df, cfg)
    if step not in decision_steps(cfg, len(df)):
        raise DashboardError(f"Step {step} is outside the valid range for this scenario.")
    x = windows(framed, feats, cfg, [step])
    nodes = list(cfg["ml"]["target_nodes"])
    fc = hub.predict(model, sid, [step], x, feats)[0]
    a, cur = assess_step(cfg, df, step, fc, feats, x[0])
    nxt = df.iloc[step + int(cfg["ml"]["horizon_steps"])]
    return StepView(sid=sid, step=step, model=model, now_ts=df["timestamp"].iloc[step], next_ts=nxt["timestamp"],
                    forecast=dict(zip(nodes, map(float, fc))), actual_next={n: float(nxt[f"{n}_pressure"]) for n in nodes},
                    current_pressure={n: float(df[f"{n}_pressure"].iloc[step]) for n in ["J1"] + nodes},
                    assessment=a, current_assessment=cur, window=x[0], feats=feats, node_status=node_status(cfg, a, cur))


def notable_step(cfg: dict, src: DataSource, hub: ModelHub, sid: str, model: str) -> int:
    """The step with the highest predicted-risk severity (ties -> the earliest). Used to open a scenario at its event."""
    df = scenario_series(src, sid)
    framed, feats = feature_table(df, cfg)
    steps = decision_steps(cfg, len(df))
    x = windows(framed, feats, cfg, steps)
    fc = hub.predict(model, sid, steps, x, feats)
    ks, nodes = RiskSettings.from_config(cfg), list(cfg["ml"]["target_nodes"])
    sev = [a.severity_score for a in assess_windows(x, fc, feats, nodes, ks)]
    return steps[int(np.argmax(sev))]


# =============================================================================================== model comparison table
def comparison_table(path: Path | None) -> pd.DataFrame | None:
    """Rows: model, validation MAE, test MAE, test RMSE, risk macro F1 (test), inference time, notes."""
    if path is None or not Path(path).exists():
        return None
    d = pd.read_csv(path)
    rows = []
    for name in d["model"].unique():
        g = d[d["model"] == name].set_index("split")
        get = lambda col, sp: (float(g.loc[sp, col]) if sp in g.index and pd.notna(g.loc[sp, col]) else np.nan)  # noqa: E731
        rows.append({"model": name, "validation MAE (m)": get("MAE", "val"), "test MAE (m)": get("MAE", "test"),
                     "test RMSE (m)": get("RMSE", "test"), "risk macro F1 (test)": get("risk_macro_f1", "test"),
                     "inference (ms / 1000 windows)": get("inference_ms_per_1000_windows", "test"),
                     "notes": (next((str(x) for x in g["notes"].dropna()), "") if "notes" in g else "") or MODEL_NOTES.get(name, "")})
    return pd.DataFrame(rows)


def comparison_summary(t: pd.DataFrame) -> str:
    """One honest sentence computed from the table (no hard-coded claim about which model wins)."""
    f = t.dropna(subset=["test MAE (m)"])
    if f.empty:
        return ""
    best = f.loc[f["test MAE (m)"].idxmin()]
    base = f[f["model"] == "persistence"]
    s = f"Lowest test MAE on this synthetic test split: {best['model']} ({best['test MAE (m)']:.3f} m)."
    if "gru" in set(f["model"]) and best["model"] != "gru":
        gru = f[f["model"] == "gru"].iloc[0]
        s += f" The GRU ({gru['test MAE (m)']:.3f} m) does not have the lowest error here."
        if len(base) and gru["test MAE (m)"] < base.iloc[0]["test MAE (m)"]:
            s += " It does beat persistence."
    return s
