"""Run hydraulic simulations with EPANET (through WNTR's EpanetSimulator).

EPANET is the only solver used. There is deliberately no fallback simulator:
if EPANET cannot run, this module raises an error or marks the scenario as failed
instead of producing numbers.

Phase 1 helpers : run_hydraulic_simulation, get_junction_pressures, pressure_summary
Phase 2 helpers : build_scenario_model, run_scenario, extract_timeseries, validate_timeseries
"""
from __future__ import annotations

import math
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wntr
from wntr.network import LinkStatus
from wntr.network.controls import Control, ControlAction, SimTimeCondition


class SimulationError(Exception):
    """Raised with a readable message when an EPANET run fails."""


# =============================================================================== Phase 1 helpers
def run_hydraulic_simulation(inp_path: str | Path, duration_hours: float | None = None):
    """Run EPANET on an .inp file. Returns (model, results).

    duration_hours overrides the duration stored in the file (useful for short checks).
    """
    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise SimulationError(f"INP file not found: {inp_path}. Run: python scripts/build_network.py")
    try:
        wn = wntr.network.WaterNetworkModel(str(inp_path))
        if duration_hours is not None:
            wn.options.time.duration = int(round(duration_hours * 3600))
        with tempfile.TemporaryDirectory(prefix="aquabalance_") as tmp:  # EPANET scratch files live here
            results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "run"))
    except Exception as exc:
        raise SimulationError(f"EPANET simulation failed for {inp_path}: {type(exc).__name__}: {exc}") from exc
    return wn, results


def get_junction_pressures(results, junction_ids: list[str]) -> pd.DataFrame:
    """Pressure (m) per timestep for the given junctions, straight from EPANET."""
    pressure = results.node["pressure"]
    missing = [j for j in junction_ids if j not in pressure.columns]
    if missing:
        raise SimulationError(f"EPANET results contain no pressure for: {missing}")
    df = pressure[junction_ids]
    if df.empty or df.isna().any().any():
        raise SimulationError("EPANET returned empty or NaN pressures.")
    return df


def pressure_summary(pressures: pd.DataFrame, p_min: float, p_max: float) -> pd.DataFrame:
    """Per-junction min/mean/max and the share of timesteps outside [p_min, p_max]."""
    pressures = pressures.astype("float64")  # EPANET returns float32; avoids 110.800003-style output
    return pd.DataFrame({
        "min_m": pressures.min(),
        "mean_m": pressures.mean(),
        "max_m": pressures.max(),
        "pct_below_p_min": (pressures < p_min).mean() * 100,
        "pct_above_p_max": (pressures > p_max).mean() * 100,
    }).round(1)


# =============================================================================== Phase 2: scenarios
LEAK_NODE = "LK"    # helper junction that carries the leak emitter
LEAK_PIPE = "LKP"   # short helper pipe that is opened/closed by EPANET time controls

PUMP_STATUS_CODES = {0: "closed/off", 1: "open/on"}
VALVE_STATUS_CODES = {0: "closed", 1: "open (not regulating)", 2: "active (regulating)"}

# Column order of data/synthetic/raw_timeseries.csv
RAW_COLUMNS = [
    "scenario_id", "timestamp", "scenario_type",
    "J1_pressure", "J2_pressure", "J3_pressure", "J4_pressure",
    "P1_flow", "P2_flow", "P3_flow", "P4_flow",
    "J2_demand", "J3_demand", "J4_demand",
    "tank_level", "reservoir_head", "pump_speed", "pump_status",
    "valve_setting", "valve_status", "leak_flag", "leak_severity",
]
# tank_level is intentionally empty: the Phase 1 network has no tank (nothing is invented to fill it).
EMPTY_BY_DESIGN = ("tank_level",)


def _ids(cfg: dict) -> dict:
    net = cfg["network"]
    return {
        "reservoir": net["reservoir"]["id"],
        "pump": net["pump"]["id"],
        "valve": net["valve"]["id"],
        "pressure_nodes": [j["id"] for j in net["junctions"] if j.get("role") in ("hub", "demand")],
        "demand_nodes": [j["id"] for j in net["junctions"] if j.get("role") == "demand"],
        "pipes": [p["id"] for p in net["pipes"]],
    }


def build_scenario_model(spec: dict, cfg: dict, base_inp: str | Path):
    """Load the base EPANET network and change ONLY the settings listed in the scenario spec."""
    ids = _ids(cfg)
    sim = cfg["simulation"]
    wn = wntr.network.WaterNetworkModel(str(base_inp))

    if str(sim.get("demand_model", "DDA")).upper() == "PDD":
        h = wn.options.hydraulic
        h.demand_model = "PDD"
        h.minimum_pressure = float(sim["pdd_minimum_pressure_m"])
        h.required_pressure = float(sim["pdd_required_pressure_m"])
        h.pressure_exponent = float(sim["pdd_pressure_exponent"])

    # source head (constant value x slow smooth pattern)
    res = wn.get_node(ids["reservoir"])
    res.base_head = float(spec["reservoir_head"])
    wn.add_pattern("HEADP", list(map(float, spec["head_pattern"])))
    res.head_pattern_name = "HEADP"

    # pump speed (constant, optionally shaped by a pattern for degradation / trips)
    pump = wn.get_link(ids["pump"])
    pump.speed_timeseries.base_value = float(spec["pump_speed"])
    if spec.get("pump_pattern") is not None:
        # In EPANET a pump speed pattern value IS the speed setting (not a multiplier of the base speed),
        # so spec["pump_pattern"] holds absolute relative speeds.
        wn.add_pattern("SPEEDP", list(map(float, spec["pump_pattern"])))
        pump.speed_pattern_name = "SPEEDP"

    # valve setting (PRV downstream pressure target, metres)
    wn.get_link(ids["valve"]).initial_setting = float(spec["valve_setting"])

    # demand: smooth time-of-day multipliers per consumer junction
    for node_id, mult in spec["demand_multipliers"].items():
        junction = wn.get_node(node_id)
        base = junction.demand_timeseries_list[0].base_value
        wn.add_pattern(f"DEM_{node_id}", list(map(float, mult)))
        junction.demand_timeseries_list.clear()
        junction.add_demand(base, f"DEM_{node_id}")

    # partial blockage: shrink the open cross-section of one pipe
    if spec.get("blockage"):
        pipe = wn.get_link(spec["blockage"]["pipe"])
        pipe.diameter = pipe.diameter * math.sqrt(1.0 - float(spec["blockage"]["factor"]))

    # leak: pressure-dependent emitter behind a pipe that EPANET opens/closes at set times
    if spec.get("leak"):
        leak = spec["leak"]
        host = wn.get_node(leak["node"])
        wn.add_junction(LEAK_NODE, base_demand=0.0, elevation=host.elevation, coordinates=host.coordinates)
        wn.get_node(LEAK_NODE).emitter_coefficient = float(leak["emitter_coefficient"])
        wn.add_pipe(LEAK_PIPE, leak["node"], LEAK_NODE, length=1.0, diameter=0.3, roughness=130.0,
                    initial_status="Closed")
        helper = wn.get_link(LEAK_PIPE)
        wn.add_control("leak_on", Control(SimTimeCondition(wn, "=", int(leak["start_s"])),
                                          ControlAction(helper, "status", LinkStatus.Open)))
        if int(leak["end_s"]) < wn.options.time.duration:
            wn.add_control("leak_off", Control(SimTimeCondition(wn, "=", int(leak["end_s"])),
                                               ControlAction(helper, "status", LinkStatus.Closed)))
    return wn


# --------------------------------------------------------------------------- EPANET report warnings
_WARN_RULES = (
    ("pump_beyond_max_flow", re.compile(r"exceeds maximum flow", re.I)),
    ("negative_pressures", re.compile(r"negative pressures", re.I)),
    ("disconnected", re.compile(r"disconnected", re.I)),
    ("unbalanced", re.compile(r"unbalanced|ill-conditioned|did not converge", re.I)),
)


def parse_report_warnings(report_text: str, has_leak: bool) -> dict[str, int]:
    """Count EPANET report warnings by type. Warnings caused only by the closed leak helper pipe are
    counted separately as 'expected_leak_helper' (EPANET notes the closed helper node as disconnected)."""
    counts: dict[str, int] = {}
    for line in report_text.splitlines():
        if "WARNING" not in line.upper() and "ERROR" not in line.upper():
            continue
        kind = next((name for name, rx in _WARN_RULES if rx.search(line)), "other")
        if has_leak and kind == "disconnected" and re.search(rf"\b{LEAK_NODE}P?\b", line):
            kind = "expected_leak_helper"
        elif "ERROR" in line.upper():
            kind = "error"
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _format_warnings(counts: dict[str, int]) -> str:
    return "; ".join(f"{k} x{v}" for k, v in sorted(counts.items()) if k != "expected_leak_helper")


# --------------------------------------------------------------------------- extraction and validation
def extract_timeseries(wn, results, spec: dict, cfg: dict) -> pd.DataFrame:
    """Turn EPANET results into the raw time-series frame (96 rows, units: m, L/s, relative speed)."""
    ids = _ids(cfg)
    steps = int(cfg["scenarios"]["steps"])
    node, link = results.node, results.link
    seconds = np.asarray(node["pressure"].index)
    keep = seconds < wn.options.time.duration          # drop the extra row at t = 24:00
    t = seconds[keep]
    if len(t) != steps:
        raise SimulationError(f"expected {steps} time steps but EPANET returned {len(t)}")

    def n(table, col, scale=1.0):
        return np.round(np.asarray(node[table][col], dtype="float64")[keep] * scale, 4)

    def l(table, col, scale=1.0):
        return np.round(np.asarray(link[table][col], dtype="float64")[keep] * scale, 4)

    df = pd.DataFrame({
        "scenario_id": spec["scenario_id"],
        "timestamp": pd.Timestamp(cfg["simulation"]["start_datetime"]) + pd.to_timedelta(t, unit="s"),
        "scenario_type": spec["scenario_type"],
    })
    for j in ids["pressure_nodes"]:
        df[f"{j}_pressure"] = n("pressure", j)
    for p in ids["pipes"]:
        df[f"{p}_flow"] = l("flowrate", p, 1000.0)           # m3/s -> L/s
    for j in ids["demand_nodes"]:
        df[f"{j}_demand"] = n("demand", j, 1000.0) + 0.0      # delivered demand, L/s (+0.0 removes -0.0)
    df["tank_level"] = np.nan                                  # no tank in this network
    df["reservoir_head"] = n("head", ids["reservoir"])
    df["pump_speed"] = l("setting", ids["pump"])
    df["pump_status"] = l("status", ids["pump"]).astype(int)
    df["valve_setting"] = l("setting", ids["valve"])
    df["valve_status"] = l("status", ids["valve"]).astype(int)

    leak = spec.get("leak")
    if leak:
        df["leak_flag"] = ((t >= leak["start_s"]) & (t < leak["end_s"])).astype(int)
        df["leak_severity"] = np.clip(l("flowrate", LEAK_PIPE, 1000.0), 0.0, None)
    else:
        df["leak_flag"] = 0
        df["leak_severity"] = 0.0
    return df[RAW_COLUMNS]


def validate_timeseries(df: pd.DataFrame, cfg: dict, spec: dict | None = None) -> list[str]:
    """Return a list of problems. An empty list means every row is acceptable."""
    problems: list[str] = []
    v = cfg["validation"]
    if len(df) != int(cfg["scenarios"]["steps"]):
        problems.append(f"{len(df)} rows instead of {cfg['scenarios']['steps']}")
    must_be_filled = [c for c in RAW_COLUMNS if c not in EMPTY_BY_DESIGN and c not in ("scenario_id", "scenario_type")]
    numeric = df[[c for c in must_be_filled if c != "timestamp"]]
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        problems.append("NaN or infinite value in a column that must be filled")
        return problems
    step = pd.Timedelta(minutes=float(cfg["simulation"]["report_timestep_min"]))
    if not (df["timestamp"].diff().dropna() == step).all():
        problems.append("timestamps are not evenly spaced at the report timestep")
    for c in [c for c in df.columns if c.endswith("_pressure")]:
        lo, hi = float(df[c].min()), float(df[c].max())
        if lo < v["min_valid_pressure_m"] or hi > v["max_valid_pressure_m"]:
            problems.append(f"{c} outside valid range [{v['min_valid_pressure_m']}, {v['max_valid_pressure_m']}] m "
                            f"(min {lo:.1f}, max {hi:.1f})")
    if (df[[c for c in df.columns if c.endswith("_demand")]] < -1e-6).any().any():
        problems.append("negative delivered demand")
    if not df["pump_status"].isin(PUMP_STATUS_CODES).all():
        problems.append("unknown pump status code")
    if not df["valve_status"].isin(VALVE_STATUS_CODES).all():
        problems.append("unknown valve status code")
    if (df["pump_speed"] < 0).any():
        problems.append("negative pump speed")
    if spec and spec.get("leak"):
        active = df["leak_flag"] == 1
        if active.any() and not (df.loc[active, "leak_severity"] > 0).any():
            problems.append("a leak was scheduled but EPANET produced no leak outflow")
    return problems


# --------------------------------------------------------------------------- one scenario
def run_scenario(spec: dict, cfg: dict, base_inp: str | Path) -> dict:
    """Simulate one scenario in EPANET. Never returns made-up data.

    Returns a dict: status ('ok' | 'ok_with_warnings' | 'rejected' | 'failed'), data (DataFrame or None),
    warnings (dict), warning_text, error (str), runtime_s.
    """
    started = time.time()
    out = {"status": "failed", "data": None, "warnings": {}, "warning_text": "", "error": "", "runtime_s": 0.0}
    has_leak = bool(spec.get("leak"))
    try:
        wn = build_scenario_model(spec, cfg, base_inp)
        with tempfile.TemporaryDirectory(prefix="aquabalance_scn_") as tmp:
            prefix = os.path.join(tmp, "run")
            results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefix)
            rpt = Path(prefix + ".rpt")
            report_text = rpt.read_text(errors="ignore") if rpt.exists() else ""
        df = extract_timeseries(wn, results, spec, cfg)
    except Exception as exc:  # EPANET/WNTR error: report it, never hide it
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["runtime_s"] = round(time.time() - started, 3)
        return out

    warnings = parse_report_warnings(report_text, has_leak)
    problems = validate_timeseries(df, cfg, spec)
    # EPANET's generic "negative pressures" warning also fires for the closed leak helper node.
    # It is only expected if every real junction is non-negative.
    if has_leak and warnings.get("negative_pressures"):
        real_min = float(df[[c for c in df.columns if c.endswith("_pressure")]].min().min())
        if real_min >= 0.0:
            warnings["expected_leak_helper"] = warnings.get("expected_leak_helper", 0) + warnings.pop("negative_pressures")
    out["warnings"] = warnings
    out["warning_text"] = _format_warnings(warnings)
    out["runtime_s"] = round(time.time() - started, 3)
    if "error" in warnings or "unbalanced" in warnings:
        out["status"], out["error"] = "rejected", "EPANET reported hydraulic errors: " + out["warning_text"]
    elif problems:
        out["status"], out["error"] = "rejected", "; ".join(problems)
    else:
        out["status"] = "ok_with_warnings" if out["warning_text"] else "ok"
        out["data"] = df
    return out
