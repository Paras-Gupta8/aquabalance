"""Phase 2: generate synthetic training scenarios with EPANET (via WNTR).

How it works
------------
1. A scenario "spec" is sampled: a dictionary of INPUT settings (source head, pump speed/pattern, valve
   setting, smooth demand multipliers, leak, blockage). Random numbers choose these settings ONLY.
2. The spec is applied to the Phase 1 EPANET network and EPANET (wntr.sim.EpanetSimulator) solves it.
3. Every pressure, flow, demand, head, speed and status value in the dataset is read from the EPANET result.
   No pressure or flow value is ever drawn from a random generator.

Reproducibility: scenario i, attempt a uses seed = SeedSequence([master_seed, i, a]); the full spec of every
scenario is written to data/synthetic/scenario_specs.jsonl so any scenario can be re-simulated exactly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import wntr

from src.labels import label_settings, label_timeseries, scenario_label
from src.simulation import (PUMP_STATUS_CODES, RAW_COLUMNS, VALVE_STATUS_CODES, _ids, run_scenario)
from src.utils import get_logger, resolve_path

FAMILIES = (
    "normal", "high_demand_stress", "high_source_pressure", "valve_misconfiguration", "reduced_supply",
    "small_leak", "burst_leak", "pump_degradation", "partial_blockage",
)
# Grouping used only to compare with the target distribution in the project brief.
FAMILY_GROUPS = {
    "normal": ("normal",),
    "high-demand / low-pressure": ("high_demand_stress", "reduced_supply", "pump_degradation", "partial_blockage"),
    "high-pressure": ("high_source_pressure",),
    "valve imbalance": ("valve_misconfiguration",),
    "leak-like / burst-like": ("small_leak", "burst_leak"),
}
METADATA_COLUMNS = [
    "scenario_id", "scenario_type", "random_seed", "reservoir_head", "tank_initial_level", "pump_speed",
    "pump_schedule", "valve_setting", "leak_node", "leak_start_time", "leak_end_time", "leak_severity",
    "blockage_pipe", "blockage_factor", "demand_profile_id", "notes",
]
LEAK_REFERENCE_PRESSURE_M = 50.0   # leak_severity (L/s) is the leak outflow at this pressure
STEP_S = 900                       # 15 minutes


# =============================================================================== small helpers
def allocate_counts(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Split n scenarios across families (largest-remainder rounding, every family at least once if n allows)."""
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"scenarios.family_fractions must sum to 1.0 (got {total:.4f})")
    names = [f for f in FAMILIES if f in fractions]
    if set(names) != set(fractions):
        raise ValueError(f"unknown scenario families in config: {sorted(set(fractions) - set(FAMILIES))}")
    raw = {f: fractions[f] * n for f in names}
    counts = {f: int(np.floor(raw[f])) for f in names}
    if n >= len(names):
        for f in names:
            counts[f] = max(counts[f], 1)
    while sum(counts.values()) < n:
        pick = max(names, key=lambda f: (raw[f] - counts[f], -names.index(f)))
        counts[pick] += 1
    while sum(counts.values()) > n:
        # remove from the most over-allocated family that still has more than one scenario
        reducible = [f for f in names if counts[f] > 1]
        if not reducible:
            break
        pick = max(reducible, key=lambda f: (counts[f] - raw[f], names.index(f)))
        counts[pick] -= 1
    return counts


def scenario_seed(master_seed: int, slot: int, attempt: int) -> int:
    return int(np.random.SeedSequence([int(master_seed), int(slot), int(attempt)]).generate_state(1)[0])


def plan_slots(cfg: dict, n: int, master_seed: int) -> list[tuple[int, str]]:
    """Return [(slot_number, family)] in a shuffled (but reproducible) order, so IDs mix all families."""
    counts = allocate_counts(n, cfg["scenarios"]["family_fractions"])
    families = [f for f in FAMILIES for _ in range(counts.get(f, 0))]
    order = np.random.default_rng(np.random.SeedSequence([int(master_seed), 0, 0])).permutation(len(families))
    return [(i + 1, families[k]) for i, k in enumerate(order)]


def _u(rng, bounds):
    return float(rng.uniform(bounds[0], bounds[1]))


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _window(t_h, start_h, end_h, ramp_h=1.0):
    """Smooth 0 -> 1 -> 0 bump: rises over ramp_h after start_h, falls over ramp_h before end_h."""
    return _smoothstep((t_h - start_h) / ramp_h) * _smoothstep((end_h - t_h) / ramp_h)


def smooth_noise(rng, n: int, sigma_steps: float, sd: float) -> np.ndarray:
    """Smooth random wiggle with standard deviation sd (Gaussian-filtered white noise, wrapped around the day)."""
    k = np.arange(-int(4 * sigma_steps) - 1, int(4 * sigma_steps) + 2)
    kernel = np.exp(-0.5 * (k / max(sigma_steps, 1e-6)) ** 2)
    kernel /= kernel.sum()
    white = rng.standard_normal(n)
    padded = np.concatenate([white[-len(k):], white, white[:len(k)]])
    smooth = np.convolve(padded, kernel, mode="same")[len(k):len(k) + n]
    smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-12)
    return smooth * sd


# =============================================================================== demand / head / pump patterns
def _demand_shapes(t_h: np.ndarray, shift: float) -> list[np.ndarray]:
    g = lambda mu, s: np.exp(-0.5 * ((t_h - mu) / s) ** 2)  # noqa: E731
    residential = 0.45 + 1.00 * g(7.0 + shift, 1.3) + 1.25 * g(19.0 + shift, 2.0)
    school = 0.25 + 1.50 * _window(t_h, 8.0 + shift, 15.5 + shift, 0.5)
    hospital = 0.85 + 0.25 * g(10.0 + shift, 2.5) + 0.20 * g(20.0 + shift, 3.0)
    return [s / s.mean() for s in (residential, school, hospital)]   # each has daily mean 1


def make_demand_multipliers(rng, cfg: dict, demand_nodes: list[str], level: float) -> dict[str, list[float]]:
    """Smooth time-of-day demand multipliers (96 values per junction); daily mean ~ level."""
    steps = int(cfg["scenarios"]["steps"])
    dp = cfg["scenarios"]["demand_profile"]
    t_h = np.arange(steps) * STEP_S / 3600.0
    shift = _u(rng, dp["peak_shift_h"])
    amp = _u(rng, dp["amplitude_scale"])
    shapes = _demand_shapes(t_h, shift)
    out = {}
    for i, node in enumerate(demand_nodes):
        shape = shapes[i] if i < len(shapes) else np.ones(steps)
        shape = 1.0 + amp * (shape - 1.0)
        shape = shape * (1.0 + smooth_noise(rng, steps, dp["smooth_noise_sigma_steps"], dp["smooth_noise_sd"]))
        mult = np.clip(shape, 0.05, None) * level * _u(rng, dp["junction_jitter"])
        out[node] = [round(float(v), 5) for v in mult]
    return out


def make_head_pattern(rng, cfg: dict, drop: dict | None = None) -> list[float]:
    """Slow +/- drift of the source head, optionally with a smooth supply drop (reduced-supply scenarios)."""
    steps = int(cfg["scenarios"]["steps"])
    hd = cfg["scenarios"]["head_drift"]
    pattern = 1.0 + smooth_noise(rng, steps, hd["smooth_sigma_steps"], hd["amplitude"])
    if drop:
        t_h = np.arange(steps) * STEP_S / 3600.0
        pattern = pattern * (1.0 - drop["fraction"] * _window(t_h, drop["start_h"], drop["end_h"], 1.0))
    return [round(float(v), 5) for v in pattern]


# =============================================================================== spec sampling
def sample_spec(cfg: dict, family: str, slot: int, attempt: int, master_seed: int) -> dict:
    """Sample the INPUT settings of one scenario. No pressure or flow values are created here."""
    sc = cfg["scenarios"]
    rg = sc["ranges"][family]
    steps = int(sc["steps"])
    duration_s = int(float(cfg["simulation"]["duration_hours"]) * 3600)
    ids = _ids(cfg)
    seed = scenario_seed(master_seed, slot, attempt)
    rng = np.random.default_rng(seed)
    t_h = np.arange(steps) * STEP_S / 3600.0

    notes: list[str] = []
    pump_pattern = None
    pump_schedule = "constant"
    leak = None
    blockage = None
    drop = None

    # ---- family-specific parameter ranges
    if family == "high_source_pressure":
        if rng.random() < rg["head_variant_share"]:
            v = rg["head_variant"]; notes.append("excess source head")
        else:
            v = rg["pump_variant"]; notes.append("excess pump speed")
        head, speed, valve = _u(rng, v["head"]), _u(rng, v["pump_speed"]), _u(rng, rg["valve_setting"])
        level = _u(rng, rg["demand_level"])
    elif family == "valve_misconfiguration":
        if rng.random() < rg["low_setting_share"]:
            v = rg["low_variant"]; notes.append("valve setting far too low (starves J2)")
        else:
            v = rg["high_variant"]; notes.append("valve setting far too high (J2 unregulated)")
        head, valve = _u(rng, v["head"]), _u(rng, v["valve_setting"])
        speed, level = _u(rng, rg["pump_speed"]), _u(rng, rg["demand_level"])
    else:
        head, speed = _u(rng, rg["head"]), _u(rng, rg["pump_speed"])
        valve, level = _u(rng, rg["valve_setting"]), _u(rng, rg["demand_level"])

    if family == "reduced_supply":
        start = round(_u(rng, rg["drop_start_h"]) * 4) / 4
        end = min(start + round(_u(rng, rg["drop_duration_h"]) * 4) / 4, duration_s / 3600.0)
        drop = {"fraction": _u(rng, rg["head_drop_fraction"]), "start_h": start, "end_h": end}
        notes.append(f"no tank in the network: reduced supply modelled as a {drop['fraction']:.0%} source-head drop "
                     f"from {start:g} h to {end:g} h")

    if family == "pump_degradation":
        if rng.random() < rg["gradual_share"]:
            start = round(_u(rng, rg["degrade_start_h"]) * 4) / 4
            end_speed = _u(rng, rg["degrade_end_speed"])
            progress = np.clip((t_h - start) / (duration_s / 3600.0 - start), 0, 1)
            # EPANET uses a pump speed pattern value AS the speed setting (it replaces the base speed;
            # it is not a multiplier), so the pattern holds absolute relative speeds.
            pump_pattern = speed + (end_speed - speed) * progress
            pump_schedule = f"degrade {speed:.2f}->{end_speed:.2f} from {start:g}h"
            notes.append("gradual pump wear")
        else:
            start = round(_u(rng, rg["trip_start_h"]) * 4) / 4
            end = min(start + round(_u(rng, rg["trip_duration_h"]) * 4) / 4, duration_s / 3600.0)
            floor = _u(rng, rg["trip_floor_speed"])
            pump_pattern = speed - (speed - floor) * _window(t_h, start, end, 1.0)   # absolute speeds
            pump_schedule = f"slow-down to {floor:.2f} from {start:g}h to {end:g}h"
            notes.append("temporary severe pump slow-down (a full stop is excluded: no tank, EPANET leaves the "
                         "isolated network undefined)")

    if family in ("small_leak", "burst_leak"):
        nominal = _u(rng, rg["leak_lps"])
        start_s = int(round(_u(rng, rg["leak_start_h"]) * 4)) * STEP_S
        end_s = min(start_s + int(round(_u(rng, rg["leak_duration_h"]) * 4)) * STEP_S, duration_s)
        node = str(rng.choice(rg["leak_nodes"]))
        k = nominal / 1000.0 / np.sqrt(LEAK_REFERENCE_PRESSURE_M)   # emitter law: Q = k * sqrt(P)
        leak = {"node": node, "start_s": start_s, "end_s": end_s, "nominal_lps": round(nominal, 3),
                "emitter_coefficient": float(k)}
        notes.append(f"leak at {node}, {nominal:.1f} L/s at {LEAK_REFERENCE_PRESSURE_M:g} m")

    if family == "partial_blockage":
        pipe = str(rng.choice(rg["pipes"]))
        factor = _u(rng, rg["blockage_factor"])
        blockage = {"pipe": pipe, "factor": round(factor, 4)}
        notes.append(f"{factor:.0%} of {pipe} cross-section blocked")

    demand = make_demand_multipliers(rng, cfg, ids["demand_nodes"], level)
    head_pattern = make_head_pattern(rng, cfg, drop)
    profile_id = f"DP{slot:04d}" + (f"r{attempt}" if attempt > 1 else "")

    return {
        "scenario_id": f"S{slot:04d}", "slot": slot, "attempt": attempt, "scenario_type": family,
        "random_seed": seed,
        "reservoir_head": round(head, 3), "head_pattern": head_pattern,
        "tank_initial_level": None,
        "pump_speed": round(speed, 4),
        "pump_pattern": None if pump_pattern is None else [round(float(v), 5) for v in pump_pattern],
        "pump_schedule": pump_schedule,
        "valve_setting": round(valve, 3),
        "demand_profile_id": profile_id, "demand_level": round(level, 4), "demand_multipliers": demand,
        "leak": leak, "blockage": blockage,
        "notes": "; ".join(notes) if notes else "normal operation within the safe envelope",
    }


def spec_to_metadata(spec: dict) -> dict:
    leak, block = spec.get("leak"), spec.get("blockage")
    return {
        "scenario_id": spec["scenario_id"], "scenario_type": spec["scenario_type"],
        "random_seed": spec["random_seed"], "reservoir_head": spec["reservoir_head"],
        "tank_initial_level": np.nan,                         # the network has no tank
        "pump_speed": spec["pump_speed"], "pump_schedule": spec["pump_schedule"],
        "valve_setting": spec["valve_setting"],
        "leak_node": leak["node"] if leak else "",
        "leak_start_time": leak["start_s"] / 3600.0 if leak else np.nan,   # hours from simulation start
        "leak_end_time": leak["end_s"] / 3600.0 if leak else np.nan,
        "leak_severity": leak["nominal_lps"] if leak else 0.0,             # L/s at 50 m (nominal)
        "blockage_pipe": block["pipe"] if block else "",
        "blockage_factor": block["factor"] if block else np.nan,
        "demand_profile_id": spec["demand_profile_id"], "notes": spec["notes"],
    }


# =============================================================================== optional sensor noise
def apply_sensor_noise(raw: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Return a NOISY COPY of the time series (added after the hydraulic simulation). The raw data is untouched."""
    sn = cfg["sensor_noise"]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 99, 99]))
    noisy = raw.copy()
    for c in [c for c in noisy.columns if c.endswith("_pressure")]:
        noisy[c] = (noisy[c] + rng.normal(0.0, sn["pressure_sd_m"], len(noisy))).round(4)
    for c in [c for c in noisy.columns if c.endswith("_flow") or c.endswith("_demand")]:
        noisy[c] = (noisy[c] * (1.0 + rng.normal(0.0, sn["flow_relative_sd"], len(noisy)))).round(4)
    return noisy


# =============================================================================== summaries and provenance
def summarize_attempt(spec: dict, result: dict, labelled: pd.DataFrame | None, cfg: dict) -> dict:
    row = {"scenario_id": spec["scenario_id"], "attempt": spec["attempt"], "scenario_type": spec["scenario_type"],
           "random_seed": spec["random_seed"], "status": result["status"], "warnings": result["warning_text"],
           "error": result["error"], "runtime_s": result["runtime_s"]}
    df = result["data"]
    if df is None or labelled is None:
        return row
    th = cfg["thresholds"]
    pcols = [c for c in df.columns if c.endswith("_pressure")]
    row["n_rows"] = len(df)
    for c in pcols:
        node = c.split("_")[0]
        row[f"{node}_min_pressure"] = round(float(df[c].min()), 3)
        row[f"{node}_max_pressure"] = round(float(df[c].max()), 3)
    row["pressure_spread_max"] = round(float(labelled["pressure_spread"].max()), 3)
    row["pressure_spread_mean"] = round(float(labelled["pressure_spread"].mean()), 3)
    row["min_J4_pressure"] = round(float(df[f"{th['critical_junction']}_pressure"].min()), 3)
    row["n_low_pressure_steps"] = int(labelled["low_pressure_flag"].sum())
    row["n_high_pressure_steps"] = int(labelled["high_pressure_flag"].sum())
    row["n_imbalance_steps"] = int(labelled["imbalance_flag"].sum())
    violating = (labelled[["low_pressure_flag", "high_pressure_flag", "imbalance_flag"]].sum(axis=1) > 0)
    row["n_pressure_violations"] = int(violating.sum())          # time steps with any pressure-related violation
    row["max_leak_flow_lps"] = round(float(df["leak_severity"].max()), 3)
    row["scenario_risk_label"] = scenario_label(labelled, label_settings(cfg)["priority"])
    return row


COLUMN_DOCS = {
    "scenario_id": ("-", "scenario identifier (S0001...)", "generator"),
    "timestamp": ("datetime", "nominal clock; 96 steps of 15 min from simulation start", "EPANET report times"),
    "scenario_type": ("-", "scenario family", "generator"),
    "J1_pressure..J4_pressure": ("m", "junction pressure", "EPANET"),
    "P1_flow..P4_flow": ("L/s", "pipe flow; positive = from start node to end node", "EPANET"),
    "J2_demand..J4_demand": ("L/s", "demand actually delivered (pressure-driven demand)", "EPANET"),
    "tank_level": ("m", "EMPTY BY DESIGN: the network has no tank", "-"),
    "reservoir_head": ("m", "source head of R1", "EPANET"),
    "pump_speed": ("relative", "PU1 speed setting (1.0 = nominal)", "EPANET"),
    "pump_status": ("code", f"PU1 status {PUMP_STATUS_CODES}", "EPANET"),
    "valve_setting": ("m", "V1 PRV downstream pressure target", "EPANET"),
    "valve_status": ("code", f"V1 status {VALVE_STATUS_CODES}", "EPANET"),
    "leak_flag": ("0/1", "1 while the scheduled leak is open", "scenario schedule"),
    "leak_severity": ("L/s", "leak outflow computed by EPANET (0 when no leak). In scenario_metadata.csv the same "
                              "column name holds the NOMINAL leak at 50 m", "EPANET"),
}


def write_network_config(cfg: dict, base_inp: Path, n: int, seed: int, path: Path, counts: dict) -> None:
    ids = _ids(cfg)
    net = cfg["network"]
    config = {
        "project": "AquaBalance",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "simulator": {"engine": "EPANET (engine bundled with WNTR) run through wntr.sim.EpanetSimulator",
                      "wntr_version": wntr.__version__, "python": platform.python_version(),
                      "numpy": np.__version__, "pandas": pd.__version__},
        "base_inp": {"path": str(Path(cfg["paths"]["base_inp"])), "sha256": hashlib.sha256(base_inp.read_bytes()).hexdigest()},
        "simulation": {k: cfg["simulation"][k] for k in cfg["simulation"]},
        "network": {"reservoir": net["reservoir"], "junctions": net["junctions"], "pipes": net["pipes"],
                    "pump": net["pump"], "valve": net["valve"], "has_tank": False,
                    "monitored_pressure_nodes": ids["pressure_nodes"], "critical_junction": cfg["thresholds"]["critical_junction"]},
        "thresholds_note": "Configurable lab-prototype thresholds, not universal real-world standards.",
        "thresholds": cfg["thresholds"], "labels": cfg["labels"], "validation": cfg["validation"],
        "scenario_generation": {"master_seed": seed, "n_scenarios": n, "scenarios_per_family": counts,
                                "family_fractions": cfg["scenarios"]["family_fractions"],
                                "ranges": cfg["scenarios"]["ranges"]},
        "columns": {k: {"unit": v[0], "meaning": v[1], "source": v[2]} for k, v in COLUMN_DOCS.items()},
        "notes": ["All pressure/flow/demand/head/speed/status values come from EPANET simulations.",
                  "Random numbers only choose scenario inputs (settings), never outputs.",
                  "Leaks are EPANET emitters (pressure-dependent) behind a time-controlled helper pipe LKP/node LK.",
                  "The final EPANET row at t = 24:00 is dropped so every scenario has exactly 96 time steps."],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


# =============================================================================== the whole pipeline
def generate_dataset(cfg: dict, n: int, seed: int | None = None, paths: dict | None = None,
                     sensor_noise: bool | None = None, quiet: bool = False) -> dict:
    """Generate n scenarios with EPANET and write all Phase 2 files. Returns the frames and a status report."""
    seed = int(cfg["project"]["seed"] if seed is None else seed)
    merged = {**cfg["paths"], **(paths or {})}
    P = {k: resolve_path(v) for k, v in merged.items()}
    base_inp = P["base_inp"]
    if not base_inp.exists():
        raise FileNotFoundError(f"{base_inp} not found. Run: python scripts/build_network.py")
    scn_logger = logging.getLogger("scenario_generation")
    for handler in list(scn_logger.handlers):      # re-attach to THIS run's log folder (matters for repeated runs)
        scn_logger.removeHandler(handler)
        handler.close()
    log = get_logger("scenario_generation", merged.get("log_dir"), console=False)
    max_attempts = int(cfg["scenarios"]["max_attempts_per_slot"])
    slots = plan_slots(cfg, n, seed)
    counts = allocate_counts(n, cfg["scenarios"]["family_fractions"])
    log.info("Generating %d scenarios (seed %d). Per family: %s", n, seed, counts)

    specs, frames, labelled_frames, summary_rows, profile_rows = [], [], [], [], []
    for done, (slot, family) in enumerate(slots, start=1):
        for attempt in range(1, max_attempts + 1):
            spec = sample_spec(cfg, family, slot, attempt, seed)
            result = run_scenario(spec, cfg, base_inp)
            labelled = label_timeseries(result["data"], cfg) if result["data"] is not None else None
            summary_rows.append(summarize_attempt(spec, result, labelled, cfg))
            log.info("scenario=%s attempt=%d type=%s status=%s settings=%s", spec["scenario_id"], attempt, family,
                     result["status"], json.dumps({
                         "seed": spec["random_seed"], "head": spec["reservoir_head"], "pump_speed": spec["pump_speed"],
                         "pump_schedule": spec["pump_schedule"], "valve": spec["valve_setting"],
                         "demand_level": spec["demand_level"], "leak": spec["leak"], "blockage": spec["blockage"],
                         "warnings": result["warning_text"], "error": result["error"]}))
            if result["data"] is not None:
                specs.append(spec); frames.append(result["data"]); labelled_frames.append(labelled)
                for step in range(int(cfg["scenarios"]["steps"])):
                    profile_rows.append({"demand_profile_id": spec["demand_profile_id"], "step": step,
                                         **{f"{j}_multiplier": m[step] for j, m in spec["demand_multipliers"].items()}})
                break
        if not quiet and (done % 10 == 0 or done == len(slots)):
            print(f"  simulated {done}/{len(slots)} scenarios", flush=True)

    summary = pd.DataFrame(summary_rows)
    metadata = pd.DataFrame([spec_to_metadata(s) for s in specs], columns=METADATA_COLUMNS)
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        labels = pd.concat(labelled_frames, ignore_index=True)
    else:
        raw, labels = pd.DataFrame(columns=RAW_COLUMNS), pd.DataFrame()
    profiles = pd.DataFrame(profile_rows)

    for key in ("scenario_metadata", "raw_timeseries", "labels", "demand_profiles", "scenario_specs", "summary"):
        P[key].parent.mkdir(parents=True, exist_ok=True)
    fmt = "%Y-%m-%d %H:%M:%S"
    metadata.to_csv(P["scenario_metadata"], index=False)
    raw.to_csv(P["raw_timeseries"], index=False, date_format=fmt)
    labels.to_csv(P["labels"], index=False, date_format=fmt)
    profiles.to_csv(P["demand_profiles"], index=False)
    summary.to_csv(P["summary"], index=False)
    with open(P["scenario_specs"], "w", encoding="utf-8") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")
    write_network_config(cfg, base_inp, n, seed, P["network_config_json"], counts)

    use_noise = cfg["sensor_noise"]["enabled"] if sensor_noise is None else sensor_noise
    if use_noise and len(raw):
        apply_sensor_noise(raw, cfg, seed).to_csv(P["sensor_timeseries"], index=False, date_format=fmt)

    failed = summary[~summary["status"].isin(["ok", "ok_with_warnings"])] if len(summary) else summary
    complete = len(specs) == n
    log.info("Done: %d/%d scenarios accepted, %d failed/rejected attempts.", len(specs), n, len(failed))
    return {"complete": complete, "n_requested": n, "n_accepted": len(specs), "n_failed_attempts": len(failed),
            "metadata": metadata, "raw": raw, "labels": labels, "summary": summary, "profiles": profiles,
            "specs": specs, "paths": P, "seed": seed}
