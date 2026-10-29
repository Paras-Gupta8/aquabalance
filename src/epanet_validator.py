"""EPANET simulated validation of AquaBalance recommendations.

This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility.

For every recommendation the validator
  1. loads the base EPANET network (.inp) afresh for every run (a clean copy, never edited in place);
  2. re-applies the scenario conditions from the saved scenario spec: demand patterns, reservoir head, pump speed/status,
     valve setting, leak event, partial blockage (the network has no tank, so there are no tank conditions);
  3. simulates it with WNTR's EpanetSimulator and records the BASELINE metrics over the next `horizon_steps` steps;
  4. re-simulates the same scenario with each bounded candidate action applied from the decision step onward
     (valve -15..+15 %, pump -10..+10 %), including the engine's own recommended action;
  5. scores every candidate with the objective below, rejects unsafe ones, and compares before and after.

objective = high_pressure_penalty   x high-pressure violation node-steps
          + low_pressure_penalty    x low-pressure violation node-steps
          + critical_j4_penalty     x J4 critical-low node-steps          (very high: J4 is protected first)
          + imbalance_penalty       x steps whose branch spread exceeds the limit
          + spread_penalty          x mean branch pressure spread (m)
          + adjustment_penalty      x |control change| (percentage points)

A candidate is SAFE only if EPANET runs it cleanly, it stays inside the component bounds, no node gets more or deeper
violations than in the baseline, and J4 is never pushed down toward its critical minimum to help another node.
If no safe action improves a problem the answer is exactly: "No safe action found; operator inspection recommended."

Recommendations of leak-like / burst-like / uncertain situations are inspection recommendations: EPANET simulated
validation records their baseline (including the leak outflow) but no control change is tried for them.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wntr  # noqa: E402

from src.recommendation_engine import (INCREASE_PUMP, INCREASE_VALVE, NO_ACTION_INSPECT, NO_SAFE_ACTION_MESSAGE,  # noqa: E402
                                       REDUCE_PUMP, REDUCE_VALVE, VALIDATION_DISCLAIMER, VALIDATION_LABEL, Recommendation,
                                       RecommendationSettings, _within_bounds, apply_validation, recommend)
from src.risk_engine import RiskInputs, RiskSettings, assess_risk  # noqa: E402
from src.scenario_generator import STEP_S  # noqa: E402
from src.simulation import run_scenario  # noqa: E402
from src.utils import resolve_path  # noqa: E402

SIMULATOR = "WNTR EpanetSimulator (EPANET 2.2)"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BEFORE_C, AFTER_C = "#eb6834", "#2a78d6"                # validated categorical slots 2 and 1 (orange = before, blue = after)

FINAL_RESULTS = ("validated_improvement", "no_action_needed", "no_safe_action_found", "inspection_recommended")
METRIC_KEYS = ["J1_pressure_mean", "J2_pressure_mean", "J3_pressure_mean", "J4_pressure_mean", "J4_pressure_min",
               "J4_status", "high_pressure_violation_count", "low_pressure_violation_count",
               "j4_critical_violation_count", "imbalance_step_count", "branch_pressure_spread_mean",
               "branch_pressure_spread_max", "max_pressure", "min_pressure", "safe_node_coverage", "inflow_mean_lps",
               "delivered_demand_mean_lps", "loss_proxy_lps", "demand_served_fraction", "problem_resolved"]


class ValidationError(RuntimeError):
    """A readable problem with the validation inputs (missing network, spec, bad step ...)."""


# =============================================================================================== settings
@dataclass(frozen=True)
class ValidationSettings:
    horizon_steps: int
    valve_pcts: tuple
    pump_pcts: tuple
    pen_high: float
    pen_low: float
    pen_critical: float
    pen_imbalance: float
    pen_spread: float
    pen_adjust: float
    j4_margin_m: float
    tol_m: float
    min_improvement: float
    inspect_only_for: tuple
    nodes: tuple
    hub: str
    critical: str
    p_min: float
    p_max: float
    max_spread: float
    critical_min: float
    warning_margin_m: float
    inflow_pipe: str
    base_demand_lps: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "ValidationSettings":
        try:
            ev, rk, th = cfg["epanet_validation"], cfg["risk"], cfg["thresholds"]
            pen = ev["penalties"]
            s = cls(horizon_steps=int(ev["horizon_steps"]), valve_pcts=tuple(float(p) for p in ev["valve_candidates_pct"]),
                    pump_pcts=tuple(float(p) for p in ev["pump_candidates_pct"]), pen_high=float(pen["high_pressure"]),
                    pen_low=float(pen["low_pressure"]), pen_critical=float(pen["critical_j4"]),
                    pen_imbalance=float(pen["imbalance"]), pen_spread=float(pen["spread"]),
                    pen_adjust=float(pen["adjustment"]), j4_margin_m=float(ev["j4_safety_margin_m"]),
                    tol_m=float(ev["safe_tolerance_m"]), min_improvement=float(ev["min_improvement"]),
                    inspect_only_for=tuple(ev["inspect_only_for"]), nodes=tuple(rk["nodes"]), hub=str(rk["hub_node"]),
                    critical=str(th["critical_junction"]), p_min=float(th["p_min_m"]), p_max=float(th["p_max_m"]),
                    max_spread=float(th["max_pressure_spread_m"]), critical_min=float(th["critical_p_min_m"]),
                    warning_margin_m=float(rk["critical_warning_margin_m"]), inflow_pipe=str(rk["inflow_pipe"]),
                    base_demand_lps={j["id"]: float(j["base_demand_lps"]) for j in cfg["network"]["junctions"]})
        except KeyError as exc:
            raise ValidationError(f"config is missing the key {exc}; see the 'epanet_validation' section of "
                                  f"config/config.yaml") from None
        if s.horizon_steps < 1:
            raise ValidationError("epanet_validation.horizon_steps must be at least 1")
        if 0.0 not in s.valve_pcts or 0.0 not in s.pump_pcts:
            raise ValidationError("the candidate lists must include 0 (no change) so 'no action' can be chosen")
        if s.critical not in s.nodes:
            raise ValidationError("the critical junction must be one of risk.nodes")
        return s

    def min_for(self, node: str) -> float:
        return self.critical_min if node == self.critical else self.p_min


@dataclass(frozen=True)
class Action:
    component: str          # "none" | "pump" | "valve"
    pct: float = 0.0        # signed percent change of the current setting / speed

    @property
    def action_type(self) -> str:
        if self.component == "pump":
            return INCREASE_PUMP if self.pct > 0 else REDUCE_PUMP
        if self.component == "valve":
            return INCREASE_VALVE if self.pct > 0 else REDUCE_VALVE
        return "no_change"

    @property
    def label(self) -> str:
        return "no change" if self.component == "none" else f"{self.component} {self.pct:+g}%"


NO_CHANGE = Action("none", 0.0)


# =============================================================================================== loading
def load_base_network(base_inp: str | Path):
    """Check that the base EPANET .inp loads with WNTR and return its path. A readable error otherwise."""
    p = resolve_path(base_inp)
    if not Path(p).exists():
        raise ValidationError(f"base EPANET network {p} not found. Run: python scripts/build_network.py")
    try:
        wntr.network.WaterNetworkModel(str(p))
    except Exception as exc:                                    # WNTR/EPANET parse error: show it
        raise ValidationError(f"{p} could not be loaded as an EPANET network: {exc}") from exc
    return Path(p)


def load_spec(specs_path: str | Path, scenario_id: str) -> dict:
    p = resolve_path(specs_path)
    if not Path(p).exists():
        raise ValidationError(f"{p} not found. Run: python scripts/generate_synthetic_data.py --mode standard")
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                spec = json.loads(line)
                if spec["scenario_id"] == scenario_id:
                    return spec
    raise ValidationError(f"scenario {scenario_id} not found in {p}")


# =============================================================================================== applying an action
def base_pump_speeds(spec: dict, steps: int) -> np.ndarray:
    """Absolute relative pump speed at every step (EPANET reads a speed pattern value AS the speed)."""
    if spec.get("pump_pattern") is not None:
        return np.asarray(spec["pump_pattern"], dtype="float64")
    return np.full(steps, float(spec["pump_speed"]))


def apply_action(spec: dict, action: Action, start_step: int, steps: int) -> tuple[dict, dict]:
    """Copy the scenario spec with the action applied from `start_step` onward. Returns (new_spec, info)."""
    new = copy.deepcopy(spec)
    info = {"component": "", "unit": "", "current": None, "new": None, "pct": action.pct,
            "action_type": action.action_type}
    if action.component == "pump":
        speeds = base_pump_speeds(spec, steps).copy()
        info.update(component="PU1", unit="x nominal speed", current=float(speeds[start_step]))
        speeds[start_step:] = speeds[start_step:] * (1.0 + action.pct / 100.0)
        new["pump_pattern"] = [round(float(v), 5) for v in speeds]
        info["new"] = float(speeds[start_step])
    elif action.component == "valve":
        cur = float(spec["valve_setting"])
        info.update(component="V1", unit="m", current=cur)
        new_setting = round(cur * (1.0 + action.pct / 100.0), 4)
        new["valve_change"] = {"start_s": int(start_step * STEP_S), "setting": new_setting}
        info["new"] = new_setting
    return new, info


def _bounds_ok(spec: dict, action: Action, info: dict, start: int, end: int, rs: RecommendationSettings, steps: int) -> bool:
    if action.component == "none":
        return True
    if action.component == "pump":
        base = base_pump_speeds(spec, steps)[start:end + 1]
        return all(_within_bounds(float(b), float(b) * (1 + action.pct / 100.0), rs.pump_min, rs.pump_max) for b in base)
    return _within_bounds(info["current"], info["new"], rs.valve_min, rs.valve_max)


# =============================================================================================== metrics
def window_metrics(df: pd.DataFrame, spec: dict, vs: ValidationSettings, i0: int, i1: int) -> dict:
    """EPANET-simulated metrics over rows i0..i1 (inclusive) of a scenario time series."""
    w = df.iloc[i0:i1 + 1].reset_index(drop=True)
    nodes = list(vs.nodes)
    P = {n: w[f"{n}_pressure"].to_numpy(float) for n in [vs.hub] + nodes}
    m: dict = {f"{n}_pressure_mean": float(P[n].mean()) for n in [vs.hub] + nodes}
    low_total = high_total = violating = 0
    node_detail = {}
    for n in nodes:
        low, high = P[n] < vs.min_for(n), P[n] > vs.p_max
        node_detail[n] = {"low_count": int(low.sum()), "high_count": int(high.sum()),
                          "low_depth": float(max(0.0, vs.min_for(n) - P[n].min())),
                          "high_depth": float(max(0.0, P[n].max() - vs.p_max))}
        low_total += int(low.sum())
        high_total += int(high.sum())
        violating += int((low | high).sum())
    stack = np.vstack([P[n] for n in nodes])
    spread = stack.max(axis=0) - stack.min(axis=0)
    crit = P[vs.critical]
    m.update({
        f"{vs.critical}_pressure_min": float(crit.min()),
        f"{vs.critical}_status": ("CRITICAL LOW (below critical minimum)" if (crit < vs.critical_min).any() else
                                  "WARNING (close to critical minimum)" if crit.min() < vs.critical_min + vs.warning_margin_m
                                  else "OK"),
        "high_pressure_violation_count": high_total, "low_pressure_violation_count": low_total,
        "j4_critical_violation_count": node_detail[vs.critical]["low_count"],
        "imbalance_step_count": int((spread > vs.max_spread).sum()),
        "branch_pressure_spread_mean": float(spread.mean()), "branch_pressure_spread_max": float(spread.max()),
        "max_pressure": float(stack.max()), "min_pressure": float(stack.min()),
        "safe_node_coverage": float(1.0 - violating / stack.size),
    })
    inflow = w[f"{vs.inflow_pipe}_flow"].to_numpy(float)
    delivered = np.sum([w[f"{n}_demand"].to_numpy(float) for n in nodes], axis=0)
    required = np.sum([vs.base_demand_lps[n] * np.asarray(spec["demand_multipliers"][n], float)[i0:i1 + 1] for n in nodes], axis=0)
    m.update({"inflow_mean_lps": float(inflow.mean()), "delivered_demand_mean_lps": float(delivered.mean()),
              # valid here: this network has no consumer other than the metered demands, so inflow - delivered demand = leak outflow
              "loss_proxy_lps": float(max(0.0, (inflow - delivered).mean())),
              "demand_served_fraction": float(delivered.sum() / required.sum()) if required.sum() > 0 else 1.0})
    m["problem_resolved"] = bool(m["low_pressure_violation_count"] + m["high_pressure_violation_count"]
                                 + m["imbalance_step_count"] == 0)
    m["_nodes"] = node_detail
    m["_step1"] = {n: float(P[n][0]) for n in nodes}
    return m


def objective(m: dict, pct_abs: float, vs: ValidationSettings) -> tuple[float, dict]:
    comp = {"high_pressure": vs.pen_high * m["high_pressure_violation_count"],
            "low_pressure": vs.pen_low * m["low_pressure_violation_count"],
            "critical_j4": vs.pen_critical * m["j4_critical_violation_count"],
            "imbalance": vs.pen_imbalance * m["imbalance_step_count"],
            "spread": vs.pen_spread * m["branch_pressure_spread_mean"],
            "adjustment": vs.pen_adjust * abs(pct_abs)}
    return float(sum(comp.values())), {k: float(v) for k, v in comp.items()}


def check_safety(before: dict, after: dict, vs: ValidationSettings, run_ok: bool = True, bounds_ok: bool = True) -> tuple[bool, list]:
    """Pure safety rules (unit-testable). Returns (safe, reasons-for-rejection)."""
    why = []
    if not run_ok:
        why.append("EPANET run was rejected or reported a hydraulic problem")
    if not bounds_ok:
        why.append("outside the allowed component bounds")
    j = vs.critical
    bj, aj = before["_nodes"][j], after["_nodes"][j]
    floor = min(before[f"{j}_pressure_min"], vs.critical_min + vs.j4_margin_m)
    if (after["j4_critical_violation_count"] > before["j4_critical_violation_count"]
            or after[f"{j}_pressure_min"] < floor - vs.tol_m):
        why.append(f"would sacrifice {j} (its minimum pressure would fall to {after[f'{j}_pressure_min']:.1f} m; "
                   f"floor {floor:.1f} m)")
    for n, a in after["_nodes"].items():
        b = before["_nodes"][n]
        if a["low_count"] > b["low_count"] or a["low_depth"] > b["low_depth"] + vs.tol_m:
            why.append(f"would worsen low pressure at {n}")
        if a["high_count"] > b["high_count"] or a["high_depth"] > b["high_depth"] + vs.tol_m:
            why.append(f"would worsen high pressure at {n}")
    if after["imbalance_step_count"] > before["imbalance_step_count"]:
        why.append("would add imbalance steps")
    return not why, why


def addresses_problem(problem: str, before: dict, after: dict, vs: ValidationSettings) -> bool:
    """Does the action reduce the problem the engine flagged (not just some other term of the objective)?

    Without this, an action that merely narrows the spread could be accepted for a HIGH-pressure problem.
    """
    t, nb, na = vs.tol_m, before["_nodes"], after["_nodes"]
    j = vs.critical

    def less(key, nodes, count_key):
        return (sum(nb[n][key] for n in nodes) - sum(na[n][key] for n in nodes) > t
                or sum(nb[n][count_key] for n in nodes) > sum(na[n][count_key] for n in nodes))

    if problem == "critical_low":
        return less("low_depth", [j], "low_count")
    if problem == "other_low":
        return less("low_depth", list(vs.nodes), "low_count")
    if problem == "high":
        return less("high_depth", list(vs.nodes), "high_count")
    if problem == "imbalance":
        return (before["branch_pressure_spread_mean"] - after["branch_pressure_spread_mean"] > t
                or after["imbalance_step_count"] < before["imbalance_step_count"])
    return (less("low_depth", list(vs.nodes), "low_count") or less("high_depth", list(vs.nodes), "high_count")
            or after["imbalance_step_count"] < before["imbalance_step_count"])


# =============================================================================================== simulation helpers
def simulate(spec: dict, cfg: dict, base_inp: Path) -> dict:
    """One EPANET run of a scenario spec (fresh network copy from the .inp). Returns run_scenario's dict."""
    return run_scenario(spec, cfg, base_inp)


def _clean_run(res: dict) -> bool:
    return res["status"] in ("ok", "ok_with_warnings") and res["data"] is not None and "negative_pressures" not in res["warnings"]


def _flat(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def candidate_grid(vs: ValidationSettings) -> list[Action]:
    return ([NO_CHANGE] + [Action("valve", p) for p in vs.valve_pcts if p != 0]
            + [Action("pump", p) for p in vs.pump_pcts if p != 0])


def engine_action(rec: Recommendation) -> Action | None:
    """The control change the recommendation engine proposed, as a candidate Action (None for inspection / no action)."""
    t = rec.recommended_action_type
    if t in (INCREASE_PUMP, REDUCE_PUMP):
        return Action("pump", float(rec.adjustment_percent))
    if t in (INCREASE_VALVE, REDUCE_VALVE):
        return Action("valve", float(rec.adjustment_percent))
    return None


# =============================================================================================== the validation
def validate_recommendation(spec: dict, rec: Recommendation, decision_step: int, cfg: dict, base_inp: str | Path,
                            rs: RecommendationSettings | None = None, vs: ValidationSettings | None = None) -> dict:
    """EPANET simulated validation of one recommendation made at `decision_step` ("now" row) of a scenario.

    The action (if any) takes effect from decision_step + 1 and is held for `horizon_steps` steps; before/after metrics
    are measured over exactly those steps.
    """
    rs, vs = rs or RecommendationSettings.from_config(cfg), vs or ValidationSettings.from_config(cfg)
    base_path = load_base_network(base_inp)
    steps = int(cfg["scenarios"]["steps"])
    start = int(decision_step) + 1
    if decision_step < 0 or start > steps - 1:
        raise ValidationError(f"decision step {decision_step} leaves no later step to validate (scenario has {steps} steps)")
    end = min(start + vs.horizon_steps - 1, steps - 1)

    base_res = simulate(spec, cfg, base_path)                                  # steps 2-3: scenario re-applied, baseline run
    if not _clean_run(base_res):
        raise ValidationError(f"baseline EPANET run of {spec['scenario_id']} failed: {base_res['error'] or base_res['warning_text']}")
    before = window_metrics(base_res["data"], spec, vs, start, end)
    base_obj, base_comp = objective(before, 0.0, vs)
    problems_before = (before["low_pressure_violation_count"] + before["high_pressure_violation_count"]
                       + before["imbalance_step_count"])

    inspect_only = rec.addresses in vs.inspect_only_for
    eng = engine_action(rec)
    rows: list[dict] = []
    grid = [] if inspect_only else candidate_grid(vs)
    if eng is not None and eng not in grid and not inspect_only:
        grid.append(eng)                                                         # always validate the engine's own action
    for act in grid:                                                             # steps 4-6: candidate runs on fresh copies
        if act.component == "none":
            m, res, info, run_ok, b_ok = before, base_res, {"component": "", "unit": "", "current": None, "new": None,
                                                              "pct": 0.0, "action_type": "no_change"}, True, True
        else:
            new_spec, info = apply_action(spec, act, start, steps)
            res = simulate(new_spec, cfg, base_path)
            run_ok = _clean_run(res)
            b_ok = _bounds_ok(spec, act, info, start, end, rs, steps)
            m = window_metrics(res["data"], spec, vs, start, end) if run_ok else None
        if m is None:
            rows.append({"action": act, "info": info, "metrics": None, "objective": float("inf"), "components": {},
                         "safe": False, "reasons": ["EPANET run was rejected or reported a hydraulic problem"],
                         "run_status": res["status"], "warnings": res["warning_text"]})
            continue
        obj, comp = objective(m, abs(act.pct), vs)
        safe, why = (True, []) if act.component == "none" else check_safety(before, m, vs, run_ok, b_ok)
        rows.append({"action": act, "info": info, "metrics": m, "objective": obj, "components": comp, "safe": safe,
                     "reasons": why, "run_status": res["status"], "warnings": res["warning_text"]})

    # ---------------------------------------------------------------- choose: lowest objective among SAFE, then lowest change
    improving = [r for r in rows if r["safe"] and r["action"].component != "none"
                 and base_obj - r["objective"] > vs.min_improvement
                 and addresses_problem(rec.addresses, before, r["metrics"], vs)]
    chosen = min(improving, key=lambda r: (round(r["objective"], 6), abs(r["action"].pct))) if improving else None
    if inspect_only:
        result, message = "inspection_recommended", ("Inspection recommended; no control change is validated for this "
                                                     "situation (a pump or valve change must not be used to mask a leak).")
    elif problems_before == 0:
        result, message = "no_action_needed", "EPANET shows no violation in the baseline; no control change is needed."
    elif chosen is not None:
        result, message = "validated_improvement", f"Validated: {chosen['action'].label} improves the EPANET objective safely."
    else:
        result, message = "no_safe_action_found", NO_SAFE_ACTION_MESSAGE
    after_row = chosen if (result == "validated_improvement") else None
    after = after_row["metrics"] if after_row else before
    level = ("resolved" if after["problem_resolved"] else "partial") if after_row else \
        ("not_applicable" if result in ("no_action_needed", "inspection_recommended") else "none")

    # ---------------------------------------------------------------- the engine's own action, checked in EPANET
    eng_row = next((r for r in rows if eng is not None and r["action"].component == eng.component
                    and abs(r["action"].pct - eng.pct) < 1e-6), None)
    if eng is None:
        eng_status = "no_control_action"
    elif problems_before == 0:
        eng_status = "no_violation_in_epanet"          # the forecast flagged a problem that EPANET did not produce
    elif eng_row is None or eng_row["metrics"] is None:
        eng_status = "not_evaluated"
    elif not eng_row["safe"]:
        eng_status = "unsafe"
    elif (base_obj - eng_row["objective"] > vs.min_improvement
          and addresses_problem(rec.addresses, before, eng_row["metrics"], vs)):
        eng_status = "confirmed" if (chosen is not None and chosen["action"] == eng) else "confirmed_but_better_action_found"
    else:
        eng_status = "safe_no_improvement"
    est_err = np.nan
    if eng_row is not None and eng_row["metrics"] is not None and rec.predicted_pressure_after:
        est_delta = {n: rec.predicted_pressure_after[n] - rec.predicted_pressure_before[n] for n in vs.nodes}
        ep_delta = {n: eng_row["metrics"]["_step1"][n] - before["_step1"][n] for n in vs.nodes}
        est_err = float(np.mean([abs(est_delta[n] - ep_delta[n]) for n in vs.nodes]))

    chosen_detail = None
    if after_row:
        a = after_row["action"]
        chosen_detail = {**after_row["info"], "pct": a.pct}
    leak, block = spec.get("leak"), spec.get("blockage")
    note = _safety_note(result, chosen, before, after, vs)
    out = {
        "scenario_id": spec["scenario_id"], "scenario_family_analysis_only": spec["scenario_type"],
        "decision_step": int(decision_step), "decision_time": str(base_res["data"]["timestamp"].iloc[decision_step]),
        "window_first_step": start, "window_last_step": end, "window_steps": end - start + 1,
        "network_file": str(base_path), "simulator": SIMULATOR, "network_loaded_from_inp": True,
        "conditions_reapplied": "; ".join([
            "demand patterns", f"reservoir head {spec['reservoir_head']:.1f} m (+ drift pattern)",
            "tank: none (network has no tank)", f"pump speed {spec['pump_speed']} ({spec['pump_schedule']}), pump status on",
            f"valve V1 setting {spec['valve_setting']:.1f} m",
            f"leak at {leak['node']} ({leak['start_s'] / 3600:g}-{leak['end_s'] / 3600:g} h)" if leak else "no leak",
            f"blockage {block['pipe']} {block['factor']:.0%}" if block else "no blockage"]),
        "engine_action_type": rec.recommended_action_type, "engine_problem": rec.addresses,
        "engine_target_component": rec.target_component, "engine_adjustment_percent": float(rec.adjustment_percent),
        "engine_confidence": float(rec.confidence), "engine_status_in_epanet": eng_status,
        "engine_estimate_delta_error_m": est_err,
        "n_candidates_evaluated": len(rows), "baseline_objective": base_obj,
        "chosen_action": "no change" if after_row is None else after_row["action"].label,
        "chosen_action_detail": chosen_detail, "chosen_objective": float(after_row["objective"]) if after_row else base_obj,
        "final_validation_result": result, "improvement_level": level, "final_message": message, "safety_note": note,
        "before": _flat(before), "after": _flat(after),
        "validation_type": VALIDATION_LABEL, "validation_disclaimer": VALIDATION_DISCLAIMER,
        "candidates": [_candidate_row(spec, rec, decision_step, base_obj, r, chosen) for r in rows],
    }
    out["validated_recommendation"] = apply_validation(rec, out).to_dict()
    return out


def _safety_note(result: str, chosen, before: dict, after: dict, vs: ValidationSettings) -> str:
    j = vs.critical
    if result == "inspection_recommended":
        s = (f"No pump or valve change was tried: leak-like, burst-like or uncertain situations call for inspection, "
             f"and more pressure would increase leakage. Baseline {j} minimum {before[f'{j}_pressure_min']:.1f} m "
             f"(critical minimum {vs.critical_min:g} m).")
    elif result == "no_action_needed":
        s = "No violation in the EPANET baseline, so every candidate change was only a cost; nothing is changed."
    elif result == "no_safe_action_found":
        s = (f"Every candidate either failed to improve the EPANET objective or was unsafe (J4 is never sacrificed for "
             f"another node; no node may get worse). {NO_SAFE_ACTION_MESSAGE}")
    else:
        s = (f"{chosen['action'].label} was re-simulated in EPANET: no node got more or deeper violations, {j} minimum "
             f"{before[f'{j}_pressure_min']:.1f} -> {after[f'{j}_pressure_min']:.1f} m (critical minimum {vs.critical_min:g} m), "
             f"change kept within the allowed bounds.")
        if not after["problem_resolved"]:
            s += " The problem is reduced but NOT fully cleared; re-check after applying and inspect the supply side."
    return f"{VALIDATION_LABEL}: {s} {VALIDATION_DISCLAIMER}"


def _candidate_row(spec, rec, step, base_obj, r, chosen) -> dict:
    m = r["metrics"]
    row = {"scenario_id": spec["scenario_id"], "decision_step": int(step), "candidate": r["action"].label,
           "component": r["action"].component, "adjustment_percent": r["action"].pct,
           "current_value": r["info"]["current"], "new_value": r["info"]["new"], "epanet_run_status": r["run_status"],
           "safe": bool(r["safe"]), "rejection_reasons": "; ".join(r["reasons"]), "objective": r["objective"],
           "objective_change_vs_no_change": r["objective"] - base_obj, "is_engine_action": bool(
               engine_action(rec) is not None and r["action"] == engine_action(rec)),
           "is_chosen": bool(chosen is not None and r is chosen),
           "validation_type": VALIDATION_LABEL}
    row.update({f"objective_{k}": v for k, v in r["components"].items()})
    if m is not None:
        row.update({k: v for k, v in _flat(m).items() if k != "problem_resolved"})
        row["problem_resolved"] = m["problem_resolved"]
    return row


# =============================================================================================== recommendations from an EPANET series
def oracle_forecast(df: pd.DataFrame, step: int, nodes) -> dict:
    """The true EPANET pressure at the next step. Only for tests/examples that need a clean, known forecast."""
    return {n: float(df[f"{n}_pressure"].iloc[step + 1]) for n in nodes}


def risk_inputs_from_timeseries(df: pd.DataFrame, step: int, forecast: dict, window: int = 12) -> RiskInputs:
    """RiskInputs for the decision at row `step`, from the previous `window` rows of an EPANET series (sensors only)."""
    w = df.iloc[max(0, step - window + 1):step + 1]
    last = w.iloc[-1]
    return RiskInputs(
        pressure_history={c.split("_")[0]: w[c].to_numpy(float) for c in df.columns if c.endswith("_pressure")},
        predicted_pressure=dict(forecast),
        flow_history={c.split("_")[0]: w[c].to_numpy(float) for c in df.columns if c.endswith("_flow")},
        demand_history={c.split("_")[0]: w[c].to_numpy(float) for c in df.columns if c.endswith("_demand")},
        tank_level=None, pump_speed=float(last["pump_speed"]), pump_status=float(last["pump_status"]),
        valve_setting=float(last["valve_setting"]), valve_status=float(last["valve_status"]))


def recommend_from_timeseries(df: pd.DataFrame, step: int, forecast: dict, cfg: dict):
    """Risk assessment + recommendation for one decision step. Returns (assessment, recommendation)."""
    a = assess_risk(risk_inputs_from_timeseries(df, step, forecast), RiskSettings.from_config(cfg))
    return a, recommend(a, RecommendationSettings.from_config(cfg))


# =============================================================================================== output tables
RESULT_ID_COLUMNS = ["validation_type", "scenario_id", "scenario_family_analysis_only", "decision_step", "decision_time",
                     "window_first_step", "window_last_step", "window_steps", "network_file", "simulator",
                     "conditions_reapplied", "engine_problem", "engine_action_type", "engine_target_component",
                     "engine_adjustment_percent", "engine_confidence", "engine_status_in_epanet",
                     "engine_estimate_delta_error_m", "n_candidates_evaluated", "baseline_objective"]
RESULT_TAIL_COLUMNS = ["chosen_action", "chosen_objective", "final_validation_result", "improvement_level", "final_message", "safety_note",
                       "validation_disclaimer"]


def result_row(v: dict) -> dict:
    """Flatten one validation result into a CSV row: identifiers, before_*, after_*, chosen action, final result."""
    row = {k: v[k] for k in RESULT_ID_COLUMNS}
    row.update({f"before_{k}": v["before"][k] for k in METRIC_KEYS})
    row.update({f"after_{k}": v["after"][k] for k in METRIC_KEYS})
    row.update({k: v[k] for k in RESULT_TAIL_COLUMNS})
    return row


def results_frame(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([result_row(v) for v in results])


def effectiveness_table(results: list[dict]) -> pd.DataFrame:
    """Aggregate outcome per engine problem type and overall. Honest counts: not every recommendation improves."""
    df = results_frame(results)

    def agg(g: pd.DataFrame, name: str) -> dict:
        n = len(g)
        fin = g["final_validation_result"]
        acted = g[fin == "validated_improvement"]
        return {
            "validation_type": VALIDATION_LABEL, "group": name, "n_recommendations": n,
            "n_validated_improvement": int((fin == "validated_improvement").sum()),
            "n_no_action_needed": int((fin == "no_action_needed").sum()),
            "n_no_safe_action_found": int((fin == "no_safe_action_found").sum()),
            "n_inspection_recommended": int((fin == "inspection_recommended").sum()),
            "n_engine_action_confirmed": int(g["engine_status_in_epanet"].isin(["confirmed", "confirmed_but_better_action_found"]).sum()),
            "n_engine_action_unsafe": int((g["engine_status_in_epanet"] == "unsafe").sum()),
            "n_engine_action_no_improvement": int((g["engine_status_in_epanet"] == "safe_no_improvement").sum()),
            "n_engine_flag_not_seen_in_epanet": int((g["engine_status_in_epanet"] == "no_violation_in_epanet").sum()),
            "n_improvement_resolved": int((g["improvement_level"] == "resolved").sum()),
            "n_improvement_partial": int((g["improvement_level"] == "partial").sum()),
            "n_j4_critical_before": int((g["before_j4_critical_violation_count"] > 0).sum()),
            "n_j4_critical_after": int((g["after_j4_critical_violation_count"] > 0).sum()),
            "mean_low_violations_before": g["before_low_pressure_violation_count"].mean(),
            "mean_low_violations_after": g["after_low_pressure_violation_count"].mean(),
            "mean_high_violations_before": g["before_high_pressure_violation_count"].mean(),
            "mean_high_violations_after": g["after_high_pressure_violation_count"].mean(),
            "mean_spread_before_m": g["before_branch_pressure_spread_mean"].mean(),
            "mean_spread_after_m": g["after_branch_pressure_spread_mean"].mean(),
            "mean_safe_node_coverage_before": g["before_safe_node_coverage"].mean(),
            "mean_safe_node_coverage_after": g["after_safe_node_coverage"].mean(),
            "mean_first_order_estimate_error_m": g["engine_estimate_delta_error_m"].mean(),
            "validation_disclaimer": VALIDATION_DISCLAIMER}

    rows = [agg(g, str(k)) for k, g in df.groupby("engine_problem", sort=True)] + [agg(df, "ALL")]
    return pd.DataFrame(rows)


def write_outputs(results: list[dict], paths: dict) -> dict:
    """Write the result, candidate-search and effectiveness tables."""
    out = {}
    frames = {"validation_results": results_frame(results),
              "validation_candidates": pd.DataFrame([c for v in results for c in v["candidates"]]),
              "recommendation_effectiveness": effectiveness_table(results)}
    for key, frame in frames.items():
        p = Path(paths[key])
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False, float_format="%.6g")
        out[key] = p
    return out


# =============================================================================================== figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def _panel_title(r) -> str:
    fam = str(r["scenario_family_analysis_only"]).replace("_", " ")
    return f"{r['scenario_id']} step {r['decision_step']}\n{fam}"


def plot_pressure_comparison(results: list[dict], vs: ValidationSettings, path: str | Path) -> Path:
    """Mean J1-J4 pressure over the validated window, before (orange) vs after (blue), with the configured limits."""
    n = len(results)
    cols = min(4, n)
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.6 * cols, 3.5 * rows_ + 1.2), squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    names = [vs.hub] + list(vs.nodes)
    x = np.arange(len(names))
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, v in zip(axes.ravel(), results):
        ax.set_visible(True)
        _style(ax)
        b = [v["before"][f"{k}_pressure_mean"] for k in names]
        a = [v["after"][f"{k}_pressure_mean"] for k in names]
        ax.bar(x - 0.2, b, 0.38, color=BEFORE_C, label="before (EPANET)")
        ax.bar(x + 0.2, a, 0.38, color=AFTER_C, label="after chosen action (EPANET)")
        ax.axhline(vs.p_max, color=INK, lw=1, ls=(0, (4, 3)))
        ax.axhline(vs.p_min, color=INK, lw=1, ls=(0, (4, 3)))
        ji = names.index(vs.critical)
        ax.plot([ji - 0.45, ji + 0.45], [vs.critical_min] * 2, color=INK, lw=2.2)
        ax.set_xticks(x, names, fontsize=9, color=INK)
        ax.set_title(_panel_title(v), fontsize=9, color=INK, loc="left")
        act = v["chosen_action"] if v["final_validation_result"] == "validated_improvement" else \
            v["final_validation_result"].replace("_", " ")
        ax.text(0.02, 0.97, f"action: {act}", transform=ax.transAxes, fontsize=8, color=INK, va="top",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    for r_ in range(rows_):
        axes[r_, 0].set_ylabel("mean pressure (m)", fontsize=9, color=INK)
    h, l = [], []
    for ax in axes.ravel():
        if ax.get_visible():
            h, l = ax.get_legend_handles_labels()
            break
    from matplotlib.lines import Line2D
    h += [Line2D([0], [0], color=INK, lw=1, ls=(0, (4, 3))), Line2D([0], [0], color=INK, lw=2.2)]
    l += [f"limits {vs.p_min:g} / {vs.p_max:g} m", f"{vs.critical} critical minimum {vs.critical_min:g} m"]
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"{VALIDATION_LABEL}: mean pressure over the validated window, before vs after", x=0.01, ha="left",
                 fontsize=11, color=INK, y=0.995)
    fig.text(0.01, 0.005, VALIDATION_DISCLAIMER, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    return _save(fig, path)


def plot_violation_comparison(results: list[dict], vs: ValidationSettings, path: str | Path) -> Path:
    """Violation node-steps and spread before vs after for every validated recommendation."""
    labels = [f"{v['scenario_id']}\n{str(v['engine_problem']).replace('_', ' ')[:14]}" for v in results]
    panels = [("low-pressure violations\n(node-steps)", "low_pressure_violation_count"),
              ("high-pressure violations\n(node-steps)", "high_pressure_violation_count"),
              (f"{vs.critical} critical-low\n(steps)", "j4_critical_violation_count"),
              ("mean branch spread (m)", "branch_pressure_spread_mean")]
    fig, axes = plt.subplots(2, 2, figsize=(max(8.5, 0.95 * len(results) + 4), 7.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(results))
    for ax, (title, key) in zip(axes.ravel(), panels):
        _style(ax)
        ax.bar(x - 0.2, [v["before"][key] for v in results], 0.38, color=BEFORE_C, label="before (EPANET)")
        ax.bar(x + 0.2, [v["after"][key] for v in results], 0.38, color=AFTER_C, label="after chosen action (EPANET)")
        if key == "branch_pressure_spread_mean":
            ax.axhline(vs.max_spread, color=INK, lw=1, ls=(0, (4, 3)))
            ax.text(len(results) - 0.5, vs.max_spread, f" limit {vs.max_spread:g} m", fontsize=8, color=INK, va="bottom", ha="right")
        ax.set_xticks(x, labels, fontsize=7, color=MUTED, rotation=0 if len(results) <= 6 else 60)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{VALIDATION_LABEL}: violations and spread, before vs after", x=0.01, ha="left", fontsize=11, color=INK, y=0.995)
    fig.text(0.01, 0.005, VALIDATION_DISCLAIMER, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return _save(fig, path)


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
