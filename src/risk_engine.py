"""Transparent, rule-based risk detection from forecast pressure (simulated decision support, not control).

Input : current pressures, GRU-forecast pressures for the next interval, recent flow / demand history, optional tank
        level, pump speed, valve setting - plus the configured thresholds.
Output: a RiskAssessment (label, low/high nodes, spread, imbalance flag, possible-leak flag, severity 0-100 and a
        plain-language explanation).

Every rule is a visible comparison against a configured number. The engine never receives scenario type, leak flags,
leak severity or any other hidden simulation label: its only inputs are things a sensor or a forecast could provide.

How a leak looks in these data (checked against the EPANET output): the metered inflow equals the metered demand at
every step of every non-leak scenario, so "inflow minus demand" is ~0 normally and equals the leak outflow during a leak.
A leak-like pattern therefore needs BOTH a pressure drop AND an inflow/demand mismatch (as specified); a mismatch
without a recent pressure drop is reported as an uncertain "flow inconsistency" and handled conservatively.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Primary label priority (same order as the Phase 2 labels): events first, then pressure problems.
LABEL_PRIORITY = ("burst_like_pattern", "leak_like_pattern", "low_pressure_risk", "high_pressure_risk",
                  "pressure_imbalance", "normal")
UNCERTAIN_LABEL = "uncertain_insufficient_data"
RISK_LABELS = LABEL_PRIORITY + (UNCERTAIN_LABEL,)
VALVE_STATUS_ACTIVE = 2          # EPANET status code: a PRV that is actively throttling


class RiskError(ValueError):
    """Readable problem with the risk-engine inputs or settings."""


# =============================================================================================== settings
@dataclass(frozen=True)
class RiskSettings:
    p_min_m: float
    p_max_m: float
    max_spread_m: float
    critical_node: str
    critical_p_min_m: float
    nodes: tuple
    horizon_minutes: float
    inflow_pipe: str
    branch_pipes: dict
    hub_node: str
    critical_warning_margin_m: float
    near_limit_margin_m: float
    trend_steps: int
    leak_residual_min_lps: float
    leak_residual_rel_tol: float
    burst_residual_min_lps: float
    pressure_drop_min_m: float
    pressure_drop_lookback_steps: int
    max_forecast_change_m: float
    tank_low_level_m: float | None
    severity: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskSettings":
        try:
            th, rk = cfg["thresholds"], cfg["risk"]
            nodes = tuple(rk["nodes"])
            s = cls(p_min_m=float(th["p_min_m"]), p_max_m=float(th["p_max_m"]),
                    max_spread_m=float(th["max_pressure_spread_m"]), critical_node=str(th["critical_junction"]),
                    critical_p_min_m=float(th["critical_p_min_m"]), nodes=nodes,
                    horizon_minutes=float(rk["horizon_minutes"]), inflow_pipe=str(rk["inflow_pipe"]),
                    branch_pipes=dict(rk["branch_pipes"]), hub_node=str(rk["hub_node"]),
                    critical_warning_margin_m=float(rk["critical_warning_margin_m"]),
                    near_limit_margin_m=float(rk["near_limit_margin_m"]), trend_steps=int(rk["trend_steps"]),
                    leak_residual_min_lps=float(rk["leak_residual_min_lps"]),
                    leak_residual_rel_tol=float(rk["leak_residual_rel_tol"]),
                    burst_residual_min_lps=float(rk["burst_residual_min_lps"]),
                    pressure_drop_min_m=float(rk["pressure_drop_min_m"]),
                    pressure_drop_lookback_steps=int(rk["pressure_drop_lookback_steps"]),
                    max_forecast_change_m=float(rk["max_forecast_change_m"]),
                    tank_low_level_m=None if rk.get("tank_low_level_m") is None else float(rk["tank_low_level_m"]),
                    severity=dict(rk["severity"]))
        except KeyError as exc:
            raise RiskError(f"config is missing the key {exc}; see the 'risk' and 'thresholds' sections of "
                            f"config/config.yaml") from None
        s.validate()
        return s

    def validate(self) -> None:
        if not self.p_min_m < self.p_max_m:
            raise RiskError("thresholds.p_min_m must be below thresholds.p_max_m")
        if self.critical_p_min_m < self.p_min_m:
            raise RiskError("the critical minimum must not be looser than the general minimum (J4 gets STRICTER rules)")
        if self.critical_node not in self.nodes:
            raise RiskError(f"critical junction {self.critical_node} must be one of risk.nodes {list(self.nodes)}")
        if self.max_spread_m <= 0:
            raise RiskError("thresholds.max_pressure_spread_m must be positive")

    def min_for(self, node: str) -> float:
        """Lower pressure limit of a node: the strict critical minimum for J4, the general minimum otherwise."""
        return self.critical_p_min_m if node == self.critical_node else self.p_min_m


# =============================================================================================== inputs
@dataclass
class RiskInputs:
    """Everything the engine may look at. Histories end at 'now' (last element)."""
    pressure_history: dict                      # node -> (T,) metres, includes the hub if available
    predicted_pressure: dict                    # node -> forecast metres for the next interval
    flow_history: dict = field(default_factory=dict)      # pipe id -> (T,) L/s
    demand_history: dict = field(default_factory=dict)    # node -> (T,) L/s
    tank_level: float | None = None
    pump_speed: float | None = None             # relative to nominal
    pump_status: float | None = None
    valve_setting: float | None = None          # PRV downstream pressure setting, metres
    valve_status: float | None = None

    def current(self, node: str) -> float:
        return float(np.asarray(self.pressure_history[node], dtype="float64")[-1])


def inputs_from_window(x_raw: np.ndarray, feature_names: list[str], predicted: np.ndarray, nodes: list[str]) -> RiskInputs:
    """Turn ONE raw model window (W, F) and its forecast (K,) into RiskInputs. Uses sensor/control columns only."""
    x_raw = np.asarray(x_raw, dtype="float64")
    if x_raw.ndim != 2 or x_raw.shape[1] != len(feature_names):
        raise RiskError(f"expected one window of shape (W, {len(feature_names)}); got {x_raw.shape}")
    predicted = np.asarray(predicted, dtype="float64").ravel()
    if predicted.size != len(nodes):
        raise RiskError(f"expected {len(nodes)} forecast values (one per node); got {predicted.size}")
    col = {n: i for i, n in enumerate(feature_names)}

    def hist(name):
        return x_raw[:, col[name]].copy() if name in col else None

    def last(name):
        return float(x_raw[-1, col[name]]) if name in col else None

    press = {k.split("_")[0]: hist(k) for k in feature_names if k.endswith("_pressure")}
    flows = {k.split("_")[0]: hist(k) for k in feature_names if k.endswith("_flow")}
    dem = {k.split("_")[0]: hist(k) for k in feature_names if k.endswith("_demand")}
    return RiskInputs(pressure_history=press, predicted_pressure=dict(zip(nodes, predicted.tolist())),
                      flow_history=flows, demand_history=dem, tank_level=last("tank_level"),
                      pump_speed=last("pump_speed"), pump_status=last("pump_status"),
                      valve_setting=last("valve_setting"), valve_status=last("valve_status"))


# =============================================================================================== output
@dataclass
class RiskAssessment:
    predicted_risk_label: str
    low_pressure_nodes: list
    high_pressure_nodes: list
    pressure_spread: float
    pressure_imbalance_flag: bool
    possible_leak_pattern_flag: bool
    severity_score: float
    human_readable_explanation: str
    # ---- supporting detail (transparency / used by the recommendation engine) ----
    all_labels: list = field(default_factory=list)
    burst_like_flag: bool = False
    critical_node_at_risk: bool = False
    uncertain: bool = False
    uncertainty_reasons: list = field(default_factory=list)
    explanation_points: list = field(default_factory=list)
    severity_components: dict = field(default_factory=dict)
    predicted_pressure: dict = field(default_factory=dict)
    current_pressure: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)              # current controls / flows needed to propose a change
    unavailable_inputs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "predicted_risk_label", "low_pressure_nodes", "high_pressure_nodes", "pressure_spread",
            "pressure_imbalance_flag", "possible_leak_pattern_flag", "severity_score", "human_readable_explanation")}
        d.update({k: getattr(self, k) for k in (
            "all_labels", "burst_like_flag", "critical_node_at_risk", "uncertain", "uncertainty_reasons",
            "severity_components", "predicted_pressure", "current_pressure", "evidence", "state", "unavailable_inputs")})
        return _jsonable(d)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# =============================================================================================== helpers
def _join(names: list[str]) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _slope(y, steps: int) -> float | None:
    y = np.asarray(y, dtype="float64")[-steps:]
    if y.size < 2 or not np.isfinite(y).all():
        return None
    return float(np.polyfit(np.arange(y.size), y, 1)[0])


def _component(base: float, size: float, scale: float) -> float:
    return float(base + (100.0 - base) * min(1.0, max(size, 0.0) / scale))


def _finite(v) -> bool:
    return v is not None and bool(np.isfinite(v))


# =============================================================================================== the engine
def assess_risk(inp: RiskInputs, s: RiskSettings) -> RiskAssessment:
    """Apply the configured rules to one situation. Pure function of `inp` and `s`; no hidden labels exist here."""
    nodes = list(s.nodes)
    unavailable: list[str] = []
    reasons: list[str] = []

    pred = {n: inp.predicted_pressure.get(n) for n in nodes}
    cur = {n: (float(np.asarray(inp.pressure_history[n], "float64")[-1]) if n in inp.pressure_history else None)
           for n in nodes}
    if not all(_finite(pred[n]) for n in nodes):
        missing = [n for n in nodes if not _finite(pred[n])]
        text = (f"No usable forecast pressure for {_join(missing)}, so no risk can be judged. "
                f"Treat the situation as unknown and inspect the data feed.")
        return RiskAssessment(UNCERTAIN_LABEL, [], [], float("nan"), False, False, 0.0, text, all_labels=[UNCERTAIN_LABEL],
                              uncertain=True, uncertainty_reasons=[f"missing/invalid forecast for {_join(missing)}"],
                              explanation_points=[text], predicted_pressure={n: pred[n] for n in nodes},
                              current_pressure=cur, unavailable_inputs=[f"forecast:{n}" for n in missing])
    pred = {n: float(pred[n]) for n in nodes}
    for n in nodes:
        if cur[n] is None or not np.isfinite(cur[n]):
            unavailable.append(f"current_pressure:{n}")

    # ------------------------------------------------------------------ pressure rules on the FORECAST
    low = [n for n in nodes if pred[n] < s.min_for(n)]
    high = [n for n in nodes if pred[n] > s.p_max_m]
    spread = max(pred.values()) - min(pred.values())
    imbalance = spread > s.max_spread_m
    critical_low = s.critical_node in low
    forecast_warning = (not critical_low) and pred[s.critical_node] < s.critical_p_min_m + s.critical_warning_margin_m

    # ------------------------------------------------------------------ pressure drop evidence (history + forecast)
    drops = {}
    for n in nodes:
        h = np.asarray(inp.pressure_history.get(n, []), "float64")
        window_drop = float(np.nanmax(h[-s.pressure_drop_lookback_steps:]) - h[-1]) if h.size and np.isfinite(h).any() else 0.0
        fc_drop = (cur[n] - pred[n]) if cur[n] is not None else 0.0
        drops[n] = max(window_drop, fc_drop, 0.0)
    drop_node = max(drops, key=drops.get)
    pressure_drop = drops[drop_node]
    has_drop = pressure_drop >= s.pressure_drop_min_m

    # ------------------------------------------------------------------ flow / demand consistency (mass balance)
    flows, dems = inp.flow_history, inp.demand_history
    q_in = np.asarray(flows[s.inflow_pipe], "float64") if s.inflow_pipe in flows else None
    have_demand = all(n in dems for n in nodes)
    residual = None
    branch_res = {}
    if q_in is not None and have_demand and np.isfinite(q_in[-1]) and all(np.isfinite(dems[n][-1]) for n in nodes):
        residual = float(q_in[-1] - sum(float(dems[n][-1]) for n in nodes))
        for n in nodes:
            pipe = s.branch_pipes.get(n)
            if pipe in flows and np.isfinite(flows[pipe][-1]):
                branch_res[n] = float(flows[pipe][-1] - dems[n][-1])
        if all(p in flows for p in s.branch_pipes.values()):       # hub: inflow not accounted for by the branches
            branch_res[s.hub_node] = float(q_in[-1] - sum(float(flows[p][-1]) for p in s.branch_pipes.values()))
    else:
        unavailable.append("flow/demand consistency check (inflow or demand history missing)")
    threshold = None
    flow_inconsistent = False
    if residual is not None:
        threshold = max(s.leak_residual_min_lps, s.leak_residual_rel_tol * max(float(q_in[-1]), 0.0))
        flow_inconsistent = residual >= threshold
        if residual <= -threshold:
            reasons.append(f"metered flow is {abs(residual):.2f} L/s below metered demand (possible meter fault)")
    leak_flag = bool(flow_inconsistent and has_drop)
    burst_flag = bool(leak_flag and residual >= s.burst_residual_min_lps)
    if flow_inconsistent and not has_drop:
        reasons.append("flow/demand inconsistency without a recent pressure drop (long-running leak or meter fault?)")
    suspect = None
    if flow_inconsistent and branch_res:
        suspect = max(branch_res, key=branch_res.get)

    # ------------------------------------------------------------------ trends (context)
    flow_trend = _slope(q_in, s.trend_steps) if q_in is not None else None
    dsum = None
    if have_demand:
        dsum = np.sum([np.asarray(dems[n], "float64") for n in nodes], axis=0)
    demand_trend = _slope(dsum, s.trend_steps) if dsum is not None else None

    # ------------------------------------------------------------------ tank (optional input)
    tank_low = False
    if _finite(inp.tank_level):
        if s.tank_low_level_m is not None and inp.tank_level < s.tank_low_level_m:
            tank_low = True
    else:
        unavailable.append("tank_level (this network has no tank)")

    # ------------------------------------------------------------------ forecast reliability
    for n in nodes:
        if cur[n] is not None and abs(pred[n] - cur[n]) > s.max_forecast_change_m:
            reasons.append(f"forecast for {n} is {abs(pred[n] - cur[n]):.1f} m away from the current pressure (unreliable?)")
    if inp.pump_status is not None and np.isfinite(inp.pump_status) and inp.pump_status < 0.5:
        reasons.append("pump reported as off")

    # ------------------------------------------------------------------ labels
    active = {"burst_like_pattern": burst_flag, "leak_like_pattern": leak_flag, "low_pressure_risk": bool(low),
              "high_pressure_risk": bool(high), "pressure_imbalance": imbalance}
    labels = [k for k in LABEL_PRIORITY[:-1] if active[k]] or ["normal"]
    label = labels[0]

    # ------------------------------------------------------------------ severity (0-100), every part visible
    sv = s.severity
    comp = {}
    if low:
        non_crit = [n for n in low if n != s.critical_node]
        if non_crit:
            comp["low_pressure"] = _component(sv["low_pressure"]["base"], max(s.p_min_m - pred[n] for n in non_crit),
                                              sv["low_pressure"]["scale_m"])
        if critical_low:
            comp["critical_low_" + s.critical_node] = _component(
                sv["critical_low_J4"]["base"], s.critical_p_min_m - pred[s.critical_node], sv["critical_low_J4"]["scale_m"])
    if high:
        comp["high_pressure"] = _component(sv["high_pressure"]["base"], max(pred[n] - s.p_max_m for n in high),
                                           sv["high_pressure"]["scale_m"])
    if imbalance:
        comp["imbalance"] = _component(sv["imbalance"]["base"], spread - s.max_spread_m, sv["imbalance"]["scale_m"])
    if leak_flag:
        key = "burst_like" if burst_flag else "leak_like"
        comp[key] = _component(sv[key]["base"], residual, sv[key]["scale_lps"])
    if flow_inconsistent and not leak_flag:
        comp["flow_inconsistency"] = _component(sv["flow_inconsistency"]["base"], residual, sv["flow_inconsistency"]["scale_lps"])
    if not comp:                                   # normal: how close is the closest limit?
        margins = [pred[n] - s.min_for(n) for n in nodes] + [s.p_max_m - pred[n] for n in nodes] + [s.max_spread_m - spread]
        margin = min(margins)
        comp["closeness_to_limit"] = float(sv["closeness_max"] * max(0.0, 1.0 - margin / s.near_limit_margin_m))
    if tank_low:
        comp["low_tank"] = float(sv["low_pressure"]["base"])
    vals = sorted(comp.values(), reverse=True)
    severity = float(min(100.0, vals[0] + sv["others_weight"] * sum(vals[1:]))) if vals else 0.0

    # ------------------------------------------------------------------ explanation
    pts: list[str] = []
    mins = f"in the next {s.horizon_minutes:g}-minute interval"
    if leak_flag:
        where = f" (largest on {suspect})" if suspect else ""
        pts.append(
            f"Pressure decline ({drop_node}: -{pressure_drop:.1f} m) combined with abnormal flow deviation "
            f"(inflow exceeds metered demand by {residual:.1f} L/s{where}) indicates a possible "
            f"{'burst-like' if burst_flag else 'leak-like'} simulated pattern. "
            + ("Urgent inspection is recommended." if burst_flag else "Inspection is recommended."))
    if critical_low:
        pts.append(f"Predicted pressure at {s.critical_node} is below the configured critical minimum "
                   f"({s.critical_p_min_m:g} m; forecast {pred[s.critical_node]:.1f} m) {mins}.")
    other_low = [n for n in low if n != s.critical_node]
    if other_low:
        pts.append(f"Predicted pressure at {_join(other_low)} is below the configured minimum ({s.p_min_m:g} m; forecast "
                   + ", ".join(f"{n} {pred[n]:.1f} m" for n in other_low) + f") {mins}.")
    for n in high:
        pts.append(f"{n} is forecast to exceed its configured maximum pressure ({s.p_max_m:g} m; forecast {pred[n]:.1f} m); "
                   f"excessive pressure may increase leakage vulnerability.")
    if imbalance:
        pts.append(f"Predicted pressure spread across {_join(nodes)} ({spread:.1f} m) exceeds the configured imbalance "
                   f"threshold ({s.max_spread_m:g} m).")
    if forecast_warning:
        pts.append(f"Early warning: forecast pressure at {s.critical_node} ({pred[s.critical_node]:.1f} m) is within "
                   f"{s.critical_warning_margin_m:g} m of its critical minimum ({s.critical_p_min_m:g} m).")
    if tank_low:
        pts.append(f"Tank level ({inp.tank_level:.2f} m) is below the configured low level ({s.tank_low_level_m:g} m).")
    for r in reasons:
        pts.append(f"Uncertainty: {r}.")
    if label == "normal" and not reasons:
        pts.append(f"All forecast pressures stay inside the configured range ({s.p_min_m:g}-{s.p_max_m:g} m; "
                   f"{s.critical_node} at least {s.critical_p_min_m:g} m) {mins}; spread {spread:.1f} m "
                   f"(limit {s.max_spread_m:g} m); no leak-like pattern detected.")
    explanation = " ".join(pts)

    uncertain = bool(reasons)
    state = {"pump_speed": inp.pump_speed, "pump_status": inp.pump_status, "valve_setting": inp.valve_setting,
             "valve_status": inp.valve_status, "tank_level": inp.tank_level,
             "pump_flow_lps": float(q_in[-1]) if q_in is not None and np.isfinite(q_in[-1]) else None,
             "hub_pressure": (float(np.asarray(inp.pressure_history[s.hub_node], "float64")[-1])
                              if s.hub_node in inp.pressure_history else None)}
    evidence = {"pressure_drop_m": pressure_drop, "pressure_drop_node": drop_node, "flow_demand_residual_lps": residual,
                "residual_threshold_lps": threshold, "branch_residuals_lps": branch_res, "suspected_location": suspect,
                "flow_inconsistency_flag": bool(flow_inconsistent), "inflow_trend_lps_per_step": flow_trend,
                "demand_trend_lps_per_step": demand_trend, "forecast_warning_critical": bool(forecast_warning),
                "tank_low": tank_low}
    return RiskAssessment(
        predicted_risk_label=label, low_pressure_nodes=low, high_pressure_nodes=high, pressure_spread=float(spread),
        pressure_imbalance_flag=bool(imbalance), possible_leak_pattern_flag=leak_flag, severity_score=round(severity, 1),
        human_readable_explanation=explanation, all_labels=labels, burst_like_flag=burst_flag,
        critical_node_at_risk=bool(critical_low), uncertain=uncertain, uncertainty_reasons=reasons,
        explanation_points=pts, severity_components={k: round(v, 1) for k, v in comp.items()},
        predicted_pressure=pred, current_pressure=cur, evidence=evidence, state=state, unavailable_inputs=unavailable)


def assess_windows(x_raw: np.ndarray, predicted: np.ndarray, feature_names: list[str], nodes: list[str],
                   s: RiskSettings) -> list[RiskAssessment]:
    """Assess many windows (N, W, F) with their forecasts (N, K). Uses only the window and the forecast."""
    if len(x_raw) != len(predicted):
        raise RiskError("x_raw and predicted must have the same number of windows")
    return [assess_risk(inputs_from_window(x_raw[i], feature_names, predicted[i], nodes), s) for i in range(len(x_raw))]
