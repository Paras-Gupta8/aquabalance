"""Bounded, rule-based recommendations for the AquaBalance risk engine.

THESE ARE SIMULATED DECISION-SUPPORT SUGGESTIONS ONLY. Nothing here controls, or claims to control, real equipment;
every output says so, and every suggested change must still be tested in EPANET (Phase 6) and reviewed by an operator.

How a suggestion is made (all visible in the code below):
  1. The risk assessment decides WHICH problem is being addressed, in this safety order:
     burst-like -> leak-like -> J4 critical low -> other low -> high -> imbalance.
  2. Only the actions allowed for that problem are tried, in small steps (valve +-5/10/15 %, pump +5/10 % up, 5 % down).
  3. A transparent first-order "what-if" estimates the pressures after each step (pump: affinity law on the configured
     pump curve; PRV: J2 is pinned at its setting while the valve regulates). It is an approximation, NOT a simulation.
  4. A step is rejected if it leaves a component bound, makes any node's violation worse, or pushes J4 below its minimum.
  5. The smallest step that clears the problem wins; if no step fully clears it, the best partial improvement is offered
     with lower confidence; if nothing safe helps (or the situation is uncertain) the answer is "no action / inspect".

Network facts the rules respect: V1 is a PRV on the J2 branch only, so J3 and J4 can only be moved by the pump; a leak
is never answered with a blind pump-speed increase; a burst-like pattern gets an inspection/isolation workflow, never an
automatic closure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.risk_engine import UNCERTAIN_LABEL, RiskAssessment, RiskSettings, _jsonable

DISCLAIMER = ("Simulated decision-support suggestion only: no real equipment is controlled, and the estimate is a "
              "first-order approximation that must be tested in the EPANET model and reviewed by an operator.")

NO_ACTION_MONITOR = "no_action_monitor"
NO_ACTION_INSPECT = "no_action_inspect"
REDUCE_VALVE = "reduce_valve_setting"
INCREASE_VALVE = "increase_valve_setting"
REDUCE_PUMP = "reduce_pump_speed"
INCREASE_PUMP = "increase_pump_speed"
INSPECT_LEAK = "inspect_possible_leak"
URGENT_BURST = "urgent_inspect_isolate_burst"
ACTION_TYPES = (NO_ACTION_MONITOR, NO_ACTION_INSPECT, REDUCE_VALVE, INCREASE_VALVE, REDUCE_PUMP, INCREASE_PUMP,
                INSPECT_LEAK, URGENT_BURST)
CONTROL_ACTIONS = (REDUCE_VALVE, INCREASE_VALVE, REDUCE_PUMP, INCREASE_PUMP)


class RecommendationError(ValueError):
    """Readable problem with the recommendation settings."""


# =============================================================================================== settings
@dataclass(frozen=True)
class RecommendationSettings:
    risk: RiskSettings
    pump_id: str
    valve_id: str
    valve_node: str                 # the node the PRV feeds (J2)
    pump_min: float
    pump_max: float
    valve_min: float
    valve_max: float
    valve_pct: tuple
    pump_up_pct: tuple
    pump_down_pct: tuple
    target_margin_m: float
    regulating_tol_m: float
    worsen_tol_m: float
    confidence: dict
    pump_curve: tuple               # (h0, r, n): EPANET three-point curve  h = h0 - r * q^n  (q in L/s, h in m)
    elevation: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "RecommendationSettings":
        try:
            rc, net = cfg["recommendation"], cfg["network"]
            pts = net["pump"]["head_curve_lps_m"]
            (q0, h0), (q1, h1), (q2, h2) = pts
            if not (q0 == 0 and h0 > h1 > h2 > 0 and 0 < q1 < q2):
                raise RecommendationError("pump head curve must fall with flow (shut-off, design, maximum points)")
            n = math.log((h0 - h2) / (h0 - h1)) / math.log(q2 / q1)       # EPANET 3-point curve fit
            r = (h0 - h1) / (q1 ** n)
            s = cls(risk=RiskSettings.from_config(cfg), pump_id=str(net["pump"]["id"]), valve_id=str(net["valve"]["id"]),
                    valve_node=str(net["valve"]["to_node"]),
                    pump_min=float(rc["bounds"]["pump_speed"]["min"]), pump_max=float(rc["bounds"]["pump_speed"]["max"]),
                    valve_min=float(rc["bounds"]["valve_setting_m"]["min"]),
                    valve_max=float(rc["bounds"]["valve_setting_m"]["max"]),
                    valve_pct=tuple(sorted(float(p) for p in rc["valve_adjust_pct"])),
                    pump_up_pct=tuple(sorted(float(p) for p in rc["pump_increase_pct"])),
                    pump_down_pct=tuple(sorted(float(p) for p in rc["pump_decrease_pct"])),
                    target_margin_m=float(rc["target_margin_m"]), regulating_tol_m=float(rc["regulating_tolerance_m"]),
                    worsen_tol_m=float(rc["worsen_tolerance_m"]), confidence=dict(rc["confidence"]),
                    pump_curve=(float(h0), float(r), float(n)),
                    elevation={j["id"]: float(j["elevation_m"]) for j in net["junctions"]})
        except KeyError as exc:
            raise RecommendationError(f"config is missing the key {exc}; see the 'recommendation' section of "
                                      f"config/config.yaml") from None
        s.validate()
        return s

    def validate(self) -> None:
        if not 0 < self.pump_min < self.pump_max:
            raise RecommendationError("pump speed bounds must satisfy 0 < min < max")
        if not 0 < self.valve_min < self.valve_max:
            raise RecommendationError("valve setting bounds must satisfy 0 < min < max")
        for name, vals, limit in (("valve_adjust_pct", self.valve_pct, 15), ("pump_increase_pct", self.pump_up_pct, 10),
                                  ("pump_decrease_pct", self.pump_down_pct, 5)):
            if not vals or any(v <= 0 or v > limit for v in vals):
                raise RecommendationError(f"{name} must be non-empty percentages in (0, {limit}]")

    def pump_head(self, q_lps: float, speed: float) -> float:
        """Pump head (m) at flow q and relative speed (affinity laws on the configured curve)."""
        h0, r, n = self.pump_curve
        q_eq = max(q_lps, 0.0) / speed
        return float(speed ** 2 * max(h0 - r * q_eq ** n, 0.0))


# =============================================================================================== output
@dataclass
class Recommendation:
    recommended_action_type: str
    target_component: str
    current_value: float | None
    recommended_value: float | None
    adjustment_percent: float
    rationale: str
    safety_note: str
    confidence: float
    expected_effect: str
    # ---- supporting detail ----
    unit: str = ""
    addresses: str = ""
    predicted_pressure_before: dict = field(default_factory=dict)
    predicted_pressure_after: dict = field(default_factory=dict)
    alternatives_considered: list = field(default_factory=list)
    is_simulated_decision_support: bool = True
    controls_real_equipment: bool = False

    def to_dict(self) -> dict:
        return _jsonable({k: getattr(self, k) for k in self.__dataclass_fields__})


# =============================================================================================== what-if model
def _regulating(a: RiskAssessment, s: RecommendationSettings) -> bool:
    """Is the PRV (V1) holding J2 at its setting? (EPANET status 'active', or J2 forecast sits on the setting.)"""
    st = a.state
    j2 = a.predicted_pressure.get(s.valve_node)
    setting = st.get("valve_setting")
    if setting is None or j2 is None:
        return False
    return bool(st.get("valve_status") == 2 or abs(j2 - setting) <= s.regulating_tol_m)


def _ceiling(a: RiskAssessment, s: RecommendationSettings, delta_h: float = 0.0) -> float:
    """Highest pressure J2 can reach: hub pressure now minus the static height difference (plus any pump head change)."""
    hub = a.state.get("hub_pressure")
    if hub is None:
        return float("inf")
    rise = s.elevation.get(s.valve_node, 0.0) - s.elevation.get(s.risk.hub_node, 0.0)
    return hub - rise + delta_h


def what_if(a: RiskAssessment, s: RecommendationSettings, kind: str, new_value: float) -> dict:
    """First-order estimate of next-interval pressures after ONE change. Approximation only (see module docstring)."""
    after = dict(a.predicted_pressure)
    st = a.state
    j2n = s.valve_node
    regulating = _regulating(a, s)
    if kind in (REDUCE_PUMP, INCREASE_PUMP):
        q = st.get("pump_flow_lps") or 0.0
        dh = s.pump_head(q, new_value) - s.pump_head(q, st["pump_speed"])
        for n in after:
            if n != j2n:
                after[n] += dh                       # hub-fed nodes move with the pump head (friction change ignored)
        if regulating and st.get("valve_setting") is not None:
            after[j2n] = min(st["valve_setting"], _ceiling(a, s, dh))      # PRV keeps J2 at its setting if it can
        elif j2n in after:
            after[j2n] += dh
    elif kind in (REDUCE_VALVE, INCREASE_VALVE):
        if regulating:
            after[j2n] = min(new_value, _ceiling(a, s))
        elif new_value < after[j2n]:
            after[j2n] = new_value                   # a setting below the free pressure throttles the valve
    return after


def violation_parts(p: dict, s: RecommendationSettings) -> dict:
    """Per-node distance outside the limits (metres) and spread excess. All zero = no rule broken."""
    r = s.risk
    parts = {f"low_{n}": max(0.0, r.min_for(n) - v) for n, v in p.items()}
    parts.update({f"high_{n}": max(0.0, v - r.p_max_m) for n, v in p.items()})
    parts["spread"] = max(0.0, (max(p.values()) - min(p.values())) - r.max_spread_m)
    return parts


# =============================================================================================== candidates
@dataclass
class _Candidate:
    kind: str
    component: str
    unit: str
    current: float
    new: float
    pct: float
    after: dict
    parts_before: dict
    parts_after: dict
    v_before: float
    v_after: float
    admissible: bool
    reject_reason: str = ""


def _within_bounds(cur: float, new: float, lo: float, hi: float) -> bool:
    """New value must be inside [lo, hi]. If the CURRENT value is already outside the range (a misconfigured
    component), a step that moves it toward the range without overshooting is the only other thing allowed."""
    if lo <= new <= hi:
        return True
    if cur > hi:
        return lo <= new < cur
    if cur < lo:
        return cur < new <= hi
    return False


def _make_candidate(a: RiskAssessment, s: RecommendationSettings, kind: str, pct: float) -> _Candidate | None:
    st = a.state
    if kind in (REDUCE_PUMP, INCREASE_PUMP):
        cur = st.get("pump_speed")
        if cur is None or not np.isfinite(cur) or cur <= 0 or (st.get("pump_status") is not None and st["pump_status"] < 0.5):
            return None
        new = cur * (1 + pct / 100.0 * (1 if kind == INCREASE_PUMP else -1))
        comp, unit, lo, hi = s.pump_id, "x nominal speed", s.pump_min, s.pump_max
    else:
        cur = st.get("valve_setting")
        if cur is None or not np.isfinite(cur) or cur <= 0:
            return None
        new = cur * (1 + pct / 100.0 * (1 if kind == INCREASE_VALVE else -1))
        comp, unit, lo, hi = s.valve_id, "m", s.valve_min, s.valve_max
    new = round(float(new), 4)
    before = a.predicted_pressure
    pb = violation_parts(before, s)
    reason = ""
    if not _within_bounds(float(cur), new, lo, hi):
        reason = f"{new:.3g} would leave the allowed range {lo:g}-{hi:g}"
        after = dict(before)
    else:
        after = what_if(a, s, kind, new)
    pa = violation_parts(after, s)
    if not reason:
        worse = [k for k in pa if pa[k] > pb[k] + s.worsen_tol_m]
        if worse:
            reason = "would make " + ", ".join(w.replace("_", " ") for w in worse) + " worse"
    return _Candidate(kind, comp, unit, float(cur), new, pct, after, pb, pa, sum(pb.values()), sum(pa.values()),
                      admissible=not reason, reject_reason=reason)


def _pool(a: RiskAssessment, s: RecommendationSettings, problem: str) -> list[_Candidate]:
    kinds: list[tuple[str, tuple]] = []
    j2n = s.valve_node
    if problem in ("critical_low", "other_low"):
        kinds.append((INCREASE_PUMP, s.pump_up_pct))
        if problem == "critical_low" or j2n in a.low_pressure_nodes:
            kinds.append((INCREASE_VALVE, s.valve_pct))        # evaluated honestly: only helps if J2 is the low node
    elif problem == "high":
        kinds.append((REDUCE_VALVE, s.valve_pct))
        network_wide = len(a.high_pressure_nodes) >= 2 or any(n != j2n for n in a.high_pressure_nodes)
        if network_wide:
            kinds.append((REDUCE_PUMP, s.pump_down_pct))
    elif problem == "imbalance":
        kinds.append((REDUCE_VALVE, s.valve_pct))
        kinds.append((INCREASE_VALVE, s.valve_pct))
    out = []
    for kind, pcts in kinds:
        for p in pcts:
            c = _make_candidate(a, s, kind, p)
            if c is not None:
                out.append(c)
    return out


def _target_cleared(problem: str, c: _Candidate, a: RiskAssessment, s: RecommendationSettings) -> bool:
    r, m = s.risk, s.target_margin_m
    p = c.after
    if problem == "critical_low":
        return p[r.critical_node] >= r.critical_p_min_m + m
    if problem == "other_low":
        return all(p[n] >= r.min_for(n) + m for n in a.low_pressure_nodes)
    if problem == "high":
        return all(p[n] <= r.p_max_m - m for n in a.high_pressure_nodes)
    if problem == "imbalance":
        return (max(p.values()) - min(p.values())) <= r.max_spread_m - m
    return False


# =============================================================================================== recommendations
def _pack(kind, component, current, new, pct, rationale, safety, confidence, effect, unit="", addresses="",
          before=None, after=None, alts=None) -> Recommendation:
    return Recommendation(kind, component, current, new, float(pct), rationale, f"{safety} {DISCLAIMER}".strip(),
                          round(float(confidence), 2), effect, unit=unit, addresses=addresses,
                          predicted_pressure_before=before or {}, predicted_pressure_after=after or {},
                          alternatives_considered=alts or [])


def _fmt(p: dict) -> str:
    return ", ".join(f"{n} {v:.1f} m" for n, v in p.items())


def conservative_default(a: RiskAssessment, s: RecommendationSettings, why: str) -> Recommendation:
    """The safe fallback: change nothing, check the situation."""
    return _pack(NO_ACTION_INSPECT, "network", None, None, 0.0,
                 f"No control change is recommended because {why}. Inspect the sensors, the forecast inputs and the "
                 f"network before any adjustment.",
                 "No equipment change is proposed; nothing here should be applied to real equipment.",
                 s.confidence["inspect"], "None: the conservative default leaves all settings unchanged.",
                 addresses="uncertain", before=a.predicted_pressure)


def recommend(a: RiskAssessment, s: RecommendationSettings) -> Recommendation:
    """One bounded, explained, simulated recommendation for an assessed situation."""
    r = s.risk
    before = a.predicted_pressure
    # -------------------------------------------------------------------- uncertain -> conservative default
    if a.predicted_risk_label == UNCERTAIN_LABEL or a.uncertain:
        why = "; ".join(a.uncertainty_reasons) or "the inputs are insufficient"
        rec = conservative_default(a, s, f"the situation is uncertain ({why})")
        if a.critical_node_at_risk or a.low_pressure_nodes or a.high_pressure_nodes or a.pressure_imbalance_flag:
            rec.rationale += (" Forecast pressure rules are also triggered (" + a.human_readable_explanation.split(" Uncertainty:")[0]
                              + "), but a control change is not proposed while the data are in doubt; in particular pump "
                              "speed is not increased while a leak or a meter fault cannot be ruled out.")
        return rec
    # -------------------------------------------------------------------- burst-like: urgent inspection, no closure
    if a.burst_like_flag:
        where = a.evidence.get("suspected_location") or "the network"
        return _pack(URGENT_BURST, where, None, None, 0.0,
                     f"A burst-like simulated pattern is indicated ({a.evidence['flow_demand_residual_lps']:.1f} L/s of inflow "
                     f"is not matched by metered demand, with a pressure drop of {a.evidence['pressure_drop_m']:.1f} m). "
                     f"Start the urgent simulated inspection / isolation workflow for {where}.",
                     "No automatic valve closure and no pump change are recommended: isolation must be planned by an "
                     "operator who confirms the network layout and which customers (e.g. the hospital node J4) lose supply.",
                     s.confidence["burst_like"], "Faster location of the loss and a controlled isolation decision; "
                     "no pressure change is claimed.", addresses="burst_like_pattern", before=before)
    # -------------------------------------------------------------------- leak-like: inspect, never a blind pump increase
    if a.possible_leak_pattern_flag:
        where = a.evidence.get("suspected_location") or "the network"
        extra = ""
        if a.critical_node_at_risk:
            extra = (f" {r.critical_node} is also forecast below its critical minimum; raising pump speed would raise leak "
                     f"outflow and is deliberately not recommended until the leak is located.")
        return _pack(INSPECT_LEAK, where, None, None, 0.0,
                     f"A possible leak-like simulated pattern was detected (inflow exceeds metered demand by "
                     f"{a.evidence['flow_demand_residual_lps']:.1f} L/s; pressure drop {a.evidence['pressure_drop_m']:.1f} m). "
                     f"Inspection of {where} is recommended.{extra}",
                     "Do not increase pump speed blindly: more pressure increases leakage and can hide the cause.",
                     s.confidence["leak_like"], "Locating the leak is expected to explain the pressure loss; "
                     "no pressure change is claimed.", addresses="leak_like_pattern", before=before)
    # -------------------------------------------------------------------- normal
    if a.predicted_risk_label == "normal":
        return _pack(NO_ACTION_MONITOR, "network", None, None, 0.0,
                     "All forecast pressures are inside the configured limits and no leak-like pattern is detected; "
                     "keep monitoring.", "No change proposed.", s.confidence["resolves"],
                     "None needed: settings stay unchanged.", addresses="normal", before=before)
    # -------------------------------------------------------------------- pressure problems, safety order
    problems = []
    if a.critical_node_at_risk:
        problems.append("critical_low")
    if [n for n in a.low_pressure_nodes if n != r.critical_node]:
        problems.append("other_low")
    if a.high_pressure_nodes:
        problems.append("high")
    if a.pressure_imbalance_flag:
        problems.append("imbalance")
    problem = problems[0]
    cands = _pool(a, s, problem)
    alts = [f"{c.kind} {c.pct:g}% -> {c.new:.3g}: rejected, {c.reject_reason}" for c in cands if not c.admissible]
    ok = [c for c in cands if c.admissible]
    full = sorted([c for c in ok if _target_cleared(problem, c, a, s)], key=lambda c: (c.pct, c.v_after))
    partial = sorted([c for c in ok if c.v_after < c.v_before - s.worsen_tol_m and c not in full],
                     key=lambda c: (c.v_after, c.pct))
    ok_no_help = [c for c in ok if c not in full and c not in partial]
    alts += [f"{c.kind} {c.pct:g}%: no predicted benefit" for c in ok_no_help]
    pick, level = (full[0], "resolves") if full else (partial[0], "partial") if partial else (None, "")
    if pick is None:
        why = {"critical_low": f"no allowed pump step is predicted to lift {r.critical_node} to its critical minimum safely "
                              f"(the valve V1 only serves {s.valve_node})",
               "other_low": "no allowed adjustment is predicted to raise the low node without creating a new violation",
               "high": "no allowed adjustment is predicted to lower the high pressure without creating a new violation",
               "imbalance": "no allowed valve adjustment is predicted to improve the spread without creating a new violation"}[problem]
        rec = conservative_default(a, s, why)
        rec.addresses, rec.alternatives_considered = problem, alts
        rec.rationale += " " + a.human_readable_explanation
        return rec
    conf = s.confidence["resolves"] if level == "resolves" else s.confidence["partial"]
    if any(abs(before[n] - (a.current_pressure.get(n) or before[n])) > 0.5 * r.max_forecast_change_m for n in before):
        conf = max(0.1, conf - s.confidence["unreliable_forecast_penalty"])
    remaining = [p for p in problems[1:] if p]
    direction = "reduce" if pick.kind in (REDUCE_PUMP, REDUCE_VALVE) else "increase"
    what = "pump speed" if pick.kind in (REDUCE_PUMP, INCREASE_PUMP) else "PRV setting"
    why_txt = {"critical_low": f"{r.critical_node} is forecast below its critical minimum",
               "other_low": f"{_join_nodes([n for n in a.low_pressure_nodes if n != r.critical_node])} forecast below the minimum",
               "high": f"{_join_nodes(a.high_pressure_nodes)} forecast above the maximum",
               "imbalance": "the forecast pressure spread exceeds the imbalance threshold"}[problem]
    rationale = (f"{a.human_readable_explanation.split(' Uncertainty:')[0]} Suggested simulated step: {direction} {what} of "
                 f"{pick.component} by {pick.pct:g}% ({pick.current:.3g} -> {pick.new:.3g} {pick.unit}) because {why_txt}.")
    if level == "partial":
        rationale += (" The step is predicted to reduce but not fully clear the problem; re-run the forecast after it and "
                      "inspect the supply side if the problem persists.")
    if remaining:
        rationale += " Other rules still triggered (handled after this one): " + ", ".join(remaining) + "."
    safety = (f"Step limited to {pick.pct:g}% and kept inside the allowed range; checked that no node gets worse and that "
              f"{r.critical_node} is not pushed below its minimum.")
    if problem == "critical_low":
        safety += f" {r.critical_node} (critical service) is protected first."
    if _outside(pick, s):
        safety += " The current value is already outside the configured operating range; this step moves it toward the range."
    effect = (f"Model estimate (first-order, not yet verified in EPANET): {_fmt(before)} -> {_fmt(pick.after)}; "
              f"total rule violation {pick.v_before:.1f} m -> {pick.v_after:.1f} m.")
    return _pack(pick.kind, pick.component, round(pick.current, 4), round(pick.new, 4),
                 round(pick.pct, 1) * (-1 if direction == "reduce" else 1), rationale, safety, conf, effect,
                 unit=pick.unit, addresses=problem, before=before, after=pick.after, alts=alts)


def _outside(c: _Candidate, s: RecommendationSettings) -> bool:
    lo, hi = ((s.pump_min, s.pump_max) if c.kind in (REDUCE_PUMP, INCREASE_PUMP) else (s.valve_min, s.valve_max))
    return not (lo <= c.current <= hi)


def _join_nodes(names: list[str]) -> str:
    from src.risk_engine import _join
    return _join(names)
