"""Phase 5 tests: transparent risk rules (high / low / spread / J4 critical / leak-like) on controlled situations,
plus one check on real EPANET windows. The engine must never need hidden simulation labels."""
import copy
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import risk_engine  # noqa: E402
from src.baseline_models import PersistenceBaseline  # noqa: E402
from src.risk_engine import (RiskError, RiskInputs, RiskSettings, assess_risk, assess_windows,  # noqa: E402
                             inputs_from_window)
from src.utils import load_config  # noqa: E402
from tests.phase5_helpers import make_inputs  # noqa: E402
from tests.test_dataset import built, cfg as _cfg  # noqa: E402,F401  (real-data fixture)

NODES = ["J2", "J3", "J4"]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def S(cfg):
    return RiskSettings.from_config(cfg)


# ------------------------------------------------------------------------------------------------ normal
def test_normal_situation_has_no_risk(S):
    a = assess_risk(make_inputs(), S)
    assert a.predicted_risk_label == "normal"
    assert a.low_pressure_nodes == [] and a.high_pressure_nodes == []
    assert not a.pressure_imbalance_flag and not a.possible_leak_pattern_flag and not a.uncertain
    assert a.severity_score < 25
    assert "inside the configured range" in a.human_readable_explanation


# ------------------------------------------------------------------------------------------------ high pressure
def test_high_pressure_rule_uses_the_configured_pmax(S):
    a = assess_risk(make_inputs(pred={"J3": 65.0}), S)
    assert a.high_pressure_nodes == ["J3"] and a.predicted_risk_label == "high_pressure_risk"
    assert "J3 is forecast to exceed its configured maximum pressure" in a.human_readable_explanation
    assert "leakage vulnerability" in a.human_readable_explanation


def test_high_pressure_boundary_is_strictly_above_pmax(S):
    assert assess_risk(make_inputs(pred={"J3": S.p_max_m}), S).high_pressure_nodes == []
    assert assess_risk(make_inputs(pred={"J3": S.p_max_m + 0.01}), S).high_pressure_nodes == ["J3"]


def test_thresholds_are_configurable(cfg):
    c = copy.deepcopy(cfg)
    c["thresholds"]["p_max_m"] = 50.0
    s2 = RiskSettings.from_config(c)
    a = assess_risk(make_inputs(pred={"J3": 52.0}), s2)
    assert a.high_pressure_nodes == ["J3"]
    assert assess_risk(make_inputs(pred={"J3": 52.0}), RiskSettings.from_config(cfg)).high_pressure_nodes == []


# ------------------------------------------------------------------------------------------------ low pressure
def test_low_pressure_rule_uses_the_configured_pmin(S):
    a = assess_risk(make_inputs(pred={"J2": 18.0}), S)
    assert a.low_pressure_nodes == ["J2"] and a.predicted_risk_label == "low_pressure_risk"
    assert "below the configured minimum" in a.human_readable_explanation


def test_low_pressure_boundary_is_strictly_below_pmin(S):
    assert assess_risk(make_inputs(pred={"J2": S.p_min_m}), S).low_pressure_nodes == []
    assert assess_risk(make_inputs(pred={"J2": S.p_min_m - 0.01}), S).low_pressure_nodes == ["J2"]


# ------------------------------------------------------------------------------------------------ spread
def test_pressure_spread_rule(S):
    a = assess_risk(make_inputs(pred={"J2": 30.0, "J3": 55.0, "J4": 50.0}), S)
    assert a.pressure_spread == pytest.approx(25.0)
    assert a.pressure_imbalance_flag and a.predicted_risk_label == "pressure_imbalance"
    assert "Predicted pressure spread across J2, J3, and J4" in a.human_readable_explanation
    assert "exceeds the configured imbalance threshold" in a.human_readable_explanation


def test_spread_boundary_is_strictly_above_the_limit(S):
    a = assess_risk(make_inputs(pred={"J2": 35.0, "J3": 55.0, "J4": 50.0}), S)        # spread exactly 20
    assert a.pressure_spread == pytest.approx(20.0) and not a.pressure_imbalance_flag
    b = assess_risk(make_inputs(pred={"J2": 34.9, "J3": 55.0, "J4": 50.0}), S)
    assert b.pressure_imbalance_flag


# ------------------------------------------------------------------------------------------------ J4 critical
def test_j4_critical_threshold_is_stricter_than_the_general_minimum(S):
    a = assess_risk(make_inputs(pred={"J4": 29.0}), S)           # above p_min (20) but below J4's critical 30
    assert a.low_pressure_nodes == ["J4"] and a.critical_node_at_risk
    assert "Predicted pressure at J4 is below the configured critical minimum" in a.human_readable_explanation
    assert "next 15-minute interval" in a.human_readable_explanation
    same_value_elsewhere = assess_risk(make_inputs(pred={"J3": 29.0}), S)
    assert same_value_elsewhere.low_pressure_nodes == [] and not same_value_elsewhere.critical_node_at_risk


def test_j4_boundary_and_early_warning(S):
    assert not assess_risk(make_inputs(pred={"J4": S.critical_p_min_m}), S).critical_node_at_risk
    warn = assess_risk(make_inputs(pred={"J2": 40.0, "J3": 45.0, "J4": S.critical_p_min_m + 1.0}), S)
    assert warn.predicted_risk_label == "normal" and warn.evidence["forecast_warning_critical"]
    assert "Early warning" in warn.human_readable_explanation


def test_j4_shortfall_scores_higher_than_the_same_shortfall_elsewhere(S):
    j4 = assess_risk(make_inputs(pred={"J2": 40.0, "J3": 40.0, "J4": S.critical_p_min_m - 5.0}), S)
    j2 = assess_risk(make_inputs(pred={"J2": S.p_min_m - 5.0, "J3": 22.0, "J4": 32.0}), S)
    assert not j4.pressure_imbalance_flag and not j2.pressure_imbalance_flag     # isolate the low-pressure part
    assert j4.severity_score > j2.severity_score


def test_critical_minimum_may_not_be_looser_than_the_general_one(cfg):
    c = copy.deepcopy(cfg)
    c["thresholds"]["critical_p_min_m"] = 10.0
    with pytest.raises(RiskError, match="STRICTER"):
        RiskSettings.from_config(c)
    c = copy.deepcopy(cfg)
    c["thresholds"]["p_min_m"], c["thresholds"]["p_max_m"] = 70.0, 60.0
    with pytest.raises(RiskError):
        RiskSettings.from_config(c)


# ------------------------------------------------------------------------------------------------ leak-like
def test_leak_like_pattern_needs_pressure_drop_and_flow_inconsistency(S):
    a = assess_risk(make_inputs(leak_lps=2.0, drop_m=1.5), S)
    assert a.possible_leak_pattern_flag and not a.burst_like_flag
    assert a.predicted_risk_label == "leak_like_pattern"
    assert "possible leak-like simulated pattern" in a.human_readable_explanation
    assert "Inspection is recommended" in a.human_readable_explanation
    assert a.evidence["suspected_location"] == "P3" or a.evidence["suspected_location"] == "J3"


def test_flow_inconsistency_without_pressure_drop_is_uncertain_not_a_leak_flag(S):
    a = assess_risk(make_inputs(leak_lps=2.0, drop_m=0.0), S)
    assert not a.possible_leak_pattern_flag
    assert a.evidence["flow_inconsistency_flag"] and a.uncertain
    assert a.severity_score >= 40


def test_pressure_drop_with_consistent_flows_is_not_a_leak(S):
    """A demand peak lowers pressure but flow still matches demand: no leak flag."""
    a = assess_risk(make_inputs(leak_lps=0.0, drop_m=3.0, demand={"J2": 2.5, "J3": 2.2, "J4": 2.0}), S)
    assert not a.possible_leak_pattern_flag and not a.evidence["flow_inconsistency_flag"]


def test_burst_like_pattern_when_the_mismatch_is_large(S):
    a = assess_risk(make_inputs(leak_lps=7.0, drop_m=5.0), S)
    assert a.burst_like_flag and a.possible_leak_pattern_flag
    assert a.predicted_risk_label == "burst_like_pattern" and a.severity_score >= 90
    assert "burst-like" in a.human_readable_explanation


def test_event_label_outranks_pressure_labels_but_pressure_nodes_are_still_reported(S):
    a = assess_risk(make_inputs(pred={"J4": 25.0}, leak_lps=2.0, drop_m=2.0), S)
    assert a.predicted_risk_label == "leak_like_pattern"
    assert a.low_pressure_nodes == ["J4"] and "low_pressure_risk" in a.all_labels


def test_leak_check_is_reported_unavailable_without_flows(S):
    a = assess_risk(make_inputs(flows=False), S)
    assert not a.possible_leak_pattern_flag
    assert any("flow/demand" in u for u in a.unavailable_inputs)


def test_sensor_noise_size_mismatch_does_not_trigger_a_leak(S):
    a = assess_risk(make_inputs(leak_lps=0.1, drop_m=2.0), S)            # 0.1 L/s is below the 0.5 L/s tolerance
    assert not a.possible_leak_pattern_flag and not a.evidence["flow_inconsistency_flag"]


def test_metered_flow_below_demand_is_flagged_uncertain(S):
    a = assess_risk(make_inputs(leak_lps=-1.5, leak_branch="none"), S)
    assert a.uncertain and any("below metered demand" in r for r in a.uncertainty_reasons)


# ------------------------------------------------------------------------------------------------ severity, outputs
def test_severity_is_between_0_and_100_and_grows_with_the_problem(S):
    scores = [assess_risk(make_inputs(pred={"J3": 60 + d}), S).severity_score for d in (1, 5, 10, 20, 60)]
    assert all(0 <= x <= 100 for x in scores) and scores == sorted(scores) and scores[-1] > scores[0]
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = {n: float(rng.uniform(-5, 130)) for n in NODES}
        a = assess_risk(make_inputs(pred=p, leak_lps=float(rng.choice([0, 1, 3, 8])), drop_m=float(rng.choice([0, 2]))), S)
        assert 0 <= a.severity_score <= 100 and a.human_readable_explanation


def test_normal_severity_rises_near_a_limit(S):
    far = assess_risk(make_inputs(pred={"J3": 52.0}), S).severity_score
    near = assess_risk(make_inputs(pred={"J3": 58.5}), S).severity_score
    assert near > far


def test_output_has_every_required_field_and_is_json_serialisable(S):
    a = assess_risk(make_inputs(pred={"J4": 27.0}), S)
    d = a.to_dict()
    for k in ("predicted_risk_label", "low_pressure_nodes", "high_pressure_nodes", "pressure_spread",
              "pressure_imbalance_flag", "possible_leak_pattern_flag", "severity_score", "human_readable_explanation"):
        assert k in d
    json.dumps(d)


def test_missing_forecast_gives_an_uncertain_label_not_a_crash(S):
    inp = make_inputs()
    inp.predicted_pressure["J3"] = float("nan")
    a = assess_risk(inp, S)
    assert a.predicted_risk_label == risk_engine.UNCERTAIN_LABEL and a.uncertain
    assert "No usable forecast" in a.human_readable_explanation


def test_tank_level_is_optional_and_can_be_checked(cfg):
    s0 = RiskSettings.from_config(cfg)
    assert any("tank_level" in u for u in assess_risk(make_inputs(), s0).unavailable_inputs)
    c = copy.deepcopy(cfg)
    c["risk"]["tank_low_level_m"] = 2.0
    s1 = RiskSettings.from_config(c)
    a = assess_risk(make_inputs(tank=1.0), s1)
    assert a.evidence["tank_low"] and "Tank level" in a.human_readable_explanation
    assert not assess_risk(make_inputs(tank=3.0), s1).evidence["tank_low"]


def test_trends_are_reported(S):
    inp = make_inputs()
    inp.flow_history["P1"] = np.linspace(3.0, 3.7, 12)
    a = assess_risk(inp, S)
    assert a.evidence["inflow_trend_lps_per_step"] > 0
    assert a.evidence["demand_trend_lps_per_step"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------------------------------------ no hidden labels
def test_engine_has_no_way_to_receive_hidden_labels(S):
    forbidden = {"leak_flag", "leak_severity", "scenario_type", "scenario_id", "event_label", "label"}
    assert not forbidden & set(RiskInputs.__dataclass_fields__)
    assert list(inspect.signature(assess_risk).parameters) == ["inp", "s"]
    body = inspect.getsource(risk_engine).split('"""', 2)[2]            # code after the module docstring
    for bad in ("scenario_type", "leak_severity", "meta_", "scenario_id", "np.load", "read_csv", '"leak_flag"', "'leak_flag'"):
        assert bad not in body, f"risk_engine code refers to {bad}"


def test_window_adapter_ignores_hidden_columns(S):
    feats = ["J1_pressure", "J2_pressure", "J3_pressure", "J4_pressure", "P1_flow", "P2_flow", "P3_flow", "P4_flow",
             "J2_demand", "J3_demand", "J4_demand", "pump_speed", "pump_status", "valve_setting", "valve_status"]
    x = np.tile(np.array([57, 48, 52, 47, 3.7, 1.4, 1.2, 1.1, 1.4, 1.2, 1.1, 1.0, 1, 48, 2], float), (12, 1))
    a1 = assess_risk(inputs_from_window(x, feats, np.array([48, 52, 47.0]), NODES), S)
    x2 = np.hstack([x, np.ones((12, 2))])                              # extra hidden columns appended
    a2 = assess_risk(inputs_from_window(x2, feats + ["leak_flag", "scenario_type"], np.array([48, 52, 47.0]), NODES), S)
    assert a1.to_dict() == a2.to_dict()
    assert a1.state["pump_speed"] == 1.0 and a1.state["valve_setting"] == 48.0
    with pytest.raises(RiskError):
        inputs_from_window(x[:, :5], feats, np.array([48, 52, 47.0]), NODES)
    with pytest.raises(RiskError):
        inputs_from_window(x, feats, np.array([48, 52.0]), NODES)


# ------------------------------------------------------------------------------------------------ real EPANET windows
def test_real_epanet_windows_leaks_are_flagged_and_normal_scenarios_are_not(built, cfg, S):
    """Uses the real simulated windows and a trivial forecast (persistence); hidden labels are used ONLY to score."""
    d, feats = built["data"]["test"], built["meta"]["input_features"]
    # use all splits to get enough leak windows from the small fixture
    xs, types, actives = [], [], []
    for sp in ("train", "val", "test"):
        xs.append(built["data"][sp]["X_raw"])
        types.append(built["data"][sp]["meta_scenario_type"])
        actives.append(built["data"][sp]["meta_leak_active"].astype(bool))
    x, types, active = np.concatenate(xs), np.concatenate(types), np.concatenate(actives)
    pred = PersistenceBaseline(feats, NODES).predict(x)
    A = assess_windows(x, pred, feats, NODES, S)
    flag = np.array([a.possible_leak_pattern_flag for a in A])
    flow = np.array([a.evidence["flow_inconsistency_flag"] for a in A])
    assert active.sum() > 20, "fixture should contain leak windows"
    assert flow[active].mean() > 0.9                                       # mass balance sees leaks
    assert flag[active].mean() > 0.4                                        # leak-like pattern needs a recent pressure drop too
    clean = np.isin(types, ["normal", "high_demand_stress", "high_source_pressure", "valve_misconfiguration",
                            "reduced_supply", "pump_degradation", "partial_blockage"])
    assert flag[clean].sum() == 0                                           # no false leak flags in leak-free scenarios
    assert all(0 <= a.severity_score <= 100 and a.human_readable_explanation for a in A)
