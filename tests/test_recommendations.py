"""Phase 5 tests: bounded, explained, simulated recommendations."""
import copy
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src import recommendation_engine as re_  # noqa: E402
from src.recommendation_engine import (DISCLAIMER, RecommendationError, RecommendationSettings, recommend,  # noqa: E402
                                       what_if)
from src.risk_engine import RiskSettings, assess_risk  # noqa: E402
from src.utils import load_config  # noqa: E402
from tests.phase5_helpers import make_inputs  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def S(cfg):
    return RiskSettings.from_config(cfg)


@pytest.fixture(scope="module")
def RS(cfg):
    return RecommendationSettings.from_config(cfg)


def rec_for(S, RS, **kw):
    a = assess_risk(make_inputs(**kw), S)
    return a, recommend(a, RS)


# ------------------------------------------------------------------------------------------------ high pressure
def test_branch_high_pressure_reduces_the_valve_by_5_to_15_percent(S, RS):
    a, r = rec_for(S, RS, now={"J2": 62.0}, pred={"J2": 62.0, "J3": 52.0, "J4": 47.0}, valve_setting=62.0)
    assert r.recommended_action_type == "reduce_valve_setting" and r.target_component == "V1"
    assert -15.0 <= r.adjustment_percent <= -5.0
    assert r.recommended_value == pytest.approx(r.current_value * (1 + r.adjustment_percent / 100), rel=1e-3)
    assert r.predicted_pressure_after["J2"] <= 60.0 and r.confidence >= 0.8


def test_network_wide_high_pressure_reduces_pump_speed_by_5_percent(S, RS):
    a, r = rec_for(S, RS, now={"J3": 66.0, "J4": 64.0}, pred={"J3": 66.0, "J4": 64.0, "J2": 50.0}, pump_speed=1.0)
    assert r.recommended_action_type == "reduce_pump_speed" and r.target_component == "PU1"
    assert r.adjustment_percent == -5.0 and r.recommended_value == pytest.approx(0.95)
    assert r.predicted_pressure_after["J3"] < 66.0 and r.predicted_pressure_after["J4"] < 64.0


# ------------------------------------------------------------------------------------------------ J4 low
def test_j4_low_pressure_increases_pump_speed_within_5_to_10_percent(S, RS):
    a, r = rec_for(S, RS, now={"J4": 28.8, "J3": 38.0, "J2": 40.0}, pred={"J4": 28.8, "J3": 38.0, "J2": 40.0},
                   pump_speed=1.0, valve_setting=40.0)
    assert a.critical_node_at_risk
    assert r.recommended_action_type == "increase_pump_speed"
    assert 5.0 <= r.adjustment_percent <= 10.0
    assert r.predicted_pressure_after["J4"] >= 30.0                       # clears the critical minimum
    assert "J4" in r.rationale and "critical" in r.rationale.lower()
    assert "J4" in r.safety_note


def test_a_larger_deficit_is_only_partly_fixed_and_confidence_drops(S, RS):
    full = rec_for(S, RS, now={"J4": 29.0, "J3": 40.0, "J2": 40.0}, pred={"J4": 29.0, "J3": 40.0, "J2": 40.0},
                   valve_setting=40.0)[1]
    part = rec_for(S, RS, now={"J4": 24.0, "J3": 40.0, "J2": 40.0}, pred={"J4": 24.0, "J3": 40.0, "J2": 40.0},
                   valve_setting=40.0)[1]
    assert full.recommended_action_type == part.recommended_action_type == "increase_pump_speed"
    assert part.confidence < full.confidence
    assert "reduce but not fully clear" in part.rationale
    assert part.predicted_pressure_after["J4"] < 30.0 + 0.5                # honest: not claimed fixed


def test_j4_is_handled_before_other_problems(S, RS):
    a, r = rec_for(S, RS, now={"J2": 40.0, "J3": 52.0, "J4": 28.0}, pred={"J2": 40.0, "J3": 52.0, "J4": 28.0},
                   valve_setting=40.0)
    assert a.critical_node_at_risk and r.addresses == "critical_low"
    assert r.recommended_action_type == "increase_pump_speed"


def test_pump_increase_that_would_push_another_node_over_pmax_is_rejected(S, RS):
    a, r = rec_for(S, RS, now={"J4": 28.0, "J3": 59.6, "J2": 45.0}, pred={"J4": 28.0, "J3": 59.6, "J2": 45.0},
                   valve_setting=45.0)
    assert r.recommended_action_type == "no_action_inspect"
    assert any("worse" in alt for alt in r.alternatives_considered)


# ------------------------------------------------------------------------------------------------ imbalance
def test_imbalance_uses_the_smallest_valve_step_that_clears_it(S, RS):
    a, r = rec_for(S, RS, now={"J2": 29.0, "J3": 50.0, "J4": 48.0}, pred={"J2": 29.0, "J3": 50.0, "J4": 48.0},
                   valve_setting=29.0)
    assert a.pressure_imbalance_flag
    assert r.recommended_action_type == "increase_valve_setting" and r.target_component == "V1"
    assert r.adjustment_percent == 10.0                                   # 5 % is not enough, 10 % is
    after = r.predicted_pressure_after
    assert max(after.values()) - min(after.values()) <= 20.0
    assert after["J4"] >= 30.0


def test_imbalance_fix_never_places_j4_below_its_minimum(S, RS):
    for j4 in (30.5, 32.0, 40.0):
        a, r = rec_for(S, RS, now={"J2": 12.0 + j4 / 2, "J3": j4 + 22.0, "J4": j4},
                       pred={"J2": 12.0 + j4 / 2, "J3": j4 + 22.0, "J4": j4}, valve_setting=12.0 + j4 / 2)
        if r.recommended_action_type in re_.CONTROL_ACTIONS:
            assert r.predicted_pressure_after["J4"] >= 30.0


def test_imbalance_that_a_valve_cannot_improve_falls_back_to_inspect(S, RS):
    # J3 high relative to J2 AND J2's valve already at its lowest allowed setting direction: nothing admissible helps
    a, r = rec_for(S, RS, now={"J2": 40.0, "J3": 61.0, "J4": 40.5}, pred={"J2": 40.0, "J3": 61.0, "J4": 40.5},
                   valve_setting=15.0, valve_status=1)
    assert r.recommended_action_type in ("no_action_inspect", "reduce_pump_speed")
    assert r.recommended_action_type != "increase_pump_speed"


# ------------------------------------------------------------------------------------------------ bounds
def test_valve_adjustments_stay_inside_allowed_bounds(S, RS):
    lo, hi = RS.valve_min, RS.valve_max
    seen = 0
    for j2, j3, j4, setting in itertools.product((14, 18, 25, 45, 62, 75, 88), (35, 50, 70), (28, 45), (lo, 20, 50, hi)):
        a, r = rec_for(S, RS, now={"J2": j2, "J3": j3, "J4": j4}, pred={"J2": j2, "J3": j3, "J4": j4},
                       valve_setting=float(setting), valve_status=2)
        if r.target_component == "V1" and r.recommended_action_type in re_.CONTROL_ACTIONS:
            seen += 1
            assert lo <= r.recommended_value <= hi, (r.recommended_value, r.current_value)
            assert 5.0 <= abs(r.adjustment_percent) <= 15.0
    assert seen > 10


def test_valve_increase_at_the_upper_bound_is_not_offered(S, RS):
    a, r = rec_for(S, RS, now={"J2": 20.0, "J3": 50.0, "J4": 48.0}, pred={"J2": 18.0, "J3": 50.0, "J4": 48.0},
                   valve_setting=RS.valve_max)
    assert not (r.target_component == "V1" and r.recommended_value and r.recommended_value > RS.valve_max)


def test_pump_adjustments_stay_inside_allowed_bounds(S, RS):
    seen = 0
    for speed in (RS.pump_min, 0.65, 0.9, 1.0, 1.05, RS.pump_max):
        for p in ({"J4": 27.0, "J3": 40.0, "J2": 40.0}, {"J3": 66.0, "J4": 64.0, "J2": 50.0}, {"J4": 29.5, "J3": 45.0, "J2": 45.0}):
            a, r = rec_for(S, RS, now=p, pred=p, pump_speed=speed, valve_setting=p["J2"])
            if r.target_component == "PU1" and r.recommended_action_type in re_.CONTROL_ACTIONS:
                seen += 1
                assert RS.pump_min <= r.recommended_value <= RS.pump_max, (speed, r.recommended_value)
                if r.recommended_action_type == "increase_pump_speed":
                    assert 5.0 <= r.adjustment_percent <= 10.0
                else:
                    assert -5.0 <= r.adjustment_percent <= -5.0
    assert seen >= 8


def test_pump_increase_at_the_maximum_is_refused(S, RS):
    a, r = rec_for(S, RS, now={"J4": 28.0, "J3": 40.0, "J2": 40.0}, pred={"J4": 28.0, "J3": 40.0, "J2": 40.0},
                   pump_speed=RS.pump_max, valve_setting=40.0)
    assert r.recommended_action_type == "no_action_inspect"
    assert any("allowed range" in alt for alt in r.alternatives_considered)


def test_pump_reduction_at_the_minimum_is_refused(S, RS):
    a, r = rec_for(S, RS, now={"J3": 66.0, "J4": 64.0, "J2": 50.0}, pred={"J3": 66.0, "J4": 64.0, "J2": 50.0},
                   pump_speed=RS.pump_min, valve_setting=50.0)
    assert r.recommended_action_type != "reduce_pump_speed"


def test_a_component_already_outside_its_range_is_only_moved_toward_it(S, RS):
    a, r = rec_for(S, RS, now={"J3": 80.0, "J4": 75.0, "J2": 50.0}, pred={"J3": 80.0, "J4": 75.0, "J2": 50.0},
                   pump_speed=1.3, valve_setting=50.0)               # e.g. the 'fast pump' misconfiguration
    assert r.recommended_action_type == "reduce_pump_speed" and r.recommended_value < 1.3
    assert "already outside" in r.safety_note


def test_bounds_are_validated_from_config(cfg):
    c = copy.deepcopy(cfg)
    c["recommendation"]["bounds"]["pump_speed"] = {"min": 1.2, "max": 1.0}
    with pytest.raises(RecommendationError):
        RecommendationSettings.from_config(c)
    c = copy.deepcopy(cfg)
    c["recommendation"]["pump_increase_pct"] = [5, 25]
    with pytest.raises(RecommendationError, match="pump_increase_pct"):
        RecommendationSettings.from_config(c)


# ------------------------------------------------------------------------------------------------ leak / burst
def test_leak_like_pattern_recommends_inspection_and_no_pump_increase(S, RS):
    a, r = rec_for(S, RS, now={"J4": 27.0, "J3": 40.0, "J2": 40.0}, pred={"J4": 27.0, "J3": 40.0, "J2": 40.0},
                   leak_lps=2.0, drop_m=2.0, valve_setting=40.0)
    assert a.possible_leak_pattern_flag and a.critical_node_at_risk
    assert r.recommended_action_type == "inspect_possible_leak"
    assert r.recommended_value is None and r.adjustment_percent == 0.0
    assert "not recommended" in r.rationale and "blindly" in r.safety_note


def test_no_scenario_with_leak_evidence_ever_gets_a_pump_increase(S, RS):
    for lps, drop, j4 in itertools.product((0.8, 2.0, 7.0), (0.0, 2.0, 6.0), (20.0, 28.0, 45.0)):
        a, r = rec_for(S, RS, now={"J4": j4, "J3": 45.0, "J2": 45.0}, pred={"J4": j4, "J3": 45.0, "J2": 45.0},
                       leak_lps=lps, drop_m=drop, valve_setting=45.0)
        assert r.recommended_action_type != "increase_pump_speed", (lps, drop, j4)
        assert r.recommended_action_type not in re_.CONTROL_ACTIONS


def test_burst_like_pattern_gets_urgent_inspection_without_automatic_closure(S, RS):
    a, r = rec_for(S, RS, leak_lps=7.0, drop_m=5.0)
    assert a.burst_like_flag and r.recommended_action_type == "urgent_inspect_isolate_burst"
    assert r.recommended_value is None and r.adjustment_percent == 0.0
    assert "No automatic valve closure" in r.safety_note
    assert r.recommended_action_type not in re_.CONTROL_ACTIONS


# ------------------------------------------------------------------------------------------------ conservative default
def test_uncertain_conditions_give_the_conservative_default(S, RS):
    a, r = rec_for(S, RS, now={"J4": 27.0, "J3": 40.0, "J2": 40.0}, pred={"J4": 27.0, "J3": 40.0, "J2": 40.0},
                   leak_lps=2.0, drop_m=0.0, valve_setting=40.0)           # flow mismatch but no pressure drop
    assert a.uncertain and r.recommended_action_type == "no_action_inspect"
    assert r.recommended_value is None and r.confidence <= 0.4
    assert "uncertain" in r.rationale


def test_missing_forecast_gives_the_conservative_default(S, RS):
    inp = make_inputs()
    inp.predicted_pressure["J3"] = float("nan")
    r = recommend(assess_risk(inp, S), RS)
    assert r.recommended_action_type == "no_action_inspect"


def test_normal_situation_needs_no_action(S, RS):
    a, r = rec_for(S, RS)
    assert r.recommended_action_type == "no_action_monitor" and r.adjustment_percent == 0.0
    assert r.recommended_value is None


def test_pump_off_never_gets_a_pump_recommendation(S, RS):
    a, r = rec_for(S, RS, now={"J4": 28.0, "J3": 40.0, "J2": 40.0}, pred={"J4": 28.0, "J3": 40.0, "J2": 40.0},
                   pump_status=0.0, valve_setting=40.0)
    assert r.recommended_action_type == "no_action_inspect"


# ------------------------------------------------------------------------------------------------ wording / honesty
SITUATIONS = [
    dict(), dict(pred={"J3": 66.0, "J4": 64.0}), dict(pred={"J4": 27.0}), dict(pred={"J2": 18.0}),
    dict(pred={"J2": 28.0, "J3": 55.0, "J4": 50.0}, now={"J2": 28.0}, valve_setting=28.0),
    dict(leak_lps=2.0, drop_m=2.0), dict(leak_lps=7.0, drop_m=5.0), dict(leak_lps=2.0),
    dict(pred={"J2": 62.0}, now={"J2": 62.0}, valve_setting=62.0),
]


@pytest.mark.parametrize("kw", SITUATIONS)
def test_every_recommendation_has_a_clear_explanation_and_safety_note(S, RS, kw):
    a, r = rec_for(S, RS, **kw)
    assert r.recommended_action_type in re_.ACTION_TYPES
    assert len(r.rationale) > 40 and len(r.expected_effect) > 10 and len(r.safety_note) > 40
    assert "Simulated decision-support suggestion only" in r.safety_note and "no real equipment is controlled" in r.safety_note
    assert 0.0 <= r.confidence <= 1.0
    assert r.is_simulated_decision_support and not r.controls_real_equipment
    assert r.target_component
    d = r.to_dict()
    for k in ("recommended_action_type", "target_component", "current_value", "recommended_value", "adjustment_percent",
              "rationale", "safety_note", "confidence", "expected_effect"):
        assert k in d
    json.dumps(d)
    text = (r.rationale + " " + r.safety_note + " " + r.expected_effect).lower()
    assert "will control" not in text and "automatically close" not in text.replace("no automatic", "")


def test_estimates_are_labelled_as_unverified(S, RS):
    a, r = rec_for(S, RS, now={"J4": 28.8, "J3": 38.0, "J2": 40.0}, pred={"J4": 28.8, "J3": 38.0, "J2": 40.0},
                   valve_setting=40.0)
    assert "first-order" in r.expected_effect and "not yet verified in EPANET" in r.expected_effect
    assert DISCLAIMER.split(":")[0] in r.safety_note


# ------------------------------------------------------------------------------------------------ the what-if model
def test_pump_curve_fit_reproduces_the_configured_points(RS):
    for q, h in ((0, 12.0), (5, 10.0), (10, 6.0)):
        assert RS.pump_head(q, 1.0) == pytest.approx(h, abs=1e-6)
    assert RS.pump_head(5, 1.1) > RS.pump_head(5, 1.0) > RS.pump_head(5, 0.9)           # affinity: faster = more head


def test_whatif_pump_moves_hub_fed_nodes_but_not_a_regulated_j2(S, RS):
    a = assess_risk(make_inputs(), S)
    up = what_if(a, RS, re_.INCREASE_PUMP, 1.05)
    assert up["J3"] > a.predicted_pressure["J3"] and up["J4"] > a.predicted_pressure["J4"]
    assert up["J2"] == pytest.approx(a.predicted_pressure["J2"])            # PRV holds J2 at its setting


def test_whatif_valve_change_only_affects_j2(S, RS):
    a = assess_risk(make_inputs(), S)
    dn = what_if(a, RS, re_.REDUCE_VALVE, 40.0)
    assert dn["J2"] == pytest.approx(40.0)
    assert dn["J3"] == a.predicted_pressure["J3"] and dn["J4"] == a.predicted_pressure["J4"]
    up = what_if(a, RS, re_.INCREASE_VALVE, 80.0)
    assert up["J2"] <= a.state["hub_pressure"] - 2.0 + 1e-9                  # cannot exceed the upstream pressure
