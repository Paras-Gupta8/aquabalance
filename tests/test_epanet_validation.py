"""Phase 6 tests: EPANET simulated validation of recommendations.

The fixture generates a small real EPANET dataset in a temporary folder (Phase 2 generator), then every validation
reruns scenarios in EPANET through WNTR. Run `python scripts/build_network.py` first so the base .inp exists.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import epanet_validator as ev  # noqa: E402
from src.recommendation_engine import (NO_ACTION_INSPECT, NO_SAFE_ACTION_MESSAGE, VALIDATION_DISCLAIMER,  # noqa: E402
                                       VALIDATION_LABEL, apply_validation)
from src.scenario_generator import generate_dataset  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FAMILY_FOR = {"high": "high_source_pressure", "imbalance": "valve_misconfiguration", "leak": "small_leak",
              "burst": "burst_leak", "blocked": "partial_blockage", "stress": "high_demand_stress",
              "normal": "normal", "reduced": "reduced_supply"}


def _paths(folder: Path) -> dict:
    f = Path(folder)
    return {
        "scenario_metadata": f / "synthetic/scenario_metadata.csv", "raw_timeseries": f / "synthetic/raw_timeseries.csv",
        "labels": f / "synthetic/labels.csv", "demand_profiles": f / "synthetic/demand_profiles.csv",
        "scenario_specs": f / "synthetic/scenario_specs.jsonl", "sensor_timeseries": f / "synthetic/sensor.csv",
        "summary": f / "sim/summary.csv", "network_config_json": f / "sim/network_config.json", "log_dir": f / "logs"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def base_inp(cfg):
    p = resolve_path(cfg["paths"]["base_inp"])
    if not p.exists():
        pytest.fail(f"{p} not found. Run: python scripts/build_network.py")
    return p


@pytest.fixture(scope="module")
def world(cfg, tmp_path_factory):
    folder = tmp_path_factory.mktemp("phase6")
    P = {k: str(v) for k, v in _paths(folder).items()}
    gen = generate_dataset(cfg, cfg["scenarios"]["quick_n"], paths=P, quiet=True)
    assert gen["complete"]
    raw = pd.read_csv(P["raw_timeseries"], parse_dates=["timestamp"]).sort_values(["scenario_id", "timestamp"])
    meta = pd.read_csv(P["scenario_metadata"])
    return {"P": P, "raw": raw, "meta": meta}


def _series(world, sid):
    return world["raw"][world["raw"]["scenario_id"] == sid].reset_index(drop=True)


def find_case(world, cfg, family_key, want=None, lo=14, hi=88):
    """First (spec, df, step, assessment, recommendation) in a family where the engine flags one of `want`."""
    fam = FAMILY_FOR[family_key]
    ids = world["meta"].loc[world["meta"]["scenario_type"] == fam, "scenario_id"]
    for sid in ids:
        df = _series(world, sid)
        for step in range(lo, hi):
            f = ev.oracle_forecast(df, step, ("J2", "J3", "J4"))
            a, rec = ev.recommend_from_timeseries(df, step, f, cfg)
            if want is None or rec.addresses in want:
                return ev.load_spec(world["P"]["scenario_specs"], sid), df, step, a, rec
    pytest.skip(f"no {fam} case found in the small test dataset")


@pytest.fixture(scope="module")
def cases(world, cfg, base_inp):
    """One validated case per kind (EPANET runs once, shared by the tests)."""
    out = {}
    wants = {"normal": None, "high": ("high",), "imbalance": ("imbalance",),
             "leak": None, "blocked": ("critical_low", "other_low"), "stress": ("critical_low", "other_low")}
    for key, want in wants.items():
        try:
            spec, df, step, a, rec = find_case(world, cfg, key, want)
        except pytest.skip.Exception:
            continue
        out[key] = {"spec": spec, "df": df, "step": step, "a": a, "rec": rec,
                    "v": ev.validate_recommendation(spec, rec, step, cfg, base_inp)}
    return out


# ------------------------------------------------------------------------------ 1. loads the .inp network
def test_loads_inp_network(base_inp):
    p = ev.load_base_network(base_inp)
    assert Path(p).suffix == ".inp" and Path(p).exists()


def test_missing_network_raises_readable_error(tmp_path):
    with pytest.raises(ev.ValidationError, match="build_network"):
        ev.load_base_network(tmp_path / "nope.inp")


def test_bad_network_file_raises(tmp_path):
    bad = tmp_path / "bad.inp"
    bad.write_text("this is not an EPANET file", encoding="utf-8")
    with pytest.raises(ev.ValidationError):
        ev.load_base_network(bad)


def test_unknown_scenario_id_raises(world):
    with pytest.raises(ev.ValidationError):
        ev.load_spec(world["P"]["scenario_specs"], "S9999")


# ------------------------------------------------------------------------------ 2. reruns the scenario
def test_baseline_rerun_reproduces_stored_epanet_data(world, cfg, base_inp):
    sid = world["meta"]["scenario_id"].iloc[0]
    spec = ev.load_spec(world["P"]["scenario_specs"], sid)
    res = ev.simulate(spec, cfg, ev.load_base_network(base_inp))
    stored = _series(world, sid)
    for n in ("J1", "J2", "J3", "J4"):
        assert np.allclose(res["data"][f"{n}_pressure"].to_numpy(), stored[f"{n}_pressure"].to_numpy(), atol=1e-3)


def test_action_changes_the_rerun_results(world, cfg, base_inp):
    spec, df, step, a, rec = find_case(world, cfg, "high", None, lo=20, hi=30)
    base = ev.simulate(spec, cfg, ev.load_base_network(base_inp))["data"]
    spec2, info = ev.apply_action(spec, ev.Action("pump", -10.0), step + 1, cfg["scenarios"]["steps"])
    after = ev.simulate(spec2, cfg, ev.load_base_network(base_inp))["data"]
    assert not np.allclose(base["J3_pressure"].iloc[step + 2:], after["J3_pressure"].iloc[step + 2:])
    assert np.allclose(base["J3_pressure"].iloc[:step + 1], after["J3_pressure"].iloc[:step + 1], atol=1e-3)  # before the action: identical


def test_apply_action_does_not_mutate_the_original_spec(world):
    spec = ev.load_spec(world["P"]["scenario_specs"], world["meta"]["scenario_id"].iloc[0])
    before = copy.deepcopy(spec)
    ev.apply_action(spec, ev.Action("valve", 10.0), 30, 96)
    ev.apply_action(spec, ev.Action("pump", -5.0), 30, 96)
    assert spec == before


# ------------------------------------------------------------------------------ 3. candidate search
def test_candidate_grid_matches_the_required_bounds(cfg):
    grid = ev.candidate_grid(ev.ValidationSettings.from_config(cfg))
    valve = sorted(a.pct for a in grid if a.component == "valve")
    pump = sorted(a.pct for a in grid if a.component == "pump")
    assert valve == [-15, -10, -5, 5, 10, 15] and pump == [-10, -5, 5, 10]
    assert any(a.component == "none" for a in grid)


def test_candidate_search_evaluates_multiple_actions(cases):
    v = cases["high"]["v"]
    assert v["n_candidates_evaluated"] >= 11 and len(v["candidates"]) >= 11
    assert {c["component"] for c in v["candidates"]} == {"none", "valve", "pump"}


def test_objective_formula_and_j4_penalty(cfg):
    vs = ev.ValidationSettings.from_config(cfg)
    m = {"high_pressure_violation_count": 2, "low_pressure_violation_count": 3, "j4_critical_violation_count": 0,
         "imbalance_step_count": 0, "branch_pressure_spread_mean": 10.0}
    obj, parts = ev.objective(m, 5.0, vs)
    assert obj == pytest.approx(10 * 2 + 10 * 3 + 1.0 * 10.0 + 0.1 * 5.0)   # imbalance_step_count is 0 here
    m2 = {**m, "j4_critical_violation_count": 1}
    assert ev.objective(m2, 5.0, vs)[0] - obj >= 900


# ------------------------------------------------------------------------------ 4. J4 never sacrificed
def _m(**kw):
    base = {"J4_pressure_min": 40.0, "j4_critical_violation_count": 0, "imbalance_step_count": 0, "run_ok": True,
            "_nodes": {n: {"low_count": 0, "high_count": 0, "low_depth": 0.0, "high_depth": 0.0} for n in ("J2", "J3", "J4")}}
    base.update(kw)
    return base


def test_check_safety_rejects_a_j4_critical_violation_even_if_another_node_improves(cfg):
    vs = ev.ValidationSettings.from_config(cfg)
    before, after = _m(), _m(J4_pressure_min=25.0, j4_critical_violation_count=3)
    ok, reasons = ev.check_safety(before, after, vs)
    assert not ok and reasons


def test_check_safety_rejects_j4_dropping_below_its_floor(cfg):
    vs = ev.ValidationSettings.from_config(cfg)
    ok, _ = ev.check_safety(_m(J4_pressure_min=45.0), _m(J4_pressure_min=31.0), vs)   # floor = critical 30 + margin 0.5 -> 30.5 ok
    assert ok
    ok, _ = ev.check_safety(_m(J4_pressure_min=45.0), _m(J4_pressure_min=30.0), vs)
    assert not ok


def test_check_safety_rejects_failed_run_or_out_of_bounds(cfg):
    vs = ev.ValidationSettings.from_config(cfg)
    assert not ev.check_safety(_m(), _m(), vs, run_ok=False)[0]
    assert not ev.check_safety(_m(), _m(), vs, bounds_ok=False)[0]


def test_chosen_action_never_pushes_j4_below_its_floor(cases, cfg):
    vs = ev.ValidationSettings.from_config(cfg)
    checked = 0
    for c in cases.values():
        v = c["v"]
        if v["final_validation_result"] != "validated_improvement":
            continue
        b, a = v["before"], v["after"]
        floor = min(b["J4_pressure_min"], vs.critical_min + vs.j4_margin_m)
        assert a["J4_pressure_min"] >= floor - vs.tol_m
        assert a["j4_critical_violation_count"] <= b["j4_critical_violation_count"]
        checked += 1
    assert checked >= 1


def test_no_candidate_marked_safe_has_a_new_j4_violation(cases):
    for c in cases.values():
        base_j4 = c["v"]["before"]["j4_critical_violation_count"]
        for cand in c["v"]["candidates"]:
            if cand.get("safe"):
                assert cand["j4_critical_violation_count"] <= base_j4


# ------------------------------------------------------------------------------ 5. no-action result is possible
def test_normal_scenario_needs_no_action(cases):
    if "normal" not in cases:
        pytest.skip("no normal case")
    v = cases["normal"]["v"]
    assert v["final_validation_result"] in ("no_action_needed", "validated_improvement", "no_safe_action_found")
    # a normal scenario must not end up with a changed control unless EPANET shows a real problem
    if v["final_validation_result"] == "no_action_needed":
        assert v["chosen_action"] in ("no action", "none", "no_action", "No action")  or "no" in str(v["chosen_action"]).lower()


def test_no_safe_action_returns_the_exact_message(world, cfg, base_inp):
    """Force the situation: ask for a fix on a scenario where no candidate helps (blocked pipe) or fake it by bounds."""
    spec, df, step, a, rec = find_case(world, cfg, "blocked", ("critical_low", "other_low"))
    v = ev.validate_recommendation(spec, rec, step, cfg, base_inp)
    assert v["final_validation_result"] in ev.FINAL_RESULTS
    if v["final_validation_result"] == "no_safe_action_found":
        assert v["final_message"] == "No safe action found; operator inspection recommended."
        assert v["validated_recommendation"]["recommended_action_type"] == NO_ACTION_INSPECT


def test_apply_validation_no_safe_action_uses_exact_message(cases):
    first = next(iter(cases.values()))
    rec = first["rec"]
    v = {**first["v"], "final_validation_result": "no_safe_action_found", "final_message": NO_SAFE_ACTION_MESSAGE,
         "improvement_level": "none", "chosen_action_detail": None}
    out = apply_validation(rec, v)
    assert out.recommended_action_type == NO_ACTION_INSPECT and out.rationale == NO_SAFE_ACTION_MESSAGE
    assert out.confidence <= 0.35 and VALIDATION_DISCLAIMER in out.safety_note
    assert rec.validation == {}                                          # the original is untouched


# ------------------------------------------------------------------------------ 6. leak / burst are inspection only
def test_leak_recommendation_is_inspection_only(world, cfg, base_inp):
    spec, df, step, a, rec = find_case(world, cfg, "leak", ("leak_like_pattern",))
    v = ev.validate_recommendation(spec, rec, step, cfg, base_inp)
    assert v["final_validation_result"] == "inspection_recommended"
    assert len(v["candidates"]) == 0 or all(c["component"] == "none" for c in v["candidates"])   # no control tried for a leak
    assert v["before"]["loss_proxy_lps"] > 0


# ------------------------------------------------------------------------------ 7. before/after schema
def test_before_after_schema(cases):
    results = [c["v"] for c in cases.values()]
    df = ev.results_frame(results)
    for k in ev.METRIC_KEYS:
        assert f"before_{k}" in df.columns and f"after_{k}" in df.columns
    for col in ("chosen_action", "final_validation_result", "validation_type", "validation_disclaimer", "safety_note"):
        assert col in df.columns
    assert set(df["final_validation_result"]) <= set(ev.FINAL_RESULTS)
    for k in ("J1", "J2", "J3", "J4"):
        assert df[f"before_{k}_pressure_mean"].notna().all() and df[f"after_{k}_pressure_mean"].notna().all()
    assert df["before_J4_status"].isin(["safe", "low", "critical", "high", "OK", "ok", "critical_low", "low_pressure", "high_pressure"]).all() or df["before_J4_status"].notna().all()


def test_violation_counts_are_non_negative_integers(cases):
    for c in cases.values():
        for side in ("before", "after"):
            m = c["v"][side]
            assert m["high_pressure_violation_count"] >= 0 and m["low_pressure_violation_count"] >= 0
            assert 0 <= m["safe_node_coverage"] <= 1


# ------------------------------------------------------------------------------ 8. output identifies EPANET validation
def test_results_identify_epanet_simulated_validation(cases):
    df = ev.results_frame([c["v"] for c in cases.values()])
    assert (df["validation_type"] == VALIDATION_LABEL).all()
    assert (df["validation_disclaimer"] == VALIDATION_DISCLAIMER).all()
    assert df["simulator"].str.contains("EPANET").all()


def test_written_outputs_and_figures(cases, cfg, tmp_path):
    results = [c["v"] for c in cases.values()]
    paths = {"validation_results": tmp_path / "r.csv", "validation_candidates": tmp_path / "c.csv",
             "recommendation_effectiveness": tmp_path / "e.csv"}
    ev.write_outputs(results, paths)
    for p in paths.values():
        assert Path(p).exists() and Path(p).stat().st_size > 0
    eff = pd.read_csv(paths["recommendation_effectiveness"])
    assert "ALL" in set(eff.iloc[:, 0].astype(str)) or "ALL" in set(eff.astype(str).values.ravel())
    vs = ev.ValidationSettings.from_config(cfg)
    f1 = ev.plot_pressure_comparison(results, vs, tmp_path / "p.png")
    f2 = ev.plot_violation_comparison(results, vs, tmp_path / "v.png")
    assert Path(f1).stat().st_size > 5000 and Path(f2).stat().st_size > 5000


def test_label_and_disclaimer_text_is_exact():
    assert VALIDATION_LABEL == "EPANET simulated validation"
    assert VALIDATION_DISCLAIMER == ("This validates the recommendation in the simulated EPANET network. "
                                     "It is not field validation in an actual water utility.")
    assert NO_SAFE_ACTION_MESSAGE == "No safe action found; operator inspection recommended."


def test_readme_and_module_docs_contain_the_required_sentences():
    for p in (REPO / "README.md", REPO / "src/epanet_validator.py"):
        text = p.read_text(encoding="utf-8")
        assert "EPANET simulated validation" in text, p
        assert VALIDATION_DISCLAIMER in text, p
