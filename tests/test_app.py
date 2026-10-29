"""Phase 7 tests: dashboard data layer and the Streamlit app (headless, via streamlit.testing).

The app tests force demo mode (AQUABALANCE_FORCE_DEMO=1) so they depend only on files committed in the repository:
data/raw/aquabalance_base.inp and data/sample/*. Run `python scripts/build_network.py` first if the .inp is missing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import dashboard_data as dd  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
APP = str(REPO / "app" / "app.py")
PROTOTYPE = "This is a simulation-based decision-support prototype, not autonomous real-world utility control."
DISCLAIMER = "This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility."


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def demo(cfg):
    src = dd.load_source(cfg, force_demo=True)
    return src, dd.ModelHub(cfg, src)


# ------------------------------------------------------------------------------ demo data
def test_sample_files_are_committed_in_the_repo():
    for f in ("sample_timeseries.csv", "sample_scenario_specs.jsonl", "sample_scenario_metadata.csv", "sample_forecasts.csv",
              "sample_model_comparison.csv"):
        assert (REPO / "data" / "sample" / f).exists(), f


def test_demo_source_is_labelled_synthetic_epanet(demo):
    src, _ = demo
    assert src.mode == "demo" and src.label == "Synthetic EPANET simulation example."
    assert len(dd.scenario_ids(src)) >= 8 and "normal" in dd.scenario_types(src)


def test_missing_data_gives_a_readable_error(cfg, tmp_path):
    bad = dict(cfg)
    bad["paths"] = {**cfg["paths"], "raw_timeseries": str(tmp_path / "x.csv")}
    with pytest.raises(dd.DashboardError, match="No simulation data found"):
        dd.load_source(bad, force_demo=False, sample_dir=tmp_path / "empty")


def _same_generation(cfg, src, sid):
    """True when the locally generated dataset is the very run the demo sample was cut from (same scenario seed, type and source head)."""
    import json
    def seed_of(path):
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            if d["scenario_id"] == sid:
                return (d["random_seed"], d["scenario_type"], d["reservoir_head"])
    full = resolve_path(cfg["paths"]["scenario_specs"]) if "scenario_specs" in cfg["paths"] else None
    if full is None or not full.exists():
        return False
    return seed_of(full) == seed_of(src.specs_path)


def test_sample_values_are_real_epanet_output(cfg, demo):
    """The sample must be a subset of the real EPANET dataset, not made-up numbers."""
    src, _ = demo
    full = resolve_path(cfg["paths"]["raw_timeseries"])
    if not full.exists():
        pytest.skip("full dataset not generated here")
    raw = pd.read_csv(full)
    sid = dd.scenario_ids(src)[0]
    if not _same_generation(cfg, src, sid):
        pytest.skip("local dataset was generated with different settings than the demo sample")
    a = raw[raw["scenario_id"] == sid]["J4_pressure"].to_numpy()
    b = src.timeseries[src.timeseries["scenario_id"] == sid]["J4_pressure"].to_numpy()
    assert np.allclose(a, b)


# ------------------------------------------------------------------------------ models
def test_decision_steps_have_a_full_window_and_a_next_step(cfg):
    s = dd.decision_steps(cfg, 96)
    assert s[0] == 11 and s[-1] == 94


def test_persistence_forecast_equals_the_current_pressure(cfg, demo):
    src, hub = demo
    sid = dd.scenario_ids(src)[0]
    v = dd.build_step_view(cfg, src, hub, sid, 40, "persistence")
    for n in ("J2", "J3", "J4"):
        assert v.forecast[n] == pytest.approx(v.current_pressure[n], abs=1e-4)


def test_demo_offers_only_models_with_real_forecasts(demo):
    """Every offered model is either live (persistence) or backed by stored forecasts of a TRAINED model; none is faked."""
    src, hub = demo
    stored = set(src.forecasts["model"].unique())
    for m in hub.available():
        assert m == "persistence" or m in stored
    assert {"persistence", "random_forest", "gru"} <= set(hub.available())
    for m, why in hub.unavailable().items():
        assert m not in hub.available() and why


def test_untrained_model_is_reported_unavailable_not_faked(cfg, demo, monkeypatch):
    src, _ = demo
    reduced = dd.DataSource(**{**src.__dict__, "forecasts": src.forecasts[src.forecasts["model"] != "tcn"]})
    hub = dd.ModelHub(cfg, reduced)
    assert "tcn" not in hub.available() and "tcn" in hub.unavailable()


def test_forecast_never_uses_the_true_next_step(cfg, demo):
    src, hub = demo
    sid = dd.scenario_ids(src)[0]
    v = dd.build_step_view(cfg, src, hub, sid, 40, "persistence")
    assert v.window.shape == (12, len(v.feats))
    assert not any(f in v.feats for f in ("leak_flag", "leak_severity", "scenario_type", "tank_level"))


def test_stored_forecast_matches_the_live_model(cfg, demo):
    src, hub = demo
    full_src = dd.load_source(cfg)
    if full_src.mode != "full":
        pytest.skip("full dataset/models not available here")
    live = dd.ModelHub(cfg, full_src)
    if live.info["random_forest"]["mode"] != "live":
        pytest.skip("random forest not trained here")
    sid = [i for i in dd.scenario_ids(src) if hub.can_forecast("random_forest", i)][0]
    if not _same_generation(cfg, src, sid):
        pytest.skip("local models/data differ from the ones the demo sample was cut from")
    a = dd.build_step_view(cfg, src, hub, sid, 40, "random_forest").forecast
    b = dd.build_step_view(cfg, full_src, live, sid, 40, "random_forest").forecast
    for n in a:
        assert a[n] == pytest.approx(b[n], abs=0.02)


def test_missing_torch_does_not_crash_the_hub(cfg, monkeypatch):
    src = dd.load_source(cfg)
    if src.mode != "full":
        pytest.skip("full mode needed")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "src.gru_model", raising=False)
    hub = dd.ModelHub(cfg, src)
    assert hub.info["gru"]["mode"] == "unavailable" and "persistence" in hub.available()


# ------------------------------------------------------------------------------ risk and node status
def _first(src, family):
    return dd.scenario_ids(src, family)[0]


def test_node_status_flags_the_critical_node(cfg, demo):
    src, hub = demo
    sid = "S0105"
    st = dd.notable_step(cfg, src, hub, sid, "persistence")
    v = dd.build_step_view(cfg, src, hub, sid, st, "persistence")
    assert v.node_status["J4"][0] == "risk" and "J4" in v.assessment.low_pressure_nodes
    assert v.assessment.predicted_risk_label == "low_pressure_risk"


def test_normal_scenario_has_no_risk_nodes_at_a_calm_step(cfg, demo):
    src, hub = demo
    sid = _first(src, "normal")
    v = dd.build_step_view(cfg, src, hub, sid, 20, "persistence")
    assert all(k != "risk" for k, _ in v.node_status.values())


def test_leak_scenario_is_found_by_the_notable_step(cfg, demo):
    src, hub = demo
    sid = _first(src, "small_leak")
    st = dd.notable_step(cfg, src, hub, sid, "persistence")
    v = dd.build_step_view(cfg, src, hub, sid, st, "persistence")
    assert v.assessment.severity_score > 0 and st in dd.decision_steps(cfg, 96)


# ------------------------------------------------------------------------------ model comparison table
def test_comparison_table_columns_and_honest_summary(demo):
    src, _ = demo
    t = dd.comparison_table(src.comparison_path)
    for c in ("model", "validation MAE (m)", "test MAE (m)", "test RMSE (m)", "risk macro F1 (test)", "inference (ms / 1000 windows)", "notes"):
        assert c in t.columns
    assert {"persistence", "random_forest", "gru"} <= set(t["model"])
    s = dd.comparison_summary(t)
    best = t.dropna(subset=["test MAE (m)"]).sort_values("test MAE (m)").iloc[0]["model"]
    assert best in s
    if best != "gru":
        assert "does not have the lowest error" in s


# ------------------------------------------------------------------------------ the Streamlit app (headless)
@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("AQUABALANCE_FORCE_DEMO", "1")
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [h.value for h in at.header] + [t.value for t in at.title]
    return "\n".join(parts)


def _outcome(at) -> str:
    for m in at.markdown:
        if "ab-outcome" in m.value and "<style>" not in m.value:
            return m.value.split(">", 1)[1].rsplit("</div", 1)[0]
    return ""


def _select(at, label, value):
    [w for w in at.sidebar.selectbox if w.label == label][0].select(value).run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_launches_with_title_all_sections_and_labels(app):
    assert app.title[0].value == "AquaBalance"
    text = _text(app)
    assert "Deep Learning–Assisted Pressure Risk Prediction for Water Distribution Networks" in text
    heads = [h.value for h in app.header]
    for n in range(1, 9):
        assert any(h.startswith(f"{n}. ") for h in heads), n
    assert "Synthetic EPANET simulation example." in text and PROTOTYPE in text
    assert "no tank" in " ".join(i.value for i in app.info).lower()


def test_sidebar_has_the_required_controls(app):
    labels = [w.label for w in app.sidebar.selectbox]
    assert {"Scenario-type filter", "Scenario", "Forecast model"} <= set(labels)
    assert [w.label for w in app.sidebar.select_slider] == ["Time step ('now')"]
    assert [w.label for w in app.sidebar.toggle] == ["Show next-step forecast"]
    assert any(b.label == "Run EPANET Simulated Validation" for b in app.sidebar.button)
    assert any("Help / Methodology" in e.label for e in app.sidebar.expander)
    model_widget = [w for w in app.sidebar.selectbox if w.label == "Forecast model"][0]
    assert {"Persistence", "Random Forest"} <= set(model_widget.options)


def test_every_model_and_filter_runs_without_error(app):
    for fam in ("All", "high_demand_stress", "valve_misconfiguration", "small_leak"):
        _select(app, "Scenario-type filter", fam)
    _select(app, "Scenario-type filter", "All")
    for m in [w for w in app.sidebar.selectbox if w.label == "Forecast model"][0].options:
        _select(app, "Forecast model", m)
    app.sidebar.toggle[0].set_value(False).run()
    assert not app.exception


def test_validation_not_shown_before_the_button_is_clicked(app):
    assert _outcome(app) == ""
    assert "Click **Run EPANET Simulated Validation**" in " ".join(i.value for i in app.info)


def test_validation_improvement_case(app):
    _select(app, "Scenario", "S0105")
    [b for b in app.sidebar.button if b.label == "Run EPANET Simulated Validation"][0].click().run()
    assert not app.exception and not app.error
    assert _outcome(app) == "Simulated improvement observed."
    text = _text(app)
    assert "EPANET simulated validation" in text and DISCLAIMER in text


def test_validation_no_safe_action_case(app):
    _select(app, "Scenario", "S0015")
    [b for b in app.sidebar.button if b.label == "Run EPANET Simulated Validation"][0].click().run()
    assert _outcome(app) == "No safe action found; operator inspection recommended."


def test_validation_partial_case_says_not_all_constraints_improved(app):
    _select(app, "Scenario", "S0147")
    [b for b in app.sidebar.button if b.label == "Run EPANET Simulated Validation"][0].click().run()
    assert _outcome(app) == "Recommendation did not improve all constraints."


def test_validation_leak_case_is_inspection_only(app):
    _select(app, "Scenario", "S0083")
    [b for b in app.sidebar.button if b.label == "Run EPANET Simulated Validation"][0].click().run()
    assert _outcome(app).startswith("Inspection recommended")


def test_stale_validation_is_not_shown_for_another_scenario(app):
    _select(app, "Scenario", "S0105")
    [b for b in app.sidebar.button if b.label == "Run EPANET Simulated Validation"][0].click().run()
    assert _outcome(app)
    _select(app, "Scenario", "S0015")
    assert _outcome(app) == ""


def test_app_has_no_hidden_fake_metrics():
    """No hard-coded result numbers in the app source: values must come from the data/model layer."""
    src = (REPO / "app" / "app.py").read_text(encoding="utf-8")
    assert "np.random" not in src and "random." not in src
