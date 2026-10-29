"""Phase 3 tests: leakage-safe dataset and forecasting baselines.

The fixture runs the REAL pipeline on a small EPANET dataset (50 scenarios) inside a temporary folder:
Phase 2 generation -> dataset build -> baseline training/evaluation. Nothing depends on stale files.
Run `python scripts/build_network.py` first so the base .inp exists.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.baseline_models import PersistenceBaseline, RandomForestBaseline, RuleBasedBaseline  # noqa: E402
from src.dataset_builder import build_dataset, build_scenario_windows, load_split  # noqa: E402
from src.evaluation import regression_metrics, risk_metrics, run_baseline_experiment  # noqa: E402
from src.feature_engineering import (FeatureError, add_time_features, hour_of_day, make_rf_features,  # noqa: E402
                                     resolve_input_features, rf_feature_names)
from src.labels import PRESSURE_RISK_CLASSES, label_settings, pressure_risk_classes  # noqa: E402
from src.preprocessing import DataError, assert_disjoint, scenario_table, split_scenarios  # noqa: E402
from src.scenario_generator import generate_dataset  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

NODES = ["J2", "J3", "J4"]


def _paths(folder: Path) -> dict:
    """Send every Phase 2 and Phase 3 output to a temporary folder."""
    f = Path(folder)
    return {
        "scenario_metadata": f / "synthetic/scenario_metadata.csv", "raw_timeseries": f / "synthetic/raw_timeseries.csv",
        "labels": f / "synthetic/labels.csv", "demand_profiles": f / "synthetic/demand_profiles.csv",
        "scenario_specs": f / "synthetic/scenario_specs.jsonl", "sensor_timeseries": f / "synthetic/sensor.csv",
        "summary": f / "sim/summary.csv", "network_config_json": f / "sim/network_config.json", "log_dir": f / "logs",
        "train_npz": f / "processed/train.npz", "val_npz": f / "processed/val.npz", "test_npz": f / "processed/test.npz",
        "feature_names": f / "processed/feature_names.json", "split_manifest": f / "processed/split_manifest.csv",
        "input_scaler": f / "scalers/input_scaler.joblib", "target_scaler": f / "scalers/target_scaler.joblib",
        "random_forest": f / "trained/random_forest.joblib",
        "baseline_metrics": f / "reports/baseline_metrics.csv",
        "baseline_error_analysis": f / "reports/baseline_error_analysis.csv",
        "fig_prediction_comparison": f / "figures/baseline_prediction_comparison.png",
        "fig_confusion_matrix": f / "figures/baseline_confusion_matrix.png",
    }


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def built(cfg, tmp_path_factory):
    """Real pipeline once: EPANET scenarios -> ML dataset -> baselines (small forest to keep the tests fast)."""
    base_inp = resolve_path(cfg["paths"]["base_inp"])
    if not base_inp.exists():
        pytest.fail(f"{base_inp} not found. Run: python scripts/build_network.py")
    folder = tmp_path_factory.mktemp("phase3")
    P = {k: str(v) for k, v in _paths(folder).items()}
    gen = generate_dataset(cfg, cfg["scenarios"]["quick_n"], paths=P, quiet=True)
    assert gen["complete"], "Phase 2 generation did not accept every scenario"
    ds = build_dataset(cfg, paths=P)
    exp = run_baseline_experiment(cfg, paths=P, n_estimators=10)
    raw = pd.read_csv(P["raw_timeseries"], parse_dates=["timestamp"]).sort_values(["scenario_id", "timestamp"])
    return {"cfg": cfg, "P": P, "folder": folder, "ds": ds, "exp": exp, "raw": raw,
            "manifest": pd.read_csv(P["split_manifest"]),
            "meta": json.loads(Path(P["feature_names"]).read_text(encoding="utf-8")),
            "data": {s: load_split(P[f"{s}_npz"]) for s in ("train", "val", "test")}}


def _feature_frame(raw: pd.DataFrame, features: list[str]) -> dict:
    framed = add_time_features(raw, fractional=True)
    return {sid: g for sid, g in framed.groupby("scenario_id")}


# ------------------------------------------------------------------------------ 1. split by scenario
def test_no_scenario_overlap_across_splits(built):
    m, data = built["manifest"], built["data"]
    sets = {s: set(g["scenario_id"]) for s, g in m.groupby("split")}
    assert sets["train"].isdisjoint(sets["val"]) and sets["train"].isdisjoint(sets["test"]) \
        and sets["val"].isdisjoint(sets["test"])
    assert sets["train"] | sets["val"] | sets["test"] == set(built["raw"]["scenario_id"])
    for s in ("train", "val", "test"):                       # the saved windows really belong to that split's scenarios
        assert set(data[s]["meta_scenario_id"]) == sets[s]
    assert_disjoint(m)                                        # the built-in guard accepts the real manifest


def test_overlap_guard_rejects_leaky_manifest():
    leaky = pd.DataFrame({"scenario_id": ["S1", "S2", "S1"], "split": ["train", "val", "test"]})
    with pytest.raises(DataError, match="more than once|overlap"):
        assert_disjoint(leaky)
    disjoint_but_missing = pd.DataFrame({"scenario_id": ["S1", "S2"], "split": ["train", "val"]})
    with pytest.raises(DataError, match="empty"):
        assert_disjoint(disjoint_but_missing)


def test_split_is_deterministic_stratified_and_proportional(built, cfg):
    raw = built["raw"]
    table = scenario_table(raw)
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    a = split_scenarios(table, ratios, seed=42)
    b = split_scenarios(table, ratios, seed=42)
    c = split_scenarios(table, ratios, seed=43)
    pd.testing.assert_frame_equal(a, b)                       # same seed -> same split
    assert not a["split"].equals(c["split"])                  # another seed -> another split
    pd.testing.assert_frame_equal(a[["scenario_id", "split"]], built["manifest"][["scenario_id", "split"]])
    counts = a["split"].value_counts()
    n = len(table)
    assert counts["train"] == round(0.70 * n) and abs(counts["val"] - counts["test"]) <= 1
    by_type = a.groupby("scenario_type")["split"].agg(set)
    big = table["scenario_type"].value_counts()
    for t in big[big >= 6].index:                             # families with enough scenarios appear in every split
        assert by_type[t] == {"train", "val", "test"}, t


# ------------------------------------------------------------------------------ 2. windows never cross scenarios
def test_no_window_crosses_a_scenario_boundary(built):
    meta, raw, data = built["meta"], built["raw"], built["data"]
    features, w, h = meta["input_features"], meta["window_steps"], meta["horizon_steps"]
    frames = _feature_frame(raw, features)
    steps = int(built["cfg"]["scenarios"]["steps"])
    for split, d in data.items():
        ids = d["meta_scenario_id"]
        for sid in np.unique(ids):
            m = ids == sid
            ends = d["meta_end_step"][m].astype(int)
            assert ends.min() == w - 1 and ends.max() == steps - 1 - h          # first/last window stay inside the scenario
            assert len(ends) == steps - w - h + 1
            assert np.array_equal(ends, np.arange(w - 1, steps - h))            # consecutive, no gaps
            g = frames[sid]
            feats = g[features].to_numpy("float64")
            tgt = g[[f"{n}_pressure" for n in NODES]].to_numpy("float64")
            for x, y, e in zip(d["X_raw"][m], d["y_raw"][m], ends):
                assert np.allclose(x, feats[e - w + 1:e + 1], atol=1e-3), f"{sid}: window is not a slice of ONE scenario"
                assert np.allclose(y, tgt[e + h], atol=1e-3), f"{sid}: target is not the future of its own window"
        assert (np.diff(np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])) > 0).all()
        order = pd.Series(ids).ne(pd.Series(ids).shift()).cumsum()
        assert order.nunique() == len(np.unique(ids)), "windows of one scenario are not stored contiguously"


def test_window_builder_alone_cannot_mix_scenarios():
    """Scenario A is all 1s, scenario B all 2s. Windows built per scenario are pure; windowing the glued array is not."""
    a, b = np.full((20, 3), 1.0), np.full((20, 3), 2.0)
    for scen, value in ((a, 1.0), (b, 2.0)):
        x, y, end = build_scenario_windows(scen, scen[:, :2], window=12, horizon=1)
        assert (x == value).all() and (y == value).all() and len(x) == 20 - 12 - 1 + 1
    glued_x, _, _ = build_scenario_windows(np.vstack([a, b]), np.vstack([a, b])[:, :2], window=12, horizon=1)
    assert any(len(np.unique(w)) > 1 for w in glued_x), "sanity: the naive approach WOULD mix scenarios"
    with pytest.raises(DataError):
        build_scenario_windows(a[:5], a[:5, :2], window=12, horizon=1)


def test_window_counts_and_shapes(built):
    meta, data, m = built["meta"], built["data"], built["manifest"]
    per_scenario = 96 - meta["window_steps"] - meta["horizon_steps"] + 1
    assert (meta["window_steps"], meta["horizon_steps"]) == (12, 1)        # 3 hours of history, 15 minutes ahead
    for s, d in data.items():
        n_scen = int((m["split"] == s).sum())
        assert d["X"].shape == (n_scen * per_scenario, 12, len(meta["input_features"]))
        assert d["X_raw"].shape == d["X"].shape and d["y"].shape == d["y_raw"].shape
        assert meta["counts"][s] == {"scenarios": n_scen, "windows": n_scen * per_scenario}


# ------------------------------------------------------------------------------ 3. targets
def test_targets_are_exactly_three_future_pressures(built):
    meta, data = built["meta"], built["data"]
    assert meta["target_names"] == ["J2_pressure", "J3_pressure", "J4_pressure"]
    for d in data.values():
        assert d["y"].ndim == 2 and d["y"].shape[1] == 3 and d["y_raw"].shape[1] == 3
        assert np.isfinite(d["y"]).all() and np.isfinite(d["y_raw"]).all()
    scaler = joblib.load(built["P"]["target_scaler"])
    back = scaler.inverse_transform(data["val"]["y"])
    assert np.allclose(back, data["val"]["y_raw"], atol=1e-3)                  # scaled targets decode to metres


# ------------------------------------------------------------------------------ 4. scalers fit on training data only
def test_scalers_are_fit_on_training_data_only(built):
    meta, raw, m, data = built["meta"], built["raw"], built["manifest"], built["data"]
    feats = _feature_frame(raw, meta["input_features"])
    train_ids = set(m.loc[m["split"] == "train", "scenario_id"])
    rows = lambda ids: np.vstack([feats[s][meta["input_features"]].to_numpy("float64") for s in sorted(ids)])  # noqa: E731
    train_rows, all_rows = rows(train_ids), rows(set(m["scenario_id"]))
    sc = joblib.load(built["P"]["input_scaler"])
    assert sc.n_samples_seen_ == len(train_rows) < len(all_rows)               # exactly the training rows
    assert np.allclose(sc.mean_, train_rows.mean(axis=0), atol=1e-6)
    assert np.allclose(sc.scale_[sc.scale_ > 0], np.where(train_rows.std(axis=0) == 0, 1, train_rows.std(axis=0))[sc.scale_ > 0],
                       atol=1e-6)
    assert not np.allclose(sc.mean_, all_rows.mean(axis=0), atol=1e-9)          # would differ if val/test had leaked in
    tsc = joblib.load(built["P"]["target_scaler"])
    assert tsc.n_samples_seen_ == len(data["train"]["y_raw"])
    assert np.allclose(tsc.mean_, data["train"]["y_raw"].mean(axis=0), atol=1e-3)
    for s in ("train", "val", "test"):                                         # saved X is raw X scaled with the TRAIN scaler
        x = data[s]["X_raw"].reshape(-1, data[s]["X_raw"].shape[2])
        assert np.allclose(sc.transform(x).reshape(data[s]["X"].shape), data[s]["X"], atol=1e-3)


def test_scalers_ignore_validation_and_test_data(built, cfg, tmp_path):
    """Strongest check: make validation/test scenarios absurd, rebuild, and the scalers must not move."""
    P, raw, m = built["P"], pd.read_csv(built["P"]["raw_timeseries"]), built["manifest"]
    held_out = set(m.loc[m["split"] != "train", "scenario_id"])
    poisoned = raw.copy()
    mask = poisoned["scenario_id"].isin(held_out)
    for col in [c for c in poisoned.columns if c.endswith("_pressure") or c.endswith("_flow")]:
        poisoned.loc[mask, col] = poisoned.loc[mask, col] + 1000.0
    P2 = {k: str(v) for k, v in _paths(tmp_path).items()}
    Path(P2["raw_timeseries"]).parent.mkdir(parents=True, exist_ok=True)
    poisoned.to_csv(P2["raw_timeseries"], index=False)
    build_dataset(cfg, paths=P2)
    pd.testing.assert_frame_equal(pd.read_csv(P2["split_manifest"]), m)        # same split
    a, b = joblib.load(P["input_scaler"]), joblib.load(P2["input_scaler"])
    assert np.array_equal(a.mean_, b.mean_) and np.array_equal(a.scale_, b.scale_)
    ta, tb = joblib.load(P["target_scaler"]), joblib.load(P2["target_scaler"])
    assert np.array_equal(ta.mean_, tb.mean_) and np.array_equal(ta.scale_, tb.scale_)
    assert load_split(P2["val_npz"])["X_raw"].max() > 900                      # (and the poison did reach val)


def test_scalers_saved_with_joblib(built):
    for key in ("input_scaler", "target_scaler"):
        p = Path(built["P"][key])
        assert p.exists() and p.stat().st_size > 0
        assert hasattr(joblib.load(p), "transform")


# ------------------------------------------------------------------------------ 5. features: no hidden labels
def test_hidden_simulation_labels_are_not_inputs(built, cfg):
    meta = built["meta"]
    banned = {"leak_flag", "leak_severity", "scenario_type", "scenario_id", "timestamp"}
    assert not banned & set(meta["input_features"])
    assert banned <= set(meta["excluded_hidden_columns"])
    assert built["data"]["train"]["X"].shape[2] == len(meta["input_features"])
    rf_names = built["exp"]["model"].rf_feature_names
    assert rf_names and not any(b in n for n in rf_names for b in ("leak", "scenario"))
    bad = copy.deepcopy(cfg["ml"]); bad["input_features"] = bad["input_features"] + ["leak_flag"]
    with pytest.raises(FeatureError, match="leak_flag"):
        resolve_input_features(built["raw"], bad)
    bad["input_features"] = ["J2_pressure", "scenario_type"]
    with pytest.raises(FeatureError, match="scenario_type"):
        resolve_input_features(built["raw"], bad)


def test_empty_tank_feature_is_dropped_and_recorded_not_invented(built):
    meta = built["meta"]
    assert "tank_level" not in meta["input_features"] and "tank_level" in meta["dropped_features"]
    assert built["raw"]["tank_level"].isna().all()
    assert len(meta["input_features"]) == 18


def test_partly_empty_feature_is_an_error_not_silently_filled(built, cfg):
    df = add_time_features(built["raw"].copy())
    df.loc[df.index[3], "P1_flow"] = np.nan
    with pytest.raises(FeatureError, match="P1_flow"):
        resolve_input_features(df, cfg["ml"])


def test_time_features_follow_the_formula(built):
    ts = pd.Series(pd.to_datetime(["2026-01-05 00:00", "2026-01-05 06:00", "2026-01-05 06:45", "2026-01-05 18:00"]))
    out = add_time_features(pd.DataFrame({"timestamp": ts}), fractional=True)
    hours = np.array([0.0, 6.0, 6.75, 18.0])
    assert np.allclose(out["hour_sin"], np.sin(2 * np.pi * hours / 24)) and np.allclose(out["hour_cos"], np.cos(2 * np.pi * hours / 24))
    assert out["hour_sin"][1] == pytest.approx(1.0) and out["hour_cos"][1] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(hour_of_day(ts, fractional=False), [0, 6, 6, 18])
    feats = built["meta"]["input_features"]                                   # and they are really in the saved windows
    d = built["data"]["test"]
    i = int(np.argmax(d["meta_end_step"] == 11))                              # a window ending at step 11 = 02:45
    assert d["X_raw"][i, -1, feats.index("hour_sin")] == pytest.approx(np.sin(2 * np.pi * 2.75 / 24), abs=1e-5)
    assert d["X_raw"][i, -1, feats.index("hour_cos")] == pytest.approx(np.cos(2 * np.pi * 2.75 / 24), abs=1e-5)


# ------------------------------------------------------------------------------ 6. baselines
def test_persistence_baseline_runs(built):
    meta, d = built["meta"], built["data"]["test"]
    model = PersistenceBaseline(meta["input_features"], NODES)
    pred = model.predict(d["X_raw"])
    assert pred.shape == d["y_raw"].shape == (len(d["X_raw"]), 3) and np.isfinite(pred).all()
    cols = [meta["input_features"].index(f"{n}_pressure") for n in NODES]
    assert np.allclose(pred, d["X_raw"][:, -1, cols])                         # next pressure = current pressure
    m = regression_metrics(d["y_raw"], pred, NODES)
    assert 0 < m["MAE"] < 5 and m["RMSE"] >= m["MAE"] and m["R2"] > 0.9
    flat = d["X_raw"].copy(); flat[:] = flat[:, -1:, :]                      # constant history, then future == current
    assert regression_metrics(flat[:, -1, cols], model.predict(flat), NODES)["MAE"] == 0.0


def test_rule_based_baseline_is_reactive_and_follows_thresholds(built, cfg):
    meta, d = built["meta"], built["data"]["test"]
    settings = label_settings(cfg)
    rule = RuleBasedBaseline(meta["input_features"], NODES, settings)
    assert "reactive" in rule.description and not hasattr(rule, "fit") and not hasattr(rule, "predict")
    pers = PersistenceBaseline(meta["input_features"], NODES)
    expected = pressure_risk_classes(pers.predict(d["X_raw"]), NODES, settings)
    assert np.array_equal(rule.predict_risk_class(d["X_raw"]), expected)       # judges NOW: same as persisting the state
    p_min, p_max, spread, crit = (settings[k] for k in ("p_min_m", "p_max_m", "max_spread_m", "critical_p_min_m"))
    ok = np.array([[45.0, 45.0, 45.0]])
    cls = lambda arr: pressure_risk_classes(arr, NODES, settings)[0]  # noqa: E731
    assert cls(ok) == "normal"
    assert cls(np.array([[p_min - 0.1, 45.0, 45.0]])) == "low_pressure_risk"
    assert cls(np.array([[45.0, 45.0, crit - 0.1]])) == "low_pressure_risk"    # J4 has the stricter minimum
    assert cls(np.array([[45.0, p_max + 0.1, 45.0]])) == "high_pressure_risk"
    assert cls(np.array([[30.0, 30.0 + spread + 0.1, 45.0]])) == "pressure_imbalance"
    assert cls(np.array([[p_min - 0.1, p_max + 0.1, 45.0]])) == "low_pressure_risk"   # priority: low before high
    strict = copy.deepcopy(cfg); strict["thresholds"]["p_min_m"] = 50.0
    assert cls(np.array([[45.0, 45.0, 45.0]])) == "normal"
    assert pressure_risk_classes(np.array([[45.0, 45.0, 45.0]]), NODES, label_settings(strict))[0] == "low_pressure_risk"


def test_random_forest_trains_and_predicts(built):
    exp, d, meta = built["exp"], built["data"], built["meta"]
    rf = exp["model"]
    assert isinstance(rf, RandomForestBaseline) and len(rf.model.estimators_) == 10
    pred = rf.predict(d["test"]["X_raw"])
    assert pred.shape == (len(d["test"]["X_raw"]), 3) and np.isfinite(pred).all()
    feats, names = make_rf_features(d["test"]["X_raw"], meta["input_features"], built["cfg"]["ml"]["random_forest"])
    assert feats.shape[1] == len(names) == rf.model.n_features_in_ and np.isfinite(feats).all()
    pers = PersistenceBaseline(meta["input_features"], NODES)
    train = d["train"]
    assert regression_metrics(train["y_raw"], rf.predict(train["X_raw"]), NODES)["MAE"] < \
        regression_metrics(train["y_raw"], pers.predict(train["X_raw"]), NODES)["MAE"]   # it learned something on TRAIN data
    path = Path(built["P"]["random_forest"])
    assert path.exists()
    again = RandomForestBaseline.load(path)                                    # the saved file reproduces the predictions
    assert np.allclose(again.predict(d["test"]["X_raw"]), pred)


def test_random_forest_is_reproducible_with_a_seed(built):
    meta, d = built["meta"], built["data"]["train"]
    x, y = d["X_raw"][:600], d["y_raw"][:600].astype("float64")
    rf_cfg = built["cfg"]["ml"]["random_forest"]
    a = RandomForestBaseline(meta["input_features"], NODES, rf_cfg, seed=7, n_estimators=5).fit(x, y)
    b = RandomForestBaseline(meta["input_features"], NODES, rf_cfg, seed=7, n_estimators=5).fit(x, y)
    # same seed -> same forest; parallel tree averaging may differ in the last floating-point digit, so compare closely
    assert np.allclose(a.predict(x), b.predict(x), rtol=0, atol=1e-6)


def test_rf_features_come_from_the_window_only(built):
    meta, cfg = built["meta"], built["cfg"]["ml"]["random_forest"]
    names = rf_feature_names(meta["input_features"], cfg)
    assert len(names) == len(set(names)) > 40
    x = built["data"]["test"]["X_raw"][:50].copy()
    f1, _ = make_rf_features(x, meta["input_features"], cfg)
    assert np.array_equal(f1, make_rf_features(x.copy(), meta["input_features"], cfg)[0])
    too_deep = dict(cfg, lags=[12])
    with pytest.raises(FeatureError, match="history"):
        make_rf_features(x, meta["input_features"], too_deep)


# ------------------------------------------------------------------------------ 7. metrics
def test_risk_metrics_known_example():
    true = ["normal", "normal", "low_pressure_risk", "low_pressure_risk", "high_pressure_risk", "pressure_imbalance"]
    pred = ["normal", "low_pressure_risk", "low_pressure_risk", "low_pressure_risk", "high_pressure_risk", "normal"]
    m, cm = risk_metrics(true, pred)
    assert m["risk_macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1 + 0) / 4)
    assert m["risk_macro_recall"] == pytest.approx((0.5 + 1 + 1 + 0) / 4)
    assert m["risk_macro_f1"] == pytest.approx((0.5 + 0.8 + 1 + 0) / 4)
    assert m["risk_accuracy"] == pytest.approx(4 / 6)
    assert cm.tolist() == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    assert cm.shape == (len(PRESSURE_RISK_CLASSES),) * 2


def test_regression_metrics_known_example():
    y = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
    p = np.array([[11.0, 20.0, 28.0], [9.0, 20.0, 32.0]])
    m = regression_metrics(y, p, NODES)
    assert m["MAE"] == pytest.approx(1.0) and m["MAE_J2"] == pytest.approx(1.0) and m["MAE_J3"] == 0.0 \
        and m["MAE_J4"] == pytest.approx(2.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((1 + 0 + 4) * 2 / 6))
    with pytest.raises(ValueError):
        regression_metrics(y, p[:, :2], NODES)


def test_metrics_and_report_files_are_created(built):
    P = built["P"]
    metrics = pd.read_csv(P["baseline_metrics"])
    assert set(metrics["model"]) == {"persistence", "rule_based_reactive", "random_forest"}
    assert set(metrics["split"]) == {"train", "val", "test"}
    for col in ("MAE", "RMSE", "R2", "MAE_J2", "MAE_J3", "MAE_J4", "risk_macro_precision", "risk_macro_recall",
                "risk_macro_f1", "inference_ms_per_1000_windows", "mae_skill_vs_persistence"):
        assert col in metrics.columns, col
    rule = metrics[metrics["model"] == "rule_based_reactive"]
    assert rule["MAE"].isna().all() and rule["risk_macro_f1"].notna().all()  # it does not forecast pressure; honest NaN
    pers = metrics[(metrics["model"] == "persistence") & (metrics["split"] == "test")].iloc[0]
    assert pers["mae_skill_vs_persistence"] == 0.0
    errors = pd.read_csv(P["baseline_error_analysis"])
    assert {"scenario_type", "node", "leak_context", "all"} <= set(errors["group_kind"])
    assert {"model", "split", "group", "n_windows", "MAE", "max_abs_error"} <= set(errors.columns)
    for key in ("fig_prediction_comparison", "fig_confusion_matrix", "random_forest", "feature_names", "split_manifest"):
        p = Path(P[key])
        assert p.exists() and p.stat().st_size > 1000, key
    for key in ("fig_prediction_comparison", "fig_confusion_matrix"):
        assert Path(P[key]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_missing_raw_data_gives_a_readable_error(cfg, tmp_path):
    P = {k: str(v) for k, v in _paths(tmp_path).items()}
    with pytest.raises(DataError, match="generate_synthetic_data"):
        build_dataset(cfg, paths=P)
    with pytest.raises(DataError, match="build_ml_dataset"):
        run_baseline_experiment(cfg, paths=P, n_estimators=2)
