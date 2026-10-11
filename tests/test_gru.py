"""Phase 4 tests: GRU model, training loop, checkpoint, evaluation outputs.

They reuse the Phase 3 fixture (`built`): a small REAL EPANET dataset built inside a temporary folder, plus a 10-tree
random forest. The GRU is then trained for only a couple of epochs; these tests check mechanics and honesty rules,
not final accuracy (the real run is `python scripts/train_gru.py`).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from src.evaluation import metric_verdicts, run_gru_experiment  # noqa: E402
from src.gru_model import GRUForecaster, GRUPredictor, ModelError, pick_device  # noqa: E402
from src.labels import PRESSURE_RISK_CLASSES, label_settings, pressure_risk_classes  # noqa: E402
from src.training import TrainingError, make_loss, train_gru  # noqa: E402
from tests.test_dataset import cfg, built  # noqa: E402,F401  (fixtures)

NODES = ["J2", "J3", "J4"]
GRU_PATH_KEYS = {
    "gru_checkpoint": "trained/gru_best.pt", "gru_config": "trained/gru_config.json",
    "gru_metrics": "reports/gru_metrics.json", "model_comparison": "reports/model_comparison.csv",
    "gru_error_by_scenario_type": "reports/gru_error_by_scenario_type.csv",
    "fig_gru_history": "figures/gru_training_history.png", "fig_gru_prediction": "figures/gru_prediction_vs_actual.png",
    "fig_gru_per_node": "figures/gru_per_node_error.png", "fig_gru_confusion": "figures/gru_risk_confusion_matrix.png",
}


@pytest.fixture(scope="module")
def gru_run(built, cfg):
    """Train the GRU for 2 epochs on the small real dataset and write every Phase 4 file into the temp folder."""
    folder = Path(built["folder"]) / "phase4"
    paths = {**built["P"], **{k: str(folder / v) for k, v in GRU_PATH_KEYS.items()}}
    out = run_gru_experiment(cfg, paths=paths, max_epochs=2, log=lambda s: None)
    out["folder"] = folder
    return out


def _train(built, cfg, tmp_path, overrides=None, epochs=1, data=None, seed=3):
    d = data if data is not None else built["data"]
    meta = built["meta"]
    from src.preprocessing import load_scalers
    sc, _ = load_scalers(built["P"]["input_scaler"], built["P"]["target_scaler"])
    gcfg = {**cfg["gru"], **(overrides or {})}
    return train_gru(gcfg, d, meta["input_features"], NODES, sc.mean_, sc.scale_, tmp_path / "ck.pt", seed,
                     log=lambda s: None, max_epochs=epochs)


# ------------------------------------------------------------------------------------------- the network
def test_forward_pass_has_correct_output_shape(built):
    n_feat = len(built["meta"]["input_features"])
    window = built["data"]["train"]["X"].shape[1]
    net = GRUForecaster(n_feat, 3)
    out = net(torch.randn(7, window, n_feat))
    assert tuple(out.shape) == (7, 3)
    assert torch.isfinite(out).all()


def test_architecture_matches_the_specification():
    net = GRUForecaster(18, 3)
    assert net.gru1.hidden_size == 64 and net.gru1.batch_first and net.gru1.num_layers == 1
    assert net.gru2.hidden_size == 32 and net.gru2.batch_first
    assert abs(net.drop.p - 0.20) < 1e-9
    linears = [m for m in net.head if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(32, 32), (32, 3)]
    assert any(isinstance(m, torch.nn.ReLU) for m in net.head)


def test_forward_rejects_wrong_input_shape():
    net = GRUForecaster(18, 3)
    with pytest.raises(ModelError, match="expected input"):
        net(torch.randn(4, 12, 17))
    with pytest.raises(ModelError, match="expected input"):
        net(torch.randn(4, 18))


def test_loss_is_selectable_and_validated():
    assert isinstance(make_loss("huber"), torch.nn.HuberLoss)
    assert isinstance(make_loss("mse"), torch.nn.MSELoss)
    with pytest.raises(TrainingError, match="unknown loss"):
        make_loss("mae")


def test_device_defaults_to_cpu_without_gpu():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(ModelError):
        pick_device("tpu")


# ------------------------------------------------------------------------------------------- training
def test_one_training_epoch_completes(built, cfg, tmp_path):
    res = _train(built, cfg, tmp_path, epochs=1)
    assert res["epochs_run"] == 1 and len(res["history"]) == 1
    h = res["history"][0]
    assert np.isfinite([h["train_loss"], h["val_loss"], h["val_mae_m"]]).all()
    assert res["best_epoch"] == 1


def test_checkpoint_is_saved_and_contains_what_inference_needs(built, cfg, tmp_path):
    _train(built, cfg, tmp_path, epochs=1)
    ck_path = tmp_path / "ck.pt"
    assert ck_path.exists() and ck_path.stat().st_size > 0
    assert not (tmp_path / "ck.pt.tmp").exists()                       # no half-written leftovers
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    for key in ("model_state", "architecture", "feature_names", "target_nodes", "input_mean", "input_scale",
                "out_mean", "out_std", "predict_delta", "best_epoch"):
        assert key in ck
    assert ck["target_nodes"] == NODES


def test_model_reload_produces_valid_prediction(gru_run, built):
    x = built["data"]["test"]["X_raw"]
    reloaded = GRUPredictor.from_checkpoint(gru_run["paths"]["gru_checkpoint"])
    pred = reloaded.predict(x)
    assert pred.shape == (len(x), 3) and np.isfinite(pred).all()
    assert np.allclose(pred, gru_run["predictor"].predict(x), atol=1e-4)      # same as the model used in the report
    # metres on a sensible scale: far closer to the true pressure than to zero
    y = built["data"]["test"]["y_raw"]
    assert np.abs(pred - y).mean() < 0.5 * np.abs(y).mean()


def test_missing_checkpoint_gives_readable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_gru.py"):
        GRUPredictor.from_checkpoint(tmp_path / "nope.pt")


def test_nan_loss_is_never_silently_accepted(built, cfg, tmp_path, monkeypatch):
    import src.training as training

    class NanLoss(torch.nn.Module):
        def forward(self, pred, target):
            return pred.sum() * float("nan")

    monkeypatch.setattr(training, "make_loss", lambda *a, **k: NanLoss())
    with pytest.raises(TrainingError, match="loss became"):
        _train(built, cfg, tmp_path, epochs=1)


def test_nan_in_training_data_is_rejected(built, cfg, tmp_path):
    bad = {s: dict(v) for s, v in built["data"].items()}
    bad["train"]["X"] = bad["train"]["X"].copy()
    bad["train"]["X"][3, 2, 1] = np.nan
    with pytest.raises(TrainingError, match="NaN"):
        _train(built, cfg, tmp_path, data=bad)
    bad2 = {s: dict(v) for s, v in built["data"].items()}
    bad2["val"]["y_raw"] = bad2["val"]["y_raw"].astype("float64").copy()
    bad2["val"]["y_raw"][0, 0] = np.inf
    with pytest.raises(TrainingError, match="infinite|NaN"):
        _train(built, cfg, tmp_path, data=bad2)


def test_predictor_rejects_nan_windows(gru_run, built):
    x = built["data"]["test"]["X_raw"][:5].copy()
    x[0, 0, 0] = np.nan
    with pytest.raises(ModelError, match="NaN"):
        gru_run["predictor"].predict(x)


def test_early_stopping_stops_and_restores_best(built, cfg, tmp_path):
    # a huge min_delta means no later epoch can count as an improvement -> stop after 1 + patience epochs
    res = _train(built, cfg, tmp_path, overrides={"early_stopping_patience": 2, "min_delta": 1e9}, epochs=20)
    assert res["stopped_early"] and res["epochs_run"] == 3 and res["best_epoch"] == 1
    ck = torch.load(tmp_path / "ck.pt", map_location="cpu", weights_only=True)
    assert ck["best_epoch"] == 1                                    # the saved file is the best epoch, not the last


def test_training_never_touches_the_test_split(built, cfg, tmp_path):
    only_tv = {"train": built["data"]["train"], "val": built["data"]["val"]}      # no 'test' key at all
    res = _train(built, cfg, tmp_path, data=only_tv, epochs=1)
    assert res["epochs_run"] == 1


def test_training_is_reproducible_with_the_same_seed(built, cfg, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _train(built, cfg, tmp_path / "a", epochs=2, seed=11)
    b = _train(built, cfg, tmp_path / "b", epochs=2, seed=11)
    x = built["data"]["val"]["X_raw"][:300]
    assert np.allclose(a["predictor"].predict(x), b["predictor"].predict(x), atol=1e-4)
    assert [h["train_loss"] for h in a["history"]] == pytest.approx([h["train_loss"] for h in b["history"]], rel=1e-5)


def test_output_normalisation_comes_from_train_rows_only(built, cfg, tmp_path):
    res = _train(built, cfg, tmp_path, epochs=1)
    tr = built["data"]["train"]
    cols = [built["meta"]["input_features"].index(f"{n}_pressure") for n in NODES]
    delta = tr["y_raw"].astype("float64") - tr["X_raw"][:, -1, cols].astype("float64")
    assert np.allclose(res["meta"]["out_mean"], delta.mean(axis=0))
    assert np.allclose(res["meta"]["out_std"], delta.std(axis=0))


# ------------------------------------------------------------------------------------------- no hidden labels
def test_gru_inputs_are_sensor_and_control_features_only(gru_run, built, cfg):
    forbidden = set(cfg["ml"]["forbidden_inputs"]) | {"leak_flag", "leak_severity", "scenario_type"}
    feats = set(gru_run["predictor"].feature_names)
    assert not feats & forbidden
    assert gru_run["predictor"].model.num_features == len(feats) == built["data"]["train"]["X"].shape[2]
    conf = json.loads(Path(gru_run["paths"]["gru_config"]).read_text(encoding="utf-8"))
    assert not set(conf["input_features"]) & forbidden and conf["target_nodes"] == NODES


def test_predictions_depend_only_on_the_window(gru_run, built):
    """predict() takes nothing but the window: it has no way to read scenario type or leak labels."""
    import inspect
    assert list(inspect.signature(gru_run["predictor"].predict).parameters) == ["x_raw"]
    x = built["data"]["test"]["X_raw"][:50]
    assert np.array_equal(gru_run["predictor"].predict(x), gru_run["predictor"].predict(x.copy()))


def test_risk_label_from_prediction_uses_prediction_only(gru_run, built, cfg):
    settings = label_settings(cfg)
    x = built["data"]["test"]["X_raw"][:200]
    got = gru_run["predictor"].predict_risk_class(x, settings)
    assert np.array_equal(got, pressure_risk_classes(gru_run["predictor"].predict(x), NODES, settings))


# ------------------------------------------------------------------------------------------- evaluation outputs
def test_all_phase4_files_are_created(gru_run):
    for key in GRU_PATH_KEYS:
        p = Path(gru_run["paths"][key])
        assert p.exists() and p.stat().st_size > 0, f"missing {key}: {p}"


def test_gru_metrics_json_contains_every_required_metric(gru_run):
    gm = json.loads(Path(gru_run["paths"]["gru_metrics"]).read_text(encoding="utf-8"))
    required = ["MAE", "RMSE", "R2", "MAE_J2", "MAE_J3", "MAE_J4", "risk_macro_precision", "risk_macro_recall",
                "risk_macro_f1", "inference_ms_per_1000_windows"]
    for split in ("train", "val", "test"):
        for k in required:
            assert k in gm["metrics"][split] and np.isfinite(gm["metrics"][split][k]), (split, k)
    cm = np.array(gm["risk_confusion_matrix_test"]["rows_actual_cols_predicted"])
    assert cm.shape == (4, 4) and cm.sum() == gm["metrics"]["test"]["n_windows"]
    assert gm["risk_confusion_matrix_test"]["classes"] == list(PRESSURE_RISK_CLASSES)
    assert gm["inference"]["ms_single_window_median"] > 0
    assert len(gm["history"]) == gm["training"]["epochs_run"] == 2


def test_model_comparison_has_the_three_models_on_every_split(gru_run):
    t = pd.read_csv(gru_run["paths"]["model_comparison"])
    assert set(t["model"]) == {"persistence", "random_forest", "gru"}
    assert set(t["split"]) == {"train", "val", "test"} and len(t) == 9
    for col in ("MAE", "RMSE", "R2", "MAE_J2", "MAE_J3", "MAE_J4", "risk_macro_precision", "risk_macro_recall",
                "risk_macro_f1", "inference_ms_per_1000_windows"):
        assert col in t.columns and t[col].notna().all()


def test_error_by_scenario_type_csv(gru_run, built):
    t = pd.read_csv(gru_run["paths"]["gru_error_by_scenario_type"])
    assert "all" in set(t["scenario_type"])
    assert set(t["scenario_type"]) - {"all"} == set(built["data"]["test"]["meta_scenario_type"])
    assert {"gru_MAE", "persistence_MAE", "random_forest_MAE"} <= set(t.columns)


def test_gru_is_scored_on_the_same_windows_as_the_baselines(gru_run, built):
    m = gru_run["metrics"]
    assert set(m[m["split"] == "test"]["n_windows"]) == {len(built["data"]["test"]["y_raw"])}


# ------------------------------------------------------------------------------------------- honesty of the verdict
def _fake_metrics(gru_values: dict):
    base = {"MAE": 0.10, "RMSE": 0.5, "R2": 0.99, "MAE_J2": 0.04, "MAE_J3": 0.15, "MAE_J4": 0.12,
            "risk_macro_precision": 0.99, "risk_macro_recall": 0.99, "risk_macro_f1": 0.99}
    rows = [{"model": "persistence", "split": "test", **base}, {"model": "random_forest", "split": "test", **base},
            {"model": "gru", "split": "test", **{**base, **gru_values}}]
    return pd.DataFrame(rows)


def test_verdict_does_not_claim_superiority_when_a_metric_is_worse():
    mixed = metric_verdicts(_fake_metrics({"MAE": 0.05, "RMSE": 0.4, "R2": 0.995, "MAE_J2": 0.01, "MAE_J3": 0.05,
                                           "MAE_J4": 0.05, "risk_macro_precision": 0.995, "risk_macro_recall": 0.98,
                                           "risk_macro_f1": 0.995}))
    v = mixed["persistence"]
    assert "risk_macro_recall" in v["worse"] and not v["improves_every_metric"]


def test_verdict_requires_strict_improvement_on_every_metric():
    better = {"MAE": 0.05, "RMSE": 0.4, "R2": 0.995, "MAE_J2": 0.01, "MAE_J3": 0.05, "MAE_J4": 0.05,
              "risk_macro_precision": 0.995, "risk_macro_recall": 0.995, "risk_macro_f1": 0.995}
    assert metric_verdicts(_fake_metrics(better))["random_forest"]["improves_every_metric"]
    tied = {**better, "MAE_J2": 0.04}                                     # one metric merely equal -> not "every"
    assert not metric_verdicts(_fake_metrics(tied))["random_forest"]["improves_every_metric"]


def test_verdict_in_json_matches_the_table(gru_run):
    gm = json.loads(Path(gru_run["paths"]["gru_metrics"]).read_text(encoding="utf-8"))
    t = pd.read_csv(gru_run["paths"]["model_comparison"])
    t = t[t["split"] == "test"].set_index("model")
    for other in ("persistence", "random_forest"):
        v = gm["verdict_test_split"][other]
        assert set(v["better"]) == {m for m in ("MAE", "RMSE", "MAE_J2", "MAE_J3", "MAE_J4")
                                    if t.loc["gru", m] < t.loc[other, m] - 1e-9} | \
            {m for m in ("R2", "risk_macro_precision", "risk_macro_recall", "risk_macro_f1")
             if t.loc["gru", m] > t.loc[other, m] + 1e-9}
