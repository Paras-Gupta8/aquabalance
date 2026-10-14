"""Phase 8 tests: LSTM and TCN comparison models.

They reuse the Phase 3/4 fixtures (a small REAL EPANET dataset in a temp folder, a 2-epoch GRU). The LSTM/TCN are trained
for only a couple of epochs: these tests check mechanics and fairness rules, not final accuracy.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("torch", reason="PyTorch is not installed (optional: pip install -r requirements-deep.txt)")
import torch  # noqa: E402

from src.evaluation import merge_keep_other_models, run_sequence_experiment  # noqa: E402
from src.gru_model import GRUForecaster, GRUPredictor, ModelError  # noqa: E402
from src.sequence_models import (LSTMForecaster, TCNForecaster, architecture_from_config, build_model,  # noqa: E402
                                 count_parameters)
from tests.test_dataset import cfg, built  # noqa: E402,F401  (fixtures)
from tests.test_gru import gru_run  # noqa: E402,F401  (fixture: 2-epoch GRU + all Phase 4 files)

F, W = 18, 12


def _nets(cfg):
    return {t: build_model(t, F, 3, architecture_from_config(t, cfg[t] if t != "gru" else cfg["gru"])) for t in ("gru", "lstm", "tcn")}


# ------------------------------------------------------------------------------ architecture
def test_all_three_models_have_the_same_input_and_output(cfg):
    x = torch.randn(5, W, F)
    for t, net in _nets(cfg).items():
        out = net(x)
        assert tuple(out.shape) == (5, 3) and torch.isfinite(out).all(), t


def test_parameter_counts_are_similar_to_the_gru(cfg):
    n = {t: count_parameters(net) for t, net in _nets(cfg).items()}
    assert n["gru"] == 26691
    assert abs(n["lstm"] - n["gru"]) / n["gru"] < 0.02, n
    assert abs(n["tcn"] - n["gru"]) / n["gru"] < 0.05, n


def test_lstm_matches_the_gru_structure():
    net = LSTMForecaster(F, 3)
    assert net.lstm1.batch_first and net.lstm2.batch_first and abs(net.drop.p - 0.2) < 1e-9
    assert [(l.in_features, l.out_features) for l in net.head if isinstance(l, torch.nn.Linear)] == [(26, 32), (32, 3)]


def test_wrong_input_shape_is_rejected(cfg):
    for net in _nets(cfg).values():
        with pytest.raises(ModelError, match="expected input"):
            net(torch.randn(4, W, F - 1))


def test_unknown_model_type_is_rejected():
    with pytest.raises(ModelError, match="unknown model type"):
        build_model("transformer", F, 3, {})
    with pytest.raises(ModelError, match="unknown model type"):
        architecture_from_config("transformer", {})


# ------------------------------------------------------------------------------ TCN: causal, dilated, residual
def test_tcn_is_causal_future_inputs_do_not_change_past_outputs():
    torch.manual_seed(0)
    net = TCNForecaster(F, 3, dropout=0.0).eval()
    x = torch.randn(3, W, F)
    t = 6
    x2 = x.clone()
    x2[:, t + 1:, :] += 100.0                                    # change ONLY the future
    h1 = net.blocks(x.transpose(1, 2))
    h2 = net.blocks(x2.transpose(1, 2))
    assert torch.allclose(h1[:, :, :t + 1], h2[:, :, :t + 1], atol=1e-6)
    assert not torch.allclose(h1[:, :, t + 1:], h2[:, :, t + 1:])    # and the future positions DO change


def test_tcn_uses_dilation_residual_blocks_and_covers_the_window(cfg):
    net = _nets(cfg)["tcn"]
    dil = [b.c1.conv.dilation[0] for b in net.blocks]
    assert dil == [1, 2, 4]
    assert all(b.c1.conv.kernel_size[0] == 3 for b in net.blocks)
    assert net.receptive_field >= W
    assert any(isinstance(b.skip, torch.nn.Conv1d) for b in net.blocks)         # residual path exists
    assert all(c.conv.padding == (0,) for b in net.blocks for c in (b.c1, b.c2))   # padding is manual and left-only


def test_lstm_and_tcn_do_not_use_future_or_label_columns(built):
    feats = built["meta"]["input_features"]
    assert not {"leak_flag", "leak_severity", "scenario_type", "scenario_id", "tank_level"} & set(feats)


# ------------------------------------------------------------------------------ training + checkpoint + comparison table
@pytest.fixture(scope="module")
def seq_run(built, cfg, gru_run):
    paths = {**gru_run["paths"]}
    folder = Path(built["folder"]) / "phase8"
    extra = {"lstm_checkpoint": "trained/lstm_best.pt", "tcn_checkpoint": "trained/tcn_best.pt",
             "lstm_metrics": "reports/lstm_metrics.json", "tcn_metrics": "reports/tcn_metrics.json",
             "fig_lstm_history": "figures/lstm_history.png", "fig_tcn_history": "figures/tcn_history.png"}
    paths.update({k: str(folder / v) for k, v in extra.items()})
    gru_ck = Path(paths["gru_checkpoint"])
    before = hashlib.sha256(gru_ck.read_bytes()).hexdigest()
    out = run_sequence_experiment(cfg, ("lstm", "tcn"), paths=paths, max_epochs=2, log=lambda s: None)
    out["gru_hash_before"], out["gru_hash_after"] = before, hashlib.sha256(gru_ck.read_bytes()).hexdigest()
    out["P"] = {k: Path(v) for k, v in paths.items()}
    return out


def test_gru_checkpoint_is_untouched_by_phase_8(seq_run):
    assert seq_run["gru_hash_before"] == seq_run["gru_hash_after"]


def test_models_trained_from_the_same_scenario_level_split(seq_run, built):
    for t, fit in seq_run["fits"].items():
        assert fit["meta"]["feature_names"] == built["meta"]["input_features"]
        assert fit["meta"]["model_type"] == t and fit["seed"] == built["cfg"]["project"]["seed"]
    d = seq_run["data"]
    ids = {s: set(map(str, np.unique(d[s]["meta_scenario_id"]))) for s in ("train", "val", "test")}
    assert ids["train"] and ids["val"] and ids["test"]
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])


def test_checkpoints_reload_into_working_predictors(seq_run, built):
    x = built["data"]["test"]["X_raw"][:20]
    for t in ("lstm", "tcn"):
        p = GRUPredictor.from_checkpoint(seq_run["P"][f"{t}_checkpoint"])
        assert p.name == t
        out = p.predict(x)
        assert out.shape == (20, 3) and np.isfinite(out).all()
        assert np.allclose(out, seq_run["models"][t].predict(x))


def test_comparison_table_has_all_five_models_on_all_splits_with_the_same_columns(seq_run):
    t = seq_run["comparison"]
    assert set(t["model"]) == {"persistence", "random_forest", "gru", "lstm", "tcn"}
    for sp in ("train", "val", "test"):
        assert len(t[t["split"] == sp]) == 5
    saved = pd.read_csv(seq_run["P"]["model_comparison"])
    assert set(saved["model"]) == set(t["model"])
    for c in ("MAE", "RMSE", "risk_macro_f1", "inference_ms_per_1000_windows"):
        assert saved[c].notna().all()


def test_every_model_is_scored_on_identical_test_windows(seq_run):
    t = seq_run["comparison"]
    assert t[t["split"] == "test"]["n_windows"].nunique() == 1


def test_scenario_error_table_has_lstm_and_tcn_columns(seq_run):
    cols = seq_run["by_type"].columns
    for m in ("persistence", "random_forest", "gru", "lstm", "tcn"):
        assert f"{m}_MAE" in cols


def test_retraining_the_gru_keeps_the_lstm_and_tcn_rows(built, cfg, seq_run):
    """Re-running run_gru_experiment must not erase rows written by the Phase 8 script."""
    from src.evaluation import run_gru_experiment
    out = run_gru_experiment(cfg, paths={k: str(v) for k, v in seq_run["P"].items()}, max_epochs=1, make_figures=False, log=lambda s: None)
    saved = pd.read_csv(seq_run["P"]["model_comparison"])
    assert {"lstm", "tcn", "gru"} <= set(saved["model"])
    assert "lstm_MAE" in pd.read_csv(seq_run["P"]["gru_error_by_scenario_type"]).columns


# ------------------------------------------------------------------------------ backward compatibility + merge helper
def test_old_gru_checkpoints_without_model_type_still_load(gru_run, tmp_path):
    ck = torch.load(gru_run["paths"]["gru_checkpoint"], map_location="cpu", weights_only=True)
    ck.pop("model_type", None)
    p = tmp_path / "old.pt"
    torch.save(ck, p)
    pred = GRUPredictor.from_checkpoint(p)
    assert pred.name == "gru" and isinstance(pred.model, GRUForecaster)


def test_merge_keeps_rows_of_other_models(tmp_path):
    old = pd.DataFrame({"model": ["gru", "lstm"], "split": ["test", "test"], "MAE": [1.0, 2.0]})
    p = tmp_path / "t.csv"
    old.to_csv(p, index=False)
    new = pd.DataFrame({"model": ["gru"], "split": ["test"], "MAE": [0.5]})
    out = merge_keep_other_models(new, p, ["model", "split"])
    assert dict(zip(out["model"], out["MAE"])) == {"gru": 0.5, "lstm": 2.0}
