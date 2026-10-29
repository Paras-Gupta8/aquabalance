"""Phase 4: train and evaluate the GRU pressure forecaster (pressure forecasting and decision support).

Usage (from the repository folder; run build_ml_dataset.py and train_baselines.py first):
    python scripts/train_gru.py
    python scripts/train_gru.py --epochs 5            # quick smoke run
    python scripts/train_gru.py --loss mse --seed 7
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation import run_gru_experiment  # noqa: E402
from src.feature_engineering import FeatureError  # noqa: E402
from src.gru_model import ModelError  # noqa: E402
from src.preprocessing import DataError  # noqa: E402
from src.training import TrainingError  # noqa: E402
from src.utils import ConfigError, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and evaluate the AquaBalance GRU forecaster.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="maximum epochs (default gru.max_epochs = 100)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--loss", choices=["huber", "mse"], default=None)
    ap.add_argument("--absolute", action="store_true", help="predict absolute pressure instead of the change from now")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = ap.parse_args()
    over = {}
    if args.loss:
        over["loss"] = args.loss
    if args.absolute:
        over["predict_delta"] = False
    if args.device:
        over["device"] = args.device
    try:
        cfg = load_config(args.config)
        out = run_gru_experiment(cfg, max_epochs=args.epochs, seed=args.seed, overrides=over)
    except (ConfigError, DataError, FeatureError, ModelError, TrainingError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    pd.set_option("display.width", 220)
    m = out["metrics"]
    show = ["model", "split", "n_windows", "MAE", "RMSE", "R2", "MAE_J2", "MAE_J3", "MAE_J4"]
    print("\nPressure forecast (metres):")
    print(out["comparison"][show].round(4).to_string(index=False))
    risk = ["model", "split", "risk_macro_precision", "risk_macro_recall", "risk_macro_f1", "risk_accuracy",
            "inference_ms_per_1000_windows"]
    print("\nRisk classes derived from the predicted pressures:")
    print(out["comparison"][risk].round(4).to_string(index=False))
    print(f"\nGRU single-window latency: {out['latency_ms']:.2f} ms (median)")
    print("\nGRU vs the others on the TEST split (per metric):")
    for other, v in out["verdict"].items():
        print(f"  vs {other}: better on {len(v['better'])}, worse on {len(v['worse'])}, tied on {len(v['tied'])}"
              f" -> improves EVERY metric: {v['improves_every_metric']}")
        if v["worse"]:
            print(f"      worse on: {', '.join(v['worse'])}")
    print("\nOutput files:")
    for k in ("gru_checkpoint", "gru_config", "gru_metrics", "model_comparison", "gru_error_by_scenario_type",
              "fig_gru_history", "fig_gru_prediction", "fig_gru_per_node", "fig_gru_confusion"):
        print(f"  {out['paths'][k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
