"""Phase 8: train the LSTM and the lightweight TCN and add them to the model-comparison table.

Usage (from the repository folder; run build_ml_dataset.py, train_baselines.py and train_gru.py first):
    python scripts/train_sequence_models.py                    # train LSTM and TCN, then re-score every saved model
    python scripts/train_sequence_models.py --models lstm      # only the LSTM
    python scripts/train_sequence_models.py --epochs 5         # quick smoke run
    python scripts/train_sequence_models.py --score-only       # no training: re-score the saved checkpoints

The GRU is not retrained or overwritten. All models use the same input features, scenario-level splits, scalers, loss,
optimiser, seed and early stopping, and are scored with the same code.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation import run_sequence_experiment  # noqa: E402
from src.feature_engineering import FeatureError  # noqa: E402
from src.gru_model import ModelError  # noqa: E402
from src.preprocessing import DataError  # noqa: E402
from src.training import TrainingError  # noqa: E402
from src.utils import ConfigError, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the AquaBalance LSTM and TCN comparison models.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", choices=["lstm", "tcn"], default=["lstm", "tcn"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--score-only", action="store_true", help="do not train; re-score the saved checkpoints")
    args = ap.parse_args()
    try:
        cfg = load_config(args.config)
        out = run_sequence_experiment(cfg, tuple(args.models), max_epochs=args.epochs, seed=args.seed, train=not args.score_only)
    except (ConfigError, DataError, FeatureError, ModelError, TrainingError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    pd.set_option("display.width", 220)
    t = out["comparison"]
    t = t[t["split"] == "test"]
    print("\nTEST split (same scenarios for every model; metres):")
    print(t[["model", "MAE", "RMSE", "R2", "risk_macro_f1", "inference_ms_per_1000_windows"]].round(4).to_string(index=False))
    print("\nParameters:", {m: sum(p.numel() for p in o.model.parameters()) for m, o in out["models"].items() if m in ("gru", "lstm", "tcn")})
    print("\nNo model is declared best by assumption: compare the columns above.")
    print(f"Updated: {out['paths']['model_comparison']}\n         {out['paths']['gru_error_by_scenario_type']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
