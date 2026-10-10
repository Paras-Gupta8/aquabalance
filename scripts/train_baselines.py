"""Phase 3: train and evaluate the baselines (persistence, rule-based, random forest).

Usage (from the repository folder; run build_ml_dataset.py first):
    python scripts/train_baselines.py
    python scripts/train_baselines.py --n-estimators 50      # faster, for a quick look
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation import run_baseline_experiment  # noqa: E402
from src.feature_engineering import FeatureError  # noqa: E402
from src.preprocessing import DataError  # noqa: E402
from src.utils import ConfigError, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the AquaBalance forecasting baselines.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-estimators", type=int, default=None, help="random forest trees (default from config)")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        print("Training the random forest on the TRAIN split and evaluating all baselines ...")
        out = run_baseline_experiment(cfg, n_estimators=args.n_estimators)
    except (ConfigError, DataError, FeatureError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    m = out["metrics"]
    pd.set_option("display.width", 200)
    show = ["model", "split", "n_windows", "MAE", "RMSE", "R2", "MAE_J2", "MAE_J3", "MAE_J4", "mae_skill_vs_persistence"]
    print(f"\nPressure forecast (metres; random forest fitted in {out['fit_seconds']:.1f} s):")
    print(m[m["MAE"].notna()][show].round(4).to_string(index=False))
    risk = ["model", "split", "risk_macro_precision", "risk_macro_recall", "risk_macro_f1", "risk_accuracy",
            "inference_ms_per_1000_windows"]
    print("\nRisk classes derived from the (predicted) pressures:")
    print(m[risk].round(4).to_string(index=False))
    print("\nRandom forest: 10 most important features")
    for name, imp in list(out["model"].feature_importances().items())[:10]:
        print(f"  {name:<28}{imp:.3f}")
    print("\nOutput files:")
    for key in ("random_forest", "baseline_metrics", "baseline_error_analysis", "fig_prediction_comparison",
                "fig_confusion_matrix"):
        print(f"  {out['paths'][key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
