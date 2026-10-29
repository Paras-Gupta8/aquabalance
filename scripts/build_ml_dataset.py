"""Phase 3: build the leakage-safe machine-learning dataset from the EPANET raw time series.

Usage (from the repository folder):
    python scripts/generate_synthetic_data.py --mode standard    # Phase 2 data first (300 scenarios)
    python scripts/build_ml_dataset.py
    python scripts/build_ml_dataset.py --window 12 --horizon 1 --seed 42
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_builder import build_dataset  # noqa: E402
from src.feature_engineering import FeatureError  # noqa: E402
from src.preprocessing import DataError  # noqa: E402
from src.utils import ConfigError, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the AquaBalance ML dataset (windows, split, scalers).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None, help="split seed (default: project.seed)")
    parser.add_argument("--window", type=int, default=None, help="history steps (default ml.window_steps = 12)")
    parser.add_argument("--horizon", type=int, default=None, help="forecast steps ahead (default ml.horizon_steps = 1)")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        out = build_dataset(cfg, seed=args.seed, window=args.window, horizon=args.horizon)
    except (ConfigError, DataError, FeatureError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Task: previous {out['window']} steps -> pressure at J2, J3, J4 {out['horizon']} step(s) ahead "
          f"(split seed {out['seed']})")
    print(f"Input features ({len(out['features'])}): {', '.join(out['features'])}")
    for name, why in out["dropped"].items():
        print(f"Dropped feature: {name} -- {why}")
    print(f"\nScenarios in the raw data: {out['n_scenarios']}")
    print(f"{'split':<6}{'scenarios':>10}{'windows':>9}   X shape               y shape")
    for s in ("train", "val", "test"):
        c, sh = out["counts"][s], out["shapes"][s]
        print(f"{s:<6}{c['scenarios']:>10}{c['windows']:>9}   {str(sh['X']):<21} {sh['y']}")
    print("\nScenario types per split:")
    print(out["manifest"].pivot_table(index="scenario_type", columns="split", values="scenario_id",
                                      aggfunc="count", fill_value=0)[["train", "val", "test"]].to_string())
    print("\nOutput files:")
    for key in ("train_npz", "val_npz", "test_npz", "feature_names", "split_manifest", "input_scaler", "target_scaler"):
        print(f"  {out['paths'][key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
