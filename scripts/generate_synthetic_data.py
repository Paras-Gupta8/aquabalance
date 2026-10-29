"""Phase 2: generate synthetic scenarios with EPANET (via WNTR).

Usage (from the repository folder):
    python scripts/generate_synthetic_data.py --mode quick       # 50 scenarios
    python scripts/generate_synthetic_data.py --mode standard    # 300 scenarios
    python scripts/generate_synthetic_data.py --mode quick --seed 7
    python scripts/generate_synthetic_data.py --n 120            # any custom number
    python scripts/generate_synthetic_data.py --mode quick --sensor-noise   # also write a noisy COPY

Every pressure/flow value in the output comes from EPANET simulations, not from random numbers.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.labels import RISK_LABELS  # noqa: E402
from src.scenario_generator import FAMILIES, FAMILY_GROUPS, generate_dataset  # noqa: E402
from src.utils import ConfigError, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AquaBalance synthetic scenarios with EPANET.")
    parser.add_argument("--mode", choices=["quick", "standard"], default="quick")
    parser.add_argument("--n", type=int, default=None, help="custom number of scenarios (overrides --mode)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: project.seed in the config)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sensor-noise", action="store_true", help="also write a noisy COPY of the time series")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        n = args.n or cfg["scenarios"][f"{args.mode}_n"]
        print(f"AquaBalance Phase 2: generating {n} scenarios with EPANET ...")
        out = generate_dataset(cfg, n, seed=args.seed, sensor_noise=True if args.sensor_noise else None)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary, meta, labels = out["summary"], out["metadata"], out["labels"]
    print(f"\nAccepted {out['n_accepted']}/{out['n_requested']} scenarios "
          f"({out['n_failed_attempts']} failed/rejected attempts, all logged in summary.csv)")
    print("\nScenario counts by type:")
    print(meta["scenario_type"].value_counts().reindex(list(FAMILIES)).fillna(0).astype(int).to_string())
    print("\nScenario counts by group (target: normal 35-45%, low-pressure 15-20%, high-pressure 15-20%, "
          "valve 10-15%, leak/burst 10-15%):")
    for group, members in FAMILY_GROUPS.items():
        k = int(meta["scenario_type"].isin(members).sum())
        print(f"  {group:<28} {k:>4}  ({k / max(len(meta), 1):.0%})")
    ok = summary[summary["status"].isin(["ok", "ok_with_warnings"])]
    print("\nPrimary risk label per scenario (what EPANET actually produced):")
    print(ok["scenario_risk_label"].value_counts().reindex(list(RISK_LABELS)).fillna(0).astype(int).to_string())
    print("\nOutput files:")
    for key in ("scenario_metadata", "raw_timeseries", "labels", "demand_profiles", "scenario_specs", "summary",
                "network_config_json"):
        print(f"  {out['paths'][key]}")
    if not out["complete"]:
        print("\nWARNING: fewer scenarios than requested were accepted. See outputs/simulations/summary.csv.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
