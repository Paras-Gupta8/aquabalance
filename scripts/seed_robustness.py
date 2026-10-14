"""Phase 8: how much do the neural-network results change with the random seed?

Trains the GRU, LSTM and TCN again with several seeds (checkpoints go to a temporary folder, so the real checkpoints and the
model-comparison table are NOT touched) and reports the mean and spread of the TEST MAE. This tells you whether a difference
between two networks is bigger than seed-to-seed noise.

Usage (after Phases 3-4):
    python scripts/seed_robustness.py                       # seeds 1 2 3, all three networks (about 15-25 min on a laptop CPU)
    python scripts/seed_robustness.py --models tcn --seeds 1 2
Output: outputs/reports/seed_robustness.csv
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataset_builder import load_split  # noqa: E402
from src.preprocessing import load_scalers  # noqa: E402
from src.training import train_gru  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed robustness of the GRU / LSTM / TCN (test MAE).")
    ap.add_argument("--models", nargs="+", choices=["gru", "lstm", "tcn"], default=["gru", "lstm", "tcn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None, help="output CSV (default: outputs/reports/seed_robustness.csv, or ..._<model>.csv if --models is a subset)")
    args = ap.parse_args()
    cfg = load_config()
    P = {k: resolve_path(v) for k, v in cfg["paths"].items()}
    try:
        meta = json.loads(Path(P["feature_names"]).read_text(encoding="utf-8"))
        data = {s: load_split(P[f"{s}_npz"]) for s in ("train", "val", "test")}
        sc, _ = load_scalers(P["input_scaler"], P["target_scaler"])
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}. Run Phases 2-3 first.", file=sys.stderr)
        return 1
    feats, nodes = meta["input_features"], list(cfg["ml"]["target_nodes"])
    rows = []
    with tempfile.TemporaryDirectory(prefix="aquabalance_seeds_") as tmp:
        for t in args.models:
            section = {**cfg["gru"], **(cfg[t] if t != "gru" else {})}
            for seed in args.seeds:
                fit = train_gru(section, data, feats, nodes, sc.mean_, sc.scale_, Path(tmp) / f"{t}_{seed}.pt", seed,
                                log=lambda s: None, max_epochs=args.epochs, model_type=t)
                pred = fit["predictor"]
                err = {sp: float(np.abs(pred.predict(data[sp]["X_raw"]) - data[sp]["y_raw"]).mean()) for sp in ("val", "test")}
                rows.append({"model": t, "seed": seed, "n_parameters": fit["n_parameters"], "best_epoch": fit["best_epoch"],
                             "val_MAE": err["val"], "test_MAE": err["test"]})
                print(f"{t:5s} seed {seed}: val MAE {err['val']:.4f}  test MAE {err['test']:.4f}  (best epoch {fit['best_epoch']})", flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else resolve_path(cfg["paths"]["scenario_error_analysis"]).parent / (
        "seed_robustness.csv" if set(args.models) == {"gru", "lstm", "tcn"} else f"seed_robustness_{'_'.join(args.models)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")
    print("\nTest MAE over seeds (metres):")
    print(df.groupby("model")["test_MAE"].agg(["mean", "std", "min", "max", "count"]).round(4).to_string())
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
