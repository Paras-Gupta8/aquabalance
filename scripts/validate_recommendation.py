"""Phase 6: EPANET simulated validation of recommendations.

This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility.

Usage (from the repository folder; needs Phases 2-5 first: scenario data, dataset, GRU):
    python scripts/validate_recommendation.py                      # a representative set from the test split
    python scripts/validate_recommendation.py --per-category 3     # more examples per situation
    python scripts/validate_recommendation.py --scenario S0123 --step 40   # one decision window

For each decision window the GRU forecast -> risk engine -> recommendation engine run exactly as in Phase 5; then the
scenario is re-simulated in EPANET (WNTR EpanetSimulator) before and after every bounded candidate action.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataset_builder import load_split  # noqa: E402
from src.epanet_validator import (ValidationError, ValidationSettings, load_base_network, load_spec,  # noqa: E402
                                  plot_pressure_comparison, plot_violation_comparison, validate_recommendation,
                                  write_outputs)
from src.gru_model import GRUPredictor, ModelError  # noqa: E402
from src.preprocessing import DataError  # noqa: E402
from src.recommendation_engine import (VALIDATION_DISCLAIMER, VALIDATION_LABEL, RecommendationError,  # noqa: E402
                                       RecommendationSettings, recommend)
from src.risk_engine import RiskError, RiskSettings, assess_windows  # noqa: E402
from src.utils import ConfigError, load_config, resolve_path  # noqa: E402

import json  # noqa: E402

CATEGORIES = [("normal", 1), ("high_pressure_risk", 2), ("low_critical_J4", 2), ("low_pressure_risk", 1),
              ("pressure_imbalance", 2), ("leak_like_pattern", 2), ("burst_like_pattern", 1)]


def _key(a):
    if a.predicted_risk_label == "low_pressure_risk" and a.critical_node_at_risk:
        return "low_critical_J4"
    return a.predicted_risk_label


def main() -> int:
    ap = argparse.ArgumentParser(description="EPANET simulated validation of AquaBalance recommendations.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-category", type=int, default=None, help="examples per situation (default: 1-2)")
    ap.add_argument("--scenario", default=None, help="validate one scenario id ...")
    ap.add_argument("--step", type=int, default=None, help="... at this decision step (the 'now' row, 0-95)")
    args = ap.parse_args()
    try:
        cfg = load_config(args.config)
        P = {k: resolve_path(v) for k, v in cfg["paths"].items()}
        for need in ("feature_names", "gru_checkpoint", "test_npz", "scenario_specs"):
            if not Path(P[need]).exists():
                raise DataError(f"{P[need]} not found. Run Phases 2-4 first (generate_synthetic_data.py, "
                                f"build_ml_dataset.py, train_gru.py)")
        base_inp = load_base_network(P["base_inp"])
        feats = json.loads(Path(P["feature_names"]).read_text(encoding="utf-8"))["input_features"]
        nodes = list(cfg["ml"]["target_nodes"])
        rs, ks, vs = RecommendationSettings.from_config(cfg), RiskSettings.from_config(cfg), ValidationSettings.from_config(cfg)
        gru = GRUPredictor.from_checkpoint(P["gru_checkpoint"])
        data = {s: load_split(P[f"{s}_npz"]) for s in ("train", "val", "test")}
    except (ConfigError, DataError, ValidationError, ModelError, RiskError, RecommendationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    picks = []                                   # (scenario_id, end_step, window, forecast, assessment, recommendation, category)
    try:
        if args.scenario:
            if args.step is None:
                raise ValidationError("--scenario needs --step")
            for sp in ("test", "val", "train"):
                d = data[sp]
                hit = np.where((d["meta_scenario_id"] == args.scenario) & (d["meta_end_step"] == args.step))[0]
                if len(hit):
                    i = int(hit[0])
                    fc = gru.predict(d["X_raw"][i:i + 1])
                    a = assess_windows(d["X_raw"][i:i + 1], fc, feats, nodes, ks)[0]
                    picks.append((args.scenario, args.step, a, recommend(a, rs), _key(a), dict(zip(nodes, fc[0]))))
                    break
            else:
                raise ValidationError(f"no model window for {args.scenario} at step {args.step} "
                                      f"(valid decision steps are 11-94)")
        else:
            d = data["test"]
            fc = gru.predict(d["X_raw"])
            A = assess_windows(d["X_raw"], fc, feats, nodes, ks)
            ids, ends = d["meta_scenario_id"], d["meta_end_step"]
            for cat, default_n in CATEGORIES:
                n = args.per_category or default_n
                best = {}
                for i, a in enumerate(A):
                    if _key(a) == cat and not a.uncertain and ends[i] + 1 < 95:
                        sid = str(ids[i])
                        if sid not in best or a.severity_score > A[best[sid]].severity_score:
                            best[sid] = i
                for sid in sorted(best)[:n]:
                    i = best[sid]
                    picks.append((sid, int(ends[i]), A[i], recommend(A[i], rs), cat, dict(zip(nodes, fc[i]))))
        if not picks:
            raise ValidationError("no decision windows found to validate")
        results = []
        for sid, step, a, rec, cat, fc in picks:
            spec = load_spec(P["scenario_specs"], sid)
            results.append(validate_recommendation(spec, rec, step, cfg, base_inp, rs, vs))
    except (ValidationError, DataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    files = write_outputs(results, P)
    figs = [plot_pressure_comparison(results, vs, P["fig_before_after_pressure"]),
            plot_violation_comparison(results, vs, P["fig_before_after_violations"])]

    pd.set_option("display.width", 230)
    print(f"{VALIDATION_LABEL} of {len(results)} recommendations (each re-simulated in EPANET before and after).")
    print(VALIDATION_DISCLAIMER + "\n")
    for v, (sid, step, a, rec, cat, fc) in zip(results, picks):
        b, af = v["before"], v["after"]
        print(f"--- {v['scenario_id']} step {step} [{cat}; EPANET scenario family, analysis only: {v['scenario_family_analysis_only']}]")
        print(f"    engine: {v['engine_action_type']} ({v['engine_adjustment_percent']:+g}%) on {v['engine_target_component']} "
              f"-> EPANET says: {v['engine_status_in_epanet']}")
        print(f"    chosen: {v['chosen_action']}   result: {v['final_validation_result']}   candidates tried: {v['n_candidates_evaluated']}")
        print(f"    pressure mean J1/J2/J3/J4 (m) before: " + " / ".join(f"{b[f'{n}_pressure_mean']:.1f}" for n in ('J1', 'J2', 'J3', 'J4'))
              + "   after: " + " / ".join(f"{af[f'{n}_pressure_mean']:.1f}" for n in ('J1', 'J2', 'J3', 'J4')))
        print(f"    J4 status {b['J4_status']} -> {af['J4_status']}   J4 min {b['J4_pressure_min']:.1f} -> {af['J4_pressure_min']:.1f} m")
        print(f"    violations low {b['low_pressure_violation_count']}->{af['low_pressure_violation_count']}  "
              f"high {b['high_pressure_violation_count']}->{af['high_pressure_violation_count']}  "
              f"imbalance steps {b['imbalance_step_count']}->{af['imbalance_step_count']}  "
              f"spread {b['branch_pressure_spread_mean']:.1f}->{af['branch_pressure_spread_mean']:.1f} m  "
              f"max P {b['max_pressure']:.1f}->{af['max_pressure']:.1f} m  safe-node coverage {b['safe_node_coverage']:.0%}->{af['safe_node_coverage']:.0%}")
        print(f"    loss proxy {b['loss_proxy_lps']:.2f}->{af['loss_proxy_lps']:.2f} L/s   demand served {b['demand_served_fraction']:.1%}->{af['demand_served_fraction']:.1%}")
        print(f"    {v['final_message']}")
    print("\nEffectiveness summary:")
    eff = pd.read_csv(files["recommendation_effectiveness"])
    print(eff[["group", "n_recommendations", "n_validated_improvement", "n_no_action_needed", "n_no_safe_action_found",
               "n_inspection_recommended", "n_engine_action_confirmed", "n_engine_action_unsafe"]].to_string(index=False))
    print("\nOutput files:")
    for p in list(files.values()) + figs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
