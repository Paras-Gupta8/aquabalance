"""Phase 5 demo: run the risk engine and recommendation engine on REAL EPANET test windows with the saved GRU forecast.

Usage (from the repository folder; needs Phases 3-4 first):
    python scripts/demo_risk_engine.py

Writes outputs/reports/sample_risk_outputs.json (one example per situation) and risk_engine_summary.csv.
Simulated decision support only: nothing here controls equipment, and no EPANET validation is done yet (Phase 6).
The scenario family printed next to each sample is hidden simulation truth shown for the reader only; the engines never see it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataset_builder import load_split  # noqa: E402
from src.gru_model import GRUPredictor, ModelError  # noqa: E402
from src.preprocessing import DataError, write_json  # noqa: E402
from src.recommendation_engine import RecommendationError, RecommendationSettings, recommend  # noqa: E402
from src.risk_engine import RiskError, RiskSettings, assess_windows  # noqa: E402
from src.utils import ConfigError, load_config, resolve_path  # noqa: E402

SAMPLE_ORDER = [("normal", "Normal"), ("high_pressure_risk", "High pressure"),
                ("low_critical_J4", "Low pressure at J4 (critical)"), ("low_pressure_risk", "Low pressure (other node)"),
                ("pressure_imbalance", "Pressure imbalance"), ("leak_like_pattern", "Leak-like pattern"),
                ("burst_like_pattern", "Burst-like pattern")]


def main() -> int:
    try:
        cfg = load_config()
        P = {k: resolve_path(v) for k, v in cfg["paths"].items()}
        for need in ("test_npz", "feature_names", "gru_checkpoint"):
            if not Path(P[need]).exists():
                raise DataError(f"{P[need]} not found. Run Phases 3-4 first: scripts/build_ml_dataset.py, "
                                f"scripts/train_baselines.py, scripts/train_gru.py")
        feats = json.loads(Path(P["feature_names"]).read_text(encoding="utf-8"))["input_features"]
        nodes = list(cfg["ml"]["target_nodes"])
        test = load_split(P["test_npz"])
        gru = GRUPredictor.from_checkpoint(P["gru_checkpoint"])
        rs, ks = RecommendationSettings.from_config(cfg), RiskSettings.from_config(cfg)
    except (ConfigError, DataError, ModelError, RiskError, RecommendationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    pred = gru.predict(test["X_raw"])                                   # forecast only: the engines see window + forecast
    assessments = assess_windows(test["X_raw"], pred, feats, nodes, ks)
    recs = [recommend(a, rs) for a in assessments]

    def key(a):
        if a.predicted_risk_label == "low_pressure_risk" and a.critical_node_at_risk:
            return "low_critical_J4"
        return a.predicted_risk_label

    samples = {}
    for i in np.argsort(test["meta_scenario_id"], kind="stable"):
        k = key(assessments[i])
        if k in dict(SAMPLE_ORDER) and k not in samples and not assessments[i].uncertain:
            samples[k] = int(i)
    out = {}
    for k, title in SAMPLE_ORDER:
        if k not in samples:
            continue
        i = samples[k]
        out[k] = {"title": title, "window_index": i, "scenario_id_analysis_only": str(test["meta_scenario_id"][i]),
                  "scenario_family_analysis_only": str(test["meta_scenario_type"][i]),
                  "forecast_pressure_m": dict(zip(nodes, [round(float(v), 2) for v in pred[i]])),
                  "risk": assessments[i].to_dict(), "recommendation": recs[i].to_dict()}
    write_json({"note": "Simulated decision-support outputs from real EPANET test windows and the saved GRU forecast. "
                        "scenario_*_analysis_only fields are hidden truth for the reader; the engines never receive them.",
                "samples": out}, P["sample_risk_outputs"])

    for k, title in SAMPLE_ORDER:
        if k not in out:
            print(f"\n=== {title}: no example in the test split ===")
            continue
        o = out[k]
        r, rc = o["risk"], o["recommendation"]
        print(f"\n=== {title}  (EPANET scenario family, analysis only: {o['scenario_family_analysis_only']}) ===")
        print(f"forecast J2/J3/J4 (m): {o['forecast_pressure_m']}")
        print(f"label: {r['predicted_risk_label']}   severity: {r['severity_score']}/100   spread: {r['pressure_spread']:.1f} m   "
              f"low: {r['low_pressure_nodes']}  high: {r['high_pressure_nodes']}  imbalance: {r['pressure_imbalance_flag']}  "
              f"leak-like: {r['possible_leak_pattern_flag']}")
        print(f"explanation: {r['human_readable_explanation']}")
        print(f"recommendation: {rc['recommended_action_type']} -> {rc['target_component']}  "
              f"{rc['current_value']} -> {rc['recommended_value']} ({rc['adjustment_percent']:+g}%)  confidence {rc['confidence']}")
        print(f"  rationale: {rc['rationale']}")
        print(f"  expected effect: {rc['expected_effect']}")
        print(f"  safety: {rc['safety_note']}")

    # --------------------------------------------------------------- how often each outcome happens (truth used for scoring only)
    labels = np.array([a.predicted_risk_label for a in assessments])
    acts = np.array([r.recommended_action_type for r in recs])
    active = test["meta_leak_active"].astype(bool)
    leak = np.array([a.possible_leak_pattern_flag or a.burst_like_flag for a in assessments])
    flow = np.array([a.evidence["flow_inconsistency_flag"] for a in assessments])
    rows = [{"item": f"engine label = {l}", "windows": int((labels == l).sum())} for l in sorted(set(labels))]
    rows += [{"item": f"recommended action = {a}", "windows": int((acts == a).sum())} for a in sorted(set(acts))]
    rows += [{"item": "windows with a leak active at the target step (truth)", "windows": int(active.sum())},
             {"item": "  ...of which leak-like/burst-like flagged", "windows": int(leak[active].sum())},
             {"item": "  ...of which flow/demand inconsistency seen", "windows": int(flow[active].sum())},
             {"item": "windows without a leak (truth) flagged leak-like/burst-like", "windows": int(leak[~active].sum())},
             {"item": "windows flagged uncertain", "windows": int(sum(a.uncertain for a in assessments))},
             {"item": "total test windows", "windows": int(len(assessments))}]
    summary = pd.DataFrame(rows)
    Path(P["risk_engine_summary"]).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(P["risk_engine_summary"], index=False)
    pd.set_option("display.width", 200)
    print("\nOutcome counts over the test windows:")
    print(summary.to_string(index=False))
    print(f"\nWrote {P['sample_risk_outputs']} and {P['risk_engine_summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
