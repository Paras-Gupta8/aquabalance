"""Phase 8: generate the final reports and figures from the saved results (nothing is typed in by hand).

Usage (after Phases 2-6 and train_sequence_models.py):
    python scripts/generate_final_reports.py

Writes:
    outputs/reports/model_comparison.csv            (checked; written by the training scripts)
    outputs/reports/scenario_error_analysis.csv     per scenario family, every model, with the best model per family
    outputs/reports/final_results_summary.md
    outputs/reports/recommendation_effectiveness_report.md
    outputs/reports/limitations.md
    outputs/reports/viva_preparation.md
    outputs/final_figures/                          numbered figures + README.md (index)
Every number in the text is read from the CSV/JSON files produced by the pipeline. All results are EPANET/WNTR synthetic data.
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import report_text as rt  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

MODELS = ["persistence", "random_forest", "gru", "lstm", "tcn"]
NAMES = {"persistence": "Persistence", "random_forest": "Random Forest", "gru": "GRU", "lstm": "LSTM", "tcn": "TCN"}
COLORS = {"persistence": "#6b7a8c", "random_forest": "#2a78d6", "gru": "#eb6834", "lstm": "#1baf7a", "tcn": "#6250d6"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e8ea", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def need(p: Path, hint: str) -> Path:
    if not Path(p).exists():
        print(f"ERROR: {p} not found. {hint}", file=sys.stderr)
        sys.exit(1)
    return Path(p)


# =============================================================================================== figures
def fig_model_comparison(comp, out):
    t = comp[comp["split"] == "test"].set_index("model").loc[[m for m in MODELS if m in set(comp["model"])]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]}, facecolor=SURFACE)
    style(a)
    nodes = ["MAE_J2", "MAE_J3", "MAE_J4"]
    w = 0.16
    for i, m in enumerate(t.index):
        xs = np.arange(3) + (i - (len(t) - 1) / 2) * w
        a.bar(xs, [t.loc[m, n] for n in nodes], width=w * 0.9, color=COLORS[m], label=NAMES[m])
    a.set_xticks(range(3), ["J2 (residential)", "J3 (school)", "J4 (hospital)"])
    a.set_ylabel("Test MAE (m), lower is better", color=MUTED)
    a.set_title("Next-15-minute pressure error per junction (test scenarios)", loc="left", fontsize=11, color=INK)
    a.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(b)
    ys = np.arange(len(t))
    b.scatter(t["risk_macro_f1"], ys, s=90, c=[COLORS[m] for m in t.index], zorder=3)
    for y, (m, r) in zip(ys, t.iterrows()):
        b.text(r["risk_macro_f1"], y + 0.22, f"{r['risk_macro_f1']:.4f}", ha="center", fontsize=8.5, color=INK)
    b.set_yticks(ys, [NAMES[m] for m in t.index])
    b.invert_yaxis()
    b.set_xlabel("Risk macro F1 (test), higher is better", color=MUTED)
    b.set_title("Risk class from the forecast", loc="left", fontsize=11, color=INK)
    b.grid(axis="x", color=GRID)
    b.grid(axis="y", visible=False)
    fig.text(0.01, -0.17, "Synthetic EPANET scenarios, scenario-level test split. Differences between neural networks may be within seed noise (see seed_robustness.csv).",
             fontsize=8.5, color=MUTED)
    return save(fig, out)


def fig_training_curves(histories, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style(ax)
    for m, h in histories.items():
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["val_mae_m"] for r in h], color=COLORS[m], linewidth=2, label=NAMES[m])
        best = max(h, key=lambda r: r["is_best"] * r["epoch"])
        ax.scatter([best["epoch"]], [best["val_mae_m"]], color=COLORS[m], s=36, zorder=3, edgecolor="white")
        ax.text(ep[-1] + 0.5, h[-1]["val_mae_m"], NAMES[m], color=INK, fontsize=9, va="center")
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel("Validation MAE (m)", color=MUTED)
    ax.set_title("Training: validation error per epoch (dot = checkpoint kept)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(right=max(len(h) for h in histories.values()) * 1.15)
    return save(fig, out)


def fig_family_errors(ea, out):
    d = ea[ea["scenario_type"] != "all"].sort_values("best_model_MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(d))
    for m in MODELS:
        if f"{m}_MAE" in d.columns:
            ax.scatter(d[f"{m}_MAE"], ys, s=55, color=COLORS[m], label=NAMES[m], zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_yticks(ys, [f"{r.scenario_type.replace('_', ' ')} (n={int(r.n_scenarios)})" for r in d.itertuples()])
    ax.set_xscale("log")
    ax.set_xlabel("Test MAE (m, log scale), lower is better", color=MUTED)
    ax.set_title("Error by scenario family (family used for analysis only, never as a model input)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    return save(fig, out)


def fig_recommendation_outcomes(eff, out):
    d = eff[eff["group"] != "ALL"].copy()
    d["resolved"], d["partial"] = d["n_improvement_resolved"], d["n_improvement_partial"]
    cols = [("resolved", "Improvement: problem resolved", "#2a78d6"), ("partial", "Improvement: partial", "#9ec5f4"),
            ("n_no_action_needed", "No action needed", "#c9ced2"), ("n_inspection_recommended", "Inspection only", "#6b7a8c"),
            ("n_no_safe_action_found", "No safe action found", "#eb6834")]
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    left = np.zeros(len(d))
    for key, label, color in cols:
        v = d[key].to_numpy(float)
        ax.barh(range(len(d)), v, left=left, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for i, (l, x) in enumerate(zip(left, v)):
            if x > 0:
                ax.text(l + x / 2, i, f"{int(x)}", ha="center", va="center", fontsize=9, color=INK if color != "#2a78d6" and color != "#6b7a8c" else "white")
        left += v
    ax.set_yticks(range(len(d)), [g.replace("_", " ") for g in d["group"]])
    ax.invert_yaxis()
    ax.set_xlabel("Validated recommendations (count)", color=MUTED)
    ax.set_title("EPANET simulated validation: what happened to each recommendation", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.text(0.01, -0.2, rt.DISCLAIMER, fontsize=8.5, color=MUTED)
    return save(fig, out)


# =============================================================================================== main
def main() -> int:
    cfg = load_config()
    P = {k: resolve_path(v) for k, v in cfg["paths"].items()}
    comp = pd.read_csv(need(P["model_comparison"], "Run Phases 3-4 and python scripts/train_sequence_models.py"))
    missing = [m for m in MODELS if m not in set(comp["model"])]
    if missing:
        print(f"ERROR: model_comparison.csv has no rows for {missing}. Run: python scripts/train_sequence_models.py", file=sys.stderr)
        return 1
    by_type = pd.read_csv(need(P["gru_error_by_scenario_type"], "Run python scripts/train_sequence_models.py --score-only"))
    eff = pd.read_csv(need(P["recommendation_effectiveness"], "Run python scripts/validate_recommendation.py"))
    val = pd.read_csv(need(P["validation_results"], "Run python scripts/validate_recommendation.py"))
    jsons = {m: json.loads(need(P[f"{m}_metrics"], "Train the model first").read_text(encoding="utf-8")) for m in ("gru", "lstm", "tcn")}
    manifest = pd.read_csv(need(P["split_manifest"], "Run python scripts/build_ml_dataset.py"))
    meta_json = json.loads(Path(P["feature_names"]).read_text(encoding="utf-8"))
    risk_summary = pd.read_csv(P["risk_engine_summary"]) if Path(P["risk_engine_summary"]).exists() else None
    seed_files = sorted(Path(P["scenario_error_analysis"]).parent.glob("seed_robustness*.csv"))
    seeds = pd.concat([pd.read_csv(f) for f in seed_files], ignore_index=True).drop_duplicates(["model", "seed"]) if seed_files else None
    n_params = {"persistence": 0, "random_forest": None, **{m: int(j["training"]["n_parameters"] if "n_parameters" in j["training"] else j["n_parameters"]) for m, j in jsons.items()}}

    # ---- scenario error analysis (adds the best model per family)
    ea = by_type.copy()
    mae_cols = [f"{m}_MAE" for m in MODELS if f"{m}_MAE" in ea.columns]
    ea["best_model"] = ea[mae_cols].idxmin(axis=1).str.replace("_MAE", "", regex=False)
    ea["best_model_MAE"] = ea[mae_cols].min(axis=1)
    ea["worst_model_MAE"] = ea[mae_cols].max(axis=1)
    Path(P["scenario_error_analysis"]).parent.mkdir(parents=True, exist_ok=True)
    ea.to_csv(P["scenario_error_analysis"], index=False, float_format="%.6g")

    # ---- figures
    F = resolve_path(P["final_figures_dir"])
    F.mkdir(parents=True, exist_ok=True)
    figs = [
        (fig_model_comparison(comp, F / "01_model_comparison_test.png"), "Test error per junction and risk F1 for all five forecasters."),
        (fig_training_curves({m: jsons[m]["history"] for m in ("gru", "lstm", "tcn")}, F / "02_training_curves.png"), "Validation MAE per epoch for the three neural networks."),
        (fig_family_errors(ea, F / "03_error_by_scenario_family.png"), "Test MAE per scenario family for every model."),
        (fig_recommendation_outcomes(eff, F / "04_recommendation_outcomes.png"), "Outcome of each EPANET simulated validation, by engine problem type."),
    ]
    copies = [("network_topology.png", "05_network_topology.png", "The EPANET network: R1, PU1, J1-J4, P1-P4, V1."),
              ("gru_prediction_vs_actual.png", "06_gru_prediction_vs_actual.png", "GRU forecast against the EPANET value (test scenarios)."),
              ("gru_risk_confusion_matrix.png", "07_gru_risk_confusion_matrix.png", "Risk class of the GRU forecast against the EPANET risk class."),
              ("before_after_pressure_comparison.png", "08_before_after_pressure.png", "EPANET simulated validation: pressures before and after."),
              ("before_after_violation_comparison.png", "09_before_after_violations.png", "EPANET simulated validation: violation counts before and after.")]
    for src, dst, caption in copies:
        s = resolve_path("outputs/figures") / src
        if s.exists():
            shutil.copy(s, F / dst)
            figs.append((F / dst, caption))
    (F / "README.md").write_text("# Final figures\n\nAll figures come from EPANET/WNTR synthetic scenarios and the saved model results. "
                                 "Regenerate with `python scripts/generate_final_reports.py`.\n\n"
                                 + "\n".join(f"- `{p.name}`: {c}" for p, c in figs) + f"\n\n{rt.DISCLAIMER}\n", encoding="utf-8")

    # ---- text reports
    ctx = rt.Context(cfg=cfg, comp=comp, ea=ea, eff=eff, val=val, manifest=manifest, meta=meta_json, jsons=jsons, n_params=n_params,
                     seeds=seeds, risk_summary=risk_summary, figures=[p.name for p, _ in figs], today=date.today().isoformat())
    out = {"final_results_summary": rt.final_summary(ctx), "recommendation_effectiveness_report": rt.effectiveness_report(ctx),
           "limitations": rt.limitations(ctx), "viva_preparation": rt.viva(ctx)}
    reports = Path(P["final_results_summary"]).parent
    paths = {"final_results_summary": P["final_results_summary"], "limitations": P["limitations"], "viva_preparation": P["viva_preparation"],
             "recommendation_effectiveness_report": reports / "recommendation_effectiveness_report.md"}
    for k, text in out.items():
        Path(paths[k]).write_text(text, encoding="utf-8")
    print("Written:")
    for p in [P["model_comparison"], P["scenario_error_analysis"], P["recommendation_effectiveness"], *paths.values(), *[p for p, _ in figs]]:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
