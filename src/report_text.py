"""Text of the Phase 8 reports (final_results_summary.md, limitations.md, viva_preparation.md, ...).

Every NUMBER is computed from the result files; the surrounding prose is fixed text. Nothing here claims real-world deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.recommendation_engine import NO_SAFE_ACTION_MESSAGE, VALIDATION_DISCLAIMER, VALIDATION_LABEL

DISCLAIMER = VALIDATION_DISCLAIMER
PROTOTYPE = "This is a simulation-based decision-support prototype, not autonomous real-world utility control."
SYNTH = "All data in this project is synthetic: it comes only from EPANET simulations run through WNTR. No real utility data is used."
NAMES = {"persistence": "Persistence", "random_forest": "Random Forest", "gru": "GRU", "lstm": "LSTM", "tcn": "TCN"}
MODELS = ["persistence", "random_forest", "gru", "lstm", "tcn"]


@dataclass
class Context:
    cfg: dict
    comp: pd.DataFrame
    ea: pd.DataFrame
    eff: pd.DataFrame
    val: pd.DataFrame
    manifest: pd.DataFrame
    meta: dict
    jsons: dict
    n_params: dict
    seeds: pd.DataFrame | None
    risk_summary: pd.DataFrame | None
    figures: list = field(default_factory=list)
    today: str = ""

    # ---- derived facts
    @property
    def test(self) -> pd.DataFrame:
        return self.comp[self.comp["split"] == "test"].set_index("model").loc[MODELS]

    @property
    def val_split(self) -> pd.DataFrame:
        return self.comp[self.comp["split"] == "val"].set_index("model").loc[MODELS]

    def best(self, col="MAE") -> str:
        t = self.test
        return str(t[col].idxmax() if col in ("risk_macro_f1", "R2") else t[col].idxmin())

    def best_nn(self) -> str:
        t = self.test.loc[["gru", "lstm", "tcn"]]
        return str(t["MAE"].idxmin())

    def seed_stats(self) -> pd.DataFrame | None:
        if self.seeds is None or self.seeds.empty:
            return None
        return self.seeds.groupby("model")["test_MAE"].agg(["mean", "std", "count"])

    def all_row(self) -> pd.Series:
        return self.eff[self.eff["group"] == "ALL"].iloc[0]


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("" if (isinstance(v, float) and np.isnan(v)) else floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pct(a, b):
    return 100.0 * (1.0 - a / b)


def model_sentences(c: Context) -> list[str]:
    t, out = c.test, []
    best, best_nn = c.best("MAE"), c.best_nn()
    out.append(f"Lowest test MAE: **{NAMES[best]}** ({t.loc[best, 'MAE']:.4f} m). Lowest test RMSE: **{NAMES[c.best('RMSE')]}** "
               f"({t.loc[c.best('RMSE'), 'RMSE']:.4f} m). Highest risk macro F1: **{NAMES[c.best('risk_macro_f1')]}** "
               f"({t.loc[c.best('risk_macro_f1'), 'risk_macro_f1']:.4f}).")
    for m in ("gru", "lstm", "tcn"):
        out.append(f"{NAMES[m]} test MAE {t.loc[m, 'MAE']:.4f} m: {_pct(t.loc[m, 'MAE'], t.loc['persistence', 'MAE']):.0f}% lower than persistence"
                   + (f", {t.loc[m, 'MAE'] / t.loc['random_forest', 'MAE']:.2f}x the random forest's error." if t.loc[m, "MAE"] > t.loc["random_forest", "MAE"]
                      else f", {_pct(t.loc[m, 'MAE'], t.loc['random_forest', 'MAE']):.0f}% lower than the random forest."))
    if any(t.loc[m, "MAE"] < t.loc["random_forest", "MAE"] for m in ("gru", "lstm", "tcn")):
        out.append("At least one neural network beat the random forest on test MAE.")
    else:
        out.append("**No neural network beat the random forest on test MAE.** The forest uses engineered lag, difference and rolling features, "
                   "which suits this small, smooth, 15-minute forecasting task. The neural networks are included as the project's deep-learning comparison, not as a claim of superiority.")
    ss = c.seed_stats()
    if ss is not None and len(ss) >= 2:
        parts = ", ".join(f"{NAMES[m]} {r['mean']:.4f} ± {0 if np.isnan(r['std']) else r['std']:.4f}" for m, r in ss.iterrows())
        out.append(f"Seed check (test MAE, mean ± standard deviation over {int(ss['count'].max())} extra seeds, `seed_robustness.csv`): {parts}. "
                   "Gaps between two networks smaller than about twice this spread should not be read as a real difference.")
        sd = c.seeds.groupby("model")["test_MAE"]
        for a_, b_ in (("tcn", "gru"), ("tcn", "lstm"), ("gru", "lstm")):
            if a_ in ss.index and b_ in ss.index:
                gap = ss.loc[b_, "mean"] - ss.loc[a_, "mean"]
                noise = 2 * max(np.nan_to_num(ss.loc[a_, "std"]), np.nan_to_num(ss.loc[b_, "std"]))
                if abs(gap) > noise:
                    lo, hi = (a_, b_) if gap > 0 else (b_, a_)
                    out.append(f"{NAMES[lo]} has a lower mean test MAE than {NAMES[hi]} over the seeds ({abs(gap):.4f} m gap, larger than twice the seed spread): a consistent difference in these runs.")
                else:
                    out.append(f"{NAMES[a_]} versus {NAMES[b_]}: the {abs(gap):.4f} m gap is within twice the seed spread, so no difference is claimed.")
    else:
        out.append("Each network was trained with ONE seed in the main run, so small differences between the GRU, LSTM and TCN may be seed noise. "
                   "Run `python scripts/seed_robustness.py` to measure the spread.")
    return out


def data_facts(c: Context) -> dict:
    m = c.manifest
    counts = c.meta.get("counts", {})
    return {"n": len(m), "types": m["scenario_type"].value_counts().to_dict(), "per_split": m["split"].value_counts().to_dict(),
            "windows": {k: v.get("windows") for k, v in counts.items()}, "window_steps": c.meta.get("window_steps"),
            "horizon_min": c.meta.get("horizon_minutes"), "n_features": len(c.meta.get("input_features", []))}


# =============================================================================================== final_results_summary.md
def final_summary(c: Context) -> str:
    d, t = data_facts(c), c.test
    th = c.cfg["thresholds"]
    ta = t.copy()
    ta.insert(0, "Model", [NAMES[m] for m in ta.index])
    ta["Parameters"] = [("n/a (trees)" if c.n_params[m] is None else f"{c.n_params[m]:,}") for m in ta.index]
    tab = ta[["Model", "Parameters", "MAE", "RMSE", "R2", "risk_macro_f1", "inference_ms_per_1000_windows"]].rename(
        columns={"MAE": "Test MAE (m)", "RMSE": "Test RMSE (m)", "R2": "R²", "risk_macro_f1": "Risk macro F1",
                 "inference_ms_per_1000_windows": "Inference (ms / 1000 windows)"})
    node = ta[["Model", "MAE_J2", "MAE_J3", "MAE_J4"]].rename(columns={"MAE_J2": "J2 MAE (m)", "MAE_J3": "J3 MAE (m)", "MAE_J4": "J4 MAE (m)"})
    vs = c.val_split
    valtab = pd.DataFrame({"Model": [NAMES[m] for m in MODELS], "Validation MAE (m)": vs["MAE"].to_numpy(), "Test MAE (m)": t["MAE"].to_numpy()})
    fam = c.ea[c.ea["scenario_type"] != "all"].copy()
    famtab = pd.DataFrame({"Scenario family": fam["scenario_type"].str.replace("_", " "), "Test scenarios": fam["n_scenarios"].astype(int),
                           **{NAMES[m] + " MAE": fam[f"{m}_MAE"] for m in MODELS if f"{m}_MAE" in fam.columns},
                           "Best model": fam["best_model"].map(NAMES)})
    wins = fam["best_model"].value_counts().to_dict()
    a = c.all_row()
    lines = [
        "# AquaBalance: final results summary", "",
        f"_Generated {c.today} by `scripts/generate_final_reports.py` from the saved pipeline outputs. Re-run it after retraining; numbers below come from the files, not from typing._", "",
        f"> {SYNTH}", f"> {PROTOTYPE}", f"> {VALIDATION_LABEL}: {DISCLAIMER}", "",
        "## 1. What was built", "",
        "EPANET network → WNTR scenario generation → ML dataset → baselines → GRU (+ LSTM, TCN comparison) → transparent risk engine → bounded recommendation engine → EPANET simulated validation → Streamlit dashboard.", "",
        "## 2. Data", "",
        f"- {d['n']} synthetic scenarios (24 h at 15-minute steps) simulated in EPANET through WNTR: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(d["types"].items())) + ".",
        f"- Split **by whole scenario** (never by row): " + ", ".join(f"{k} {v} scenarios" for k, v in d["per_split"].items()) + "; windows " + ", ".join(f"{k} {v:,}" for k, v in d["windows"].items()) + ".",
        f"- Task: last {d['window_steps']} steps (3 h) of {d['n_features']} sensor/control features → pressure at J2, J3, J4 in {d['horizon_min']:.0f} minutes. Scenario type, leak flags and leak size are never model inputs.",
        f"- Lab-prototype thresholds (not utility standards): pressure {th['p_min_m']:.0f}–{th['p_max_m']:.0f} m, J4 (critical) minimum {th['critical_p_min_m']:.0f} m, imbalance when the spread of J2–J4 exceeds {th['max_pressure_spread_m']:.0f} m.", "",
        "## 3. Forecasting results (test split)", "", md_table(tab), "", md_table(node), "",
        "Validation versus test MAE:", "", md_table(valtab), "", "**Reading the table honestly**", "",
        *[f"- {s}" for s in model_sentences(c)], "",
        "- Persistence (\"next pressure = current pressure\") is already strong because pressure changes slowly in 15 minutes; a model must beat it to be worth using.",
        f"- All five models use the same {d['n_features']} input features, the same scenario-level split and the same scoring code. GRU, LSTM and TCN have similar parameter counts and identical training settings.", "",
        "## 4. Error by scenario family (test split)", "", md_table(famtab), "",
        "Best model per family: " + ", ".join(f"{NAMES.get(k, k)} {v}" for k, v in wins.items()) + ". The family label is used for analysis only.",
        "Families with only " + ", ".join(sorted({str(int(x)) for x in fam[fam["n_scenarios"] <= 2]["n_scenarios"]}) or ["few"]) + " test scenarios ("
        + ", ".join(fam[fam["n_scenarios"] <= 2]["scenario_type"].str.replace("_", " ")) + ") are too few to rank models reliably.", "",
        "## 5. Recommendation effectiveness (" + VALIDATION_LABEL + ")", "",
        f"{int(a['n_recommendations'])} recommendations were re-simulated in EPANET (a sample of high-severity test windows per situation, not every window): "
        f"{int(a['n_validated_improvement'])} validated improvements ({int(a['n_improvement_resolved'])} fully resolved, {int(a['n_improvement_partial'])} partial), "
        f"{int(a['n_no_action_needed'])} where EPANET showed no violation, {int(a['n_no_safe_action_found'])} with \"{NO_SAFE_ACTION_MESSAGE}\", "
        f"{int(a['n_inspection_recommended'])} inspection-only (leak-like / burst-like).", "",
        "**Not every recommendation improved every scenario.** See `recommendation_effectiveness_report.md` for each case.", "",
        f"{DISCLAIMER}", "",
        "## 6. Figures", "", *[f"- `outputs/final_figures/{f}`" for f in c.figures], "",
        "## 7. Limitations", "", "See `limitations.md`. In short: synthetic data from a 4-junction teaching network, one-step-ahead forecasts, validation in the same simulator that made the data, no real sensors or actuators.", "",
        "## 8. Reproduce", "", "```bash", "python scripts/build_network.py", "python scripts/generate_synthetic_data.py --mode standard",
        "python scripts/build_ml_dataset.py", "python scripts/train_baselines.py", "python scripts/train_gru.py", "python scripts/train_sequence_models.py",
        "python scripts/validate_recommendation.py", "python scripts/generate_final_reports.py", "python -m pytest tests -q", "streamlit run app/app.py", "```", ""]
    return "\n".join(lines)


# =============================================================================================== effectiveness report
def effectiveness_report(c: Context) -> str:
    e = c.eff.copy()
    cols = ["group", "n_recommendations", "n_validated_improvement", "n_improvement_resolved", "n_improvement_partial", "n_no_action_needed",
            "n_no_safe_action_found", "n_inspection_recommended", "mean_low_violations_before", "mean_low_violations_after",
            "mean_high_violations_before", "mean_high_violations_after", "mean_spread_before_m", "mean_spread_after_m"]
    e = e[[x for x in cols if x in e.columns]]
    v = c.val
    case = v[["scenario_id", "decision_time", "engine_problem", "chosen_action", "final_validation_result", "improvement_level",
              "before_low_pressure_violation_count", "after_low_pressure_violation_count", "before_high_pressure_violation_count",
              "after_high_pressure_violation_count", "before_branch_pressure_spread_mean", "after_branch_pressure_spread_mean",
              "before_J4_pressure_min", "after_J4_pressure_min", "before_loss_proxy_lps"]].copy()
    case.columns = ["Scenario", "Decision time", "Engine problem", "Chosen action", "Result", "Level", "Low viol. before", "after", "High viol. before", "after ",
                    "Spread before (m)", "after  ", "J4 min before (m)", "after   ", "Loss proxy (L/s)"]
    n_all = len(v)
    n_improved = int((v["final_validation_result"] == "validated_improvement").sum())
    worse = v[v["final_validation_result"].isin(["no_safe_action_found"])]
    partial = v[(v["final_validation_result"] == "validated_improvement") & (v["improvement_level"] != "resolved")]
    return "\n".join([
        "# Recommendation effectiveness report", "", f"> {VALIDATION_LABEL}. {DISCLAIMER}", f"> {SYNTH}", "",
        f"{n_all} recommendations (one decision window each, chosen as the highest-severity window of different test scenarios per situation) were each re-simulated in EPANET "
        "before and after every bounded candidate action (valve −15…+15 %, pump −10…+10 %). The lowest-change safe action with the best objective is kept; "
        f"if none is safe the result is \"{NO_SAFE_ACTION_MESSAGE}\"", "",
        "## Summary by situation", "", md_table(e, "{:.2f}"), "",
        "## Case by case", "", md_table(case, "{:.2f}"), "",
        "## Honest reading", "",
        f"- {n_improved} of {n_all} cases ended in a validated improvement; {len(partial)} of those only partly cleared the problem (e.g. high source head: pressure falls but violations remain).",
        f"- {len(worse)} case(s) ended with no safe action; the engine's first-order estimate can be wrong where the real cause is a blockage.",
        "- Leak-like and burst-like patterns are inspection-only: no pump/valve change is tried because that could hide a leak. The loss proxy (inflow minus delivered demand) is the leak outflow in the simulation.",
        "- A forecast false alarm shows up as \"no violation in EPANET\": the model predicted a problem that did not occur.",
        "- Improvement is measured over the next hour in the same simulator that generated the data; it says nothing about a real network.", ""])


# =============================================================================================== limitations.md
def limitations(c: Context) -> str:
    th = c.cfg["thresholds"]
    return "\n".join([
        "# Limitations", "", f"> {PROTOTYPE}", "",
        "1. **Synthetic data only.** " + SYNTH + " Real networks have noise, missing data, sensor drift, unknown demands and unmodelled equipment.",
        "2. **Tiny network.** Four junctions, four pipes, one pump, one PRV, no tank. Behaviour, forecast difficulty and recommendation options are far simpler than a real district.",
        "3. **Idealised scenarios.** Faults (leaks, blockage, valve misconfiguration, pump wear, high source head) are drawn from fixed families and ranges. Real faults are messier and can overlap.",
        "4. **Optimistic validation.** The EPANET simulated validation re-runs the same simulator that produced the training data, so agreement is expected. " + DISCLAIMER,
        f"5. **Lab-prototype thresholds.** Pressure {th['p_min_m']:.0f}–{th['p_max_m']:.0f} m, J4 minimum {th['critical_p_min_m']:.0f} m and spread {th['max_pressure_spread_m']:.0f} m are configurable teaching values, not utility standards.",
        "6. **Short horizon.** Forecasts are one step (15 minutes) ahead from a 3-hour window; no multi-hour planning, no demand forecasting.",
        "7. **Simple models win on this task.** Persistence is already strong and the random forest with engineered features beats the neural networks on test error; the networks' advantage on richer, noisier data is not shown here. Each network was trained with one main seed (see the seed check for the spread).",
        "8. **Rule-based risk and recommendations.** Transparent by design, but the first-order what-if estimate can be wrong (for example for a blocked pipe); the EPANET re-run is what is trusted.",
        "9. **Leak handling is minimal.** Leak-like patterns are flagged by a mass-balance check and always lead to inspection; leak localisation is only a suspicion.",
        "10. **No hardware.** No SCADA, no physical sensors, no actuator interface, no latency, safety interlocks or operator workflow. Recommendations are suggestions in a simulation.",
        "11. **Single critical node.** Only J4 has the stricter rule; real systems have many priority customers and conflicting constraints.",
        "12. **Reservoir head choice.** The base reservoir head is 62 m (the Phase 1 value of 110 m left every consumer far above the pressure limit); 110 m is reached only in high-source-pressure scenarios.", ""])


# =============================================================================================== viva_preparation.md
def viva(c: Context) -> str:
    t, d, th = c.test, data_facts(c), c.cfg["thresholds"]
    best, best_nn = c.best("MAE"), c.best_nn()
    a = c.all_row()
    rf, gru = t.loc["random_forest", "MAE"], t.loc["gru", "MAE"]
    pers = t.loc["persistence", "MAE"]
    ss = c.seed_stats()
    seed_line = ("Over extra seeds the spread (std) of test MAE was " + ", ".join(f"{NAMES[m]} {0 if np.isnan(r['std']) else r['std']:.4f}" for m, r in ss.iterrows()) + " m, so only gaps clearly larger than that mean something."
                 if ss is not None else "I trained each network with one main seed, so small gaps between the networks are not conclusive.")
    return "\n".join([
        "# AquaBalance: viva preparation", "",
        f"_Generated {c.today} from the project's result files. Numbers match `final_results_summary.md`._", "",
        f"> {PROTOTYPE} {SYNTH}", "",
        "## 30-second explanation", "",
        "AquaBalance is a simulation-based decision-support prototype for a small water-distribution network. I use EPANET, through the WNTR Python package, to simulate "
        f"{d['n']} scenarios: normal days, high demand, leaks, a blocked pipe, a mis-set valve. A GRU neural network forecasts the pressure at three junctions 15 minutes ahead. "
        "A transparent rule engine turns the forecast into a risk label, a bounded recommender suggests a small pump or valve change, and EPANET re-simulates that change to check it is safe. "
        "Everything is synthetic, so it is a prototype, not a control system.", "",
        "## 60-second explanation", "",
        "Water networks need pressure inside a safe band: too low and customers, especially a hospital, lose supply; too high and pipes leak or burst. Real operators react after the problem appears. "
        f"AquaBalance tries to warn earlier. Step one: EPANET/WNTR generates {d['n']} labelled-by-physics scenarios, so I have ground truth without real data. Step two: I split them by whole scenario, never by row, "
        "so the test set is truly unseen, and train a GRU on the last 3 hours of sensor and pump/valve values to predict J2, J3 and J4 pressure 15 minutes ahead. I compare it fairly with persistence, a random forest, an LSTM and a small TCN. "
        f"Step three: a rule-based risk engine classifies the forecast (low, high, imbalance, leak-like) and treats the hospital node J4 as most important. Step four: a recommendation engine proposes a bounded pump or valve change "
        "and, for leaks, only an inspection. Step five: every recommendation is re-run in EPANET before and after, and rejected if any node gets worse or J4 is sacrificed. "
        f"Result: on test scenarios the best forecaster was {NAMES[best]} (MAE {t.loc[best, 'MAE']:.3f} m); the GRU reached {gru:.3f} m, better than persistence ({pers:.3f} m) but not better than the random forest ({rf:.3f} m). "
        f"Of {int(a['n_recommendations'])} validated recommendations, {int(a['n_validated_improvement'])} improved the simulated network, and the rest were no-action, inspection or \"no safe action\". "
        "The limits are clear: synthetic data, tiny network, validation inside the same simulator.", "",
        "## Ten likely questions and concise answers", "",
        "**1. Why did you use EPANET (and WNTR)?**  ", "I have no real utility data, and real faults cannot be created on demand. EPANET is the standard, physics-based hydraulic solver, so its pressures and flows follow real hydraulics. "
        "WNTR lets me script thousands of reproducible scenarios, replay any one exactly, and re-run a recommended action to validate it. It gives labelled ground truth and a safe testbed. The price is that it is a simulation.", "",
        "**2. Why did you use a GRU?**  ", "The task is sequence-to-value: the last 3 hours of pressure, flow, demand and pump/valve values predict the next pressure. A GRU is a compact recurrent network that remembers recent trends, with about the same number of parameters as the LSTM I compared it with "
        f"({c.n_params['gru']:,} vs {c.n_params['lstm']:,}) and runs fast. It is a justified starting point for time-series forecasting. But I tested it honestly: it beat persistence yet not the random forest on this small task "
        f"(GRU {gru:.3f} m vs forest {rf:.3f} m), and the TCN/LSTM comparison is in the results. The deep-learning value here is a comparison and a path to richer data, not a proven win.", "",
        "**3. What is the difference between sensors and actuators in your project?**  ", "Sensors measure: junction pressures, pipe flows, demand, and the reported pump speed and valve setting. They feed the model. "
        "Actuators change the network: the pump (speed) and the pressure-reducing valve (setting). In AquaBalance the model only reads sensor-type values, and the recommendation engine only SUGGESTS an actuator change inside the simulation. "
        "Nothing is connected to real equipment, and an operator would approve any change.", "",
        "**4. What is the limitation of simulation data?**  ", "It is clean, small and generated by my own assumptions. There is no sensor noise by default, no missing data, no unknown demand or valve states, and the fault families are idealised. "
        "Validation uses the same simulator that made the data, so it is optimistic. Results show the method works in simulation; they do not show it works on a real utility.", "",
        "**5. Pressure imbalance versus equal pressure: what is the difference?**  ", "I do not try to make all junctions equal. Elevation and demand differ, so natural pressures differ. "
        f"A pressure imbalance means the spread between the highest and lowest of J2, J3, J4 exceeds {th['max_pressure_spread_m']:.0f} m, even if each node is inside the {th['p_min_m']:.0f}-{th['p_max_m']:.0f} m band. "
        f"Safe operation means every node inside its band (J4, the hospital, at least {th['critical_p_min_m']:.0f} m) and the spread under the limit. Equal pressure is not the goal.", "",
        "**6. How did you prevent data leakage?**  ", "I split by whole scenario ID, so no window of a test scenario is in training. Windows are cut inside one scenario. Scalers are fitted on training scenarios only. "
        "Hidden truth (scenario type, leak flag, leak size) is never a model input, and a test checks it. The EPANET validation reads only the sensor-style history.", "",
        "**7. Did the GRU beat your baselines?**  ", f"Partly. Test MAE: persistence {pers:.3f} m, GRU {gru:.3f} m, random forest {rf:.3f} m, LSTM {t.loc['lstm', 'MAE']:.3f} m, TCN {t.loc['tcn', 'MAE']:.3f} m. "
        f"The GRU beat persistence but the forest was best on error. {seed_line} I report this as it is rather than claim superiority.", "",
        "**8. How does the recommendation engine stay safe?**  ", "Changes are bounded (pump about ±10 %, valve about ±15 %) and small. J4 is protected first: no action may push it toward its minimum. Leak-like and burst-like patterns only give an inspection "
        f"recommendation. Every suggestion is re-run in EPANET; if no candidate is safe the answer is \"{NO_SAFE_ACTION_MESSAGE}\".", "",
        "**9. What does your EPANET validation prove, and what does it not?**  ", f"It proves that in the simulated network the chosen action improved (or did not worsen) the next hour's pressures under my safety rules. "
        f"{DISCLAIMER} It does not prove field performance, because the real system has noise, delays, unknown states and other equipment.", "",
        "**10. What would you need to deploy this for real?**  ", "Real SCADA data and calibrated hydraulic models, physical pressure and flow sensors with noise handling, an actuator interface with safety interlocks, much longer testing, "
        "and an operator-in-the-loop workflow where a human approves every change. Future research: graph-temporal neural networks that use the pipe topology, reinforcement learning for control inside the validated simulator, and validation on real utility data.", "",
        "## Short answers to follow-up questions", "",
        "- **How is a leak detected?** Mass balance: water entering the network minus the metered demand should be near zero; a persistent positive gap with a pressure drop means leak-like. It is flagged for inspection only.",
        "- **Why is J4 special?** It represents a hospital / critical service, so it has a stricter minimum and is never sacrificed to improve another node.",
        "- **Why rule-based risk rather than a neural classifier?** Operators must be able to see why an alert was raised; thresholds are explicit and configurable.",
        f"- **Why is a random forest good here?** Pressure changes slowly, so recent values and differences carry most of the signal; trees on engineered lag/rolling features capture it with little data.", "",
        "## Numbers to remember (test split)", "",
        md_table(pd.DataFrame({"Model": [NAMES[m] for m in MODELS], "Test MAE (m)": t["MAE"].to_numpy(), "Test RMSE (m)": t["RMSE"].to_numpy(), "Risk macro F1": t["risk_macro_f1"].to_numpy()})), "",
        f"Scenarios: {d['n']} (train/val/test split by scenario: " + ", ".join(f"{k} {v}" for k, v in d['per_split'].items()) + "). "
        f"Validated recommendations: {int(a['n_recommendations'])}; improved {int(a['n_validated_improvement'])}; no action needed {int(a['n_no_action_needed'])}; no safe action {int(a['n_no_safe_action_found'])}; inspection {int(a['n_inspection_recommended'])}.", ""])
