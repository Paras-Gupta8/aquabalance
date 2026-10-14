"""AquaBalance dashboard (Phase 7). Run from the repository folder:  streamlit run app/app.py

A simulation-based decision-support prototype. Every number shown comes from EPANET scenarios (through WNTR) or from
a model forecast of them. It is not autonomous real-world utility control and uses no real utility data.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import charts  # noqa: E402
from src import dashboard_data as dd  # noqa: E402
from src.epanet_validator import ValidationError, ValidationSettings, load_spec, validate_recommendation  # noqa: E402
from src.recommendation_engine import (NO_SAFE_ACTION_MESSAGE, VALIDATION_DISCLAIMER, VALIDATION_LABEL,  # noqa: E402
                                       RecommendationSettings, recommend)
from src.utils import load_config, resolve_path  # noqa: E402

st.set_page_config(page_title="AquaBalance", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.stApp { background: #f5f9fa; }
h1, h2, h3 { color: #0f3d4c; letter-spacing: -0.01em; }
.ab-sub { color: #1f6f7a; font-size: 1.05rem; margin-top: -0.6rem; }
.ab-banner { background: #e3f1f4; border-left: 4px solid #1f8a9a; padding: 0.55rem 0.9rem; border-radius: 4px; color: #0f3d4c; font-size: 0.92rem; margin: 0.6rem 0 1rem 0; }
.ab-card { background: #ffffff; border: 1px solid #d8e5e8; border-radius: 8px; padding: 0.7rem 0.9rem; min-height: 7.6rem; margin-bottom: 0.9rem; }
.ab-card .k { color: #52514e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.ab-card .v { color: #0b0b0b; font-size: 1.25rem; font-weight: 600; line-height: 1.25; margin-top: 0.15rem; }
.ab-card .s { color: #52514e; font-size: 0.82rem; margin-top: 0.15rem; }
.ab-tag { display: inline-block; background: #0f3d4c; color: #fff; padding: 0.15rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
.ab-outcome { border-radius: 6px; padding: 0.7rem 1rem; font-weight: 600; margin: 0.4rem 0 0.8rem 0; }
.ab-ok { background: #e4f4e6; border-left: 4px solid #0ca30c; color: #0b3d12; }
.ab-part { background: #fff4d6; border-left: 4px solid #fab219; color: #4a3500; }
.ab-bad { background: #fbe6e6; border-left: 4px solid #d03b3b; color: #5a1414; }
.ab-neutral { background: #e8eef0; border-left: 4px solid #6b7a8c; color: #1d2b33; }
.ab-note { color: #52514e; font-size: 0.85rem; }
.ab-legend span { margin-right: 1.1rem; font-size: 0.85rem; color: #52514e; }
.ab-dot { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 50%; margin-right: 0.3rem; vertical-align: -0.1rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================================== cached resources
@st.cache_resource(show_spinner="Loading data and models...")
def get_context():
    cfg = load_config()
    src = dd.load_source(cfg, force_demo=os.environ.get("AQUABALANCE_FORCE_DEMO") == "1")   # =1: use data/sample/ even if the full data exists
    return cfg, src, dd.ModelHub(cfg, src)


@st.cache_data(show_spinner=False)
def cached_notable(sid: str, model: str, mode: str) -> int:
    cfg, src, hub = get_context()
    return dd.notable_step(cfg, src, hub, sid, model)


def card(label: str, value: str, sub: str = "") -> str:
    return f'<div class="ab-card"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def clock(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")


def risk_text(label: str) -> str:
    return dd.RISK_TEXT.get(label, label)


def status_word(kind: str) -> str:
    color = {"safe": "#0a7d0a", "warning": "#b87800", "risk": "#c02f2f"}[kind]
    return f'<span style="color:{color}">' + {"safe": "● OK", "warning": "▲ WARNING", "risk": "■ HIGH RISK"}[kind] + "</span>"


# =============================================================================================== header
st.title("AquaBalance")
st.markdown('<div class="ab-sub">Deep Learning–Assisted Pressure Risk Prediction for Water Distribution Networks</div>', unsafe_allow_html=True)

try:
    cfg, src, hub = get_context()
except dd.DashboardError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:                                                      # noqa: BLE001
    st.error(f"Could not start the dashboard: {exc}")
    st.stop()

banner = dd.SYNTHETIC_BANNER
if src.mode == "demo":
    banner = f"<b>{dd.DEMO_LABEL}</b> {dd.SYNTHETIC_BANNER}"
    banner += "<br>Demo mode: using the small saved EPANET example in <code>data/sample/</code> (trained artifacts or the full dataset were not found)."
st.markdown(f'<div class="ab-banner">{banner}<br>{dd.PROTOTYPE_STATEMENT}</div>', unsafe_allow_html=True)

# =============================================================================================== sidebar
with st.sidebar:
    st.header("Controls")
    types = ["All"] + dd.scenario_types(src)
    stype = st.selectbox("Scenario-type filter", types, help="Scenario families are known because this is a simulation; the models never see them.")
    ids = dd.scenario_ids(src, stype)
    sid = st.selectbox("Scenario", ids, format_func=lambda i: f"{i}  ·  {dd.scenario_type_of(src, i).replace('_', ' ')}")
    usable = [m for m in hub.available() if hub.can_forecast(m, sid)]
    model = st.selectbox("Forecast model", usable, format_func=lambda m: dd.MODEL_NAMES[m])
    for m, why in hub.unavailable().items():
        st.caption(f"{dd.MODEL_NAMES[m]}: unavailable ({why})")
    if hub.info.get("gru", {}).get("mode") == "unavailable":
        pass
    df = dd.scenario_series(src, sid)
    steps = dd.decision_steps(cfg, len(df))
    if st.session_state.get("_sid_loaded") != sid:
        st.session_state["step"] = cached_notable(sid, model, src.mode)
        st.session_state["_sid_loaded"] = sid
    if st.session_state.get("step") not in steps:
        st.session_state["step"] = steps[len(steps) // 2]
    step = st.select_slider("Time step ('now')", options=steps, key="step",
                            format_func=lambda s: f"{clock(df['timestamp'].iloc[s])}  (step {s})")
    st.button("Jump to highest-risk step", on_click=lambda: st.session_state.update(step=cached_notable(sid, model, src.mode)),
              help="Finds the step with the highest predicted-risk severity for the selected model.")
    show_forecast = st.toggle("Show next-step forecast", value=True)
    run_validation = st.button("Run EPANET Simulated Validation", type="primary", width="stretch")
    with st.expander("Help / Methodology"):
        st.markdown(
            "- **Now** is the selected time step; the forecast is the pressure 15 minutes later.\n"
            "- Models see only the last 3 hours of sensor-type values (pressures, flows, demands, pump/valve settings).\n"
            "- Risk and recommendations come from transparent rules, not from the neural network.\n"
            "- *Run EPANET Simulated Validation* re-simulates the next hour in EPANET with and without the action.\n\n"
            "More detail is in section 8 below. " + dd.PROTOTYPE_STATEMENT)

# =============================================================================================== compute the view
try:
    view = dd.build_step_view(cfg, src, hub, sid, step, model)
except (dd.DashboardError, Exception) as exc:                                  # noqa: BLE001
    st.error(f"Could not compute the forecast: {exc}")
    st.stop()
a, cur = view.assessment, view.current_assessment
rs = RecommendationSettings.from_config(cfg)
rec = recommend(a, rs)
now_row = df.iloc[step]
j4_kind, j4_why = view.node_status["J4"]

# =============================================================================================== 1. overview
st.header("1. System overview")
st.caption(f"Scenario {sid} · time {clock(view.now_ts)} (step {step}) · forecast for {clock(view.next_ts)} · model: {dd.MODEL_NAMES[model]}"
           + (" · stored forecast from the trained model" if hub.info[model]["mode"] == "stored" else ""))
c = st.columns(6)
c[0].markdown(card("Scenario", sid, dd.scenario_type_of(src, sid).replace("_", " ") + " (simulation family)"), unsafe_allow_html=True)
c[1].markdown(card("Current risk", risk_text(cur.predicted_risk_label), f"severity {cur.severity_score:.0f}/100"), unsafe_allow_html=True)
c[2].markdown(card("Predicted next-step risk", risk_text(a.predicted_risk_label) + (" (uncertain)" if a.uncertain else ""),
                   f"{dd.MODEL_NAMES[model]} · severity {a.severity_score:.0f}/100"), unsafe_allow_html=True)
c[3].markdown(card("J4 critical node", status_word(j4_kind), f"now {view.current_pressure['J4']:.1f} m → {view.forecast['J4']:.1f} m (min {cfg['thresholds']['critical_p_min_m']:.0f} m)"), unsafe_allow_html=True)
c[4].markdown(card("Pressure spread (J2–J4)", f"{a.pressure_spread:.1f} m", f"predicted · limit {cfg['thresholds']['max_pressure_spread_m']:.0f} m"
                   + (" · imbalance" if a.pressure_imbalance_flag else "")), unsafe_allow_html=True)
rec_short = rec.recommended_action_type.replace("_", " ")
c[5].markdown(card("Recommended action", rec_short.capitalize(),
                   (f"{rec.target_component} {rec.adjustment_percent:+.0f}%" if rec.adjustment_percent else rec.target_component)), unsafe_allow_html=True)

# =============================================================================================== 2. network diagram
st.header("2. Network diagram")
st.plotly_chart(charts.network_figure(cfg, view, now_row, show_forecast), width="stretch", config={"displayModeBar": False})
st.markdown('<div class="ab-legend"><span><i class="ab-dot" style="background:#0ca30c"></i>OK (safe)</span>'
            '<span><i class="ab-dot" style="background:#fab219"></i>WARNING</span>'
            '<span><i class="ab-dot" style="background:#d03b3b"></i>HIGH RISK</span>'
            '<span><i class="ab-dot" style="background:#6b7a8c"></i>hub J1 (shown, not scored)</span></div>', unsafe_allow_html=True)
bad = [f"{n}: {w}" for n, (k, w) in view.node_status.items() if k != "safe"]
if bad:
    st.caption("Why: " + " · ".join(bad))
st.caption("Label format: current pressure → forecast pressure. Hover a node for details. Colour reflects the worse of the current and forecast state.")

# =============================================================================================== 3. trends
st.header("3. Pressure and flow trends")
st.plotly_chart(charts.pressure_figure(cfg, df, view, show_forecast), width="stretch")
cf, cd, ct = st.columns([3, 3, 2])
cf.plotly_chart(charts.flow_figure(df, view), width="stretch")
cd.plotly_chart(charts.demand_figure(df, view), width="stretch")
with ct:
    st.markdown("**Tank level**")
    if df["tank_level"].notna().any():
        st.line_chart(df.iloc[:step + 1].set_index("timestamp")["tank_level"], height=250)
    else:
        st.info("This simulated network has no tank, so no tank level exists. Nothing is shown rather than an invented value.")

# =============================================================================================== 4. AI prediction
st.header("4. AI prediction and risk analysis")
left, right = st.columns([3, 2])
with left:
    tab = pd.DataFrame({"Junction": ["J2", "J3", "J4"],
                        "Now (m)": [view.current_pressure[n] for n in ("J2", "J3", "J4")],
                        "Forecast +15 min (m)": [view.forecast[n] for n in ("J2", "J3", "J4")],
                        "EPANET next step (m)": [view.actual_next[n] for n in ("J2", "J3", "J4")]})
    tab["Forecast error (m)"] = tab["Forecast +15 min (m)"] - tab["EPANET next step (m)"]
    st.dataframe(tab.round(2), hide_index=True, width="stretch")
    st.caption("The forecast is from the selected model. 'EPANET next step' is the true simulated value, shown only to judge the forecast. The models never see it.")
    st.markdown("**Explanation**")
    st.write(a.human_readable_explanation)
    if a.uncertain:
        st.warning("The engine is uncertain about this window: " + "; ".join(a.uncertainty_reasons))
with right:
    st.markdown(f"**Selected model:** {dd.MODEL_NAMES[model]} ({hub.info[model]['note']})")
    st.markdown(f"**High-pressure nodes:** {', '.join(a.high_pressure_nodes) or 'none'}")
    st.markdown(f"**Low-pressure nodes:** {', '.join(a.low_pressure_nodes) or 'none'}")
    st.markdown(f"**Pressure imbalance:** {'YES' if a.pressure_imbalance_flag else 'no'} "
                f"(spread {a.pressure_spread:.1f} m, limit {cfg['thresholds']['max_pressure_spread_m']:.0f} m)")
    st.markdown(f"**Leak-like pattern:** {'YES' if a.possible_leak_pattern_flag else 'no'}" + (" (burst-like)" if a.burst_like_flag else ""))
    comp = dd.comparison_table(src.comparison_path)
    st.markdown("**Model metrics** (synthetic test split)")
    if comp is not None and model in set(comp["model"]):
        r = comp[comp["model"] == model].iloc[0]
        st.markdown(f"- test MAE {r['test MAE (m)']:.3f} m · test RMSE {r['test RMSE (m)']:.3f} m\n- risk macro F1 {r['risk macro F1 (test)']:.3f}")
    else:
        st.caption("No saved metrics found for this model.")

# =============================================================================================== 5. recommended action
st.header("5. Recommended action")
r1, r2, r3, r4, r5 = st.columns(5)
unit = f" {rec.unit}" if rec.unit else ""
fmt = lambda x: "—" if x is None else f"{x:.2f}{unit}"                                   # noqa: E731
r1.markdown(card("Component", rec.target_component, rec.recommended_action_type.replace("_", " ")), unsafe_allow_html=True)
r2.markdown(card("Current setting", fmt(rec.current_value)), unsafe_allow_html=True)
r3.markdown(card("Recommended setting", fmt(rec.recommended_value)), unsafe_allow_html=True)
r4.markdown(card("Adjustment", f"{rec.adjustment_percent:+.0f}%"), unsafe_allow_html=True)
r5.markdown(card("Confidence", f"{rec.confidence:.0%}", "rule-based, not a probability"), unsafe_allow_html=True)
st.markdown(f"**Rationale:** {rec.rationale}")
st.markdown(f"**Expected effect (first-order estimate, not verified):** {rec.expected_effect}")
st.markdown(f"**Safety note:** {rec.safety_note}")
st.caption("This is a suggestion for a human operator inside a simulation. Nothing is sent to any equipment.")

# =============================================================================================== 6. validation
st.header("6. Before vs after: EPANET simulated validation")
key = (sid, step, model)
if run_validation:
    with st.spinner("Re-running the scenario in EPANET for every candidate action..."):
        try:
            spec = load_spec(src.specs_path, sid)
            st.session_state["validation"] = {"key": key, "v": validate_recommendation(
                spec, rec, step, cfg, resolve_path(cfg["paths"]["base_inp"]), rs, ValidationSettings.from_config(cfg))}
        except (ValidationError, Exception) as exc:                                # noqa: BLE001
            st.session_state["validation"] = {"key": key, "error": str(exc)}
res = st.session_state.get("validation")
if not res or res["key"] != key:
    st.markdown(f'<span class="ab-tag">{VALIDATION_LABEL}</span>', unsafe_allow_html=True)
    st.info("Click **Run EPANET Simulated Validation** in the sidebar. EPANET will re-simulate the next hour of this scenario "
            "before and after the recommended (and a few alternative) pump/valve changes.")
elif "error" in res:
    st.error(f"Validation could not run: {res['error']}")
else:
    v = res["v"]
    b, af = v["before"], v["after"]
    st.markdown(f'<span class="ab-tag">{VALIDATION_LABEL}</span>', unsafe_allow_html=True)
    st.caption(f"{VALIDATION_DISCLAIMER}  Window: the {v['window_steps']} steps (15 min each) after {clock(view.now_ts)}.")
    result, level = v["final_validation_result"], v.get("improvement_level")
    if result == "validated_improvement" and level == "resolved":
        outcome, cls = "Simulated improvement observed.", "ab-ok"
    elif result == "validated_improvement":
        outcome, cls = "Recommendation did not improve all constraints.", "ab-part"
    elif result == "no_safe_action_found":
        outcome, cls = NO_SAFE_ACTION_MESSAGE, "ab-bad"
    elif result == "inspection_recommended":
        outcome, cls = "Inspection recommended: no pump or valve change is validated for a leak-like or burst-like pattern.", "ab-neutral"
    else:
        outcome, cls = "No violation found in EPANET for this window; no control change is needed.", "ab-neutral"
    st.markdown(f'<div class="ab-outcome {cls}">{outcome}</div>', unsafe_allow_html=True)
    st.write(v["final_message"])
    st.markdown(f"**Chosen action (lowest change with the best safe objective):** {v['chosen_action']} · engine's own suggestion in EPANET: "
                f"{v['engine_status_in_epanet'].replace('_', ' ')}")

    rows = [("J1 mean pressure (m)", "J1_pressure_mean", "{:.1f}"), ("J2 mean pressure (m)", "J2_pressure_mean", "{:.1f}"),
            ("J3 mean pressure (m)", "J3_pressure_mean", "{:.1f}"), ("J4 mean pressure (m)", "J4_pressure_mean", "{:.1f}"),
            ("J4 minimum pressure (m)", "J4_pressure_min", "{:.1f}"), ("J4 critical-node status", "J4_status", "{}"),
            ("High-pressure violations", "high_pressure_violation_count", "{:d}"), ("Low-pressure violations", "low_pressure_violation_count", "{:d}"),
            ("Imbalance steps", "imbalance_step_count", "{:d}"), ("Mean branch pressure spread (m)", "branch_pressure_spread_mean", "{:.1f}"),
            ("Maximum pressure (m)", "max_pressure", "{:.1f}"), ("Safe-node coverage", "safe_node_coverage", "{:.0%}"),
            ("Loss proxy (L/s)", "loss_proxy_lps", "{:.2f}"), ("Demand served", "demand_served_fraction", "{:.1%}")]
    table = []
    for label, k, f in rows:
        bv, av = b[k], af[k]
        ch = "" if isinstance(bv, str) else (f"{av - bv:+.1f}" if "%" not in f and "d" not in f else f"{av - bv:+.0f}" if "d" in f else f"{(av - bv) * 100:+.0f} pts")
        table.append({"Metric": label, "Before (EPANET)": f.format(int(bv) if "d" in f else bv), "After (EPANET)": f.format(int(av) if "d" in f else av), "Change": ch})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)
    if v["before"]["loss_proxy_lps"] > 0 or result == "inspection_recommended":
        st.caption("Loss proxy = inflow minus delivered demand; in this simulation it equals the leak outflow, so it only appears in leak-like events.")

    g1, g2, g3 = st.columns([4, 4, 3])
    g1.plotly_chart(charts.before_after_pressure(v, cfg), width="stretch")
    g2.plotly_chart(charts.before_after_counts(v), width="stretch")
    g3.plotly_chart(charts.before_after_spread(v, cfg), width="stretch")

    vr = v["validated_recommendation"]
    st.markdown("**Recommendation after EPANET validation**")
    st.write(f"{vr['recommended_action_type'].replace('_', ' ').capitalize()} on {vr['target_component']}"
             + (f": {vr['current_value']:.2f} → {vr['recommended_value']:.2f} {vr['unit']} ({vr['adjustment_percent']:+.0f}%)" if vr.get("recommended_value") is not None else ""))
    st.write(v["safety_note"])
    with st.expander(f"Candidate search ({v['n_candidates_evaluated']} EPANET runs)"):
        cand = pd.DataFrame(v["candidates"])
        if len(cand):
            show = cand[["candidate", "safe", "objective", "objective_change_vs_no_change", "rejection_reasons", "is_chosen"]].copy()
            show["objective"] = show["objective"].round(2)
            show["objective_change_vs_no_change"] = show["objective_change_vs_no_change"].round(2)
            st.dataframe(show, hide_index=True, width="stretch")
        else:
            st.write("No control candidates were tried (leak-like, burst-like and uncertain situations are inspection-only).")
        st.caption("Objective = 10 × high-pressure violations + 10 × low-pressure violations + spread + 0.1 × |control change| "
                   "(+ very large penalties for any J4 critical violation). Lower is better. A candidate is rejected if it worsens any node or pushes J4 toward its critical minimum.")
    st.caption("Not every recommendation improves every scenario: the outcome above is what EPANET showed for this window.")

# =============================================================================================== 7. model comparison
st.header("7. Model comparison")
comp = dd.comparison_table(src.comparison_path)
if comp is None:
    st.info("No model-comparison table found. Run `python scripts/train_baselines.py` and `python scripts/train_gru.py`.")
else:
    st.dataframe(comp.round(3), hide_index=True, width="stretch",
                 column_config={"inference (ms / 1000 windows)": st.column_config.NumberColumn(format="%.2f")})
    st.caption(dd.comparison_summary(comp) + " Metrics come from the saved Phase 3-4 runs on synthetic EPANET scenarios held out by scenario ID. "
               "Lower MAE/RMSE is better; higher macro F1 is better."
               + ("" if {"lstm", "tcn"} <= set(comp["model"]) else " LSTM and TCN have not been trained in this run (python scripts/train_sequence_models.py)."))

# =============================================================================================== 8. help
st.header("8. Help and methodology")
st.markdown(f"""
**{dd.PROTOTYPE_STATEMENT}**

| Term | Plain meaning |
|---|---|
| **EPANET** | The standard hydraulic simulation engine for water pipe networks. It computes pressures and flows. All data here comes from it. |
| **WNTR** | A Python package that builds and runs EPANET models, so scenarios can be generated and replayed by code. |
| **Pressure** | How hard the water pushes at a point (in metres of water head). Too low: taps run weakly or dry. Too high: pipes and joints are stressed and leak more. |
| **Flow** | How much water moves through a pipe (litres per second). |
| **Junction** | A point where pipes meet or where customers draw water (J1 hub, J2 residential, J3 school, J4 hospital). |
| **Pump (PU1)** | Adds pressure at the source. The recommendation can raise or lower its relative speed within small limits. |
| **Valve (V1)** | A pressure-reducing valve on the branch to J2. Its setting is the pressure it holds downstream. |
| **Leak-like event** | Pressure falling while the water entering the network exceeds the water customers use. The engine recommends inspection, never a control change, for these. |
| **Deep learning** | A neural network (GRU; LSTM and TCN are compared with it) forecasts the next 15-minute pressure at J2–J4 from the last 3 hours. Risk labels and recommendations come from transparent rules applied to that forecast. |
| **EPANET simulated validation** | The scenario is re-simulated in EPANET before and after each bounded candidate action; only a safe, measurably better action is accepted. |

**Limitations**
- All data is synthetic, from a small 4-junction teaching network with no tank. Thresholds are lab-prototype values, not utility standards.
- Forecasts are one step (15 min) ahead. The simple models are strong on this data; section 7 shows the actual comparison of all models, with no model assumed best.
- Validation covers the next hour only, in the same simulator that produced the data. {VALIDATION_DISCLAIMER}
- Not every recommendation improves every scenario; some end with "No safe action found" or "inspection recommended".
- No real-time sensor feed, no equipment control, no real utility data.
""")
