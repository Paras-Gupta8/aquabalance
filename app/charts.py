"""Plotly figures for the AquaBalance dashboard. Light surface, restrained blue/teal; status colours are fixed and always
come with a text label (never colour alone)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e8ea", "#ffffff"
SERIES = {"J1": "#6b7a8c", "J2": "#2a78d6", "J3": "#1baf7a", "J4": "#eb6834"}     # J1 = hub context line
FLOW_COLORS = {"P1": "#6b7a8c", "P2": "#2a78d6", "P3": "#1baf7a", "P4": "#eb6834"}
BEFORE, AFTER = "#eb6834", "#2a78d6"
STATUS = {"safe": ("#0ca30c", "OK"), "warning": ("#fab219", "WARNING"), "risk": ("#d03b3b", "HIGH RISK"), "hub": ("#6b7a8c", "hub, not scored")}
DESCR = {"J1": "Distribution hub", "J2": "Residential", "J3": "School", "J4": "Hospital / critical service"}


def _layout(fig: go.Figure, height: int, title: str | None = None, ytitle: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(color=INK, size=13), showlegend=legend,
                      title=dict(text=title, x=0, font=dict(size=14, color=INK)) if title else None,
                      legend=dict(orientation="h", y=-0.22, x=0, font=dict(color=MUTED)), hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED), title=ytitle)
    return fig


# =============================================================================================== network diagram
def network_figure(cfg: dict, view, df_now: pd.Series, show_forecast: bool) -> go.Figure:
    """R1 -PU1- JP1 -P1- J1 ; J1 -P2- JV1 -V1- J2 ; J1 -P3- J3 ; J1 -P4- J4, coloured by risk status."""
    net = cfg["network"]
    xy = {"R1": net["reservoir"]["xy"], **{j["id"]: j["xy"] for j in net["junctions"]}}
    fig = go.Figure()

    def line(a, b, w=4, color="#9fb3bf"):
        fig.add_trace(go.Scatter(x=[xy[a][0], xy[b][0]], y=[xy[a][1], xy[b][1]], mode="lines", line=dict(width=w, color=color),
                                 hoverinfo="skip", showlegend=False))

    for a, b in (("R1", "JP1"), ("JP1", "J1"), ("J1", "JV1"), ("JV1", "J2"), ("J1", "J3"), ("J1", "J4")):
        line(a, b)
    # pipe labels with the current EPANET flow
    for pid, (a, b) in {"P1": ("JP1", "J1"), "P2": ("J1", "JV1"), "P3": ("J1", "J3"), "P4": ("J1", "J4")}.items():
        mx, my = (xy[a][0] + xy[b][0]) / 2, (xy[a][1] + xy[b][1]) / 2
        fig.add_annotation(x=mx, y=my, text=f"<b>{pid}</b> {df_now[f'{pid}_flow']:.2f} L/s", showarrow=False, yshift=14,
                           font=dict(size=11, color=MUTED), bgcolor="rgba(255,255,255,0.85)")
    # equipment
    px, py = (xy["R1"][0] + xy["JP1"][0]) / 2, xy["R1"][1]
    vx, vy = (xy["JV1"][0] + xy["J2"][0]) / 2, (xy["JV1"][1] + xy["J2"][1]) / 2
    pump_on = "on" if df_now["pump_status"] >= 0.5 else "OFF"
    fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers+text", marker=dict(size=26, symbol="triangle-right", color="#1d4e89", line=dict(width=2, color="white")),
                             text=[f"<b>PU1</b> pump<br>speed {df_now['pump_speed']:.2f} ({pump_on})"], textposition="top center",
                             textfont=dict(size=11, color=INK), hovertext="PU1: pump. Speed is relative to nominal (1.0).", hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[vx], y=[vy], mode="markers+text", marker=dict(size=22, symbol="bowtie", color="#1d4e89", line=dict(width=2, color="white")),
                             text=[f"<b>V1</b> PRV valve<br>setting {df_now['valve_setting']:.1f} m"], textposition="top left",
                             textfont=dict(size=11, color=INK), hovertext="V1: pressure-reducing valve; the setting is the pressure it holds downstream (J2 side).", hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[xy["R1"][0]], y=[xy["R1"][1]], mode="markers+text", marker=dict(size=34, symbol="square", color="#2a78d6", line=dict(width=2, color="white")),
                             text=[f"<b>R1</b> reservoir<br>head {df_now['reservoir_head']:.1f} m"], textposition="bottom center",
                             textfont=dict(size=11, color=INK), hovertext="R1: reservoir (water source)", hoverinfo="text", showlegend=False))
    # junctions
    for n in ("J1", "J2", "J3", "J4"):
        if n == "J1":
            kind, why = "hub", "Hub junction: shown for information. The risk engine scores only J2-J4 (consumer junctions)."
        else:
            kind, why = view.node_status[n]
        color, word = STATUS[kind]
        cur = view.current_pressure[n]
        txt = f"<b>{n}</b> {cur:.1f} m"
        hover = f"{n} ({DESCR[n]})<br>now: {cur:.1f} m<br>status: {word}<br>{why}"
        if n != "J1" and show_forecast:
            txt += f" → {view.forecast[n]:.1f} m"
            hover += f"<br>forecast next 15 min ({view.model}): {view.forecast[n]:.1f} m"
        fig.add_trace(go.Scatter(x=[xy[n][0]], y=[xy[n][1]], mode="markers+text", marker=dict(size=34, color=color, line=dict(width=3, color="white")),
                                 text=[f"{txt}<br><span style='font-size:11px'>{word}</span>"], textposition="middle right" if n != "J1" else "top center",
                                 textfont=dict(size=12, color=INK), hovertext=hover, hoverinfo="text", showlegend=False, cliponaxis=False))
    fig.update_xaxes(visible=False, range=[-110, 900])
    fig.update_yaxes(visible=False, range=[-110, 330])
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, hovermode="closest")
    return fig


# =============================================================================================== trends
def pressure_figure(cfg: dict, df: pd.DataFrame, view, show_forecast: bool) -> go.Figure:
    ks, t = cfg["thresholds"], view.step
    h = df.iloc[:t + 1]
    fig = go.Figure()
    fig.add_hrect(y0=ks["p_min_m"], y1=ks["p_max_m"], fillcolor="#e6f4f1", opacity=0.55, line_width=0, layer="below")
    for n in ("J1", "J2", "J3", "J4"):
        fig.add_trace(go.Scatter(x=h["timestamp"], y=h[f"{n}_pressure"], mode="lines", name=f"{n}" + (" (hub)" if n == "J1" else ""),
                                 line=dict(width=1.6 if n == "J1" else 2.4, color=SERIES[n], dash="dot" if n == "J1" else "solid"),
                                 hovertemplate="%{y:.1f} m"))
        if show_forecast and n != "J1":
            fig.add_trace(go.Scatter(x=[view.now_ts, view.next_ts], y=[view.current_pressure[n], view.forecast[n]], mode="lines+markers",
                                     name=f"{n} forecast", line=dict(width=2, color=SERIES[n], dash="dash"),
                                     marker=dict(size=[0.1, 10], symbol="diamond", color=SERIES[n], line=dict(width=2, color="white")),
                                     hovertemplate="forecast %{y:.1f} m", showlegend=False))
    for y, label, color in ((ks["p_max_m"], f"safe maximum {ks['p_max_m']:.0f} m", MUTED), (ks["p_min_m"], f"safe minimum {ks['p_min_m']:.0f} m", MUTED),
                            (ks["critical_p_min_m"], f"J4 critical minimum {ks['critical_p_min_m']:.0f} m", SERIES["J4"])):
        fig.add_hline(y=y, line=dict(color=color, width=1.2, dash="dot"), annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=10, color=color))
    fig.add_vline(x=view.now_ts, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(range=[df["timestamp"].iloc[0], view.next_ts + pd.Timedelta(minutes=45)], tickformat="%H:%M")
    return _layout(fig, 400, "Pressure at the junctions (EPANET simulated; dashed diamonds = next-step forecast)", "Pressure (m)")


def flow_figure(df: pd.DataFrame, view) -> go.Figure:
    h = df.iloc[:view.step + 1]
    fig = go.Figure([go.Scatter(x=h["timestamp"], y=h[f"{p}_flow"], mode="lines", name=p, line=dict(width=2.2, color=FLOW_COLORS[p]),
                                hovertemplate="%{y:.2f} L/s") for p in ("P1", "P2", "P3", "P4")])
    fig.update_xaxes(tickformat="%H:%M")
    return _layout(fig, 300, "Pipe flows (EPANET simulated)", "Flow (L/s)")


def demand_figure(df: pd.DataFrame, view) -> go.Figure:
    h = df.iloc[:view.step + 1]
    fig = go.Figure([go.Scatter(x=h["timestamp"], y=h[f"{n}_demand"], mode="lines", name=n, line=dict(width=2.2, color=SERIES[n]),
                                hovertemplate="%{y:.2f} L/s") for n in ("J2", "J3", "J4")])
    fig.update_xaxes(tickformat="%H:%M")
    return _layout(fig, 300, "Demand at the consumer junctions (simulated input)", "Demand (L/s)")


# =============================================================================================== validation
def before_after_pressure(v: dict, cfg: dict) -> go.Figure:
    nodes = ["J1", "J2", "J3", "J4"]
    ks = cfg["thresholds"]
    fig = go.Figure()
    for name, key, color in (("Before (EPANET)", "before", BEFORE), ("After (EPANET)", "after", AFTER)):
        ys = [v[key][f"{n}_pressure_mean"] for n in nodes]
        fig.add_trace(go.Bar(x=nodes, y=ys, name=name, marker=dict(color=color, cornerradius=4), text=[f"{y:.1f}" for y in ys],
                             textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:.1f} m"))
    for y, label in ((ks["p_max_m"], f"safe max {ks['p_max_m']:.0f} m"), (ks["p_min_m"], f"safe min {ks['p_min_m']:.0f} m")):
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=10, color=MUTED))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    return _layout(fig, 340, "Mean pressure over the validated hour", "Pressure (m)")


def before_after_counts(v: dict) -> go.Figure:
    cats = [("Low pressure", "low_pressure_violation_count"), ("High pressure", "high_pressure_violation_count"),
            ("Imbalance", "imbalance_step_count")]
    fig = go.Figure()
    for name, key, color in (("Before (EPANET)", "before", BEFORE), ("After (EPANET)", "after", AFTER)):
        ys = [v[key][c] for _, c in cats]
        fig.add_trace(go.Bar(x=[c for c, _ in cats], y=ys, name=name, marker=dict(color=color, cornerradius=4), text=ys,
                             textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y}"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    fig.update_yaxes(rangemode="tozero")
    return _layout(fig, 340, "Violations during the validated hour (counts)", "Count")


def before_after_spread(v: dict, cfg: dict) -> go.Figure:
    lim = cfg["thresholds"]["max_pressure_spread_m"]
    fig = go.Figure()
    for name, key, color in (("Before (EPANET)", "before", BEFORE), ("After (EPANET)", "after", AFTER)):
        y = v[key]["branch_pressure_spread_mean"]
        fig.add_trace(go.Bar(x=["Mean branch pressure spread"], y=[y], name=name, marker=dict(color=color, cornerradius=4),
                             text=[f"{y:.1f}"], textposition="outside", cliponaxis=False, hovertemplate="%{y:.1f} m"))
    fig.add_hline(y=lim, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=f"limit {lim:.0f} m",
                  annotation_position="top left", annotation_font=dict(size=10, color=MUTED))
    fig.update_layout(barmode="group", bargap=0.5, bargroupgap=0.06)
    fig.update_yaxes(rangemode="tozero")
    return _layout(fig, 340, "Pressure spread (highest - lowest of J2-J4)", "Spread (m)")
