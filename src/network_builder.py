"""Build, validate and save the AquaBalance base EPANET network.

The network is described entirely by config/config.yaml. WNTR is used only to
assemble the model and write a standard EPANET .inp file; all hydraulics are
solved by EPANET (see src/simulation.py).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import wntr  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.utils import resolve_path  # noqa: E402


class NetworkBuildError(Exception):
    """Raised with a readable message when the network cannot be built or validated."""


# --------------------------------------------------------------------------- config helpers
def _need(mapping, key, where):
    if not isinstance(mapping, dict) or key not in mapping:
        raise NetworkBuildError(f"Missing '{key}' in config section '{where}'. Check config/config.yaml.")
    return mapping[key]


def required_node_ids(cfg: dict) -> list[str]:
    net = _need(cfg, "network", "root")
    ids = [_need(_need(net, "reservoir", "network"), "id", "network.reservoir")]
    ids += [_need(j, "id", "network.junctions[]") for j in _need(net, "junctions", "network")]
    return ids


def required_link_ids(cfg: dict) -> list[str]:
    net = _need(cfg, "network", "root")
    ids = [_need(_need(net, "pump", "network"), "id", "network.pump")]
    ids += [_need(p, "id", "network.pipes[]") for p in _need(net, "pipes", "network")]
    ids.append(_need(_need(net, "valve", "network"), "id", "network.valve"))
    return ids


# --------------------------------------------------------------------------- build
def _apply_options(wn, cfg: dict) -> None:
    sim = _need(cfg, "simulation", "root")
    hours = float(_need(sim, "duration_hours", "simulation"))
    hyd = int(round(float(_need(sim, "hydraulic_timestep_min", "simulation")) * 60))
    rep = int(round(float(_need(sim, "report_timestep_min", "simulation")) * 60))
    if hours <= 0 or hyd <= 0 or rep <= 0:
        raise NetworkBuildError("simulation duration and timesteps must all be greater than zero.")
    if rep % hyd != 0:
        raise NetworkBuildError(
            f"report_timestep_min ({rep / 60:g}) must be a multiple of hydraulic_timestep_min ({hyd / 60:g}).")
    wn.options.time.duration = int(round(hours * 3600))
    wn.options.time.hydraulic_timestep = hyd
    wn.options.time.report_timestep = rep
    wn.options.time.pattern_timestep = hyd  # keeps future demand patterns aligned with the timestep
    wn.options.hydraulic.headloss = _need(sim, "headloss_formula", "simulation")
    wn.options.hydraulic.demand_model = "DDA"


def _check_endpoints(wn, kind: str, spec: dict) -> None:
    for end in ("from_node", "to_node"):
        if spec[end] not in wn.node_name_list:
            raise NetworkBuildError(
                f"{kind} '{spec['id']}' ({end}) refers to unknown node '{spec[end]}'. "
                f"Nodes defined in config: {sorted(wn.node_name_list)}")


def build_network_model(cfg: dict) -> "wntr.network.WaterNetworkModel":
    """Assemble the WNTR model from the config (nothing is written to disk here)."""
    net = _need(cfg, "network", "root")
    wn = wntr.network.WaterNetworkModel()
    _apply_options(wn, cfg)
    try:
        res = _need(net, "reservoir", "network")
        wn.add_reservoir(res["id"], base_head=float(res["head_m"]), coordinates=tuple(res["xy"]))

        for j in _need(net, "junctions", "network"):
            wn.add_junction(
                j["id"],
                base_demand=float(j["base_demand_lps"]) / 1000.0,  # L/s -> m3/s
                elevation=float(j["elevation_m"]),
                coordinates=tuple(j["xy"]),
            )

        for p in _need(net, "pipes", "network"):
            _check_endpoints(wn, "Pipe", p)
            wn.add_pipe(
                p["id"], p["from_node"], p["to_node"],
                length=float(p["length_m"]), diameter=float(p["diameter_m"]),
                roughness=float(p["roughness_hw"]), minor_loss=0.0,
            )

        pump = _need(net, "pump", "network")
        _check_endpoints(wn, "Pump", pump)
        points = [(float(q) / 1000.0, float(h)) for q, h in pump["head_curve_lps_m"]]  # L/s -> m3/s
        wn.add_curve(pump["curve_id"], "HEAD", points)
        wn.add_pump(
            pump["id"], pump["from_node"], pump["to_node"],
            pump_type="HEAD", pump_parameter=pump["curve_id"], speed=float(pump["speed_default"]),
        )

        valve = _need(net, "valve", "network")
        _check_endpoints(wn, "Valve", valve)
        wn.add_valve(
            valve["id"], valve["from_node"], valve["to_node"],
            diameter=float(valve["diameter_m"]), valve_type=valve["type"],
            minor_loss=float(valve["minor_loss"]), initial_setting=float(valve["setting_default"]),
            initial_status="Active",
        )
    except NetworkBuildError:
        raise
    except KeyError as exc:
        raise NetworkBuildError(f"Missing config key {exc} while building the network. Check config/config.yaml.") from exc
    except (ValueError, TypeError) as exc:
        raise NetworkBuildError(f"Invalid value in config/config.yaml while building the network: {exc}") from exc
    except Exception as exc:  # WNTR rejected something (duplicate id, unknown node, bad curve, ...)
        raise NetworkBuildError(f"WNTR could not build the network: {type(exc).__name__}: {exc}") from exc
    return wn


# --------------------------------------------------------------------------- validate
def validate_model(wn, cfg: dict) -> None:
    """Check the model and thresholds; raise one NetworkBuildError listing every problem."""
    problems: list[str] = []

    for node_id in required_node_ids(cfg):
        if node_id not in wn.node_name_list:
            problems.append(f"required node '{node_id}' is missing")
    for link_id in required_link_ids(cfg):
        if link_id not in wn.link_name_list:
            problems.append(f"required link '{link_id}' is missing")

    if not nx.is_connected(wn.to_graph().to_undirected()):
        problems.append("network is not fully connected (an isolated node or link exists)")

    th = _need(cfg, "thresholds", "root")
    p_min, p_max = float(th["p_min_m"]), float(th["p_max_m"])
    crit_id, crit_min = th["critical_junction"], float(th["critical_p_min_m"])
    if not p_min < p_max:
        problems.append(f"p_min_m ({p_min}) must be lower than p_max_m ({p_max})")
    if float(th["max_pressure_spread_m"]) <= 0:
        problems.append("max_pressure_spread_m must be greater than zero")
    if crit_id not in wn.junction_name_list:
        problems.append(f"critical_junction '{crit_id}' is not a junction in the network")
    if not p_min <= crit_min < p_max:
        problems.append(f"critical_p_min_m ({crit_min}) must lie between p_min_m and p_max_m")

    if problems:
        raise NetworkBuildError("Network validation failed:\n  - " + "\n  - ".join(problems))


def validate_inp_file(inp_path: str | Path, cfg: dict):
    """Reload a saved .inp with WNTR and confirm the required elements are present."""
    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise NetworkBuildError(f"INP file not found: {inp_path}. Run: python scripts/build_network.py")
    try:
        wn = wntr.network.WaterNetworkModel(str(inp_path))
    except Exception as exc:
        raise NetworkBuildError(
            f"WNTR/EPANET could not read {inp_path} ({type(exc).__name__}: {exc}). "
            "The file may be malformed; open it in the EPANET GUI to locate the problem."
        ) from exc
    validate_model(wn, cfg)
    return wn


# --------------------------------------------------------------------------- save
def write_inp(wn, inp_path: str | Path, flow_units: str = "LPS") -> Path:
    """Write the model as a standard EPANET .inp file."""
    inp_path = Path(inp_path)
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wntr.network.write_inpfile(wn, str(inp_path), units=flow_units)
    except Exception as exc:
        raise NetworkBuildError(f"Could not write the EPANET file {inp_path}: {type(exc).__name__}: {exc}") from exc
    return inp_path


def save_topology_figure(wn, cfg: dict, fig_path: str | Path) -> Path:
    """Draw a simple static schematic (not to scale) of the network."""
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    crit_id = cfg["thresholds"]["critical_junction"]
    desc = {j["id"]: j.get("description", "") for j in cfg["network"]["junctions"]}
    coords = {n: wn.get_node(n).coordinates for n in wn.node_name_list}

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

    # links
    for name in wn.link_name_list:
        link = wn.get_link(name)
        (x0, y0), (x1, y1) = coords[link.start_node_name], coords[link.end_node_name]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if link.link_type == "Pipe":
            ax.plot([x0, x1], [y0, y1], color="#6b7280", lw=1.5 + 14 * link.diameter, solid_capstyle="round", zorder=1)
            label = f"{name}\n{link.length:.0f} m, Ø{link.diameter * 1000:.0f} mm"
        elif link.link_type == "Pump":
            ax.plot([x0, x1], [y0, y1], color="#2563eb", lw=4, zorder=1)
            ax.plot(mx, my, marker=">", ms=14, color="#2563eb", zorder=3)
            label = f"{name}\npump"
        else:
            ax.plot([x0, x1], [y0, y1], color="#d97706", lw=3, ls="--", zorder=1)
            ax.plot(mx, my, marker="D", ms=11, color="#d97706", zorder=3)
            label = f"{name}\n{link.valve_type} valve"
        ax.annotate(label, (mx, my), textcoords="offset points", xytext=(0, 16), ha="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=4)

    # nodes
    for name in wn.node_name_list:
        node = wn.get_node(name)
        x, y = coords[name]
        if node.node_type == "Reservoir":
            ax.plot(x, y, marker="s", ms=20, color="#0891b2", zorder=5)
            ax.annotate(f"{name}\nhead {node.base_head:.0f} m", (x, y), textcoords="offset points",
                        xytext=(0, 22), ha="center", fontsize=9, fontweight="bold")
            continue
        is_aux = node.base_demand == 0 and name not in ("J1",)
        crit = name == crit_id
        ax.plot(x, y, marker="o", ms=8 if is_aux else 16, color="#9ca3af" if is_aux else "#16a34a",
                markeredgecolor="#dc2626" if crit else "white", markeredgewidth=3 if crit else 1.5, zorder=5)
        if is_aux:
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=7, color="#4b5563")
            continue
        text = f"{name} – {desc.get(name, '')}\nz = {node.elevation:.0f} m"
        if node.base_demand > 0:
            text += f"\ndemand {node.base_demand * 1000:.1f} L/s"
        if crit:
            text += "\nCRITICAL"
        side = name in ("J2", "J3", "J4")
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(18, -4) if side else (0, 18),
                    ha="left" if side else "center", va="center" if side else "bottom", fontsize=9,
                    color="#dc2626" if crit else "black", fontweight="bold" if crit else "normal", zorder=6)

    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#0891b2", ms=11, label="Reservoir"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", ms=10, label="Junction"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ca3af", ms=7, label="Helper node (zero demand)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markeredgecolor="#dc2626",
               markeredgewidth=2.5, ms=10, label="Critical junction"),
        Line2D([0], [0], color="#2563eb", lw=4, label="Pump"),
        Line2D([0], [0], color="#d97706", lw=3, ls="--", label="Valve (controllable)"),
        Line2D([0], [0], color="#6b7280", lw=3, label="Pipe"),
    ], loc="lower left", fontsize=8, frameon=True)

    ax.set_xlim(-70, 900)
    ax.set_ylim(-110, 330)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("AquaBalance base network (EPANET model, schematic - not to scale)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# --------------------------------------------------------------------------- one call
def build_and_save(cfg: dict) -> dict:
    """Build -> validate -> write .inp -> reload and re-validate -> draw figure."""
    wn = build_network_model(cfg)
    validate_model(wn, cfg)
    inp_path = write_inp(wn, resolve_path(_need(_need(cfg, "paths", "root"), "base_inp", "paths")),
                         cfg["simulation"].get("inp_flow_units", "LPS"))
    loaded = validate_inp_file(inp_path, cfg)  # proves the saved file round-trips
    fig_path = save_topology_figure(loaded, cfg, resolve_path(cfg["paths"]["topology_figure"]))
    return {"inp_path": inp_path, "figure_path": fig_path, "model": loaded}
