"""Phase 1 tests: the EPANET base network exists, loads, and EPANET can simulate it.

Run `python scripts/build_network.py` first so data/raw/aquabalance_base.inp exists.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import wntr  # noqa: E402

from src.simulation import get_junction_pressures, run_hydraulic_simulation  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402

REQUIRED_NODES = ["R1", "J1", "J2", "J3", "J4"]
REQUIRED_LINKS = ["PU1", "P1", "P2", "P3", "P4", "V1"]
PRESSURE_NODES = ["J1", "J2", "J3", "J4"]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def inp_path(cfg):
    return resolve_path(cfg["paths"]["base_inp"])


@pytest.fixture(scope="module")
def wn(inp_path):
    if not inp_path.exists():
        pytest.fail(f"{inp_path} not found. Run: python scripts/build_network.py")
    return wntr.network.WaterNetworkModel(str(inp_path))


def test_inp_file_exists(inp_path):
    assert inp_path.name == "aquabalance_base.inp"
    assert inp_path.parent.name == "raw"
    assert inp_path.exists(), "Run: python scripts/build_network.py"
    assert inp_path.stat().st_size > 0


def test_wntr_can_load_inp(wn):
    assert isinstance(wn, wntr.network.WaterNetworkModel)


@pytest.mark.parametrize("node_id", REQUIRED_NODES)
def test_required_nodes_exist(wn, node_id):
    assert node_id in wn.node_name_list


@pytest.mark.parametrize("link_id", REQUIRED_LINKS)
def test_required_links_exist(wn, link_id):
    assert link_id in wn.link_name_list


def test_component_types(wn):
    assert wn.get_node("R1").node_type == "Reservoir"
    assert wn.get_link("PU1").link_type == "Pump"
    assert wn.get_link("V1").link_type == "Valve"
    assert wn.get_link("V1").valve_type == "PRV"


def test_saved_values_match_config(wn, cfg):
    net = cfg["network"]
    assert wn.get_node(net["reservoir"]["id"]).base_head == pytest.approx(net["reservoir"]["head_m"])
    for j in net["junctions"]:
        node = wn.get_node(j["id"])
        assert node.elevation == pytest.approx(j["elevation_m"])
        assert node.base_demand == pytest.approx(j["base_demand_lps"] / 1000.0, abs=1e-9)
    for p in net["pipes"]:
        pipe = wn.get_link(p["id"])
        assert pipe.length == pytest.approx(p["length_m"])
        assert pipe.diameter == pytest.approx(p["diameter_m"])
        assert pipe.roughness == pytest.approx(p["roughness_hw"])
    assert wn.get_link(net["valve"]["id"]).initial_setting == pytest.approx(net["valve"]["setting_default"])
    assert wn.options.time.duration == int(cfg["simulation"]["duration_hours"] * 3600)
    assert wn.options.time.hydraulic_timestep == int(cfg["simulation"]["hydraulic_timestep_min"] * 60)
    assert wn.options.time.report_timestep == int(cfg["simulation"]["report_timestep_min"] * 60)


def test_j4_is_highest_and_critical_in_config(wn, cfg):
    assert cfg["thresholds"]["critical_junction"] == "J4"
    assert wn.get_node("J4").elevation == max(wn.get_node(j).elevation for j in PRESSURE_NODES)


def test_short_epanet_simulation_returns_pressures(inp_path, cfg):
    """One hour with EPANET (EpanetSimulator); pressures for J1-J4 must come back and be finite."""
    wn_run, results = run_hydraulic_simulation(inp_path, duration_hours=1)
    assert isinstance(wn_run, wntr.network.WaterNetworkModel)
    pressures = get_junction_pressures(results, PRESSURE_NODES)
    step = cfg["simulation"]["report_timestep_min"] * 60
    assert list(pressures.columns) == PRESSURE_NODES
    assert len(pressures) == 3600 // step + 1
    assert all(math.isfinite(v) for v in pressures.to_numpy().ravel())
    assert (pressures > 0).all().all(), "negative or zero pressure at a junction"
