"""Shared builders for the Phase 5 tests: hand-made, fully controlled situations (no simulation needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.risk_engine import RiskInputs  # noqa: E402

T = 12
DEMAND = {"J2": 1.4, "J3": 1.2, "J4": 1.1}


def make_inputs(pred=None, now=None, leak_lps=0.0, drop_m=0.0, leak_branch="P3", pump_speed=1.0, valve_setting=None,
                valve_status=2, tank=None, demand=None, pump_status=1.0, flows=True):
    """A flat 12-step history. Optional leak: adds `leak_lps` to the inflow and one branch in the last step and makes
    all pressures fall by `drop_m` over the last 3 steps. pred defaults to the current pressure."""
    now = {"J1": 57.0, "J2": 48.0, "J3": 52.0, "J4": 47.0, **(now or {})}
    pred = {n: now[n] for n in ("J2", "J3", "J4")} if pred is None else {**{n: now[n] for n in ("J2", "J3", "J4")}, **pred}
    dem = {**DEMAND, **(demand or {})}
    ph = {}
    for n, v in now.items():
        h = np.full(T, float(v))
        if drop_m:
            h[:-3] += drop_m                       # earlier pressure was higher by drop_m
        ph[n] = h
    fh, dh = {}, {}
    if flows:
        total = sum(dem.values())
        for n, d in dem.items():
            dh[n] = np.full(T, d)
        fh = {"P1": np.full(T, total), "P2": np.full(T, dem["J2"]), "P3": np.full(T, dem["J3"]),
              "P4": np.full(T, dem["J4"])}
        if leak_lps:
            fh["P1"][-1] += leak_lps
            if leak_branch in fh:
                fh[leak_branch][-1] += leak_lps
    return RiskInputs(pressure_history=ph, predicted_pressure=pred, flow_history=fh, demand_history=dh,
                      tank_level=tank, pump_speed=pump_speed, pump_status=pump_status,
                      valve_setting=now["J2"] if valve_setting is None else valve_setting, valve_status=valve_status)
