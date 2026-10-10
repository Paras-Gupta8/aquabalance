"""Run hydraulic simulations with EPANET (through WNTR's EpanetSimulator).

EPANET is the only solver used. There is deliberately no fallback simulator:
if EPANET cannot run, this module raises an error instead of producing numbers.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import wntr


class SimulationError(Exception):
    """Raised with a readable message when an EPANET run fails."""


def run_hydraulic_simulation(inp_path: str | Path, duration_hours: float | None = None):
    """Run EPANET on an .inp file. Returns (model, results).

    duration_hours overrides the duration stored in the file (useful for short checks).
    """
    inp_path = Path(inp_path)
    if not inp_path.exists():
        raise SimulationError(f"INP file not found: {inp_path}. Run: python scripts/build_network.py")
    try:
        wn = wntr.network.WaterNetworkModel(str(inp_path))
        if duration_hours is not None:
            wn.options.time.duration = int(round(duration_hours * 3600))
        with tempfile.TemporaryDirectory(prefix="aquabalance_") as tmp:  # EPANET scratch files live here
            results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "run"))
    except Exception as exc:
        raise SimulationError(f"EPANET simulation failed for {inp_path}: {type(exc).__name__}: {exc}") from exc
    return wn, results


def get_junction_pressures(results, junction_ids: list[str]) -> pd.DataFrame:
    """Pressure (m) per timestep for the given junctions, straight from EPANET."""
    pressure = results.node["pressure"]
    missing = [j for j in junction_ids if j not in pressure.columns]
    if missing:
        raise SimulationError(f"EPANET results contain no pressure for: {missing}")
    df = pressure[junction_ids]
    if df.empty or df.isna().any().any():
        raise SimulationError("EPANET returned empty or NaN pressures.")
    return df


def pressure_summary(pressures: pd.DataFrame, p_min: float, p_max: float) -> pd.DataFrame:
    """Per-junction min/mean/max and the share of timesteps outside [p_min, p_max]."""
    pressures = pressures.astype("float64")  # EPANET returns float32; avoids 110.800003-style output
    return pd.DataFrame({
        "min_m": pressures.min(),
        "mean_m": pressures.mean(),
        "max_m": pressures.max(),
        "pct_below_p_min": (pressures < p_min).mean() * 100,
        "pct_above_p_max": (pressures > p_max).mean() * 100,
    }).round(1)
