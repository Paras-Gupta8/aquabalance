# AquaBalance

Deep Learning–Assisted Prediction and Control Recommendation for Junction-Level
Pressure Imbalance in Water Distribution Networks.

AquaBalance is a simulation-based decision-support prototype for an undergraduate
project. It makes no claim of real autonomous control. Sensors measure pressure and
flow; they do not equalize pressure. The goal is safe pressure ranges, not equal
pressure everywhere.

**EPANET is the mandatory primary hydraulic simulator for final experiments.**

## Roles of EPANET and WNTR

- **EPANET** solves the hydraulics (pressures and flows). All final data comes from it.
- **WNTR** is the Python layer that builds the network, edits settings, runs EPANET
  and reads the results. It does not replace EPANET.

## Setup

Run these from the `aquabalance/` folder.

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Phase 1

```bash
python scripts/build_network.py
python -m pytest tests/test_network.py -v
```

This writes `data/raw/aquabalance_base.inp` and `outputs/figures/network_topology.png`.
Network values (elevations, demands, pipes, reservoir head, valve and pump defaults,
pressure thresholds) are all in `config/config.yaml`.

## Run Phase 2: synthetic scenarios

Run `python scripts/build_network.py` first (Phase 1), then:

```bash
# 50 scenarios (quick mode, about 5 seconds)
python scripts/generate_synthetic_data.py --mode quick

# 300 scenarios (standard mode, about 15 seconds)
python scripts/generate_synthetic_data.py --mode standard

# options: --seed 7   --n 120   --sensor-noise (writes a separate noisy copy)
python -m pytest tests -v
```

**This dataset is generated from EPANET hydraulic simulations, not from randomly invented pressure values.**

### How scenarios are physically generated

1. Random numbers choose only the **settings** of a scenario, never a pressure or a flow.
2. The settings are applied to the Phase 1 EPANET network with WNTR: source head, pump speed
   (constant, gradual wear, or a temporary slow-down), PRV valve setting, smooth time-of-day demand
   multipliers at J2/J3/J4, a pipe blockage (smaller open cross-section) and leaks.
3. EPANET (`wntr.sim.EpanetSimulator`, EPANET 2.2) solves 24 hours at 15-minute steps. Demand is
   pressure-driven, so a starved junction receives less water instead of showing impossible negative pressure.
4. A leak is an EPANET emitter (flow rises and falls with pressure) behind a helper pipe that EPANET opens
   and closes at the scheduled times.
5. Every pressure, flow, demand, head, pump speed and valve value is read from the EPANET result.
   Each scenario has 96 time steps (the extra row at t = 24:00 is dropped).

Nine scenario families are generated: normal, high-demand stress, high source pressure / fast pump,
valve misconfiguration, reduced supply, small leak, burst leak, pump degradation and partial blockage.
The network has no tank, so "low tank level" is modelled as a temporary drop in source head, and
`tank_level` is empty on purpose.

### Output files

| File | Content |
|---|---|
| `data/synthetic/scenario_metadata.csv` | the settings of every scenario |
| `data/synthetic/raw_timeseries.csv` | EPANET pressures (m), flows and demands (L/s), 96 rows per scenario |
| `data/synthetic/labels.csv` | per-time-step risk flags and labels (kept apart from the raw data) |
| `data/synthetic/scenario_specs.jsonl` | full spec of each scenario, so any scenario can be re-simulated exactly |
| `data/synthetic/demand_profiles.csv` | the demand multipliers behind each `demand_profile_id` |
| `outputs/simulations/summary.csv` | per-scenario summary, status and EPANET warnings |
| `outputs/simulations/network_config.json` | network, thresholds, units and provenance |
| `outputs/logs/scenario_generation.log` | the settings and status of every scenario |

Scenarios that EPANET cannot solve, or whose results fail validation, are never added to the data: they are
retried with a new seed and every attempt is recorded in `summary.csv`.
In `scenario_metadata.csv`, `leak_severity` is the nominal leak (L/s at 50 m pressure); in
`raw_timeseries.csv` it is the leak flow EPANET actually computed. Leak times are in hours from the start.

### Labels

Thresholds (`p_min` 20 m, `p_max` 60 m, maximum spread 20 m, critical junction J4 minimum 30 m) are
**configurable lab-prototype values in `config/config.yaml`, not universal real-world standards.**
Flags are multi-label per time step (`low_pressure_flag`, `high_pressure_flag`, `imbalance_flag`, plus
leak/burst events). The primary `risk_label` uses this priority: burst_like_event, leak_like_event,
low_pressure_risk, high_pressure_risk, pressure_imbalance, normal. Pressure limits are applied to the consumer
junctions J2-J4; the hub J1 has no customers and is not judged (`labels.label_junctions` in the config).

## Open the network in the EPANET GUI

1. Install EPANET 2.2 from the US EPA website (https://www.epa.gov/water-research/epanet).
2. In EPANET choose **File → Open** and select `data/raw/aquabalance_base.inp`.
3. Run it with **Project → Run Analysis**, then use the Browser panel to view node pressures.

Flow units in the file are litres per second (LPS). The official EPANET GUI is a
Windows application; on macOS/Linux you can run it under Wine, or skip the GUI, because
the `.inp` is a plain text file that WNTR and EPANET run directly from Python.
