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

## Run Phase 3: ML dataset and baselines

Phase 3 turns the EPANET scenarios into a leakage-safe forecasting dataset and trains three simple baselines.
There is **no deep learning yet** (the GRU comes in Phase 4).

```bash
python scripts/generate_synthetic_data.py --mode standard   # 300 scenarios (quick mode is too small for Phase 3)
python scripts/build_ml_dataset.py
python scripts/train_baselines.py
python -m pytest tests -v
```

**Task:** previous 12 steps (3 h) -> pressure at J2, J3, J4 one step (15 min) ahead. Window, horizon, target nodes and
features are set in `config/config.yaml` under `ml:`.

**Leakage rules**
- Scenarios are split as whole units (70/15/15, fixed seed, stratified by scenario type); `split_manifest.csv` records
  the split and the code asserts no scenario is in two splits.
- Sliding windows are cut inside one scenario only, never across a boundary.
- Scalers are fitted on the training split only and saved with Joblib (`models/scalers/`).
- `leak_flag`, `leak_severity`, `scenario_type` are never model inputs. They are kept only as `meta_*` arrays for analysis.
- A feature that EPANET did not produce (here `tank_level`, the network has no tank) is dropped and recorded in
  `feature_names.json`, not invented.

**Baselines**
1. Persistence: next pressure = current pressure.
2. Rule-based: **reactive, not predictive.** It judges the current pressures against the thresholds.
3. Random forest: multi-output, on lag/difference/rolling features; predicts the change from the current pressure.

Outputs (generated, git-ignored): `data/processed/*.npz`, `models/scalers/*`, `models/trained/random_forest.joblib`,
`outputs/reports/baseline_metrics.csv`, `outputs/reports/baseline_error_analysis.csv`, and two figures in `outputs/figures/`.

## Run Phase 4: GRU pressure forecaster

Pressure forecasting and decision support (simulation-based; nothing here controls a real network).

```bash
python scripts/train_gru.py            # needs Phase 3 first: build_ml_dataset.py and train_baselines.py
python scripts/train_gru.py --epochs 5 # quick smoke run
python -m pytest tests -v
```

**Model:** input `(batch, 12, 18)` sensor/control history -> GRU(64) -> dropout 0.20 -> GRU(32) -> final hidden state
-> Dense(32)+ReLU -> 3 outputs (J2, J3, J4 pressure 15 min ahead). Adam, lr 0.001, Huber loss (`gru.loss: mse` to switch),
batch 64, up to 100 epochs, early stopping (patience 10), ReduceLROnPlateau, seeded. CPU by default, GPU if available.
The best checkpoint (lowest validation MAE) is saved and restored before any test evaluation. The network predicts the
**change from the current pressure** and adds it back (`gru.predict_delta`), exactly like the random forest, so the
comparison is fair. Scenario type, leak flags and leak severity are never inputs.

Outputs (generated, git-ignored except figures): `models/trained/gru_best.pt` and `gru_config.json`,
`outputs/reports/gru_metrics.json`, `model_comparison.csv`, `gru_error_by_scenario_type.csv`, and four figures
(`gru_training_history`, `gru_prediction_vs_actual`, `gru_per_node_error`, `gru_risk_confusion_matrix`) in `outputs/figures/`.

Risk labels for scoring come from the true future EPANET pressure; a dashboard must derive risk from the **predicted**
pressure only (`GRUPredictor.predict_risk_class`).

## Open the network in the EPANET GUI

1. Install EPANET 2.2 from the US EPA website (https://www.epa.gov/water-research/epanet).
2. In EPANET choose **File → Open** and select `data/raw/aquabalance_base.inp`.
3. Run it with **Project → Run Analysis**, then use the Browser panel to view node pressures.

Flow units in the file are litres per second (LPS). The official EPANET GUI is a
Windows application; on macOS/Linux you can run it under Wine, or skip the GUI, because
the `.inp` is a plain text file that WNTR and EPANET run directly from Python.
