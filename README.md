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

## Open the network in the EPANET GUI

1. Install EPANET 2.2 from the US EPA website (https://www.epa.gov/water-research/epanet).
2. In EPANET choose **File → Open** and select `data/raw/aquabalance_base.inp`.
3. Run it with **Project → Run Analysis**, then use the Browser panel to view node pressures.

Flow units in the file are litres per second (LPS). The official EPANET GUI is a
Windows application; on macOS/Linux you can run it under Wine, or skip the GUI, because
the `.inp` is a plain text file that WNTR and EPANET run directly from Python.
