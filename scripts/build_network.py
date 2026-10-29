"""Phase 1: build the AquaBalance base network, save the .inp, draw the topology figure.

Usage (from the aquabalance/ folder):
    python scripts/build_network.py
    python scripts/build_network.py --skip-sim     # skip the EPANET sanity run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.network_builder import NetworkBuildError, build_and_save  # noqa: E402
from src.simulation import SimulationError, get_junction_pressures, pressure_summary, run_hydraulic_simulation  # noqa: E402
from src.utils import ConfigError, get_logger, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the AquaBalance base EPANET network.")
    parser.add_argument("--config", default=None, help="path to a config YAML (default: config/config.yaml)")
    parser.add_argument("--skip-sim", action="store_true", help="skip the EPANET sanity run")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        log = get_logger("build_network", cfg.get("paths", {}).get("log_dir"))
        out = build_and_save(cfg)
        wn = out["model"]
        log.info("Saved EPANET network: %s", out["inp_path"])
        log.info("Saved topology figure: %s", out["figure_path"])
        log.info("Network: %d nodes, %d pipes, %d pump, %d valve",
                 len(wn.node_name_list), len(wn.pipe_name_list), len(wn.pump_name_list), len(wn.valve_name_list))

        if not args.skip_sim:
            th = cfg["thresholds"]
            ids = [j["id"] for j in cfg["network"]["junctions"] if j.get("role") in ("hub", "demand")]
            _, results = run_hydraulic_simulation(out["inp_path"])
            pressures = get_junction_pressures(results, ids)
            log.info("EPANET sanity run OK (%d timesteps). Junction pressures in metres:", len(pressures))
            print(pressure_summary(pressures, th["p_min_m"], th["p_max_m"]).to_string())
    except (ConfigError, NetworkBuildError, SimulationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
