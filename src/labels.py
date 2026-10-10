"""Risk labels for AquaBalance time series.

IMPORTANT: every threshold used here (p_min, p_max, maximum pressure spread, the stricter
critical-junction minimum, leak/burst flow limits) is a CONFIGURABLE LAB-PROTOTYPE value from
config/config.yaml. They are NOT universal real-world standards: real utilities define their own
service-pressure limits.

Flags are computed per time step and are multi-label internally:

  low_pressure_flag   any consumer junction below p_min, OR the critical junction below its stricter minimum
  high_pressure_flag  any consumer junction above p_max
  imbalance_flag      pressure spread (max - min over the spread junctions) above the maximum spread
  leak_event_flag     leak active with EPANET leak outflow between leak_min_lps and burst_min_lps
  burst_event_flag    leak active with EPANET leak outflow at or above burst_min_lps

Comparisons are strict: a pressure exactly equal to a limit is NOT a violation.

Primary label (one per row): the FIRST label in `labels.priority` whose condition is true.
Default priority, with the reason:

  1. burst_like_event      a known physical event; the root cause outranks its pressure symptoms
  2. leak_like_event       same reason, smaller event
  3. low_pressure_risk     loss of service, and the critical (hospital) junction is protected first
  4. high_pressure_risk    pipe/fitting stress and water loss
  5. pressure_imbalance    pressures inside limits but too uneven between branches
  6. normal                no condition triggered

Because the flags are kept separately, nothing is lost when the primary label hides a second problem.
"""
from __future__ import annotations

import pandas as pd

RISK_LABELS = (
    "normal",
    "low_pressure_risk",
    "high_pressure_risk",
    "pressure_imbalance",
    "leak_like_event",
    "burst_like_event",
)

FLAG_TO_LABEL = {
    "low_pressure_flag": "low_pressure_risk",
    "high_pressure_flag": "high_pressure_risk",
    "imbalance_flag": "pressure_imbalance",
    "leak_event_flag": "leak_like_event",
    "burst_event_flag": "burst_like_event",
}
FLAG_COLUMNS = tuple(FLAG_TO_LABEL)

DEFAULT_PRIORITY = (
    "burst_like_event",
    "leak_like_event",
    "low_pressure_risk",
    "high_pressure_risk",
    "pressure_imbalance",
    "normal",
)


class LabelError(ValueError):
    """Raised when label settings or input data are invalid."""


def label_settings(cfg: dict) -> dict:
    """Collect all label thresholds from the config into one flat dictionary."""
    try:
        th = cfg["thresholds"]
        lab = cfg["labels"]
        settings = {
            "p_min_m": float(th["p_min_m"]),
            "p_max_m": float(th["p_max_m"]),
            "max_spread_m": float(th["max_pressure_spread_m"]),
            "critical_junction": str(th["critical_junction"]),
            "critical_p_min_m": float(th["critical_p_min_m"]),
            "label_junctions": list(lab["label_junctions"]),
            "spread_junctions": list(lab["spread_junctions"]),
            "leak_min_lps": float(lab["leak_min_lps"]),
            "burst_min_lps": float(lab["burst_min_lps"]),
            "priority": list(lab.get("priority", DEFAULT_PRIORITY)),
        }
    except KeyError as exc:
        raise LabelError(f"Missing label setting {exc} in config/config.yaml.") from exc
    if not settings["p_min_m"] < settings["p_max_m"]:
        raise LabelError("p_min_m must be lower than p_max_m.")
    if set(settings["priority"]) != set(RISK_LABELS):
        raise LabelError(f"labels.priority must list exactly these labels: {list(RISK_LABELS)}")
    if settings["priority"][-1] != "normal":
        raise LabelError("'normal' must be last in labels.priority (it is the fallback).")
    if settings["leak_min_lps"] >= settings["burst_min_lps"]:
        raise LabelError("leak_min_lps must be lower than burst_min_lps.")
    return settings


def compute_flags(ts: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Per-time-step flags for a raw time-series frame (needs <J>_pressure, leak_flag, leak_severity)."""
    needed = [f"{j}_pressure" for j in set(settings["label_junctions"]) | set(settings["spread_junctions"])
              | {settings["critical_junction"]}] + ["leak_flag", "leak_severity"]
    missing = [c for c in needed if c not in ts.columns]
    if missing:
        raise LabelError(f"Time series is missing columns needed for labels: {missing}")

    label_p = ts[[f"{j}_pressure" for j in settings["label_junctions"]]]
    spread_p = ts[[f"{j}_pressure" for j in settings["spread_junctions"]]]
    spread = spread_p.max(axis=1) - spread_p.min(axis=1)
    critical_p = ts[f"{settings['critical_junction']}_pressure"]
    leak_active = (ts["leak_flag"] == 1) & (ts["leak_severity"] >= settings["leak_min_lps"])

    flags = pd.DataFrame(index=ts.index)
    flags["pressure_spread"] = spread
    flags["low_pressure_flag"] = ((label_p < settings["p_min_m"]).any(axis=1)
                                  | (critical_p < settings["critical_p_min_m"])).astype(int)
    flags["high_pressure_flag"] = (label_p > settings["p_max_m"]).any(axis=1).astype(int)
    flags["imbalance_flag"] = (spread > settings["max_spread_m"]).astype(int)
    flags["burst_event_flag"] = (leak_active & (ts["leak_severity"] >= settings["burst_min_lps"])).astype(int)
    flags["leak_event_flag"] = (leak_active & (ts["leak_severity"] < settings["burst_min_lps"])).astype(int)
    return flags


def primary_labels(flags: pd.DataFrame, priority: list[str] | tuple[str, ...] = DEFAULT_PRIORITY) -> pd.Series:
    """Pick one label per row: the first label in `priority` whose flag is set, else 'normal'."""
    label_to_flag = {v: k for k, v in FLAG_TO_LABEL.items()}
    out = pd.Series("normal", index=flags.index, dtype="object")
    assigned = pd.Series(False, index=flags.index)
    for label in priority:
        if label == "normal":
            continue
        hit = flags[label_to_flag[label]].astype(bool) & ~assigned
        out[hit] = label
        assigned |= hit
    return out


def all_labels_text(flags: pd.DataFrame) -> pd.Series:
    """Multi-label text per row, e.g. 'low_pressure_risk;pressure_imbalance' (or 'normal')."""
    names = []
    for label in RISK_LABELS[1:]:
        flag = next(k for k, v in FLAG_TO_LABEL.items() if v == label)
        names.append((label, flags[flag].astype(bool)))
    rows = []
    for i in range(len(flags)):
        active = [label for label, mask in names if mask.iloc[i]]
        rows.append(";".join(active) if active else "normal")
    return pd.Series(rows, index=flags.index)


def label_timeseries(ts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return one labelled row per time step: ids, flags, spread, multi-label text and primary label."""
    settings = label_settings(cfg)
    flags = compute_flags(ts, settings)
    out = ts[["scenario_id", "timestamp"]].copy()
    out = pd.concat([out, flags], axis=1)
    out["risk_labels"] = all_labels_text(flags)
    out["risk_label"] = primary_labels(flags, settings["priority"])
    return out


def scenario_label(labelled: pd.DataFrame, priority: list[str] | tuple[str, ...] = DEFAULT_PRIORITY) -> str:
    """One label for a whole scenario: a label applies if it occurs at ANY time step; priority picks the winner."""
    label_to_flag = {v: k for k, v in FLAG_TO_LABEL.items()}
    for label in priority:
        if label != "normal" and bool(labelled[label_to_flag[label]].any()):
            return label
    return "normal"
