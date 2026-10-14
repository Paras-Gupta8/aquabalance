"""Phase 8 audit tests: final reports exist, say the right things, and README commands are real."""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "outputs" / "reports"
DISCLAIMER = "This is a simulation-based decision-support prototype, not autonomous real-world utility control."
VALID = "This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility."
FILES = ["final_results_summary.md", "limitations.md", "viva_preparation.md", "recommendation_effectiveness_report.md"]
OVERCLAIMS = [r"deployed in", r"field[- ]proven", r"real[- ]world performance (is|was) (good|excellent)", r"autonomously controls",
              r"proves? (it|this) works (in|on) (a )?real", r"guarantee"]


def text(name):
    p = REP / name
    if not p.exists():
        pytest.skip(f"{name} not generated")
    return p.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", FILES)
def test_reports_have_required_statements_and_no_overclaims(name):
    t = text(name)
    assert "EPANET" in t
    low = t.lower()
    for pat in OVERCLAIMS:
        assert not re.search(pat, low), f"{name}: overclaim pattern {pat}"


def test_summary_and_viva_contain_disclaimers():
    assert DISCLAIMER in text("final_results_summary.md")
    assert DISCLAIMER in text("viva_preparation.md")
    assert VALID in text("viva_preparation.md")


def test_viva_has_required_parts():
    t = text("viva_preparation.md")
    for h in ["30-second", "60-second", "sensors", "actuators", "limitation of simulation", "equal pressure"]:
        assert h.lower() in t.lower()
    assert len(re.findall(r"^\*\*\d+\. ", t, flags=re.M)) >= 10


def test_comparison_table_has_five_models_and_three_splits():
    p = REP / "model_comparison.csv"
    if not p.exists():
        pytest.skip("not generated")
    df = pd.read_csv(p)
    assert set(df["model"]) >= {"persistence", "random_forest", "gru", "lstm", "tcn"}
    assert set(df["split"]) == {"train", "val", "test"}


def test_readme_commands_point_to_existing_scripts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    scripts = set(re.findall(r"python scripts/(\w+\.py)", readme))
    assert scripts
    for s in scripts:
        assert (ROOT / "scripts" / s).exists(), s
    for s in ["train_sequence_models.py", "generate_final_reports.py", "seed_robustness.py"]:
        assert s in scripts


def test_readme_has_required_sections_and_no_stale_text():
    r = (ROOT / "README.md").read_text(encoding="utf-8")
    for h in ["Architecture", "Methodology", "Installation", "Commands for every phase", "Limitations",
              "Ethics and safety notes", "Future scope"]:
        assert f"## {h}" in r
    for k in ["graph-temporal", "reinforcement learning", "SCADA", "physical sensors", "operator-in-the-loop"]:
        assert k.lower() in r.lower()
    assert "not implemented" not in r
    assert DISCLAIMER in r and VALID in r


@pytest.mark.parametrize("script", ["train_sequence_models.py", "seed_robustness.py", "generate_final_reports.py"])
def test_phase8_scripts_have_help(script):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]


def test_final_figures_exist():
    d = ROOT / "outputs" / "final_figures"
    pngs = list(d.glob("*.png"))
    assert len(pngs) >= 9 and (d / "README.md").exists()
