"""Cloud-path milestone M1: the local reference (data/cloud_reference/) and the Databricks check notebook
(databricks/cloud/m1_environment_check.py).

- The reference must still equal what the code produces locally (otherwise the Databricks comparison is against stale data).
- The notebook's code cells are executed here with a stand-in for dbutils. On the machine that made the reference every check
  must pass, so a failure on Databricks is a real environment difference, not a bug in the notebook.
- The notebook's pip pins must equal databricks/cloud/requirements-cloud.txt and the versions the model was frozen with."""
import json
import re

import pytest

from config import HOLDOUT_TAKES, ROOT, TAKE_GROUPS
from src.final import load_meta

REF_DIR = ROOT / "data" / "cloud_reference"
NOTEBOOK = ROOT / "databricks" / "cloud" / "m1_environment_check.py"
REQUIREMENTS = ROOT / "databricks" / "cloud" / "requirements-cloud.txt"
TAKES = sorted(TAKE_GROUPS) + sorted(HOLDOUT_TAKES)


@pytest.mark.parametrize("take", TAKES)
def test_reference_equals_current_local_output(take):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_cloud_reference import reference

    ev, sig, post, name = reference(take)
    assert name == json.loads((REF_DIR / "manifest.json").read_text())["takes"][take]["model"]
    for kind, df in (("signals", sig), ("posteriors", post), ("events", ev)):
        assert df.to_csv(index=False) == (REF_DIR / f"{take}_{kind}.csv").read_text(), (take, kind)


def test_notebook_pins_match_requirements_and_frozen_model():
    pins = dict(line.split("==") for line in REQUIREMENTS.read_text().splitlines() if line and not line.startswith("#"))
    src = NOTEBOOK.read_text()
    m = re.search(r"^# MAGIC %pip install -q (.+)$", src, re.M)
    assert m and dict(p.split("==") for p in m.group(1).split()) == pins
    frozen = load_meta()["libraries"]
    assert pins == {lib: frozen[lib].removeprefix("v") for lib in ("scikit-learn", "ruptures")}  # ruptures reports "v1.1.10"


class _NoContextDbutils:
    """Stand-in for dbutils: no notebook context, so the notebook falls back to the working directory."""

    class notebook:
        class entry_point:
            @staticmethod
            def getDbutils():
                raise RuntimeError("not on Databricks")


def test_notebook_code_passes_on_the_reference_machine(monkeypatch, capsys):
    source = NOTEBOOK.read_text()
    assert source.startswith("# Databricks notebook source\n")
    cells = [c.strip() for c in source.split("# COMMAND ----------")]
    code = [c for c in cells if c and not c.startswith("# MAGIC") and not c.startswith("# Databricks notebook source")
            and c != "dbutils.library.restartPython()"]
    monkeypatch.chdir(ROOT / "databricks" / "cloud")  # where a Git-folder notebook runs
    ns = {"dbutils": _NoContextDbutils}
    for cell in code:
        exec(compile(cell, str(NOTEBOOK), "exec"), ns)
    out = capsys.readouterr().out
    assert "M1 CHECKS: ALL PASSED" in out, out[-3000:]
    assert f"events identical for {len(TAKES)} of {len(TAKES)} takes" in out


# ---- the saved Databricks run (evidence/cloud_m1_environment_check.html) ----
@pytest.fixture(scope="module")
def run():
    from tests.databricks_export import read_export

    return read_export("cloud_m1_environment_check.html")


def test_databricks_run_finished_and_ran_the_repo_notebook(run):
    nb, _ = run
    local = NOTEBOOK.read_text()
    for c in nb["commands"]:
        assert c["state"] == "finished" and not c.get("error") and not c.get("errorSummary"), c["position"]
        if not c["command"].startswith("%md"):
            for line in c["command"].strip().splitlines():
                assert line.replace("%pip", "# MAGIC %pip") in local or line in local, line


def test_databricks_run_passed_with_identical_events_for_every_take(run):
    _, text = run
    assert "M1 CHECKS: ALL PASSED" in text and f"events identical for {len(TAKES)} of {len(TAKES)} takes" in text
    summary = text[text.index("status check / detail"):]
    assert not re.findall(r"^(FAIL|SKIP)\b", summary, flags=re.M)
    for take in TAKES:
        assert re.search(rf"^PASS\s+{take}: events$", summary, flags=re.M), take
        assert re.search(rf"most-likely phase agrees on 1\.0000 of \d+ samples", summary)
    assert "PASS   library scikit-learn" in summary and "PASS   library ruptures" in summary
