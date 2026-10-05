"""The saved Databricks run (evidence/databricks_phase3.html, a notebook export with outputs) is parsed and
checked, so the claim 'it passed on Databricks' rests on saved output, not on memory."""
import base64
import json
import re
import urllib.parse

import pandas as pd
import pytest

from config import ROOT

EVIDENCE = ROOT / "evidence" / "databricks_phase3.html"


@pytest.fixture(scope="module")
def nb():
    if not EVIDENCE.exists():
        pytest.fail(f"{EVIDENCE} missing: export the notebook as HTML with results")
    s = EVIDENCE.read_text(encoding="utf-8")
    m = re.search(r'__DATABRICKS_NOTEBOOK_MODEL\s*=\s*[\'"]([A-Za-z0-9+/=]+)[\'"]', s)
    assert m, "not a Databricks notebook HTML export"
    return json.loads(urllib.parse.unquote(base64.b64decode(m.group(1)).decode()))


def outputs(nb):
    texts, tables = [], []
    for c in nb["commands"]:
        r = c.get("results") or {}
        for item in r.get("data") or []:
            if isinstance(item, dict) and item.get("type") == "ansi":
                texts.append(item["data"])
            elif isinstance(item, dict) and item.get("type") == "table":
                tables.append(item)
    return "\n".join(texts), tables


def test_every_cell_finished_without_error(nb):
    assert len(nb["commands"]) == 20
    for c in nb["commands"]:
        assert c["state"] == "finished" and not c.get("error") and not c.get("errorSummary"), c["position"]


def test_notebook_code_in_the_export_is_the_repo_notebook(nb):
    """The run used the generated notebook (code cells identical to databricks/motion_pipeline_spark.py)."""
    local = (ROOT / "databricks" / "motion_pipeline_spark.py").read_text()
    ran = [c["command"] for c in nb["commands"] if not c["command"].startswith("%md")]
    for code in ran:
        if code.startswith("BASE ="):
            continue  # the one line the user edits
        assert code.strip() in local, code[:60]


def test_all_twelve_checks_passed_on_databricks(nb):
    text, _ = outputs(nb)
    assert "PHASE 3 CHECKS: ALL PASSED" in text and "12 checks run" in text
    assert "FAIL" not in text.replace("FAILURES", "")
    assert len(re.findall(r"^PASS ", text, flags=re.M)) == 12
    assert "raw keypoint rows: 83160" in text


def test_databricks_events_table_equals_the_pandas_events(nb):
    _, tables = outputs(nb)
    ev = next(t for t in tables if [c["name"] for c in t["schema"]][:3] == ["take", "event_idx", "label"] and len(t["data"]) == 101)
    cols = [c["name"] for c in ev["schema"]]
    spark = pd.DataFrame(ev["data"], columns=cols)
    ref = pd.read_csv(ROOT / "data" / "export" / "events.csv")
    m = spark.merge(ref, on=["take", "event_idx"], suffixes=("_dbx", "_pd"))
    assert len(m) == 101 == len(ref)
    assert (m["label_dbx"] == m["label_pd"]).all()
    assert (m["n_frames_dbx"] == m["n_frames_pd"]).all()
    assert ((m["start_s_dbx"] - m["start_s_pd"]).abs() < 1e-6).all() and ((m["end_s_dbx"] - m["end_s_pd"]).abs() < 1e-6).all()


def test_databricks_signal_agreement_with_pandas_is_as_reported(nb):
    _, tables = outputs(nb)
    ag = next(t for t in tables if [c["name"] for c in t["schema"]] == ["take", "n_joined", "r_speed", "r_aperture"])
    df = pd.DataFrame(ag["data"], columns=["take", "n_joined", "r_speed", "r_aperture"]).set_index("take")
    assert df["n_joined"].to_dict() == {"vid1": 976, "vid2": 1015, "vid3": 775, "vid4": 281, "vid5": 913}
    assert (df["r_speed"] > 0.97).all()
    assert (df.drop("vid4")["r_aperture"] > 0.99).all() and 0.85 < df.loc["vid4", "r_aperture"] < 0.9
