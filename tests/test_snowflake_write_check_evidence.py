"""The saved Databricks run of the Snowflake write check (evidence/databricks_snowflake_write_check.html, a notebook
export with outputs) is parsed and checked, so 'Databricks can write to Snowflake' rests on saved output."""
import base64
import json
import re
import urllib.parse

import pytest

from config import ROOT

EVIDENCE = ROOT / "evidence" / "databricks_snowflake_write_check.html"
NOTEBOOK = ROOT / "databricks" / "snowflake_write_check.py"


@pytest.fixture(scope="module")
def nb():
    if not EVIDENCE.exists():
        pytest.fail(f"{EVIDENCE} missing: export the notebook as HTML with results")
    s = EVIDENCE.read_text(encoding="utf-8")
    m = re.search(r'__DATABRICKS_NOTEBOOK_MODEL\s*=\s*[\'"]([A-Za-z0-9+/=]+)[\'"]', s)
    assert m, "not a Databricks notebook HTML export"
    return json.loads(urllib.parse.unquote(base64.b64decode(m.group(1)).decode()))


def output_text(nb):
    texts = []
    for c in nb["commands"]:
        data = (c.get("results") or {}).get("data")
        if isinstance(data, str):
            texts.append(data)
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and item.get("type") == "ansi":
                texts.append(item["data"])
    return "\n".join(texts)


def test_every_cell_finished_without_error(nb):
    for c in nb["commands"]:
        assert c["state"] == "finished" and not c.get("error") and not c.get("errorSummary"), c["position"]


def test_code_that_ran_is_the_repo_notebook(nb):
    local = NOTEBOOK.read_text()
    for c in nb["commands"]:
        code = c["command"]
        if code.startswith("%md"):
            continue
        lines = [l for l in code.strip().splitlines() if not l.startswith("SF_HOST = \"")]  # the one line the user edits
        for line in lines:
            assert line in local, line


def test_all_checks_passed_and_path_a_works(nb):
    text = output_text(nb)
    assert "SNOWFLAKE WRITE TEST: ALL PASSED" in text
    assert "PATH A WORKS" in text
    summary = text[text.index("status check / detail"):]
    statuses = re.findall(r"^(PASS|FAIL|SKIP|INFO)\s+(.+)$", summary, flags=re.M)
    assert not [c for s, c in statuses if s in ("FAIL", "SKIP")]
    passed = {c.strip() for s, c in statuses if s == "PASS"}
    assert {"network: Snowflake host", "credentials: private key from secret", "path A: Spark write (5 rows)",
            "path A: Spark read-back", "path A: events table (101 rows)", "path B: Python connector connect",
            "cross-check: Spark table read by Python connector", "path B: write_pandas (5 rows)"} <= passed
    assert "rows source 101 / Snowflake 101" in text and "per-label counts identical" in text
