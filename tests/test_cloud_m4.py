"""Cloud path M4: the job definition (databricks.yml) and the behaviour it relies on.

The job runs m2_build_tables then m3_publish_snowflake from GitHub. These tests pin the order, the source, that M3 only runs
after M2 succeeded, and that each notebook fails its task when a check fails (otherwise a failed M2 check would still let M3
publish)."""
import ast
import re

import pytest
import yaml

from config import ROOT

BUNDLE = ROOT / "databricks.yml"
NOTEBOOKS = {"m2_build_tables": "M2", "m3_publish_snowflake": "M3"}


@pytest.fixture(scope="module")
def job():
    b = yaml.safe_load(BUNDLE.read_text())
    assert b["bundle"]["name"] == "motion-intent-cloud"
    return b["resources"]["jobs"]["motion_intent_cloud_pipeline"]


def test_job_runs_m2_then_m3_from_github_main(job):
    assert job["git_source"] == {"git_url": "https://github.com/kian-hekmat/track-hand-motion", "git_provider": "gitHub",
                                 "git_branch": "main"}
    assert job["max_concurrent_runs"] == 1
    tasks = {t["task_key"]: t for t in job["tasks"]}
    assert list(tasks) == ["m2_build_tables", "m3_publish_snowflake"]
    assert "depends_on" not in tasks["m2_build_tables"]
    assert tasks["m3_publish_snowflake"]["depends_on"] == [{"task_key": "m2_build_tables"}]
    for key, t in tasks.items():
        nt = t["notebook_task"]
        assert nt["source"] == "GIT" and nt["notebook_path"] == f"databricks/cloud/{key}"
        assert (ROOT / f"{nt['notebook_path']}.py").exists()
        assert not ({"new_cluster", "existing_cluster_id", "job_cluster_key"} & set(t))  # serverless


def test_nothing_local_is_uploaded(job):
    assert yaml.safe_load(BUNDLE.read_text())["sync"] == {"exclude": ["**"]}


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_notebook_fails_its_task_when_a_check_fails(name):
    src = (ROOT / "databricks" / "cloud" / f"{name}.py").read_text()
    tree = ast.parse(src)
    last = tree.body[-1]
    assert isinstance(last, ast.If) and ast.unparse(last.test) == "failures"
    assert isinstance(last.body[0], ast.Raise)
    assert re.search(rf'raise RuntimeError\(f"{NOTEBOOKS[name]}: ', src)
    # the raise comes after the results are logged and printed, so a failed run still leaves its evidence
    assert src.rindex("saveAsTable") < src.rindex("raise RuntimeError") and src.rindex("CHECKS:") < src.rindex("raise RuntimeError")


def test_raise_triggers_only_on_failures():
    """Execute the final block on its own: no failures -> nothing raised; one failure -> RuntimeError naming it."""
    src = (ROOT / "databricks" / "cloud" / "m2_build_tables.py").read_text()
    block = ast.unparse(ast.parse(src).body[-1])
    exec(block, {"failures": []})
    with pytest.raises(RuntimeError, match="M2: 1 check\\(s\\) failed: gold vid1: events"):
        exec(block, {"failures": ["gold vid1: events"]})


# ---- the first job run (2026-10-08, job run 121730247088002), saved by scripts/save_job_run_evidence.py ----
def _export(name):
    from tests.databricks_export import read_export

    return read_export(f"cloud_m4_{name}.html")


@pytest.fixture(scope="module")
def run_record():
    import json

    return json.loads((ROOT / "evidence" / "cloud_m4_job_run.json").read_text())


def test_job_run_succeeded_m2_then_m3_on_one_commit(run_record):
    assert run_record["state"]["result_state"] == "SUCCESS"
    tasks = {t["task_key"]: t for t in run_record["tasks"]}
    assert set(tasks) == set(NOTEBOOKS)
    commits = {t["git_source"]["git_snapshot"]["used_commit"] for t in tasks.values()}
    assert commits == {"7d9be28b47d4c866101cc65c98f1d0ab42408070"}
    assert all(t["state"]["result_state"] == "SUCCESS" for t in tasks.values())
    assert tasks["m3_publish_snowflake"]["depends_on"] == [{"task_key": "m2_build_tables"}]
    assert tasks["m3_publish_snowflake"]["start_time"] >= tasks["m2_build_tables"]["end_time"]


@pytest.mark.parametrize("name,n_checks", [("m2_build_tables", 33), ("m3_publish_snowflake", 31)])
def test_each_task_ran_the_committed_notebook_and_passed_every_check(name, n_checks, run_record):
    import subprocess

    commit = run_record["tasks"][0]["git_source"]["git_snapshot"]["used_commit"]
    local = subprocess.run(["git", "show", f"{commit}:databricks/cloud/{name}.py"], capture_output=True, text=True,
                           cwd=ROOT, check=True).stdout
    nb, text = _export(name)
    for c in nb["commands"]:
        assert c["state"] == "finished" and not c.get("error") and not c.get("errorSummary"), c["position"]
        if not c["command"].startswith("%md"):
            for line in c["command"].strip().splitlines():
                assert line in local or ("# MAGIC " + line) in local, line
    assert f"{NOTEBOOKS[name]} CHECKS: ALL PASSED" in text and f"{n_checks} checks logged" in text
    summary = text[text.index("status check / detail"):]
    statuses = re.findall(r"^(PASS|FAIL)\s+\S", summary, flags=re.M)
    assert len(statuses) == n_checks and set(statuses) == {"PASS"}
    if name == "m2_build_tables":  # the job ran a checkout of the commit, not the Git folder
        assert f"repo /Workspace/Repos/.internal/89448e6966_commits/{commit}" in text


def test_m3_gate_used_the_m2_run_from_the_same_job_run():
    _, m2 = _export("m2_build_tables")
    _, m3 = _export("m3_publish_snowflake")
    m2_run = re.search(r"^run ([0-9a-f-]{36}): 33 checks logged", m2, flags=re.M).group(1)
    assert re.search(rf"PASS\s+gate: latest M2 run: run {m2_run} ", m3)
