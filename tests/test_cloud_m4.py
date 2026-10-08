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
