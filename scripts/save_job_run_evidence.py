"""Save the evidence of one cloud-pipeline job run (M4) with the Databricks CLI, which must be signed in.

  python scripts/save_job_run_evidence.py <job_run_id>

Writes evidence/cloud_m4_job_run.json (the run record: state per task, the git commit each task used) and one HTML export per
task, evidence/cloud_m4_<task_key>.html, in the same format as a notebook exported from the UI.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
KEEP = ("job_id", "run_id", "run_name", "state", "start_time", "end_time", "git_source", "trigger", "run_page_url")
TASK_KEEP = ("task_key", "run_id", "state", "depends_on", "notebook_task", "git_source", "start_time", "end_time")


def cli(*args) -> dict:
    out = subprocess.run(["databricks", *args, "-o", "json"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main(run_id: str):
    run = cli("jobs", "get-run", run_id)
    record = {k: run.get(k) for k in KEEP}
    record["tasks"] = [{k: t.get(k) for k in TASK_KEEP} for t in run["tasks"]]
    (EVIDENCE / "cloud_m4_job_run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"run {run_id}: {run['state'].get('result_state')}")
    for t in run["tasks"]:  # the commit is recorded per task (git_source.git_snapshot.used_commit)
        commit = ((t.get("git_source") or {}).get("git_snapshot") or {}).get("used_commit")
        print(f"  {t['task_key']}: {t['state'].get('result_state')}, commit {commit}")
        if t["state"].get("life_cycle_state") != "TERMINATED":
            continue
        views = cli("jobs", "export-run", str(t["run_id"]), "--views-to-export", "ALL")["views"]
        html = next(v["content"] for v in views if v.get("type") == "NOTEBOOK")
        (EVIDENCE / f"cloud_m4_{t['task_key']}.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
