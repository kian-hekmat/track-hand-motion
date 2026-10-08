"""Cloud path M2: the bronze -> silver -> gold chain (src/cloud/), run on local Spark from a landing folder laid out exactly
as the Databricks Volume will be. Every parity check in src.cloud.checks must pass here before the chain runs on Databricks."""
import shutil

import pytest

from config import GROUND_TRUTH_DIR, RAW_DIR
from src.cloud import checks


@pytest.fixture(scope="module")
def spark_tables(spark, tmp_path_factory):
    landing = tmp_path_factory.mktemp("landing")
    (landing / "raw").mkdir()
    (landing / "ground_truth").mkdir()
    for take in checks.TAKES:
        for suffix in ("_keypoints.csv", "_meta.json"):
            shutil.copy(RAW_DIR / f"{take}{suffix}", landing / "raw")
    shutil.copy(RAW_DIR / "out_of_frame_intervals.csv", landing / "raw")
    for p in GROUND_TRUTH_DIR.glob("take_*.csv"):
        shutil.copy(p, landing / "ground_truth")
    return {k: v.cache() for k, v in checks.build(spark, str(landing)).items()}


@pytest.fixture(scope="module")
def tables(spark_tables):
    return {k: v.toPandas() for k, v in spark_tables.items()}


def test_row_counts_through_the_chain(tables):
    assert len(tables["bronze.raw_keypoints"]) == 118_839          # 5,659 frames x 21 landmarks
    assert len(tables["silver.frames"]) == len(tables["gold.frame_labels"]) == 5_659
    assert len(tables["silver.signals"]) == 5_659                  # one 30 Hz sample per frame for these takes
    assert len(tables["gold.events"]) == 101 + 31 + 18             # vid1-5 canonical + vid6 + vid7


def test_every_parity_check_passes(tables):
    results = checks.compare(tables)
    failed = [(c, d) for c, s, d in results if s != "PASS"]
    assert not failed, failed
    assert len(results) == 7 + 2 + 7 * 3 + 1  # bronze per take, bronze files, (frames, signals, events) per take, scores


def test_a_wrong_model_is_caught(tables):
    """The check is not vacuous: relabel one take's model and its signals check must fail."""
    broken = dict(tables)
    s = tables["silver.signals"].copy()
    s.loc[s["take"] == "vid1", "model"] = "segmenter_v2.joblib"
    broken["silver.signals"] = s
    status = {c: st for c, st, _ in checks.compare(broken)}
    assert status["silver vid1: signals"] == "FAIL" and status["silver vid2: signals"] == "PASS"


def test_a_shifted_event_boundary_is_caught(tables):
    broken = dict(tables)
    e = tables["gold.events"].copy()
    e.loc[(e["take"] == "vid4") & (e["event_idx"] == 3), "start_s"] += 1e-9
    broken["gold.events"] = e
    assert {c: st for c, st, _ in checks.compare(broken)}["gold vid4: events"] == "FAIL"


def test_notebook_is_valid_and_uses_the_tested_code():
    import ast
    import re

    from config import ROOT

    src = (ROOT / "databricks" / "cloud" / "m2_build_tables.py").read_text()
    assert src.startswith("# Databricks notebook source\n")
    ast.parse(src)
    pins = re.search(r"^# MAGIC %pip install -q (.+)$", src, re.M).group(1).split()
    req = [l for l in (ROOT / "databricks" / "cloud" / "requirements-cloud.txt").read_text().splitlines()
           if l and not l.startswith("#")]
    assert pins == req
    # the notebook runs the same transforms and the same checks the tests above run
    for call in ("C.read_landing(", "C.build_frames(", "C.predict_signals(", "C.build_events(", "C.label_frames(",
                 "C.frame_scores(", "C.boundary_scores(", "checks.compare(", "checks.compare_scores(", "load_model_for_take("):
        assert call in src, call
    assert 'mode("overwrite")' in src and 'mode("append").saveAsTable(f"{CATALOG}.{SCHEMAS[\'gold\']}.run_log")' in src


# ---- the saved Databricks runs ----
# run1 (2026-10-08): the notebook at commit b1d6b6d (before M2b), 32 checks. The current run file is added when it exists.
RUNS = {"cloud_m2_build_tables_run1.html": ("b1d6b6d", 32), "cloud_m2_build_tables.html": (None, 33)}


def _databricks_run(fname):
    import base64
    import json
    import re
    import urllib.parse

    from config import ROOT

    s = (ROOT / "evidence" / fname).read_text(encoding="utf-8")
    m = re.search(r'__DATABRICKS_NOTEBOOK_MODEL\s*=\s*[\'"]([A-Za-z0-9+/=]+)[\'"]', s)
    assert m, "not a Databricks notebook HTML export"
    nb = json.loads(urllib.parse.unquote(base64.b64decode(m.group(1)).decode()))
    text = []
    for c in nb["commands"]:
        data = (c.get("results") or {}).get("data")
        if isinstance(data, str):
            text.append(data)
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and item.get("type") == "ansi":
                text.append(item["data"])
    return nb, "\n".join(text)


@pytest.mark.parametrize("fname", sorted(RUNS))
def test_databricks_run_finished_ran_the_repo_notebook_and_passed(fname):
    import re
    import subprocess

    from config import ROOT

    commit, n_expected = RUNS[fname]
    if not (ROOT / "evidence" / fname).exists():
        pytest.skip(f"{fname}: Databricks run not saved yet")
    nb, text = _databricks_run(fname)
    path = "databricks/cloud/m2_build_tables.py"
    local = (subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, cwd=ROOT, check=True).stdout
             if commit else (ROOT / path).read_text())
    for c in nb["commands"]:
        assert c["state"] == "finished" and not c.get("error") and not c.get("errorSummary"), c["position"]
        if not c["command"].startswith("%md"):
            for line in c["command"].strip().splitlines():
                assert line in local or ("# MAGIC " + line) in local, line
    assert "M2 CHECKS: ALL PASSED" in text and f"{n_expected} checks logged" in text
    summary = text[text.index("status check / detail"):]
    assert len(re.findall(r"^PASS\s", summary, flags=re.M)) == n_expected and not re.findall(r"^FAIL\s", summary, flags=re.M)
    for table, n in {"motion_bronze.raw_keypoints": 118839, "motion_silver.frames": 5659, "motion_silver.signals": 5659,
                     "motion_gold.events": 150, "motion_gold.frame_scores": 10}.items():
        assert f"wrote workspace.{table}: {n} rows" in text


# ---- M2b (full scores) and M3 (serving tables in the PIPELINE layout) ----
@pytest.fixture(scope="module")
def serving(spark_tables):
    return {k: v.toPandas() for k, v in checks.build_serving(spark_tables).items()}


def test_full_scores_match_the_reference(tables):
    (check, status, detail), = checks.compare_scores(tables["gold.scores"])
    assert status == "PASS", detail


def test_serving_tables_match_the_verified_exports(serving):
    results = checks.compare_serving(serving)
    assert [s for _, s, _ in results] == ["PASS"] * 7, [r for r in results if r[1] != "PASS"]


def test_serving_check_catches_a_changed_value(serving):
    broken = dict(serving)
    e = serving["events"].copy()
    e.loc[(e["take"] == "vid2") & (e["event_idx"] == 5), "n_frames"] += 1
    broken["events"] = e
    status = {c: s for c, s, _ in checks.compare_serving(broken)}
    assert status["serving events"] == "FAIL" and status["serving signals"] == "PASS"


# ---- M3: the Snowflake-side checks, run in DuckDB (PIPELINE = verified export of vid1-5, CLOUD = serving tables) ----
@pytest.fixture(scope="module")
def duck(serving):
    import duckdb

    from config import ROOT

    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA pipeline; CREATE SCHEMA cloud;")
    for name in checks.SERVING_KEYS:
        f = ROOT / "data" / "export" / (f"{name}.parquet" if name == "raw_keypoints" else f"{name}.csv")
        import pandas as pd
        ref = (pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, float_precision="round_trip")).rename(
            columns={"group": "take_group", "false": "false_boundaries"})
        con.register("ref_df", ref)
        con.execute(f"CREATE TABLE pipeline.{name} AS SELECT * FROM ref_df")
        con.unregister("ref_df")
        con.register("cloud_df", serving[name])
        con.execute(f"CREATE TABLE cloud.{name} AS SELECT * FROM cloud_df")
        con.unregister("cloud_df")
    return con


def _columns(con):
    return {n: [r[0] for r in con.execute(f"DESCRIBE pipeline.{n}").fetchall()] for n in checks.SERVING_KEYS}


def test_snowflake_parity_sql_passes_in_duckdb(duck):
    from src.cloud import snowflake_sql as Q

    res = Q.run_parity(lambda sql: duck.execute(sql).fetchdf(), _columns(duck), "pipeline.", "cloud.")
    assert [s for _, s, _ in res] == ["PASS"] * 7, [r for r in res if r[1] != "PASS"]


def test_snowflake_parity_sql_catches_a_changed_value(duck):
    from src.cloud import snowflake_sql as Q

    duck.execute("CREATE TABLE cloud.events_backup AS SELECT * FROM cloud.events")
    duck.execute("UPDATE cloud.events SET mean_speed = mean_speed + 1e-6 WHERE take = 'vid3' AND event_idx = 2")
    try:
        res = {c: (s, d) for c, s, d in Q.run_parity(lambda sql: duck.execute(sql).fetchdf(), _columns(duck),
                                                     "pipeline.", "cloud.")}
        assert res["snowflake parity events"][0] == "FAIL" and "mean_speed (1 rows)" in res["snowflake parity events"][1]
        assert res["snowflake parity signals"][0] == "PASS"
    finally:
        duck.execute("DROP TABLE cloud.events; ALTER TABLE cloud.events_backup RENAME TO events")


def test_queries_give_the_same_answers_in_both_schemas(duck):
    from src.cloud import snowflake_sql as Q

    res, cloud = Q.compare_queries(lambda sql: duck.execute(sql).fetchdf(), "USE pipeline", "USE cloud")
    assert [s for _, s, _ in res] == ["PASS"] * 5, res
    assert set(cloud["q2_most_ambiguous_takes"]["take"]) == set(checks.TAKES)  # cloud answers include vid6 and vid7
    duck.execute("USE memory.main")


def test_m3_notebook_is_valid_and_uses_the_tested_checks():
    import ast
    import re

    from config import ROOT

    src = (ROOT / "databricks" / "cloud" / "m3_publish_snowflake.py").read_text()
    assert src.startswith("# Databricks notebook source\n")
    ast.parse(src)
    assert re.search(r"^# MAGIC %pip install -q scikit-learn==1\.9\.1 ruptures==1\.1\.10$", src, re.M)
    for call in ("checks.build_serving(", "checks.compare_serving(", "Q.run_parity(", "Q.view_statements(",
                 "Q.compare_queries(", 'dbutils.secrets.get(SECRET_SCOPE, "snowflake_private_key")',
                 'dbutils.secrets.get(SECRET_SCOPE, "snowflake_host")'):
        assert call in src, call
    # publishes only into CLOUD; PIPELINE is only read
    assert '"sfSchema": SF_CLOUD' in src and "SF_PIPELINE}." in src
    assert not re.search(r"(INSERT|UPDATE|DELETE|CREATE|DROP|MERGE)[^\n]*SF_PIPELINE", src)
    # the serving tables are checked against the verified exports before anything is published
    assert src.index("compare_serving(") < src.index('format("snowflake")')


def test_tableau_views_build_on_the_cloud_tables(duck):
    from src.cloud import snowflake_sql as Q

    duck.execute("USE cloud")
    try:
        for stmt in Q.view_statements():
            duck.execute(stmt)
        n = {v: duck.execute(f"SELECT COUNT(*) FROM cloud.{v}").fetchone()[0]
             for v in ("v_tableau_phases", "v_tableau_signals", "v_tableau_accuracy")}
        assert n == {"v_tableau_phases": 150 + 145, "v_tableau_signals": 5659, "v_tableau_accuracy": 7}
    finally:
        duck.execute("USE memory.main")
