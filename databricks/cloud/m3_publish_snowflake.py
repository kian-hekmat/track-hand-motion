# Databricks notebook source
# MAGIC %md
# MAGIC # Cloud path M3: publish the gold tables to Snowflake and check them there
# MAGIC 1. **Gate:** the latest M2 run in `workspace.motion_gold.run_log` must have passed every check, including the full scores.
# MAGIC 2. **Serving tables:** the M2 Delta tables reshaped into exactly the layout of the verified Snowflake `PIPELINE` tables
# MAGIC    (`takes`, `events`, `ground_truth`, `signals`, `frames`, `scores`, `raw_keypoints`), compared with the verified exports
# MAGIC    in this Git folder **before** anything is published.
# MAGIC 3. **Publish** with the Spark Snowflake connector into `MOTION_INTENT.CLOUD` (overwrite: the run is repeatable).
# MAGIC 4. **Check inside Snowflake** (Python connector): row counts equal what Databricks wrote; every verified `PIPELINE` row has an
# MAGIC    equal `CLOUD` row (SQL from `src/cloud/snowflake_sql.py`); the Tableau views are created in `CLOUD`; the five queries in
# MAGIC    `queries.sql` give the same answers in both schemas for vid1-5.
# MAGIC
# MAGIC Credentials: the key-pair secret from the connector test (`motion/snowflake_private_key`) and the host in
# MAGIC `motion/snowflake_host`. Nothing is typed into this notebook.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.9.1 ruptures==1.1.10

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings, repo code, gate on the latest M2 run

# COMMAND ----------

import datetime
import os
import sys
import uuid
from pathlib import Path

CATALOG = "workspace"
SECRET_SCOPE = "motion"
SF_DATABASE, SF_CLOUD, SF_PIPELINE = "MOTION_INTENT", "CLOUD", "PIPELINE"
SF_ROLE, SF_WAREHOUSE, SF_USER = "DATABRICKS_PIPELINE_ROLE", "COMPUTE_WH", "DATABRICKS_SVC"
RUN_ID = str(uuid.uuid4())
RUN_AT = datetime.datetime.now(datetime.timezone.utc)


def find_repo_root():
    starts = [Path(os.getcwd())]
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        starts.append(Path("/Workspace" + nb_path).parent)
    except Exception:
        pass
    for start in starts:
        for p in (start, *start.parents):
            if (p / "config.py").exists() and (p / "src" / "cloud" / "snowflake_sql.py").exists():
                return p
    raise RuntimeError(f"repo root not found from {starts}; open this notebook from the Git folder and pull the latest commit")


REPO = find_repo_root()
sys.path.insert(0, str(REPO))
import pandas as pd

from src.cloud import checks
from src.cloud import snowflake_sql as Q

RESULTS = []


def record(check, status, detail):
    RESULTS.append((check, status, detail))
    print(f"{status:4}  {check}: {detail}")


log = spark.table(f"{CATALOG}.motion_gold.run_log").toPandas()
m2_runs = log[log["check"] == "gold: frame scores"]["run_id"].unique()  # a check only M2 runs have
log = log[log["run_id"].isin(m2_runs)]
last = log[log["run_at"] == log["run_at"].max()]
m2_ok = (last["status"] == "PASS").all() and last["check"].str.startswith("gold: scores").any()
record("gate: latest M2 run", "PASS" if m2_ok else "FAIL",
       f"run {last['run_id'].iloc[0]} at {last['run_at'].iloc[0]}: {(last['status'] == 'PASS').sum()} of {len(last)} checks passed"
       + ("" if last["check"].str.startswith("gold: scores").any() else "; it has no full-scores check (re-run M2 first)"))
assert m2_ok, "re-run databricks/cloud/m2_build_tables.py until it passes, then run this notebook"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Serving tables, checked against the verified exports before publishing

# COMMAND ----------

names = {"bronze.take_meta": "motion_bronze.take_meta", "bronze.ground_truth": "motion_bronze.ground_truth",
         "bronze.raw_keypoints": "motion_bronze.raw_keypoints", "silver.frames": "motion_silver.frames",
         "silver.signals": "motion_silver.signals", "gold.events": "motion_gold.events", "gold.scores": "motion_gold.scores"}
m2 = {k: spark.table(f"{CATALOG}.{v}") for k, v in names.items()}
serving = checks.build_serving(m2)
serving_pd = {k: v.toPandas() for k, v in serving.items()}
for c, s, d in checks.compare_serving(serving_pd):
    record(c, s, d)
assert all(s == "PASS" for _, s, _ in RESULTS), "serving tables differ from the verified exports; nothing was published"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Publish to `MOTION_INTENT.CLOUD` with the Spark connector

# COMMAND ----------

pem = dbutils.secrets.get(SECRET_SCOPE, "snowflake_private_key").strip()
SF_HOST = dbutils.secrets.get(SECRET_SCOPE, "snowflake_host").strip().removeprefix("https://").rstrip("/")
SF_OPTIONS = {"host": SF_HOST, "sfUser": SF_USER, "pem_private_key": "".join(l for l in pem.splitlines() if not l.startswith("-----")),
              "sfRole": SF_ROLE, "sfWarehouse": SF_WAREHOUSE, "sfDatabase": SF_DATABASE, "sfSchema": SF_CLOUD}

published = {}
for name, df in serving.items():
    df.write.format("snowflake").options(**SF_OPTIONS).option("dbtable", name.upper()).mode("overwrite").save()
    published[name] = len(serving_pd[name])
    print(f"published {SF_DATABASE}.{SF_CLOUD}.{name.upper()}: {published[name]} rows")
runs = pd.DataFrame([{"run_id": RUN_ID, "run_at": RUN_AT.isoformat(), "table_name": n.upper(), "rows_written": r}
                     for n, r in published.items()])
(spark.createDataFrame(runs).write.format("snowflake").options(**SF_OPTIONS).option("dbtable", "PUBLISH_RUNS")
 .mode("append").save())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Checks inside Snowflake

# COMMAND ----------

import snowflake.connector
from cryptography.hazmat.primitives import serialization

key = serialization.load_pem_private_key(pem.encode(), password=None)
conn = snowflake.connector.connect(
    account=SF_HOST.removesuffix(".snowflakecomputing.com"), user=SF_USER, role=SF_ROLE, warehouse=SF_WAREHOUSE,
    database=SF_DATABASE, schema=SF_CLOUD,
    private_key=key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))


def fetch_df(sql):
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=[d[0] for d in cur.description]) if cur.description else pd.DataFrame()


# a) the rows Databricks wrote are the rows Snowflake holds (manifest written in this run)
counts = fetch_df(f"SELECT table_name, rows_written FROM {SF_DATABASE}.{SF_CLOUD}.PUBLISH_RUNS WHERE run_id = '{RUN_ID}'")
for row in counts.itertuples():
    n = int(fetch_df(f"SELECT COUNT(*) AS n FROM {SF_DATABASE}.{SF_CLOUD}.{row.TABLE_NAME}").iloc[0, 0])
    record(f"snowflake rows {row.TABLE_NAME}", "PASS" if n == int(row.ROWS_WRITTEN) else "FAIL",
           f"{n} rows in Snowflake, {int(row.ROWS_WRITTEN)} written by Databricks (run {RUN_ID[:8]})")
record("snowflake manifest", "PASS" if len(counts) == len(serving) else "FAIL",
       f"{len(counts)} tables in PUBLISH_RUNS for this run, {len(serving)} published")

# b) every verified PIPELINE row has an equal CLOUD row (same keys and tolerances as the Python checks)
cols = fetch_df(f"""SELECT LOWER(table_name) AS t, LOWER(column_name) AS c FROM {SF_DATABASE}.INFORMATION_SCHEMA.COLUMNS
                    WHERE table_schema = '{SF_PIPELINE}' ORDER BY table_name, ordinal_position""")
columns_by_table = {n: cols[cols["T"] == n]["C"].tolist() for n in checks.SERVING_KEYS}
for c, s, d in Q.run_parity(fetch_df, columns_by_table, f"{SF_DATABASE}.{SF_PIPELINE}.", f"{SF_DATABASE}.{SF_CLOUD}."):
    record(c, s, d)

# c) the Tableau views, created in CLOUD from snowflake/05_tableau_views.sql unchanged
fetch_df(f"USE SCHEMA {SF_DATABASE}.{SF_CLOUD}")
for stmt in Q.view_statements():
    fetch_df(stmt)
expected_views = {"V_TABLEAU_PHASES": published["events"] + published["ground_truth"],
                  "V_TABLEAU_SIGNALS": published["signals"], "V_TABLEAU_ACCURACY": len(checks.TAKES)}
for view, n_expected in expected_views.items():
    n = int(fetch_df(f"SELECT COUNT(*) FROM {SF_DATABASE}.{SF_CLOUD}.{view}").iloc[0, 0])
    record(f"snowflake view {view}", "PASS" if n == n_expected else "FAIL", f"{n} rows, expected {n_expected}")

# d) queries.sql, unchanged, in both schemas
query_checks, cloud_answers = Q.compare_queries(fetch_df, f"USE SCHEMA {SF_DATABASE}.{SF_PIPELINE}",
                                                f"USE SCHEMA {SF_DATABASE}.{SF_CLOUD}")
for c, s, d in query_checks:
    record(c, s, d)
conn.close()
for qname, df in cloud_answers.items():
    print(f"\n{qname} (CLOUD, all takes):")
    display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary (also appended to `workspace.motion_gold.run_log`)

# COMMAND ----------

log_rows = pd.DataFrame([{"run_id": RUN_ID, "run_at": RUN_AT, "check": c, "status": s, "detail": d} for c, s, d in RESULTS])
spark.createDataFrame(log_rows).write.mode("append").saveAsTable(f"{CATALOG}.motion_gold.run_log")
print(f"{'status':6} check / detail")
for c, s, d in RESULTS:
    print(f"{s:6} {c}\n         {d}")
failures = [c for c, s, _ in RESULTS if s != "PASS"]
print(f"\nrun {RUN_ID}: {len(RESULTS)} checks logged")
print(f"M3 CHECKS: {'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
