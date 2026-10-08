# Databricks notebook source
# MAGIC %md
# MAGIC # Snowflake write test (first test for the cloud-only pipeline)
# MAGIC Answers one question before anything is built: **can this Databricks workspace write tables straight into Snowflake?**
# MAGIC
# MAGIC It checks, in order, and keeps going after a failure so every result is visible:
# MAGIC 1. **Network:** can this workspace reach the Snowflake host (Free Edition restricts outbound internet access)?
# MAGIC 2. **Credentials:** does the private key load from a Databricks secret?
# MAGIC 3. **Spark connector (path A):** write a 5-row table with `format("snowflake")`, read it back, compare every value.
# MAGIC 4. **Real table (path A):** write the 101-row `events.csv`, compare counts and sums computed *inside* Snowflake.
# MAGIC 5. **Python connector (path B, fallback):** connect, read the Spark table back from a second client, write with
# MAGIC    `write_pandas`, check the stage hosts from `SYSTEM$ALLOWLIST()`.
# MAGIC
# MAGIC It writes only to `MOTION_INTENT.CONNECTOR_TEST`, never to the verified `PIPELINE` schema.
# MAGIC Setup: `snowflake/10_connector_test_setup.sql` and `databricks/snowflake_write_check.md`. No credential is typed into this
# MAGIC notebook; the private key comes from a Databricks secret, and Databricks redacts secret values in output.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Settings: edit SF_HOST below, then Run all
# MAGIC Edit this in your Databricks copy only; the repo copy keeps the placeholder. The host is not a secret.

# COMMAND ----------

SF_HOST = "ORGNAME-ACCOUNTNAME.snowflakecomputing.com"  # EDIT: Snowsight > account > View account details > Server URL
SF_USER = "DATABRICKS_SVC"
SF_ROLE = "DATABRICKS_TEST_ROLE"
SF_WAREHOUSE = "COMPUTE_WH"
SECRET_SCOPE = "motion"
SECRET_KEY = "snowflake_private_key"
INPUTS_PATH = "/Volumes/workspace/default/motion"  # folder with events.csv (optional check)
DROP_TEST_TABLES = False  # True drops the three test tables at the end

SF_HOST = SF_HOST.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
INPUTS_PATH = INPUTS_PATH.rstrip("/")

SF_DATABASE = "MOTION_INTENT"
SF_SCHEMA = "CONNECTOR_TEST"  # fixed on purpose: this test must never write to the verified PIPELINE schema
SUFFIX = ".snowflakecomputing.com"
assert SF_HOST.endswith(SUFFIX) and not SF_HOST.startswith("ORGNAME-"), (
    f"Edit SF_HOST at the top of this cell to your Snowflake host (ends with {SUFFIX}); it is {SF_HOST!r}")
SF_ACCOUNT = SF_HOST[: -len(SUFFIX)]  # account identifier for the Python connector, e.g. ORGNAME-ACCOUNTNAME
print(f"Snowflake host {SF_HOST}, account {SF_ACCOUNT}, user {SF_USER}, role {SF_ROLE}, warehouse {SF_WAREHOUSE}")
print(f"Target {SF_DATABASE}.{SF_SCHEMA}; secret {SECRET_SCOPE}/{SECRET_KEY}; inputs {INPUTS_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Helpers and environment

# COMMAND ----------

import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

RESULTS = []  # (check, status, detail); status is PASS / FAIL / SKIP / INFO


def record(check, status, detail=""):
    RESULTS.append((check, status, detail))
    print(f"{status:4}  {check}" + (f": {detail}" if detail else ""))


def short_error(e, limit=1500):
    text = f"{type(e).__name__}: {e}"
    return text if len(text) <= limit else text[:limit] + " ...(truncated)"


def status_of(check):
    return next((s for c, s, _ in RESULTS if c == check), None)


record("environment", "INFO",
       f"Spark {spark.version}; DATABRICKS_RUNTIME_VERSION={os.environ.get('DATABRICKS_RUNTIME_VERSION', 'unset')}; "
       f"Python {platform.python_version()}; HTTPS_PROXY {'set' if os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') else 'not set'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Network: can this workspace reach Snowflake?
# MAGIC HTTPS first (it honours a proxy if the workspace uses one); if that fails, DNS and raw TCP say where it stops.
# MAGIC `pypi.org` matters for installing the Python connector; `example.com` is a reference for general internet access.

# COMMAND ----------


def probe(host, port=443, timeout=8):
    """Returns (reachable, detail). Any HTTP status (even 403) means the host was reached."""
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://{host}/", method="HEAD"), timeout=timeout)
        return True, "HTTPS ok"
    except urllib.error.HTTPError as e:
        return True, f"HTTPS reached (HTTP {e.code})"
    except Exception as https_error:
        https_detail = f"HTTPS failed ({type(https_error).__name__}: {https_error})"
    try:
        ip = socket.getaddrinfo(host, port)[0][4][0]
    except Exception as e:
        return False, f"{https_detail}; DNS failed ({type(e).__name__})"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return False, f"{https_detail}; DNS ok ({ip}) and TCP {port} ok"
    except Exception as e:
        return False, f"{https_detail}; DNS ok ({ip}) but TCP {port} failed ({type(e).__name__})"


ok, detail = probe(SF_HOST)
record("network: Snowflake host", "PASS" if ok else "FAIL", f"{SF_HOST}: {detail}")
for reference_host in ("pypi.org", "files.pythonhosted.org", "example.com"):
    ok, detail = probe(reference_host)
    record(f"network (reference): {reference_host}", "INFO", ("reachable, " if ok else "NOT reachable, ") + detail)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Credentials: private key from the Databricks secret
# MAGIC The printed fingerprint is public information. It must equal `RSA_PUBLIC_KEY_FP` from `DESC USER DATABRICKS_SVC` in
# MAGIC Snowflake; if it does not, the key in the secret does not match the public key set on the user.

# COMMAND ----------

import base64
import hashlib

PEM_BODY = None          # for the Spark connector: key body without BEGIN/END lines or newlines
PRIVATE_KEY_DER = None   # for the Python connector: unencrypted PKCS#8 DER bytes
try:
    scopes = [s.name for s in dbutils.secrets.listScopes()]
    if SECRET_SCOPE not in scopes:
        raise LookupError(f"secret scope {SECRET_SCOPE!r} not found (scopes visible: {scopes})")
    pem = dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY).strip()
    if "ENCRYPTED" in pem:
        raise ValueError("the key is passphrase-encrypted; generate it with -nocrypt as in the guide")
    PEM_BODY = "".join(line.strip() for line in pem.splitlines() if not line.startswith("-----"))
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(pem.encode(), password=None)
    PRIVATE_KEY_DER = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption())
    public_der = key.public_key().public_bytes(serialization.Encoding.DER,
                                               serialization.PublicFormat.SubjectPublicKeyInfo)
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(public_der).digest()).decode()
    record("credentials: private key from secret", "PASS", f"{key.key_size}-bit key; public key fingerprint {fingerprint}")
except Exception as e:
    record("credentials: private key from secret", "FAIL", short_error(e))

SF_OPTIONS = {
    "host": SF_HOST,  # serverless rejects the classic "sfURL" option (first run, 2026-10-07); it accepts "host"
    "sfUser": SF_USER,
    "pem_private_key": PEM_BODY,
    "sfRole": SF_ROLE,
    "sfWarehouse": SF_WAREHOUSE,
    "sfDatabase": SF_DATABASE,
    "sfSchema": SF_SCHEMA,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Path A: Spark connector, small table with every column type the pipeline uses
# MAGIC Includes a NULL double (the pipeline's undetected frames) and doubles that need all 17 digits, so a lossy transfer
# MAGIC would show up as a value mismatch rather than pass silently.

# COMMAND ----------

from pyspark.sql import types as T

TEST_SCHEMA = T.StructType([
    T.StructField("id", T.LongType(), False),
    T.StructField("label", T.StringType(), True),
    T.StructField("value", T.DoubleType(), True),
    T.StructField("flag", T.BooleanType(), True),
])
TEST_ROWS = [
    (0, "REST", 0.0, True),
    (1, "REACH", 1.25, False),
    (2, "GRASP", None, True),
    (3, "HOLD", 2.6533333333333333, False),
    (4, "RELEASE", 0.13333333333333333, True),
]


def normalise(row_dict):
    """Snowflake returns upper-case column names and NUMBER for integers; compare as plain Python values."""
    d = {k.lower(): v for k, v in row_dict.items()}
    return (int(d["id"]), d["label"], None if d["value"] is None else float(d["value"]),
            None if d["flag"] is None else bool(d["flag"]))


def compare_rows(got, expected=TEST_ROWS):
    if len(got) != len(expected):
        return False, f"{len(got)} rows back, expected {len(expected)}"
    mismatches = [(g, e) for g, e in zip(got, expected) if g != e]
    return (not mismatches), ("all values identical" if not mismatches else f"mismatches: {mismatches}")


if PEM_BODY is None:
    record("path A: Spark write (5 rows)", "SKIP", "no private key")
    record("path A: Spark read-back", "SKIP", "no private key")
else:
    try:
        t0 = time.time()
        (spark.createDataFrame(TEST_ROWS, TEST_SCHEMA)
         .write.format("snowflake").options(**SF_OPTIONS)
         .option("dbtable", "SPARK_WRITE_TEST").mode("overwrite").save())
        record("path A: Spark write (5 rows)", "PASS", f"wrote {SF_DATABASE}.{SF_SCHEMA}.SPARK_WRITE_TEST in {time.time() - t0:.1f} s")
    except Exception as e:
        record("path A: Spark write (5 rows)", "FAIL", short_error(e))
    if status_of("path A: Spark write (5 rows)") == "PASS":
        try:
            back = (spark.read.format("snowflake").options(**SF_OPTIONS)
                    .option("query", "SELECT * FROM SPARK_WRITE_TEST ORDER BY ID").load().collect())
            ok, detail = compare_rows([normalise(r.asDict()) for r in back])
            record("path A: Spark read-back", "PASS" if ok else "FAIL", detail)
        except Exception as e:
            record("path A: Spark read-back", "FAIL", short_error(e))
    else:
        record("path A: Spark read-back", "SKIP", "write failed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Path A: a real pipeline table (`events.csv`, 101 rows)
# MAGIC The count, total duration and per-label counts are computed **inside Snowflake** (a query, not a table scan back into
# MAGIC Spark) and compared with the same numbers computed in Spark from the source file. `group` is renamed `take_group`
# MAGIC because it is a reserved word in Snowflake (as in Phase 4).

# COMMAND ----------

from pyspark.sql import functions as F

events_path = f"{INPUTS_PATH}/events.csv"
try:
    dbutils.fs.ls(events_path)
    events_available = True
except Exception:
    events_available = False

if status_of("path A: Spark write (5 rows)") != "PASS":
    record("path A: events table (101 rows)", "SKIP", "small Spark write did not pass")
elif not events_available:
    record("path A: events table (101 rows)", "SKIP", f"{events_path} not found (optional; upload it to run this check)")
else:
    try:
        events = (spark.read.option("header", True).option("inferSchema", True).csv(events_path)
                  .withColumnRenamed("group", "take_group"))
        src_n = events.count()
        src_dur = events.agg(F.sum("duration_s")).first()[0]
        src_labels = {r["label"]: r["count"] for r in events.groupBy("label").count().collect()}
        (events.write.format("snowflake").options(**SF_OPTIONS)
         .option("dbtable", "EVENTS_WRITE_TEST").mode("overwrite").save())

        def sf_query(sql):
            return spark.read.format("snowflake").options(**SF_OPTIONS).option("query", sql).load().collect()

        totals = sf_query("SELECT COUNT(*) AS N, SUM(DURATION_S) AS DUR FROM EVENTS_WRITE_TEST")[0]
        sf_labels = {r["LABEL"]: int(r["N"]) for r in sf_query("SELECT LABEL, COUNT(*) AS N FROM EVENTS_WRITE_TEST GROUP BY LABEL")}
        dur_diff = abs(float(totals["DUR"]) - src_dur)
        ok = int(totals["N"]) == src_n and dur_diff < 1e-9 and sf_labels == src_labels
        record("path A: events table (101 rows)", "PASS" if ok else "FAIL",
               f"rows source {src_n} / Snowflake {int(totals['N'])}; total duration difference {dur_diff:.2e} s; "
               f"per-label counts {'identical' if sf_labels == src_labels else f'differ: source {src_labels}, Snowflake {sf_labels}'}")
    except Exception as e:
        record("path A: events table (101 rows)", "FAIL", short_error(e))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Path B (fallback): Python connector
# MAGIC Installs `snowflake-connector-python` if it is missing (needs PyPI access), then: reads the Spark-written table from
# MAGIC this second client (proves the rows are really in Snowflake), writes a table with `write_pandas`, and checks the
# MAGIC stage hosts listed by `SYSTEM$ALLOWLIST()` (the cloud storage both connectors upload through).

# COMMAND ----------

import importlib

conn = None


def ensure_python_connector():
    try:
        import snowflake.connector  # noqa: F401
        return "already installed"
    except ImportError:
        pass
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "snowflake-connector-python[pandas]"],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError("pip install failed: " + (r.stderr or r.stdout)[-800:])
    importlib.invalidate_caches()
    import snowflake.connector  # noqa: F401
    return "installed with pip"


if PRIVATE_KEY_DER is None:
    record("path B: Python connector connect", "SKIP", "no private key")
else:
    try:
        how = ensure_python_connector()
        import snowflake.connector

        conn = snowflake.connector.connect(account=SF_ACCOUNT, user=SF_USER, private_key=PRIVATE_KEY_DER,
                                           role=SF_ROLE, warehouse=SF_WAREHOUSE, database=SF_DATABASE, schema=SF_SCHEMA,
                                           login_timeout=60)
        version, role, warehouse = conn.cursor().execute(
            "SELECT CURRENT_VERSION(), CURRENT_ROLE(), CURRENT_WAREHOUSE()").fetchone()
        record("path B: Python connector connect", "PASS",
               f"connector {how}; Snowflake {version}; role {role}; warehouse {warehouse}")
    except Exception as e:
        record("path B: Python connector connect", "FAIL", short_error(e))

# COMMAND ----------

if conn is None:
    record("cross-check: Spark table read by Python connector", "SKIP", "no Python connection")
    record("path B: write_pandas (5 rows)", "SKIP", "no Python connection")
    record("network: stage hosts from SYSTEM$ALLOWLIST", "SKIP", "no Python connection")
else:
    # Cross-check: the Spark-written rows, read by a different client.
    if status_of("path A: Spark write (5 rows)") == "PASS":
        try:
            cur = conn.cursor(snowflake.connector.DictCursor)
            got = [normalise(r) for r in cur.execute("SELECT * FROM SPARK_WRITE_TEST ORDER BY ID").fetchall()]
            ok, detail = compare_rows(got)
            record("cross-check: Spark table read by Python connector", "PASS" if ok else "FAIL", detail)
        except Exception as e:
            record("cross-check: Spark table read by Python connector", "FAIL", short_error(e))
    else:
        record("cross-check: Spark table read by Python connector", "SKIP", "Spark write did not pass")

    # Path B write. Nullable Float64 so the missing value is a real NULL, not NaN (the same NaN/NULL trap as Phase 3).
    try:
        import pandas as pd
        from snowflake.connector.pandas_tools import write_pandas

        pdf = pd.DataFrame(TEST_ROWS, columns=["ID", "LABEL", "VALUE", "FLAG"]).astype(
            {"ID": "int64", "LABEL": "string", "VALUE": "Float64", "FLAG": "boolean"})
        success, n_chunks, n_rows, _ = write_pandas(conn, pdf, "PYTHON_WRITE_TEST", auto_create_table=True,
                                                    overwrite=True, quote_identifiers=False)
        cur = conn.cursor(snowflake.connector.DictCursor)
        got = [normalise(r) for r in cur.execute("SELECT * FROM PYTHON_WRITE_TEST ORDER BY ID").fetchall()]
        ok, detail = compare_rows(got)
        record("path B: write_pandas (5 rows)", "PASS" if (success and ok) else "FAIL",
               f"write_pandas success={success}, {n_rows} rows; read-back: {detail}")
    except Exception as e:
        record("path B: write_pandas (5 rows)", "FAIL", short_error(e))

    # Stage hosts: the connectors move data through cloud storage, which must also be reachable.
    try:
        allowlist = json.loads(conn.cursor().execute("SELECT SYSTEM$ALLOWLIST()").fetchone()[0])
        stage_hosts = sorted({entry["host"] for entry in allowlist if entry.get("type") == "STAGE"})
        if not stage_hosts:
            record("network: stage hosts from SYSTEM$ALLOWLIST", "INFO", f"no STAGE entries listed ({len(allowlist)} entries)")
        for host in stage_hosts:
            ok, detail = probe(host)
            record("network: stage hosts from SYSTEM$ALLOWLIST", "PASS" if ok else "FAIL", f"{host}: {detail}")
    except Exception as e:
        record("network: stage hosts from SYSTEM$ALLOWLIST", "INFO",
               "could not read SYSTEM$ALLOWLIST with this role (run it in Snowsight as ACCOUNTADMIN instead): " + short_error(e, 300))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Optional cleanup (DROP_TEST_TABLES in the settings cell)

# COMMAND ----------

if DROP_TEST_TABLES and conn is not None:
    for table in ("SPARK_WRITE_TEST", "EVENTS_WRITE_TEST", "PYTHON_WRITE_TEST"):
        conn.cursor().execute(f"DROP TABLE IF EXISTS {SF_DATABASE}.{SF_SCHEMA}.{table}")
    print("Dropped the three test tables.")
elif DROP_TEST_TABLES:
    print("No Python connection, so nothing was dropped. Drop the tables in Snowsight if needed.")
else:
    print("Test tables kept (set DROP_TEST_TABLES = True in the settings cell to drop them on the next run).")
if conn is not None:
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary and verdict

# COMMAND ----------

print(f"{'status':6} check / detail")
for check, status, detail in RESULTS:
    print(f"{status:6} {check}" + (f"\n         {detail}" if detail else ""))

path_a = all(status_of(c) == "PASS" for c in ("path A: Spark write (5 rows)", "path A: Spark read-back"))
path_a_events = status_of("path A: events table (101 rows)")
path_b = status_of("path B: write_pandas (5 rows)") == "PASS"
reachable = status_of("network: Snowflake host") == "PASS"
failures = [c for c, s, _ in RESULTS if s == "FAIL"]

print()
if path_a:
    verdict = ("PATH A WORKS: the Spark connector writes to and reads from Snowflake from this workspace."
               + ("" if path_a_events == "PASS" else f" (events-table check: {path_a_events})"))
elif path_b:
    verdict = "PATH B ONLY: the Spark connector failed but the Python connector can write. See the Spark error above."
elif not reachable:
    verdict = ("SNOWFLAKE NOT REACHABLE from this workspace (outbound access restricted). Options: verify the account to "
               "unlock outbound access and re-run, or keep a manual upload step.")
else:
    verdict = "REACHABLE BUT NO WRITE PATH WORKED: see the errors above (often the key, the role grants or the account identifier)."
print(verdict)
print(f"SNOWFLAKE WRITE TEST: {'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
