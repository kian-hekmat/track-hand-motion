# Databricks notebook source
# MAGIC %md
# MAGIC # Cloud path M2: build the bronze, silver and gold tables in Databricks
# MAGIC Reads the files uploaded to the Volume `/Volumes/workspace/motion_bronze/landing/`, runs the repo's code (`src/cloud/`,
# MAGIC imported from this Git folder) and writes Delta tables in Unity Catalog:
# MAGIC
# MAGIC | layer | table | what it holds |
# MAGIC |---|---|---|
# MAGIC | bronze | `workspace.motion_bronze.raw_keypoints`, `take_meta`, `ground_truth`, `out_of_frame_intervals` | the uploaded files, typed, with their source file name |
# MAGIC | silver | `workspace.motion_silver.frames` | one row per video frame (21 landmark rows pivoted with a conditional aggregation) |
# MAGIC | silver | `workspace.motion_silver.signals` | per 30 Hz sample: motion signals, phase probabilities, predicted label (the repo's Python, one take per group via `applyInPandas`) |
# MAGIC | gold | `workspace.motion_gold.events` | phase events (gaps-and-islands with window functions) |
# MAGIC | gold | `workspace.motion_gold.frame_labels`, `frame_scores` | predicted vs hand-labelled phase per frame (range joins), accuracy per take (`groupBy`/`agg`) |
# MAGIC | gold | `workspace.motion_gold.scores` | all scores incl. boundary recall/precision/timing, chance baselines, per cycle (inputs gathered per take with `collect_list`, then `src.evaluate` per take) |
# MAGIC | gold | `workspace.motion_gold.run_log` | one row per check per run, appended |
# MAGIC
# MAGIC Every table is overwritten on each run (the run is repeatable), except `run_log`, which is appended. After writing, the tables
# MAGIC are **read back from Delta** and compared with the verified local reference (`src/cloud/checks.py`, the same checks the
# MAGIC local tests run). Steps: `databricks/cloud/README.md`.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.9.1 ruptures==1.1.10

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings, repo code, schemas

# COMMAND ----------

import datetime
import os
import sys
import uuid
import zipfile
from pathlib import Path

CATALOG = "workspace"
LANDING = "/Volumes/workspace/motion_bronze/landing"
SCHEMAS = {"bronze": "motion_bronze", "silver": "motion_silver", "gold": "motion_gold"}
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
            if (p / "config.py").exists() and (p / "src" / "cloud" / "tables.py").exists():
                return p
    raise RuntimeError(f"repo root not found from {starts}; open this notebook from the Git folder and pull the latest commit")


REPO = find_repo_root()
sys.path.insert(0, str(REPO))
import pandas as pd

from src.cloud import checks
from src.cloud import tables as C

for schema in SCHEMAS.values():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
print(f"repo {REPO}\nrun {RUN_ID} at {RUN_AT:%Y-%m-%d %H:%M:%S} UTC\nlanding {LANDING}")
for sub in ("raw", "ground_truth"):
    names = sorted(f.name for f in dbutils.fs.ls(f"{LANDING}/{sub}"))
    print(f"{sub}/: {len(names)} files: {', '.join(names)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Make the repo code importable on the workers
# MAGIC `applyInPandas` runs on workers, which need `src/` and the pinned libraries. A probe checks this first; if the workers cannot
# MAGIC import `src`, the code is zipped from the Git folder and shipped with `spark.addArtifacts`, then probed again.

# COMMAND ----------


def probe(pdf):
    try:
        import ruptures
        import sklearn

        import src.cloud.tables  # noqa: F401
        detail = f"workers import src; scikit-learn {sklearn.__version__}, ruptures {ruptures.__version__}"
        return pd.DataFrame({"ok": [True], "detail": [detail]})
    except Exception as e:
        return pd.DataFrame({"ok": [False], "detail": [f"{type(e).__name__}: {e}"[:400]]})


def run_probe():
    return spark.range(1).groupBy("id").applyInPandas(probe, "ok boolean, detail string").collect()[0]


result = run_probe()
CODE_SHIPPING = "already importable on workers"
if not result.ok:
    print(f"first probe: {result.detail}; shipping the code")
    zip_path = "/tmp/motion_src.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(REPO / "config.py", "config.py")
        for p in (REPO / "src").rglob("*.py"):
            z.write(p, str(p.relative_to(REPO)))
    spark.addArtifacts(zip_path, pyfile=True)
    result = run_probe()
    CODE_SHIPPING = "shipped as a zip with spark.addArtifacts"
print(f"{'PASS' if result.ok else 'FAIL'}  code on workers ({CODE_SHIPPING}): {result.detail}")
assert result.ok, "workers cannot import the repo code; send this output"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bronze: the uploaded files

# COMMAND ----------


COUNTS = {}


def write(df, layer, name):
    full = f"{CATALOG}.{SCHEMAS[layer]}.{name}"
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full)
    COUNTS[full] = spark.table(full).count()
    print(f"wrote {full}: {COUNTS[full]} rows")
    return spark.table(full)


bronze = C.read_landing(spark, LANDING)
written = {}
for name, df in bronze.items():
    written[f"bronze.{name}"] = write(df, "bronze", name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Silver: frames, then signals and predictions (each take in parallel on the workers)
# MAGIC Models are loaded and hash-checked on the driver (`src.final.load_model_for_take`: vid1-5 their leave-one-take-out model,
# MAGIC other takes the full model) and travel to the workers inside the function.

# COMMAND ----------

from src.final import load_model_for_take

takes = sorted(r.take for r in written["bronze.take_meta"].select("take").collect())
assert takes == checks.TAKES, f"uploaded takes {takes}, expected {checks.TAKES}"
written["silver.frames"] = write(C.build_frames(written["bronze.raw_keypoints"]), "silver", "frames")

models, meta = {}, None
for take in takes:
    model, meta, name = load_model_for_take(take)
    models[take] = (model, name)
    print(f"{take}: {name}")
written["silver.signals"] = write(C.predict_signals(written["silver.frames"], models, meta), "silver", "signals")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Gold: events, labels per frame, scores

# COMMAND ----------

written["gold.events"] = write(C.build_events(written["silver.signals"], written["silver.frames"]), "gold", "events")
written["gold.frame_labels"] = write(
    C.label_frames(written["silver.frames"], written["gold.events"], written["bronze.ground_truth"],
                   written["bronze.out_of_frame_intervals"]), "gold", "frame_labels")
written["gold.frame_scores"] = write(C.frame_scores(written["gold.frame_labels"]), "gold", "frame_scores")
# M2b: all scores (boundary recall/precision/timing, chance baselines, per cycle), src.evaluate.score_take per take
written["gold.scores"] = write(
    C.boundary_scores(written["gold.events"], written["bronze.ground_truth"], written["silver.frames"],
                      written["bronze.out_of_frame_intervals"]), "gold", "scores")
display(written["gold.scores"].filter("scope = 'all'").orderBy("take"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Read every table back from Delta and compare with the verified local reference

# COMMAND ----------

tables_pd = {k: v.toPandas() for k, v in written.items()}
results = checks.compare(tables_pd) + checks.compare_scores(tables_pd["gold.scores"])
results.insert(0, ("code on workers", "PASS", f"{CODE_SHIPPING}: {result.detail}"))
log = pd.DataFrame([{"run_id": RUN_ID, "run_at": RUN_AT, "check": c, "status": s, "detail": d} for c, s, d in results])
spark.createDataFrame(log).write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMAS['gold']}.run_log")

print(f"{'status':6} check / detail")
for c, s, d in results:
    print(f"{s:6} {c}\n         {d}")
failures = [c for c, s, _ in results if s != "PASS"]
print()
print("row counts: " + ", ".join(f"{name} {n}" for name, n in COUNTS.items()))
print(f"run {RUN_ID}: {len(results)} checks logged to {CATALOG}.{SCHEMAS['gold']}.run_log")
print(f"M2 CHECKS: {'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
# A failed check fails the task, so a job run stops here and the next task does not run (M4).
if failures:
    raise RuntimeError(f"M2: {len(failures)} check(s) failed: {'; '.join(failures)}")
