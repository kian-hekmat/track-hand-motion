# Phase 3: running the Spark notebook in Databricks

`motion_pipeline_spark.py` is a Databricks notebook (source format) generated from `spark_transforms.py`
(`python scripts/build_databricks_notebook.py`). It has been run **locally** on Spark 3.5.5 (`tests/test_spark.py`) and **on Databricks** (Free Edition, serverless):
all 12 checks passed; evidence in `evidence/databricks_phase3.html`, asserted by `tests/test_databricks_evidence.py`.

## What you do

1. **Files to upload** (from `data/export/`): `raw_keypoints.parquet`, `signals.csv`, `events.csv`, `frames.csv`,
   `takes.csv` (about 5 MB in total).
2. **Upload them into one folder.** Pick the option your workspace has:
   - *Classic Community Edition (DBFS):* Catalog/Data -> Create -> Upload File -> choose the DBFS tab, target
     directory `/FileStore/tables/motion`. (If there is no DBFS option, an admin setting "DBFS File Browser" must be
     enabled under Settings -> Advanced.) Path to use: `dbfs:/FileStore/tables/motion`.
   - *Free Edition / Unity Catalog:* Catalog -> workspace -> default -> Create -> Volume (name it `motion`), then
     upload the five files into the volume. Path to use: `/Volumes/workspace/default/motion`.
3. **Import the notebook:** Workspace -> Import -> select `databricks/motion_pipeline_spark.py`. It opens as a
   notebook with markdown cells.
4. **Attach compute:** create/start a cluster (Community Edition) or use serverless (Free Edition).
5. **Set `BASE`** in the first code cell to the path from step 2.
6. **Run all.** The last lines of the checks cell should read
   `PHASE 3 CHECKS: ALL PASSED` and `12 checks run`. Any `FAIL` line is a real finding: do not edit around it; send it to me.
7. **Save the evidence:** File -> Export -> HTML (with results) and save it as `evidence/databricks_phase3.html` in
   the repo. Also screenshot or note the `agree` table (correlations) and the Spark events table.
8. Optional: download the output (`display(events)` -> Download CSV, or the CSV folder written under `BASE/spark_output`).

## What to send me

The `PASS`/`FAIL` lines, the Databricks runtime / compute type (shown in the cluster or compute page), and any error text.
Things that might differ from my local run (Spark 3.5.5, Java 11): runtime version, `saveAsTable` permissions on your
workspace (the notebook catches that and still writes CSV), and slow first execution.

## What the notebook checks

Row counts (raw = 21 x frames; Spark frames/signals = pandas frames = `takes.n_frames`), undetected frames stay NULL,
Spark vs pandas signal correlation (informational, > 0.8), and the Spark events vs the pandas events (count, label,
start, end, frames per event; every frame lands in exactly one event).

## Scope note

Only signal derivation and event construction/aggregation are reimplemented in Spark. The gradient-boosting classifier
and `ruptures` are not; their per-frame labels (`signals.csv:predicted_label`) are an input.

## Snowflake write check (not yet run)

`snowflake_write_check.py` is a separate notebook. It tests whether this workspace can write tables straight into Snowflake, the first step toward running the pipeline entirely in Databricks and Snowflake. It writes only to `MOTION_INTENT.CONNECTOR_TEST`. Steps and how to read the result: `snowflake_write_check.md`.
