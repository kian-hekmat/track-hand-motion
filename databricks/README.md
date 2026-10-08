# Databricks

Databricks runs the cloud pipeline: it builds the bronze, silver and gold Delta tables from the uploaded files, then publishes
the gold tables to Snowflake and checks them there. Everything runs as one job on serverless compute.

| Where | What |
|---|---|
| `../databricks.yml` | the job, as a Databricks Asset Bundle: `m2_build_tables`, then `m3_publish_snowflake` (only if M2 succeeded), notebooks read from GitHub `main` |
| `cloud/m2_build_tables.py` | bronze, silver and gold tables in `workspace.motion_{bronze,silver,gold}`, compared with the verified local reference |
| `cloud/m3_publish_snowflake.py` | the same tables in the verified `PIPELINE` layout, published to `MOTION_INTENT.CLOUD`, then checked inside Snowflake |
| `cloud/m1_environment_check.py` | one-off check (M1) that the frozen model gives the same answers on serverless |
| `cloud/requirements-cloud.txt` | the library versions pinned on serverless (the ones the model was saved with) |
| `../src/cloud/` | the Spark transforms, the checks and the Snowflake check SQL that the notebooks run (tested locally on Spark and DuckDB) |

Run steps, results and evidence for every milestone: [`cloud/README.md`](cloud/README.md). Plan: `../docs/cloud_pipeline_plan.md`.

## Spark techniques used (all in `src/cloud/tables.py`)

- **Conditional aggregation** (`max(when(...))` grouped by take and frame) pivots the 21 landmark rows of each frame into one row.
- **`groupBy().applyInPandas`** runs the repo's own signal, feature, model and changepoint code once per recording, in parallel
  on the workers, with the model that may label that recording honestly. This keeps the per-frame labels identical to the
  verified ones (a Spark rewrite of the 61 features would not be).
- **Gaps-and-islands** with window functions (`lag` to mark where the label changes, a running `sum` to number the events,
  `lead` for each event's end) turns per-sample labels into events.
- **Range joins** match every video frame to the event and the hand-labelled segment that contain it, and to the out-of-frame
  intervals.
- **`collect_list` of structs** gathers each recording's events, labels and frame times into one row, so the full scoring
  (boundaries, timing, chance baselines, per cycle) runs once per recording.

## Retired in the cloud cleanup (2026-10-08)

Kept at the git tag `pre-cloud-cleanup` (commit `8dc35b5`):
- **The Phase 3 local Spark demo** (`motion_pipeline_spark.py`, `spark_transforms.py`): a PySpark re-derivation of the
  signals and events, run once on Databricks Free Edition on 2026-10-05 with 12 of 12 checks passing and 101 of 101 events
  identical to pandas. The cloud notebooks replace it.
- **The Snowflake write check** (`snowflake_write_check.py`): the 2026-10-07 test showing this workspace can write to
  Snowflake. One finding from it still applies: serverless rejects the Spark connector's `sfURL` option and needs `host`.
