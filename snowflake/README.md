# Phase 4: loading and querying in Snowflake

**Status: run on Snowflake and verified (2026-10-05).** The SQL was first tested locally in DuckDB (same table definitions,
positional CSV load, portable SQL): `tests/test_snowflake_sql.py`. The six result files you downloaded from Snowflake are saved in
`actual_results/` and checked by `tests/test_snowflake_evidence.py` (see "Results of the Snowflake run" below). Nothing here needs
a credential from me; do not send me passwords or keys.

| File | What it is |
|---|---|
| `00_diagnose.sql` | troubleshooting only (stage contents, load history); statements 3 and 5 have been run on Snowflake, the others have not |
| `01_setup.sql` | warehouse, database `MOTION_INTENT`, schema `PIPELINE`, file formats, stage, 7 tables (generated) |
| `02_load.sql` | `COPY INTO` from the stage for each table (generated) |
| `03_verify.sql` | 46 row-count and integrity checks; every row must say `PASS` (generated from the export manifest) |
| `04_fingerprint.sql` | **optional** value-level check (180 sums and counts, per table and take) that catches wrong values that row counts miss |
| `../queries.sql` | the 5 analytical queries (includes `LAG` and `LEAD` window functions) |
| `expected_results/` | what each script/query should return, produced locally from the same data |
| `actual_results/` | where you save what Snowflake returns |

## What you do

1. **Create a Snowflake trial account** (snowflake.com, free trial). Choose any cloud/region. A new trial has the role
   `ACCOUNTADMIN` and usually a warehouse called `COMPUTE_WH`. Note: the trial is time-limited (about 30 days), so do the
   steps below in one sitting.
2. **Open a SQL worksheet** (Snowsight: Projects -> Worksheets, or Workspaces in newer UIs). Paste all of `01_setup.sql` and
   run it (Run all). This creates the database, schema, stage and 7 empty tables.
3. **Upload 7 files to the stage.** Files are in `data/export/` of the repo:
   `takes.csv`, `events.csv`, `ground_truth.csv`, `signals.csv`, `frames.csv`, `scores.csv`, `raw_keypoints.parquet`.
   In Snowsight: Catalog (or Data) -> Database Explorer -> `MOTION_INTENT` -> `PIPELINE` -> Stages -> `MOTION_STAGE` ->
   **+ Files**, drop the 7 files in, Upload. (Fallback: Data -> Add Data -> Load files into a Stage; or SnowSQL
   `PUT file://.../data/export/*.csv @motion_stage AUTO_COMPRESS=FALSE`.)
4. **Load:** paste and run `02_load.sql`. `LIST @motion_stage;` at the top should show 7 files; each `COPY INTO` should report
   the number of rows loaded (events 101, signals 3960, frames 3960, ground_truth 101, takes 5, scores 10, raw_keypoints 83160).
   If a `COPY` fails, send me the full error text; do not edit around it.
5. **Verify:** run `03_verify.sql`. Every row of the result must be `PASS` (46 rows). Download the result:
   results pane -> Download -> CSV, and save it as `snowflake/actual_results/verify.csv`.
6. **Run the queries:** in a worksheet paste `queries.sql`, then run **one query at a time** (select a query, Ctrl/Cmd+Enter).
   Download each result as CSV into `snowflake/actual_results/` with exactly these names:
   `q1_avg_duration_by_event_type.csv`, `q2_most_ambiguous_takes.csv`, `q3_time_between_consecutive_reaches.csv`,
   `q4_phase_transitions.csv`, `q5_prediction_agreement_by_label.csv`.
7. **Compare:** `python scripts/compare_snowflake_results.py` prints `OK` or `MISMATCH` per file. Send me the output
   (and a screenshot of the verify result if you like). A mismatch is a real finding, not something to patch.

## Notes

- Column names `group` and `false` are renamed (`take_group`, `false_boundaries`) because they are reserved words in Snowflake.
- Re-running `01_setup.sql` recreates the tables (data is dropped); re-run `02_load.sql` afterwards.
- The warehouse is XSMALL with auto-suspend after 60 s, so trial credit use is minimal.

## Troubleshooting: every "actual" is 0 in 03_verify.sql

Zero means the tables are empty. The warehouse is not the cause (it only supplies compute; `COMPUTE_WH`, XSMALL, is right, and a
missing or stopped warehouse gives an error message, not zeros). Run the numbered statements of `00_diagnose.sql` one at a time:

| What you see | Cause | Fix |
|---|---|---|
| Statement 3 (`LIST @...motion_stage`) returns no rows | the files are not in this stage (never uploaded, or uploaded to a stage in another database/schema; statement 2 shows all stages with this name) | upload the 7 files to `MOTION_INTENT.PIPELINE.MOTION_STAGE`, then re-run `02_load.sql` |
| Statement 3 lists files but statement 5 has no rows | `COPY INTO` matched no file (it reports "0 files processed" as a success) | send me the exact file names from statement 3; the load patterns assume names like `events.csv` or `events.csv.gz` |
| Statement 5 shows `LOAD_FAILED` / `PARTIALLY_LOADED` | a parse error | send me `first_error_message` |
| Statement 4 shows counts > 0 but verify still shows 0 for per-take rows | take names loaded differently (for example with extra characters) | run `SELECT DISTINCT take FROM events;` and send me the output |
| Statement 4 shows 0 and statement 5 shows successful loads | `01_setup.sql` was re-run after loading (`CREATE OR REPLACE TABLE` empties the tables) | run `02_load.sql` again, then `03_verify.sql` |
| Statement 1 shows a different database/schema/role | the worksheet context is not `MOTION_INTENT.PIPELINE` | run `USE SCHEMA MOTION_INTENT.PIPELINE;` first |

After `02_load.sql`, the last result is a row count per table; do not run `03_verify.sql` until those match
(takes 5, events 101, ground_truth 101, signals 3960, frames 3960, scores 10, raw_keypoints 83160).

## Results of the Snowflake run

Saved in `actual_results/` (downloaded from Snowsight, upper-case column names as Snowflake returns them):

| File | Result |
|---|---|
| `verify.csv` | 46 of 46 checks `PASS`, `expected` equals `actual` on every row; row counts equal the export manifest: takes 5, events 101, ground_truth 101, signals 3,960, frames 3,960, scores 10, raw_keypoints 83,160 |
| `q1` to `q5` | every file matches the locally produced expected results exactly (`python scripts/compare_snowflake_results.py` prints `ALL MATCH`) |

Beyond that comparison, `tests/test_snowflake_evidence.py` recomputes all five queries in pandas, with no SQL, from the exported CSVs
and compares them with the Snowflake output (maximum difference 0.00), so a mistake in the SQL logic would not be hidden by
Snowflake and DuckDB running the same text. The comparison script itself was tightened to an exact match (no relative tolerance).

### What this does and does not prove

- Proved: all seven tables loaded completely (counts), events and signals loaded with correct values (the queries aggregate their
  numbers, labels, booleans and NULLs), and the query logic is correct.
- Not proved yet: the **values** of `raw_keypoints` and `frames` (only row counts and a NULL check). The optional
  `04_fingerprint.sql` covers this. To run it: execute the file in Snowflake, download the result as
  `actual_results/fingerprint.csv`, then `python scripts/compare_snowflake_results.py`.
- Not captured: Snowflake edition, region, warehouse size actually used, query history. A screenshot of Snowsight's Query History
  saved under `evidence/` would document the run; it is not required.
- The first load attempt left six of seven tables empty (only `takes` had a `COPY` record in `COPY_HISTORY`). The cause was not
  recorded; the load was then completed. This is why `02_load.sql` now ends with a row-count summary.
