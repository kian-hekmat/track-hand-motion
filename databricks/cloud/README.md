# Cloud path notebooks (Databricks)

Plan and milestones: `docs/cloud_pipeline_plan.md`. These notebooks run from a **Git folder**: Databricks' own clone of
`github.com/kian-hekmat/track-hand-motion`. They import the repo's `src/` directly, so Databricks runs the committed code.
No notebook is generated or copy-pasted.

## One-time setup: the Git folder

1. In Databricks: **Workspace** → your user folder → **Create** → **Git folder**.
2. Git repository URL: `https://github.com/kian-hekmat/track-hand-motion`, provider GitHub, branch `main`.
   - If the repo is private, Databricks asks for a GitHub token first: Settings → Linked accounts → add a GitHub personal
     access token with read access to this repo. Do not send it to me.
3. Before each run, update the folder: open it, click the branch name → **Pull**, so it has the latest commit.

If Free Edition does not offer Git folders, stop and tell me; the fallback is a wheel of `src/` uploaded to a Volume.

## M1: does the frozen model give the same answers on Databricks? (`m1_environment_check.py`)

**Status: passed on Databricks 2026-10-08** (serverless, Python 3.12.3). Events byte-identical to the local reference for all 7 takes. Serverless provides numpy 2.3.4, pandas 2.3.3 and scipy 1.16.3 (local reference: 1.26.4, 3.0.6, 1.17.1); with these, the signals differ from the reference by at most 1.1e-13 and the phase probabilities by at most 3.5e-18, with the most likely phase identical on every sample. So signal tables are **not** bit-identical across environments; later parity checks on signals use a stated tolerance, while events and labels must match exactly. Evidence: `evidence/cloud_m1_environment_check.html`, checked by `tests/test_cloud_m1.py`.

1. In the Git folder open `databricks/cloud/m1_environment_check.py` (it opens as a notebook). Attach **serverless** compute.
2. **Run all.** The first cell installs the pinned scikit-learn and ruptures (`requirements-cloud.txt`), the second restarts
   Python, the rest run all seven takes and compare with `data/cloud_reference/`. Nothing needs editing: the data and models come
   from the Git folder itself.
3. The last cell prints `M1 CHECKS: ALL PASSED` or the failed checks. Export: File → Export → HTML, save as
   `evidence/cloud_m1_environment_check.html`, and send me the summary output.

How to read it:
- **PASS on every take's events:** the model and decoding give byte-identical events on Databricks. M2 can build on them.
- **Event FAIL:** look at the INFO lines above it. A difference in **signals** means the input already differs (numpy, pandas or
  scipy behave differently on serverless). Probabilities differing while signals match points at the model or scikit-learn. This is
  a real finding: send it to me and nothing gets adjusted until the cause is known.
- **Library FAIL:** the pinned scikit-learn or ruptures did not install at the pinned version.

## M2: build the bronze, silver and gold tables (`m2_build_tables.py`)

**Status: run 1 passed on Databricks 2026-10-08** (32 of 32 checks; evidence `evidence/cloud_m2_build_tables_run1.html`, checked against the notebook at commit `b1d6b6d`). The workers imported the repo code directly, so the zip fallback was not needed. Since then M2b was added: the notebook also writes `gold.scores` (boundary recall, precision, timing error, chance baselines, per-cycle rows) and missing values are stored as NULL instead of NaN. **Run 2 passed on Databricks 2026-10-08** with M2b: 33 of 33 checks (evidence `evidence/cloud_m2_build_tables.html`).

The code is in `src/cloud/tables.py` (Spark transforms) and `src/cloud/checks.py` (the comparison with the verified local reference).
The notebook only orchestrates them and writes Delta tables. The only manual step is uploading the input files to a Volume; in
M5 that upload becomes the videos themselves.

### 1. Create the schemas and the landing Volume (CLI, from the repo root)

```bash
databricks schemas create motion_bronze workspace
```

```bash
databricks schemas create motion_silver workspace
```

```bash
databricks schemas create motion_gold workspace
```

```bash
databricks volumes create workspace motion_bronze landing MANAGED
```

### 2. Upload the input files (keypoints, take metadata, out-of-frame intervals, hand labels)

```bash
databricks fs mkdir dbfs:/Volumes/workspace/motion_bronze/landing/raw && databricks fs mkdir dbfs:/Volumes/workspace/motion_bronze/landing/ground_truth
```

```bash
for f in data/raw/vid*_keypoints.csv data/raw/vid*_meta.json data/raw/out_of_frame_intervals.csv; do databricks fs cp "$f" "dbfs:/Volumes/workspace/motion_bronze/landing/raw/$(basename "$f")" --overwrite; done
```

```bash
for f in ground_truth/take_*.csv; do databricks fs cp "$f" "dbfs:/Volumes/workspace/motion_bronze/landing/ground_truth/$(basename "$f")" --overwrite; done
```

Check: `raw/` should list 15 files (7 keypoint CSVs, 7 meta JSONs, 1 interval CSV) and `ground_truth/` 7 files.

```bash
databricks fs ls dbfs:/Volumes/workspace/motion_bronze/landing/raw && databricks fs ls dbfs:/Volumes/workspace/motion_bronze/landing/ground_truth
```

### 3. Run the notebook

1. In the Git folder, **Pull** so it has the latest commit.
2. Open `databricks/cloud/m2_build_tables.py`, attach serverless compute, **Run all**. Nothing needs editing.
3. The last cell prints `M2 CHECKS: ALL PASSED` or the failed checks; every check is also appended to
   `workspace.motion_gold.run_log`. Export: File → Export → HTML as `evidence/cloud_m2_build_tables.html` and send me the summary.

What the checks require (tolerances fixed in `src/cloud/checks.py` before the run):
- **bronze:** 21 keypoint rows per frame for every take; ground-truth and interval rows equal the repo's files.
- **silver.frames:** equal to `src.signals.load_frames` on the repo's copy, within 1e-12. Spark and pandas parse CSV numbers
  slightly differently; locally the difference is at most 2.2e-16.
- **silver.signals:** signals and probabilities within 1e-9 of the reference; flags, per-sample labels and the model used per
  take identical.
- **gold.events:** label, start, end, duration and sample count identical for every event; mean confidence within 1e-12.
- **gold.frame_scores:** counts identical, accuracies within 1e-12 of `src.evaluate.score_take`.

If the "code on workers" probe fails even after shipping the zip, stop and send the output: the workers then cannot run the
model, and that changes the design.

## M3: publish to Snowflake and check it there (`m3_publish_snowflake.py`)

**Status: run 3 passed 2026-10-08, all 31 checks** (run from the CLI as a one-off serverless job on commit `32af63b`; evidence `evidence/cloud_m3_publish_snowflake.html`, checked by `tests/test_cloud_m2.py`). Inside Snowflake: row counts for all 7 tables equal what Databricks wrote; every verified `PIPELINE` row of all 7 tables has an equal `CLOUD` row; the 3 Tableau views have the expected row counts; the 5 queries in `queries.sql` give identical answers for vid1-5 in both schemas.

**Run 2 (2026-10-08): 30 of 31 checks passed; `snowflake parity scores` FAILED because of a bug in the check, not the data.** Row counts, the manifest, parity for the other 6 tables, the 3 Tableau views and all 5 queries passed inside Snowflake. The scores table differs from `PIPELINE.SCORES` in the per-phase accuracy columns (`acc_*`, 1 to 3 of 10 rows each), although the same cloud scores matched `data/export/scores.csv` in Databricks and the same SQL passed in DuckDB. Cause: Snowflake's INFORMATION_SCHEMA returns lower-case column names (`acc_rest`), the tolerance table used `acc_REST`, so the lookup missed and those columns were compared exactly; last-bit rounding in the CSV-loaded `PIPELINE` values then counted as differences. A direct comparison in Snowflake within the 1e-12 tolerance found no differing cell. Fixed in `32af63b` (case-insensitive lookups, regression test). Evidence: `evidence/cloud_m3_publish_snowflake_run2.html`. Earlier: **run 1 failed** Run 1 passed the M2 gate and all 7 serving-table checks and published the 7 tables to `CLOUD`, then stopped at `import snowflake.connector` (`ModuleNotFoundError`): after the `%pip` install and Python restart, serverless no longer had the connector the write check had found preinstalled. No Snowflake-side check ran. Fix: the connector is installed in the `%pip` cell. Evidence: `evidence/cloud_m3_publish_snowflake_run1_failed.html`. Locally the Snowflake-side SQL, the Tableau views and the five queries pass in DuckDB (`tests/test_cloud_m2.py`).

The notebook reshapes the M2 tables into exactly the layout of the verified `MOTION_INTENT.PIPELINE` tables, checks them against
the verified exports, publishes them to `MOTION_INTENT.CLOUD` with the Spark connector, then checks inside Snowflake:
- row counts equal what Databricks wrote (`CLOUD.PUBLISH_RUNS` manifest);
- every verified `PIPELINE` row (vid1-5) has an equal `CLOUD` row, per table and column (`src/cloud/snowflake_sql.py`);
- the Tableau views from `snowflake/05_tableau_views.sql` are created in `CLOUD`, with the expected row counts;
- the five queries in `queries.sql`, unchanged, return the same answers in both schemas for vid1-5.

### What you do

1. **Snowflake:** check how many trial days are left, then run `snowflake/11_cloud_setup.sql` in a worksheet as ACCOUNTADMIN.
   It creates the `CLOUD` schema and a role that can write only there and read `PIPELINE`, and gives that role to the existing
   key-pair user.
2. **Databricks secret** for the host, so no notebook needs editing (your host is the one the connector test used):

```bash
databricks secrets put-secret motion snowflake_host --string-value "SKVYGXH-FD77388.snowflakecomputing.com"
```

3. In the Git folder **Pull**, then **re-run `m2_build_tables.py`** (Run all). Expect `M2 CHECKS: ALL PASSED` with 33 checks.
   Export it as `evidence/cloud_m2_build_tables.html`.
4. Open `m3_publish_snowflake.py`, Run all. It stops before publishing if the latest M2 run did not pass or the serving tables
   differ from the verified exports. Expect `M3 CHECKS: ALL PASSED`. Export it as `evidence/cloud_m3_publish_snowflake.html`.
5. Send me both exports.

## M4: the whole cloud pipeline as one job (`databricks.yml` at the repo root)

**Status: first job run passed 2026-10-08** (job 866964811525959, run 121730247088002, started with `databricks bundle run`). Both tasks ran commit `7d9be28` from a checkout Databricks made of that commit (not the Git folder). `m2_build_tables` passed 33 of 33 checks; `m3_publish_snowflake` started after it ended, gated on that same M2 run, and passed 31 of 31. Evidence: `evidence/cloud_m4_job_run.json` (run record with the commit per task) and `evidence/cloud_m4_m2_build_tables.html`, `evidence/cloud_m4_m3_publish_snowflake.html`, saved by `scripts/save_job_run_evidence.py` and checked by `tests/test_cloud_m4.py`.

A Databricks Asset Bundle defines one job with two tasks on serverless compute: `m2_build_tables`, then `m3_publish_snowflake`
(which runs only if M2 succeeded). The tasks read the notebooks from GitHub `main` (`git_source`), so each run records the commit
it used, and nothing is uploaded from a local folder. Each notebook now raises at the end if any of its checks failed, after
logging and printing them, so a failed check fails its task and stops the job before Snowflake is touched.
`tests/test_cloud_m4.py` pins the task order, the source, serverless, and the fail-on-check behaviour.

From the repo root (the Databricks CLI must be signed in):

```bash
databricks bundle validate
```

```bash
databricks bundle deploy
```

```bash
databricks bundle run motion_intent_cloud_pipeline
```

The only manual step left in the cloud path is uploading the input files to the Volume (M2 step 2).
