# Cloud pipeline: Databricks → Snowflake → Tableau

**Status: plan decided 2026-10-07; scope change approved and written into `CLAUDE.md` 2026-10-08. M0 done (2026-10-08); M1 done on Databricks (2026-10-08); M2 passed on Databricks (2026-10-08); M2b and M3 built and passing locally, not yet run on Databricks/Snowflake.** Prerequisite done: Databricks can write to Snowflake
with the Spark connector (`databricks/snowflake_write_check.md`, evidence saved).

## Goal

Run the pipeline so the only local actions are recording the videos and uploading files: Databricks does all computation
from raw input to final tables, Snowflake holds and serves the results, and Tableau reads from Snowflake. The existing local
pipeline stays as the verified reference, and every cloud result is checked against it.

```
Unity Catalog Volume  (videos, ground-truth CSVs, frozen model files)   <- the only manual upload
        │
        ▼  Databricks job (tasks run in order; any failed check stops the job)
  bronze.raw_keypoints        MediaPipe extraction per video        [M5; until then: uploaded keypoint files]
  bronze.ground_truth         hand labels as uploaded
  silver.frames               21 landmark rows -> 1 row per frame          (Spark: conditional aggregation)
  silver.frame_predictions    signals + 61 features + model + PELT, per take (applyInPandas, existing code)
  gold.events                 per-frame labels -> events                    (Spark: gaps-and-islands)
  gold.frame_scores           frame metrics in Spark (range joins + groupBy/agg); boundary metrics later (M2b, src/score.py)
  publish                     Spark Snowflake connector -> MOTION_INTENT.CLOUD
  verify                      row counts vs a run manifest; parity with the verified local results
        │
        ▼
Snowflake  MOTION_INTENT.CLOUD: tables, verify procedure, analytical queries, Tableau views
        │
        ▼
Tableau (Desktop trial connected to Snowflake) -> published to Tableau Public as an extract
```

## Decisions

### 1. Storage: Unity Catalog, bronze / silver / gold schemas in Delta

Catalog `workspace` (the Free Edition default), schemas `motion_bronze`, `motion_silver`, `motion_gold`, and one Volume
`workspace.motion_bronze.landing` with folders `videos/`, `ground_truth/`, `models/`. Bronze tables hold data as it arrived,
silver holds cleaned and derived per-frame data, gold holds what Snowflake and Tableau use. This replaces the Postgres landing
table **for the cloud path only** (see "Spec change" below).

### 2. Model features run the existing pandas code inside Spark (`applyInPandas`), not a Spark rewrite

The frozen model was trained on features from `src/signals.py` + `src/features.py` (30 Hz grid, Savitzky-Golay smoothing,
61 rolling and past-vs-future features). Phase 3 showed the Spark-native signals are close but **not identical** (speed
r = 0.98-0.99; vid4 aperture 0.87). A model fed slightly different features gives different labels, and the difference would be
untestable against the verified results. So the per-take feature and prediction step runs the same Python code, distributed one
take per group with `groupBy("take").applyInPandas(...)`. This is the standard Spark pattern for applying a single-machine model
per partition, and it keeps the cloud labels comparable frame by frame with the local ones.

Spark-native code is used where it is exact: pivoting landmarks to frames, building events (gaps-and-islands, already verified
101/101 against pandas), frame-level scoring (joins and `groupBy`), and per-event aggregates. The existing Spark signal
derivation stays as the Phase 3 demonstration; it does not feed the model.

### 3. Which model labels which take (keeps the scores honest)

`segmenter_v2.joblib` was trained on all five development takes, so applying it to vid1-5 would score the model on its own
training data and inflate the results. The verified canonical events instead come from five leave-one-take-out models, which
`scripts/freeze_model.py` trains but does not save. Decision:

- Save the five fold models once, locally, with the same seed (`fit(..., seed=0)`), plus their hashes. Check that they reproduce
  `data/segments/v2_frozen_oof/` exactly before uploading anything.
- In the cloud, a small `model_assignment` table maps each take to its model: vid1-5 → the fold model that excluded it;
  vid6, vid7 and any new take → the full frozen model.
- Models are loaded from the Git folder (versioned with the code, so the commit fixes which model ran) and their hashes are checked before use (`src.final.load_model_for_take`). The Volume holds only uploaded data.
  MLflow registration is optional and comes later (M6); the files plus hashes are the record of which model ran.

### 4. Code reaches Databricks through a Git folder, with pinned libraries

The repo (`github.com/kian-hekmat/track-hand-motion`) is cloned into the workspace as a Git folder, and job tasks import
`src/` from it. No generated or copy-pasted notebooks, and Databricks runs the committed code. Libraries are pinned to the
versions the model was frozen with (scikit-learn 1.9.1, ruptures 1.1.10). The serverless environment ships its own numpy and
pandas (local: numpy 1.26.4, pandas 3.0.6). Unpickling and predictions may still be identical; the parity check in M1 decides
that, and any difference is reported, not tuned away.

### 5. Orchestration: one Databricks job defined in the repo (Asset Bundle)

The job and its task graph live in `databricks.yml` and are deployed with the CLI that is already set up
(`databricks bundle deploy`, `databricks bundle run`). One task per step above, so a failure shows exactly where it stopped,
and each task's checks fail the task instead of printing a warning. If Free Edition does not accept bundles, the same task graph is
created as a job in the UI from the same YAML.

### 6. Snowflake: a new `CLOUD` schema, written by Spark, verified in SQL

- The job writes gold and silver tables into `MOTION_INTENT.CLOUD` with the Spark connector (the `host` option; serverless
  rejects `sfURL`). `MOTION_INTENT.PIPELINE` stays untouched as the verified local-path result.
- A dedicated role `DATABRICKS_PIPELINE_ROLE` (write access to `CLOUD` only, read access to `PIPELINE` for the parity check) is
  granted to the existing key-pair user. The test role and schema are dropped after M3.
- Every run writes a `pipeline_runs` manifest (run id, git commit, model per take, row counts per table). A Snowflake stored
  procedure compares the loaded tables with the manifest, like the existing 46 checks but generated from the manifest instead of
  from local files. The job's last task calls it through the Python connector, which the write check also proved.
- **Parity in SQL:** `EXCEPT` queries between `CLOUD` and `PIPELINE` for events, scores and per-frame labels. For vid1-5 the
  expected result is zero differing rows. This is the main proof that the cloud pipeline reproduces the verified one.
- `queries.sql` and `05_tableau_views.sql` are pointed at `CLOUD`. Snowflake then produces the Tableau tables, not DuckDB.

### 7. Tableau: a real Snowflake connection, published as an extract

Tableau Public cannot connect to Snowflake, and a Tableau Public workbook is always an extract. Decision: build in a Tableau
Desktop trial (14 days) connected to the `CLOUD` views, then publish to Tableau Public. Whether the Desktop trial can save to
Tableau Public is **not yet verified**; it is checked at the start of M7. Fallback: download the view results from Snowsight
into Tableau Public, described as "data from Snowflake views" rather than a live connection.

### 8. Extraction (MediaPipe) moves last and may stay local

Running MediaPipe on serverless is untested (install, uploading the model file to the Volume, decoding the iPhone's HEVC
10-bit video without ffmpeg). It is the step least related to the tools being demonstrated, so it is done last. Until then
bronze is loaded from uploaded keypoint files, through the same table contract, so nothing downstream changes when extraction moves. MediaPipe on different
hardware may not reproduce keypoints bit for bit. M5 measures how much they differ and how much the labels change, and reports both.

## Spec change (approved 2026-10-08, now in `CLAUDE.md` under "Cloud path")

`CLAUDE.md` makes Postgres the raw landing zone ("Do not skip this step") and defines Phase 3 as reading data exported from
Postgres. The cloud path lands raw data in Delta instead. Proposed wording: Postgres remains the local dev/test landing table
and the source of the verified reference results; the cloud path lands in Unity Catalog bronze tables and must reproduce those
reference results.

## Milestones (each ends with a check that must pass and saved evidence)

| # | Milestone | Done when |
|---|---|---|
| M0 | Save fold models locally | **Done 2026-10-08.** `scripts/save_fold_models.py` saved `models/folds/` (5 models + `folds.json` with hashes); through `src.final.load_model_for_take`, each reproduces its take's `v2_frozen_oof` events byte for byte, and vid6/vid7 reproduce `data/holdout/` events with the full model (`tests/test_fold_models.py`, 9 tests) |
| M1 | Git folder, pinned environment, models from the Git folder (`databricks/cloud/m1_environment_check.py`) | **Done 2026-10-08.** On serverless (numpy 2.3.4, pandas 2.3.3, scipy 1.16.3 vs local 1.26.4, 3.0.6, 1.17.1) events are byte-identical for all 7 takes; signals differ by at most 1.1e-13, probabilities by at most 3.5e-18, most likely phase identical on every sample (`evidence/cloud_m1_environment_check.html`) |
| M2 | Bronze → silver → gold in Databricks (from uploaded keypoints) | Signals within a stated tolerance of the reference (M1: floating-point differences up to ~1e-13 across environments); `gold.events` = 101/101 identical to `data/export/events.csv`; `gold.frame_scores` equal to the scored frame metrics; vid6/vid7 equal to `data/holdout/`. **Scope note:** boundary metrics (recall, precision, timing error, chance baselines) are not in M2; they follow as M2b with `src/score.py` on the gold tables |
| M3 | Publish to Snowflake `CLOUD`, verify procedure | Manifest checks all pass; `CLOUD` vs `PIPELINE` parity queries return zero differing rows |
| M4 | One job end to end (bundle) | A single `databricks bundle run` goes from Volume files to verified Snowflake tables with no manual step in between |
| M5 | MediaPipe extraction in Databricks | Keypoints compared with the local extraction (difference reported); labels re-scored and reported, never pooled with earlier numbers |
| M6 | Optional: MLflow tracking and registry | Models registered with their hashes; retraining in Databricks compared with the frozen model and reported |
| M7 | Tableau from Snowflake | Dashboard built on the `CLOUD` views and published; image re-verified with `scripts/verify_dashboard_image.py`; the first-time-viewer test |

Local pytest stays the safety net: the Spark parts keep running on local Spark, the SQL in DuckDB, and every evidence export
gets a parsing test like `tests/test_databricks_evidence.py`.

## Risks

- **Snowflake trial expiry** (about 30 days). Check the remaining days before M3. If it lapses, a new trial means re-running
  the setup SQL, and evidence from the old account stays valid as a record of that run.
- **Free Edition quotas:** compute stops for the day if exceeded. The data is small (about 83k keypoint rows), so this is unlikely.
- **Library drift** (numpy/pandas versions on serverless) may change model outputs slightly. M1 measures this before anything is built on it.
- **Bundles or Git folders may be unavailable on Free Edition.** Fallbacks: a UI-defined job; a wheel of `src/` uploaded to the Volume.
