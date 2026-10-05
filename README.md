# Motion Intent Pipeline

An end-to-end data pipeline that turns recorded human hand motion into discrete "intent" events:

```
iPhone video → MediaPipe Hands → Postgres (raw) → Segmentation → Databricks (Spark) → Snowflake → Tableau
```

## Status

Phase 1 is complete (extraction, Postgres landing table, ground truth, overlay review, tests). Phases 2 and 3 are complete. Phases 4–5 are not started.

| Phase | Stage | Status |
|---|---|---|
| 1 | Capture & keypoint extraction | **Complete.** 5 videos extracted, Postgres landing table loaded (row counts match), ground truth labelled for all 5 takes, overlays reviewed, 191 pytest checks pass. Known limitations in `NOTES.md` |
| 2 | Segmentation | **Complete (v2); plots in `evidence/phase2_v2/` reviewed.** 191 pytest checks pass. Frame accuracy 0.84–0.97 under leave-one-take-out CV (v1: 0.45–0.78); caveats in `NOTES.md` |
| 3 | Databricks processing | **Complete.** PySpark signal derivation + event construction; 12/12 checks passed locally and on Databricks (evidence saved); Spark events identical to pandas events (101/101). See `databricks/README.md`, `NOTES.md` |
| 4 | Snowflake | Not started |
| 5 | Tableau dashboard | Not started |

### Phase 1: what exists (verified by `pytest`, run against real output)

- Five recordings in `vids/vid1–5.mov`, converted to H.264 (`scripts/convert_videos.py`) and run through MediaPipe Hand Landmarker (`src/extract.py`, `scripts/extract_take.py`). Output: `data/raw/vidN_keypoints.csv` (one row per frame × 21 landmarks) and `vidN_meta.json`. Real per-frame timestamps are used (iPhone video is variable frame rate); undetected frames are kept as explicit NaN rows.
- Fraction of frames with no hand detected (`data/raw/detection_report.csv`):

| video | frames | not detected | sustained dropouts (s) |
|---|---|---|---|
| vid1 | 976 | 0.1% | none |
| vid2 | 1015 | 3.0% | none |
| vid3 | 775 | 2.7% | none |
| vid4 | 281 | 5.7% | none |
| vid5 | 913 | 6.4% | 9.34–9.80, 17.10–17.24 |

- **Trimmed takes (data decision):** `vid3` and `vid4` originally ran longer and ended with footage that should not be in the dataset. `vid4.mov` was cropped by hand; `vid3.mov` was cut at the end only, to 25.805 s (775 frames, re-encoded HEVC 10-bit with the HLG colour tags kept, every frame timestamp identical to the original; checked). Neither was trimmed at the start, so timestamps still begin at 0. Untrimmed originals are kept locally in `vids_original/` (gitignored; the committed pre-trim versions are in git history). Their detection numbers changed (vid3 2.2%→2.7%, vid4 9.7%→5.7%, and the sustained dropouts previously listed for both are gone). For vid4 the old dropouts at 10.40 s and 14.77 s are past the new end; for vid3 the old 0.53–0.67 s dropout is inside the kept range and no longer appears after re-encoding. That vid3 change is **not yet explained** (likely the re-encode altering what the tracker sees on that frame range); worth a look at the overlay.
- Local Postgres landing table `raw_keypoints` (`docker-compose.yml`, `sql/001_raw_keypoints.sql`, `src/db.py`, `scripts/load_postgres.py`). Row counts match the CSVs exactly for all five takes (20496 / 21315 / 16275 / 5901 / 19173). The loader is idempotent (replaces a take in one transaction) and the table's CHECK constraint enforces that detected frames have coordinates and undetected frames have none.
- Overlay clips and contact sheets in `evidence/`. **Human review result:** vid1, vid2 and vid4 track well throughout; vid3 loses the hand on and off in the opening REST position; vid5 loses the hand at 9.335–9.835 s and 17.103–17.270 s because the hand is raised above the frame (physical absence, declared in `data/raw/out_of_frame_intervals.csv`, verified by `tests/test_data_gaps.py`). Details and consequences in `NOTES.md`.
- Ground truth: `ground_truth/take_1–5.csv`, hand-labelled from the video frames (no audio cues), scoring tolerance 0.10 s. vid5 has four cycles. Mapping: vid1–3 clean, vid4 Fast, vid5 Hard case.
- `Left` handedness on 49 of 3,834 detected frames (1.3%) of a right hand is a MediaPipe quirk (low-confidence, isolated frames in moving phases). Handedness is never used to filter frames. See `NOTES.md`.

Run it:

```
docker-compose up -d db            # Postgres 16 on localhost:5433 (separate from other projects' DBs)
python scripts/load_postgres.py    # init schema, load all takes, verify row counts
pytest                             # needs the DB loaded; Postgres tests fail (not skip) if it is down
```

### Phase 2: segmentation (built, scored, improved)

**v1 (rules + PELT on raw speed/aperture)** was the baseline; **v2 (learned frame classifier + PELT on phase probabilities)** replaces it as the working result. Details, protocol, failure modes and disclosures: `NOTES.md`. Code: `src/signals.py`, `src/features.py`, `src/learned.py`, `src/segment.py` (v1), `src/score.py`, `src/evaluate.py`; scripts `cv_segmentation.py` (nested leave-one-take-out CV), `score_segmentation.py`, `plot_segmentation.py`.

Every take is scored by a model trained on the *other four* takes only. Tolerance 0.10 s. Reported per group, never pooled; numbers from `data/segments/history.csv` (every scored version is logged there):

| take | group | frame acc v1 → v2 | balanced acc v2 | boundary recall v1 → v2 | precision v2 | MAE matched (s) v2 |
|---|---|---|---|---|---|---|
| vid1 | clean | 0.68 → **0.95** | 0.93 | 0.50 → 0.67 | 0.67 | 0.042 |
| vid2 | clean | 0.78 → **0.97** | 0.96 | 0.78 → 0.83 | 0.83 | 0.048 |
| vid3 | clean | 0.59 → **0.96** | 0.96 | 0.61 → 0.83 | 0.83 | 0.031 |
| vid4 | fast | 0.45 → **0.91** | 0.79 | 0.33 → 0.78 | 0.93 | 0.031 |
| vid5 | hard | 0.54 → **0.84** | 0.83 | 0.54 → 0.75 | 0.67 | 0.035 |

Known problems: RELEASE in the fast take is still missed (0.13 s segments); vid5's long pause before grasping is read as GRASP early; the occlusion cycle is noisy. These numbers are optimistic: the design was informed by all five takes and vid1–3 come from one session (see `NOTES.md`). A newly recorded take would give an unbiased check.

Run: `python scripts/cv_segmentation.py` (several minutes), then `python scripts/score_segmentation.py --events-dir data/segments/v2_pelt --version v2_pelt` and `python scripts/plot_segmentation.py --events-dir data/segments/v2_pelt --out-subdir phase2_v2`. v1: `scripts/run_segmentation.py`.

### Phase 2: frozen model and exported tables

The v2 model is frozen in `models/segmenter_v2.joblib` (+ `.json` with hash, training takes, penalty and feature columns). The canonical events are out-of-fold (each take predicted by a model that never saw it; scored as `v2_frozen_oof`: frame accuracy 0.95 / 0.97 / 0.97 / 0.91 / 0.84 for vid1–5). `data/export/` holds the tables for the next phases (`events`, `signals`, `frames`, `ground_truth`, `takes`, `scores`, `raw_keypoints.parquet`) with a `manifest.json` of row counts and hashes. New takes can be segmented with `src.final.segment_new_take(take)`; nothing is retrained. Details: `NOTES.md`.

Run: `python scripts/freeze_model.py && python scripts/score_segmentation.py --events-dir data/segments/v2_frozen_oof --version v2_frozen_oof && python scripts/export_tables.py`.

### Phase 1: remaining caveats

- The confidence signal is per-frame `detected` / `handedness_score`; the Hand Landmarker gives no per-landmark visibility score.
- The vid3 sustained dropout previously reported at 0.53–0.67 s disappeared after re-encoding; the reason is unexplained (`NOTES.md`).

## Goals

Convert high-frequency motion data (position + time, many points over a few seconds) into discrete intent events, then store, query, and visualize them. Recorded human hand motion serves as the data source.

**Design choice to note:** the motion data comes from self-recorded video of a hand (keypoints extracted by computer vision), not from motion-sensor telemetry.

**Non-negotiable constraint:** every claim made about this project must be true. No metric, accuracy number, or completed step is reported unless a test or a human review of the output verified it. If a stage's output looks wrong, the cause gets fixed (or documented), not tuned away.

## Tech Stack

- Python 3.11+, pandas, NumPy
- MediaPipe Hands (wrist position for speed, fingertips for hand aperture; Pose alone can't tell open from closed hand)
- `ruptures` for changepoint detection (one approach is chosen and committed to; no parallel HMM build)
- PostgreSQL (local, Docker) as the raw landing zone
- Databricks Community Edition (PySpark)
- Snowflake (trial account)
- Tableau Public
- pytest

## Pipeline Stages

### Phase 1: Data capture & keypoint extraction

**Goal:** raw time-series table of `(timestamp, keypoint_id, x, y, z, confidence)`.

**Recording protocol**

- Seated at a table, torso still, only the working arm moves; other hand in lap, out of frame.
- Two tape marks: **Home (A)** ~10 cm in front of the body, **Target (B)** ~40 cm from A, running *across* the camera view.
- One light, rigid, graspable object (cup, block, or ball), identical in every take, placed at B.
- iPhone on a stand (never handheld), landscape, roughly table height, perpendicular to the A→B path. Plain non-reflective background, even lighting, no backlighting, short sleeves.
- 1080p at 30 fps (60 fps only if the Fast take shows motion blur). Action Mode and cinematic/portrait modes off.
- Do one slow dry run first to confirm fingertips stay visible during the grasp; raise or angle the camera if the object hides them.

**Phase labels (ground-truth vocabulary for the whole project)**

| # | Label | Action | Target duration | Expected signal |
|---|---|---|---|---|
| 0 | REST | Hand flat on A, still | 2 s | Near-zero wrist speed |
| 1 | REACH | Move A → object at B, opening fingers | ~1 s | Speed rises then falls; aperture increases |
| 2 | GRASP | Close hand on object, lift ~2 cm | ~0.5 s | Aperture drops sharply; little travel |
| 3 | HOLD | Hold lifted object still | 1 s | Near-zero speed; aperture stays closed |
| 4 | RELEASE | Set object down, open hand | ~0.5 s | Aperture rises; little travel |
| 5 | RETRACT | Return to A, hand flat | ~1 s | Speed rises then falls |
| 6 | REST | Still at A | 2 s | Near-zero speed |

Aperture = distance between thumb-tip and index-fingertip keypoints. One cycle = phases 0–6. Three cycles per take, back to back (~20–25 s).

**The five takes**

1. **Takes 1–3, Clean:** protocol exactly as above; primary data. At least two are fully ground-truth labeled.
2. **Take 4, Fast:** same phases and order, no deliberate stops, ~half the time per phase.
3. **Take 5, Hard case**, a different problem per cycle:
   - Cycle 1, **Hesitation:** ~0.5 s pause mid-REACH. Correct label: one REACH.
   - Cycle 2, **Failed grasp:** close, open, re-grasp before lifting. Correct label decided in advance (default: one long GRASP).
   - Cycle 3, **Occlusion:** rotate the wrist during HOLD so the object hides the fingertips for ~1 s.

**Ground truth capture**

- Tap the table once at the start of every take (visual sync point).
- Ground truth is read off the video frames (the overlay's `t=` stamp); audio cues are not used. The scoring tolerance is 0.10 s (3 frames), chosen before any segmentation was run.
- Right after Take 5, write down what happened in each cycle and the intended correct labels.
- Save per-take ground truth to `ground_truth/take_N.csv` with columns `take, cycle, label, start_s, end_s, source` (`source` = `manual`: read off the video frames).
- Before tearing down the setup, run MediaPipe on Take 1 and review the overlay. If fingers drop out during the grasp, fix angle/lighting and re-shoot immediately.

**Processing steps**

1. Record the five takes.
2. Run each video through MediaPipe Hands (21 landmarks per frame).
3. Write one raw CSV/Parquet per video: one row per (frame, keypoint) with timestamp, keypoint_id, x, y, z, confidence.
4. Load into Postgres table `raw_keypoints`. This is the dev/test source of truth and is not skipped even though cloud stages follow.

**Required testing & verification**

- Manually inspect at least one full video's keypoint overlay; save an annotated clip or several frame screenshots as evidence.
- pytest: no null/NaN coordinates for frames above the confidence threshold; output frame count matches FPS × duration within a small tolerance.
- Log and report the fraction of frames below the confidence threshold (tracking failures). This is a real data-quality metric and is not discarded.

### Phase 2: Segmentation (time series → discrete events)

**Goal:** a labeled sequence of events (REST / REACH / GRASP / HOLD / RELEASE / RETRACT) with start/end timestamps.

**Steps**

1. Use `ruptures` (PELT or Binary Segmentation) on a derived signal.
2. Derive two channels from raw keypoints:
   - **Wrist speed:** frame-to-frame wrist displacement, smoothed.
   - **Hand aperture:** thumb-tip to index-tip distance, normalized by hand size (wrist to middle-MCP).
3. Segment each take, then assign labels with rule-based logic on speed/aperture levels (more explainable than a learned classifier at this data size).
4. Refine `ground_truth/take_N.csv` for at least Takes 1 and 2 by checking the labelled timestamps against video frames. Takes 4 and 5 also need ground truth, since they are the stress tests.

**Required testing & verification**

- Score against ground truth for every take that has it: boundary MAE (seconds), boundary recall within the stated tolerance, per-label frame accuracy.
- Report clean takes, the Fast take, and the Hard-case take **separately**; never average them into one number.
- For Take 5, report per-cycle results (hesitation, failed grasp, occlusion).
- pytest regression test: run segmentation on a synthetic signal with programmatically inserted changepoints and assert detections fall within a tolerance window. It must keep passing as parameters are tuned.
- If the hard case performs poorly, document why in `NOTES.md` rather than excluding it from the demo.

### Phase 3: Databricks processing

**Goal:** reimplement a meaningful part of the pipeline on a Spark DataFrame in a Databricks Community Edition notebook.

**Steps**

1. Load raw keypoints (exported from Postgres as CSV/Parquet) into a Spark DataFrame.
2. Reimplement signal derivation and/or segmentation in PySpark (window functions over partitions, `groupBy`/`agg`), not pandas code pasted into a cell.
3. Write segmented events to an exportable table (CSV/Delta).

**Required testing & verification**

- Confirm row counts match between the pandas (Phase 2) and PySpark outputs for the same video. If they diverge, find out why before proceeding.
- Record one Spark-specific concept actually used and why (e.g., a window function partitioned by `video_id`, ordered by timestamp, for frame-to-frame deltas).

### Phase 4: Snowflake

**Goal:** land the structured tables in Snowflake as the queryable layer.

**Steps**

1. Load the segmented-events table and a sampled/aggregated raw-keypoints table (via `COPY INTO` from staged files, or Snowsight's loader).
2. Write 3–5 analytical SQL queries a stakeholder might ask, for example:
   - Average duration of each event type across videos
   - Which video had the most ambiguous/short-duration segments
   - Time between consecutive events using `LAG`/`LEAD`
3. Save them in `queries.sql`.

**Required testing & verification**

- Explicit check that Snowflake row counts match the source files exactly (no silent drops).
- Sanity-check each query result against what was observed in the videos.

### Phase 5: Tableau dashboard

**Goal:** one dashboard a non-technical viewer can read at a glance.

**Steps**

1. Connect Tableau Public to Snowflake (or exported CSVs if the free tier makes a live connection hard).
2. Build at minimum: (a) a timeline with colored bands for the active segment, and (b) the derived signal (e.g., speed) paired with that timeline.
3. Combine into one dashboard with a plain-language title/caption.

**Required testing & verification**

- Show it to someone who hasn't seen the project and ask them to describe it unprompted. If they can't, revise it.

## Final Integration Checklist

- [ ] Full pipeline runs end to end on at least one video, from raw footage to Tableau-ready export, without manual patching of intermediate files
- [ ] Every number reported about the project has a test or saved output that produced it
- [x] `NOTES.md` documents at least one real limitation or failure mode (Phase 1 ones so far; extend after Phase 2)
- [ ] This README matches what is actually built (no planned features described as done)

## Repository Layout

Exists today:

```
vids/             original iPhone recordings
src/              extract.py, landmarks.py (MediaPipe), db.py (Postgres loader), signals/features/learned/segment/score/evaluate/ground_truth (Phase 2)
scripts/          convert, extract, detection report, overlay/contact sheets, load_postgres
sql/              Postgres DDL
data/raw/         per-take keypoint CSVs, meta JSON, detection_report.csv
databricks/       PySpark transforms, generated Databricks notebook, run instructions
models/           frozen segmenter (joblib + json metadata)
data/export/      tables for Databricks / Snowflake / Tableau + manifest.json
data/segments/    Phase 2: v1 events + params.json; v2_pelt/v2_grammar/v2_argmax (cross-validated); history.csv (run log); cv_selection.json
docs/             phase2_roadmap.md
evidence/         overlay clips and screenshots
ground_truth/     take_N.csv hand-labelled phase boundaries
NOTES.md          limitations and failure modes
tests/            pytest suites
docker-compose.yml
```

`NOTES.md` holds limitations found so far. Not yet created: `queries.sql` (Snowflake), Phase 4–5 code.

## Ground Rules for AI Assistants

See `CLAUDE.md`. In short: never mark a phase complete without its tests passing and shown; never fabricate or estimate a metric; stop and say so when a result looks wrong; flag any scope-simplifying decision explicitly.
