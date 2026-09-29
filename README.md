# Motion Intent Pipeline

An end-to-end data pipeline that turns recorded human hand motion into discrete "intent" events:

```
iPhone video → MediaPipe Hands → Postgres (raw) → Segmentation → Databricks (Spark) → Snowflake → Tableau
```

## Status

**Planning stage. No pipeline stage has been built or verified yet.** This README describes the intended design and the tests each stage must pass. Each section below should be updated to reflect reality as stages are completed, and no metric should be added here unless a saved test or output produced it.

| Phase | Stage | Status |
|---|---|---|
| 1 | Capture & keypoint extraction | Not started |
| 2 | Segmentation | Not started |
| 3 | Databricks processing | Not started |
| 4 | Snowflake | Not started |
| 5 | Tableau dashboard | Not started |

## Goals

Portfolio project built for a data-team role at Intuitive Surgical. The team's real problem: convert high-frequency robot-arm motion (position + time, tens of thousands of points over a few seconds) into discrete user-intent events, then store, query, and visualize them. This project mirrors that problem, using recorded human hand motion as a stand-in for robot-arm telemetry.

**Design choice to note:** self-recorded video of a hand is a deliberate substitute for real robot telemetry, which is not available.

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

- Tap the table once at the start of every take (audio + visual sync point).
- Say each phase aloud as it starts ("reach," "grab," "hold," "drop," "back," "rest"). The audio track is an approximate timeline; record the scoring tolerance used, since voice leads/lags motion.
- Right after Take 5, write down what happened in each cycle and the intended correct labels.
- Save per-take ground truth to `ground_truth/take_N.csv` with columns `take, cycle, label, start_s, end_s, source` (`source` = `audio` or `manual`).
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
4. Refine `ground_truth/take_N.csv` for at least Takes 1 and 2 by checking audio-derived timestamps against video frames. Takes 4 and 5 also need ground truth, since they are the stress tests.

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
- [ ] Every number used in resume bullets or conversation has a test or saved output that produced it
- [ ] `NOTES.md` documents at least one real limitation or failure mode
- [ ] This README matches what is actually built (no planned features described as done)

## Planned Repository Layout

Not yet created; subject to change.

```
ground_truth/     per-take labels (take_N.csv)
NOTES.md          limitations and failure analysis
queries.sql       Snowflake analytical queries
tests/            pytest suites
README.md
CLAUDE.md         project instructions for AI assistants
```

## Ground Rules for AI Assistants

See `CLAUDE.md`. In short: never mark a phase complete without its tests passing and shown; never fabricate or estimate a metric; stop and say so when a result looks wrong; flag any scope-simplifying decision explicitly.
