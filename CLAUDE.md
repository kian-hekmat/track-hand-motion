# Motion Intent Pipeline — Project Instructions

## Project Overview

An end-to-end data pipeline: record real motion video → extract pose keypoints via computer vision → segment the resulting time series into discrete "intent" events → process the pipeline in Databricks → store structured results in Snowflake → visualize in Tableau.

**Purpose:** Convert high-frequency motion data (position + time, many points over seconds) into discrete intent events, store/query the results, and visualize them in Tableau. Recorded human motion video is the data source (keypoints extracted by computer vision rather than sensor telemetry).

**Non-negotiable constraint:** every claim made about the project must be true. Do not report a metric, an accuracy number, or a completed step that hasn't actually been verified by a test or a human look at the output. If a stage's output looks wrong, stop and say so — do not adjust it to "look better" without fixing the actual cause.

---

## Tech Stack

- **Python 3.11+** for all pipeline code
- **MediaPipe Hands** for keypoint extraction from video (gives wrist position for speed and fingertip positions for hand opening; Pose alone cannot distinguish open from closed hand)
- **pandas / NumPy** for time-series manipulation
- **ruptures** (or a small HMM, pick one and commit — see Phase 2) for changepoint/segmentation
- **PostgreSQL** (local, Docker) as the raw landing zone — reuse existing Docker Compose setup from the Research Digest Agent if compatible
- **Databricks Community Edition** for the transformation/processing stage
- **Snowflake** (trial account) for the structured, queryable output layer
- **Tableau Public** for the final dashboard
- **pytest** for all testing

---

## Phase 1 — Data Capture & Keypoint Extraction

**Goal:** Turn a real video into a raw time-series table of (timestamp, keypoint_id, x, y, z, confidence).

### Recording protocol

**Setup**
- Seated at a table; torso still, only the working arm moves. Non-working hand in lap, out of frame if possible.
- Two tape marks on the table: **Home (A)** ~10 cm in front of the body; **Target (B)** ~40 cm from A, running *across* the camera's view (not toward the lens).
- One light, rigid, graspable object (small cup, block, or ball), identical in every take, placed at B.
- iPhone on a stable stand (never handheld), landscape, roughly at table height, perpendicular to the A→B path. Plain non-reflective background, even diffuse lighting, no backlighting, short sleeves.
- Camera settings: **1080p HD at 30 fps** (60 fps only if the Fast take shows motion blur). Turn off Action Mode and cinematic/portrait video modes.
- Before recording: one slow dry run to confirm fingertips stay visible during the grasp. If the object hides them, raise or angle the camera slightly toward the front.

**Phase labels (the ground-truth vocabulary for the whole project)**

| # | Label | Action | Target duration | Expected signal signature |
|---|---|---|---|---|
| 0 | REST | Hand flat on A, still | 2 s | Near-zero wrist speed |
| 1 | REACH | Move A → object at B, opening fingers on approach | ~1 s | Speed rises then falls; aperture increases |
| 2 | GRASP | Close hand on object, lift ~2 cm | ~0.5 s | Aperture drops sharply; little travel |
| 3 | HOLD | Hold lifted object still | 1 s | Near-zero speed; aperture stays closed |
| 4 | RELEASE | Set object down, open hand | ~0.5 s | Aperture rises; little travel |
| 5 | RETRACT | Return to A, hand flat | ~1 s | Speed rises then falls |
| 6 | REST | Still at A | 2 s | Near-zero speed |

"Aperture" = distance between thumb tip and index fingertip keypoints. One cycle = phases 0–6. **Three cycles per take**, back to back (~20–25 s per take).

**Execution rules**
- Stop cleanly at every phase boundary in the clean takes (count "one-one-thousand" silently for holds and rests).
- Same path, same marks, same arc every cycle. One smooth motion per phase — corrections will be detected as extra phases.

**The five takes**
1. **Takes 1–3 — Clean:** protocol exactly as above. Primary data. At least two are fully ground-truth labeled.
2. **Take 4 — Fast:** same six phases, same order, no deliberate stops, ~half the time per phase. Boundaries exist only as aperture/direction changes, not stillness.
3. **Take 5 — Hard case**, a different deliberate problem per cycle:
   - Cycle 1 — **Hesitation:** pause ~0.5 s halfway through REACH, then continue. Correct label: one REACH, not two.
   - Cycle 2 — **Failed grasp:** close, open, re-grasp before lifting. Correct label decided in advance and written down (default: one long GRASP).
   - Cycle 3 — **Occlusion:** rotate the wrist during HOLD so the object hides the fingertips for ~1 s. Tests handling of low-confidence frames.

**Ground-truth capture during recording**
- Tap the table once, firmly, at the start of every take (visual sync point).
- Ground truth is read off the video frames (the overlay's `t=` stamp); audio cues are not used. Record the scoring tolerance used.
- Immediately after Take 5, write down exactly what happened in each cycle and the intended correct labels. This file is the ground truth for the NOTES.md limitation analysis.
- Save per-take ground truth to `ground_truth/take_N.csv` with columns: `take, cycle, label, start_s, end_s, source` (`source` = `manual`: read off the video frames).

**Before tearing down the setup:** run MediaPipe Hands on Take 1 and review the keypoint overlay. If fingers drop out during the grasp, fix angle/lighting and re-shoot immediately.

### Processing steps

1. Record the five takes per the protocol above.
2. Run each video through MediaPipe Hands to extract per-frame keypoint coordinates (21 hand landmarks per frame).
3. Write extraction output to a raw CSV/Parquet file per video: one row per (frame, keypoint), columns for timestamp, keypoint_id, x, y, z, visibility/confidence score.
4. Load raw output into Postgres as a landing table (`raw_keypoints`). Do not skip this step even though Databricks/Snowflake come later — the local Postgres table is your dev/test source of truth and lets you iterate fast without cloud round-trips.

**Testing & verification requirements (strict):**
- Manually inspect at least one full video's keypoint overlay (draw keypoints back onto the video frames) to confirm MediaPipe is tracking correctly, not silently producing garbage. Save this as a short annotated clip or a few frame screenshots — this is your evidence, not just a claim.
- Write a pytest test asserting: no null/NaN coordinates in the output for frames where confidence exceeds a defined threshold; frame count in the output matches expected frame count from video FPS × duration (within a small tolerance).
- Explicitly log and report the fraction of frames where confidence was below threshold (i.e., tracking failed) — do not discard this number, it is a real data-quality metric worth having on hand.

---

## Phase 2 — Segmentation (Time Series → Discrete Events)

**Goal:** Convert the raw keypoint time series into a labeled sequence of discrete events using the six-label vocabulary from Phase 1 (REST / REACH / GRASP / HOLD / RELEASE / RETRACT), each with start/end timestamps.

**Steps:**
1. Choose one approach and commit — do not build both:
   - **Changepoint detection** (`ruptures` library, e.g. PELT or Binary Segmentation on a derived signal like velocity or joint-angle magnitude), or
   - **A simple HMM/sequence model**, if leaning on RL/sequence-modeling background is preferred.
2. Derive a two-channel signal from the raw keypoints: **wrist speed** (frame-to-frame displacement magnitude of the wrist landmark, smoothed) and **hand aperture** (thumb-tip to index-tip distance, normalized by hand size, e.g. wrist-to-middle-MCP distance). Speed separates moving vs. still phases; aperture separates GRASP/HOLD from REACH/RELEASE. Segmenting raw 21-landmark data directly is harder and less interpretable.
3. Run segmentation on each take's derived signal to produce boundaries, then assign one of the six labels to each segment (rule-based on speed/aperture levels is acceptable and more explainable than a learned classifier at this data size).
4. Refine the ground truth captured during recording (`ground_truth/take_N.csv`) for at least Takes 1 and 2 by checking the labelled timestamps against the video frames. Takes 4 and 5 must also have ground truth, since they are the stress tests.

**Testing & verification requirements (strict):**
- Score detected segments against ground truth for every take that has it. Report at minimum: boundary timing error (mean absolute error in seconds), boundary recall within the stated tolerance, and per-label accuracy (fraction of frames given the correct label). Report clean takes, the Fast take, and the Hard-case take **separately** — never average them into one flattering number.
- For Take 5, report per-cycle results (hesitation, failed grasp, occlusion) so each failure mode is visible on its own.
- Write a pytest test that runs the segmentation function against a synthetic signal with known, programmatically-inserted changepoints (not from your real data) and asserts detected changepoints fall within a tolerance window of the true ones. This is your regression test — it must keep passing as you tune parameters later.
- If segmentation performs poorly on the "hard case" video from Phase 1, document why (in a NOTES.md or similar) rather than quietly excluding that video from the final demo. An honest limitation is a better story than a cherry-picked result.

---

## Phase 3 — Databricks Processing

**Goal:** Move the transformation logic (or a meaningful portion of it) into a Databricks notebook running on a Spark DataFrame, demonstrating the tool rather than just Python-on-a-laptop.

**Steps:**
1. Load the raw keypoint data (from Postgres, exported as CSV/Parquet) into a Databricks Community Edition notebook as a Spark DataFrame.
2. Reimplement the signal-derivation and/or segmentation step (or a meaningful chunk of the pipeline) using PySpark operations, not just calling the same pandas code inside a notebook cell. The point is demonstrating Spark-flavored data engineering (window functions over partitions, groupBy/agg), not just relocating Python.
3. Write the segmented-events output to a table Databricks can export (CSV/Delta) for the next phase.

**Testing & verification requirements (strict):**
- Confirm row counts match between the pandas-based pipeline output (Phase 2) and the PySpark-based output for the same input video — if they diverge, find out why before proceeding; do not assume the Spark version is correct just because it ran without error.
- Keep a note of one Spark-specific concept actually used and why (e.g., "used a window function partitioned by video_id, ordered by timestamp, to compute frame-to-frame deltas") — a specific, true description of what was done rather than just "used Databricks."

---

## Phase 4 — Snowflake

**Goal:** Land the final structured tables (raw keypoints, segmented events) in Snowflake as the queryable layer.

**Steps:**
1. Load the segmented-events table and a sampled/aggregated version of the raw keypoints table into Snowflake (via `COPY INTO` from staged files, or Snowsight's UI loader).
2. Write at least 3-5 real analytical SQL queries against the Snowflake tables that answer questions a stakeholder might actually ask, e.g.:
   - Average duration of each event type across all videos
   - Which video had the most ambiguous/short-duration segments
   - A query using a window function (`LAG`/`LEAD`) to compute time between consecutive events
3. Save these queries in a `queries.sql` file in the repo — this is evidence of SQL fluency beyond "I loaded data."

**Testing & verification requirements (strict):**
- Verify row counts in Snowflake match the source files exactly (no silent drops during load) — write this as an explicit check, not an assumption.
- Each of the 3-5 queries must return a sensible, sanity-checked result — eyeball the output against what you'd expect from having watched the videos yourself.

---

## Phase 5 — Tableau Dashboard

**Goal:** One dashboard, built from Snowflake (or exported CSV) data, that a non-technical viewer could read at a glance.

**Steps:**
1. Connect Tableau Public to the Snowflake tables (or exported CSVs if a live connection proves difficult on the free tier).
2. Build minimum two views:
   - A timeline: time on x-axis, colored bands/marks showing the active intent-segment
   - The raw derived signal (e.g., velocity) overlaid or paired with the segment timeline, so a viewer can see raw signal vs. detected event side by side
3. Combine into one dashboard with a short title/caption a non-technical stakeholder could understand without explanation.

**Testing & verification requirements (strict):**
- Show the dashboard to at least one person who hasn't seen the project before and ask them to describe what it shows, in their own words, without your explanation. If they can't, the dashboard isn't done — revise it. Visualization is a communication task, not just a chart-rendering task.

---

## Final Integration Check (before calling the project "done")

- [ ] Full pipeline runs end-to-end on at least one video from raw footage to final Tableau-ready export, without manual patching of intermediate files
- [ ] Every metric or number reported about the project has a corresponding test or saved output that produced it — no number should exist only in memory
- [ ] NOTES.md documents at least one real limitation or failure mode encountered (e.g., tracking confidence dropping during fast motion, or segmentation struggling on the hard-case video) — an honest limitation is better than a polished absence of problems
- [ ] README.md written describing the pipeline stages, matching what's actually built — no aspirational/planned features described as done

## Ground Rules for Any AI Assistant Working on This Project

- Never report a phase as complete without the corresponding test in that phase's "Testing & verification requirements" actually passing and being shown, not just claimed
- Never fabricate or estimate a metric (accuracy, error rate, row count) — compute it from real output every time
- If a step fails or a result looks wrong, say so plainly and stop rather than smoothing over it or quietly adjusting until it "looks right"
- Flag any place where a design decision was made to simplify scope (e.g., using a public dataset instead of self-recorded video) so it's a visible, discussed choice — not a silent substitution
