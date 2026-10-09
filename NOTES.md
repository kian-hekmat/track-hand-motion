# Notes: limitations and failure modes

Every number here was computed from `data/raw/*` and `ground_truth/*` (see the pytest suite and
`scripts/detection_report.py`). Phase 1 only; segmentation limitations get added when Phase 2 exists.

## 1. vid5: the hand leaves the frame during GRASP (physical absence)

The hand was raised above the top of the frame, so there is genuinely no hand to track
(declared in `data/raw/out_of_frame_intervals.csv`; `tests/test_data_gaps.py` verifies the intervals
are fully undetected and that no other sustained dropout exists in any take):

| interval (s) | ground-truth phase | share of that phase with no data |
|---|---|---|
| 9.335–9.835 | cycle 2 GRASP (9.035–9.835) | 0.50 s of 0.80 s (~62%) |
| 17.103–17.270 | cycle 3 GRASP (16.037–17.270) | 0.167 s of 1.233 s (~14%) |

Consequences for later phases:
- These are not tracker errors, so they must not be "fixed" by interpolating or tuning the confidence
  threshold. Frames are left as NaN.
- The segmenter has no signal in these intervals, so any label there is a guess. Phase 2 scoring must
  report frames inside these intervals separately from frames with data.
- This is a protocol deviation: the dry run was meant to confirm fingertips stay visible through the
  grasp. In cycle 2 the grasp is mostly unobserved.

## 2. Start-of-take REST: intermittent missed detections (tracker flicker)

In the opening REST (hand flat on the table), the hand is repeatedly lost for one or two frames at a
time, then re-acquired. All of these gaps are isolated (1–2 frames, never a sustained run):

| take | opening REST (s) | frames undetected |
|---|---|---|
| vid2 | 0–1.935 | 27 of 58 |
| vid3 | 0–0.900 | 13 of 27 |
| vid5 | 0–1.067 | 10 of 32 |
| vid4 | 0–0.867 | 2 of 26 |
| vid1 | 0–1.833 | 0 of 55 |

Review verdict: vid3 "lost contact on and off at the starting rest position" (confirmed by the
numbers). vid2 was judged to track well, but its opening REST has the same pattern, at the same
rate as vid3. Likely cause (not proven): a flat, edge-on hand gives the detector little to go on.
The hand does not move in this phase, so the effect on speed is small, but the aperture channel
will be noisy there. Plan: segmentation must tolerate isolated NaN frames, not drop or zero-fill them.

Also note: vid3's earlier 0.53–0.67 s sustained dropout disappeared after the vid3 trim and
re-encode. The cause of that change is unexplained; the tracker's output on that stretch is
evidently sensitive to the re-encode.

## 3. Handedness mislabels ("Left") on a right-handed recording: a tracker quirk

All takes are of a **right hand** (known fact). MediaPipe reports `Left` on 49 of 3,834 detected
frames (1.3%): vid1 10, vid2 21, vid3 1, vid4 2, vid5 15. Investigation of the extracted data:

- All but a few are isolated 1–2 frame blips. The longest runs are 11 frames (vid2, 8.74–9.07 s)
  and 6 frames (vid5, 17.30–17.47 s).
- The classifier is unsure on those frames: mean handedness score 0.53–0.74 on `Left` frames versus
  0.88–0.93 on `Right` frames.
- They cluster in the moving phases rather than at rest. By ground-truth label: vid1 RELEASE 9,
  GRASP 1; vid2 RELEASE 9, RETRACT 8, REACH 4; vid3 RELEASE 1; vid4 RETRACT 1, REST 1; vid5 REACH 6,
  HOLD 6, GRASP 2, RETRACT 1. The wrist moves more between frames on these frames (median step
  about 1.2–4x the take median).
- They are not tied to detection dropouts (only 2 of the 49 sit next to a missed frame).

Likely cause (hypothesis, not proven): when the hand opens/rotates fast, the handedness head becomes
unreliable even though landmarks are still returned. I did not check landmark accuracy on every one
of these frames; the overlay review found tracking good in vid1, vid2, vid4.
Decision: `handedness` and `handedness_score` are kept in the raw data for transparency but are
**never used to filter frames**. Segmentation uses landmarks on every detected frame regardless of
the handedness label.

## 4. Data decisions

- vid3 and vid4 were trimmed to remove unwanted footage at the end (see README). vid3: cut at
  25.805 s, frame timestamps identical to the original. vid4: cropped by hand before processing.
- Hand Landmarker has no per-landmark confidence; "confidence" is the per-frame `detected` flag.
- The motion data is extracted from video with MediaPipe rather than recorded by motion sensors (a deliberate choice).

# Phase 2 v1 (rule-based baseline): results and failure analysis

Superseded by the v2 section at the end of this file; kept as the baseline and as the record of why v2 exists.

Full tables: `data/segments/results.md` / `scores.csv` (regenerate with `scripts/run_segmentation.py` then
`scripts/score_segmentation.py`); plots for review: `evidence/phase2/<take>_segmentation.png`.
Boundary tolerance **0.10 s**, fixed before any segmentation was run (originally 0.2 s; lowered because
0.2 s exceeds the shortest ground-truth segments). Method: ruptures PELT (L2) on sqrt(speed) and aperture,
rule-based labels, same-label neighbours merged. Reported per group, never pooled.

## Protocol and what was tuned on what

- Parameters (`data/segments/params.json`) were tuned by seeded random search (800 trials) on **vid1 and
  vid2 only**; objective = mean of boundary F1 and frame accuracy. vid3 is held-out clean data. vid4 and vid5
  were never used for tuning, and parameters were not changed after seeing their results.
- 10 parameters were tuned on 2 takes: expect optimism on vid1/vid2. The drop on vid3 (below) is consistent
  with that, but with 5 takes the size of the gap is itself noisy.
- Penalty/thresholds are one fixed set for all takes.

## Results (tolerance 0.10 s)

| take | group | true / pred / matched boundaries | recall | precision | MAE of matched (s) | chance recall (uniform / random) | frame acc | majority-label baseline |
|---|---|---|---|---|---|---|---|---|
| vid1 | clean (tuned on) | 18 / 13 / 9 | 0.50 | 0.69 | 0.033 | 0.06 / 0.07 | 0.68 | 0.27 |
| vid2 | clean (tuned on) | 18 / 18 / 14 | 0.78 | 0.78 | 0.041 | 0.17 / 0.10 | 0.78 | 0.31 |
| vid3 | clean (held out) | 18 / 17 / 11 | 0.61 | 0.65 | 0.028 | 0.17 / 0.12 | 0.59 | 0.25 |
| vid4 | fast | 18 / 6 / 6 | 0.33 | 1.00 | 0.022 | 0.17 / 0.12 | 0.45 | 0.36 |
| vid5 | hard | 24 / 22 / 13 | 0.54 | 0.59 | 0.029 | 0.04 / 0.13 | 0.54 | 0.26 |

MAE is over matched boundaries only; boundaries missed or falsely added show up in recall and precision.
Frame accuracy excludes the 20 vid5 frames with no hand (reported separately: 0.25).
Per-label frame accuracy (fraction of that label's frames predicted correctly):

| take | REST | REACH | GRASP | HOLD | RELEASE | RETRACT |
|---|---|---|---|---|---|---|
| vid1 | 1.00 | 0.97 | 0.26 | 0.37 | 0.75 | 0.90 |
| vid2 | 0.99 | 0.99 | 0.34 | 0.72 | 0.42 | 0.96 |
| vid3 | 0.99 | 0.65 | 0.55 | 0.15 | 0.44 | 0.90 |
| vid4 | 0.96 | 0.29 | 0.00 | 0.00 | 0.00 | 0.62 |
| vid5 | 0.98 | 0.72 | 0.02 | 0.25 | 0.00 | 0.96 |

Summary: boundaries that exist in speed (REST<->REACH, REACH/RETRACT<->REST) are found with ~0.03-0.04 s
error and REST, REACH and RETRACT are labelled well on clean takes. The segmenter is weak where the signal
differences are small: GRASP / HOLD / RELEASE. Every take beats the chance baselines, but the fast take
only modestly (frame accuracy 0.45 vs 0.36 majority baseline).

## Failure modes

1. **GRASP vs HOLD vs RELEASE (all takes, incl. clean).** Confusion on vid1: of 160 GRASP frames only 41 are
   labelled GRASP (54 HOLD, 40 RELEASE, 25 REACH); of 260 HOLD frames, 82 are GRASP and 81 RELEASE. Cause: in
   the real data GRASP and HOLD differ little in speed (phase medians ~0.4-0.8 vs ~0.3 hand-lengths/s, with
   within-phase standard deviations of 0.6-1.3) and have the same aperture, so PELT does not place a boundary
   between them (pinned by `test_weak_speed_only_step_is_not_detected_documented_limit`). The merged segment is
   then labelled from its start-to-end aperture trend, which is dominated by the rise at its end, so a
   GRASP+HOLD block is often called RELEASE (visible in the vid1 plot, 4.2-9.0 s).
2. **Fast take (vid4): under-segmentation.** Only 6 of 18 boundaries are detected (precision 1.00: what it finds
   is right, it just finds little); whole cycles collapse into one long RETRACT or REACH segment, so GRASP, HOLD
   and RELEASE are never labelled. The fixed penalty tuned on slow takes is too coarse for 0.13-0.5 s phases.
   A 0.13 s segment (vid4 RELEASE) is below the penalty even with a realistic aperture jump
   (`test_realistic_short_segment_is_not_detected_documented_limit`).
3. **RELEASE is never detected in vid4 or vid5** (true RELEASE lasts 0.13 s and 0.07-0.23 s). A 0.07 s segment
   cannot exist at all with the 3-sample minimum segment length (`test_minimum_segment_length_mechanics...`).
   The 0.10 s tolerance is wider than the 0.07 s vid5 RELEASE, so chance matches there are possible.
4. **Segment-level labelling on long merged segments** uses start/end trends and means; one label per segment
   is coarse when a segment spans several phases.
5. **PELT's piecewise-constant model vs bell-shaped speed.** Worked for REACH (a bell becomes one REACH after
   merging, `test_bell_shaped_reach_becomes_one_reach_after_merge`) but also contributes spurious splits.

## Hard take (vid5), per cycle

| cycle | what happened | ground truth | detected (summary) | boundaries true/pred/matched | frame acc |
|---|---|---|---|---|---|
| 1 hesitation | stopped halfway through the reach | one REACH 1.07-2.47 | **one** REACH 1.03-3.07 (hesitation handled; ends 0.6 s late because GRASP is missed), RELEASE (0.07 s) not found | 6 / 6 / 3 | 0.73 |
| 2 normal | normal cycle; hand left the frame 9.335-9.835 (inside GRASP) | REACH, GRASP, HOLD, RELEASE, RETRACT | REACH runs through the GRASP start, GRASP labelled RETRACT, then HOLD; RELEASE not found | 6 / 5 / 2 | 0.86 |
| 3 failed grasp | grasped, released without lifting, re-grasped and lifted | GRASP 16.04-17.27 (hand out of frame 17.103-17.270); the recorded rule keeps the failed attempt as REACH (see "Open question" at the end) | HOLD, then a spurious REACH (16.70-17.03), then GRASP | 6 / 6 / 2 | 0.46 |
| 4 occlusion | ball hidden by turning the hand during the grasp | one long GRASP 23.61-27.07 | HOLD (23.37-26.63), spurious REACH (26.63-27.00), GRASP | 6 / 5 / 3 | 0.21 |

Cycle 2's high frame accuracy (0.86) mostly reflects REACH, REST and RETRACT frames being easy; it says
little about the GRASP. Frames with no hand data are not scored in these numbers (they get their own row,
accuracy 0.25 over 20 frames). The two deliberately hard cycles (3, 4) are the worst, as expected: a long
GRASP with internal motion is read as HOLD/REACH.

## Disclosures

- **Tolerance:** changed from 0.2 s to 0.10 s before any segmentation was run (see `ground_truth/README.md`).
- **Synthetic test edited after a failure.** After tuning, `test_regression_known_changepoints...` failed
  because the original synthetic GRASP phase (speed 0.6 for 1.0 s) produced a GRASP->HOLD step the tuned
  segmenter could not see. The synthetic GRASP was changed to 1.5 s at speed 1.0, and the weak-step case was
  moved into its own test that pins the limitation. That is a test adjusted after seeing the result; the real
  data has the same weakness, which is reported above rather than hidden.
- **Parameters are not claimed optimal:** random search, 800 trials, one seed.
- **Human review:** the plots were looked at by the assistant (vid1, vid4, vid5); the project rules require a human
  look at `evidence/phase2/*.png` before Phase 2 is called complete.

## What would help (not done; any change after this point is informed by vid3-5 and must be reported as such)

Separate penalty or min-size handling for fast takes; segment-level labelling that looks at sub-segment
structure; a dedicated lift/grip-release detector using vertical wrist motion; using the physical sequence as
a prior (deliberately not used here so failures stay visible).


# Phase 2 v2: learned frame classifier + PELT on phase probabilities

## Why

The v1 review (yours, and the confusion matrices above) showed one structural failure: no boundary between
GRASP / HOLD / RELEASE, so merged blocks were labelled from their end-of-segment trend. Tuning thresholds could
not fix that, so v2 changes the structure instead.

## What changed

- **More signals:** height above home (`ydev`, `dy`) and off-axis offset (`q`, `dq`) in addition to speed,
  aperture and progress along the home-to-target axis.
- **Features (`src/features.py`):** 61 per-frame features: each signal at 4 window sizes (0.1-1.5 s) plus
  past-vs-future trend features (e.g. "aperture 0.5 s ahead minus 0.5 s behind"), rolling max/std of speed and
  min of aperture. Computed from one take's own signals only.
- **Frame classifier (`src/learned.py`):** gradient boosting (`HistGradientBoostingClassifier`) or logistic
  regression, class-balanced, producing a probability for each of the six phases per frame.
- **Segmentation (headline decoder `v2_pelt`):** ruptures PELT (L2) on the smoothed phase probabilities; each
  segment gets the label with the highest mean probability; adjacent equal labels merge. **The protocol's phase
  order is not used** (decision D2 kept).
- **Training augmentation:** each training take is also used time-compressed by 1.5x, 2x and 3x, so the Fast
  take's tempo is represented (training takes only; the scored take is never augmented or seen).
- **Two reported variants:** `v2_argmax` (per-frame argmax, no changepoint step; diagnostic) and `v2_grammar`
  (Viterbi over the protocol's phase order; uses protocol knowledge, so it is reported separately and is not the
  headline).

## Evaluation protocol (nested leave-one-take-out)

Each take is scored by a model trained on the other four takes only (`scripts/cv_segmentation.py`; asserted in
code and by `tests/test_v2_pipeline.py`). Classifier family, PELT penalty and grammar penalty are chosen by an inner
leave-one-take-out over those four takes; the scored take influences nothing. Gradient boosting was chosen in
every fold. Selection details: `data/segments/cv_selection.json`.

Leakage control (`tests/test_leakage_control.py`): trained on vid1-2 and applied to vid3, aligned labels give 0.94
frame accuracy; the same labels shifted in time give 0.01. The classifier learns real phase signatures.

## Results (tolerance 0.10 s; every number from `data/segments/history.csv`)

| take | group | version | frame acc | balanced acc | tolerant acc | boundary recall | precision | MAE matched (s) | pred / true boundaries |
|---|---|---|---|---|---|---|---|---|---|
| vid1 | clean | v1 | 0.680 | 0.708 | 0.717 | 0.50 | 0.69 | 0.033 | 13 / 18 |
| vid1 | clean | **v2** | **0.946** | 0.933 | 0.985 | 0.67 | 0.67 | 0.042 | 18 / 18 |
| vid2 | clean | v1 | 0.778 | 0.737 | 0.803 | 0.78 | 0.78 | 0.041 | 18 / 18 |
| vid2 | clean | **v2** | **0.967** | 0.955 | 0.999 | 0.83 | 0.83 | 0.048 | 18 / 18 |
| vid3 | clean | v1 | 0.592 | 0.615 | 0.628 | 0.61 | 0.65 | 0.028 | 17 / 18 |
| vid3 | clean | **v2** | **0.964** | 0.964 | 0.999 | 0.83 | 0.83 | 0.031 | 18 / 18 |
| vid4 | fast | v1 | 0.452 | 0.312 | 0.498 | 0.33 | 1.00 | 0.022 | 6 / 18 |
| vid4 | fast | **v2** | **0.911** | 0.785 | 1.000 | 0.78 | 0.93 | 0.031 | 15 / 18 |
| vid5 | hard | v1 | 0.542 | 0.490 | 0.579 | 0.54 | 0.59 | 0.029 | 22 / 24 |
| vid5 | hard | **v2** | **0.841** | 0.831 | 0.876 | 0.75 | 0.67 | 0.035 | 27 / 24 |

Frame accuracy excludes the 20 vid5 frames with no hand data (their own row: 0.50). Majority-label baselines are
0.25-0.36. Grammar variant (not the headline): frame accuracy 0.945 / 0.966 / 0.964 / 0.922 / 0.845 for
vid1-5; boundary recall 0.61 / 0.89 / 0.83 / 0.94 / 0.75. It helps mostly on the fast take and is otherwise
about equal to the headline; per-label tables: `data/segments/v2_*/results.md`.

How to read it: on the clean and fast takes tolerant accuracy is ~1.0, i.e. almost every remaining wrong frame
sits within 0.1 s of a true boundary (timing jitter, not a wrong phase). Boundary MAE got slightly worse
(0.03-0.05 s vs 0.02-0.04 s) because v2 matches many more boundaries (including harder ones).

Per-label frame accuracy (headline `v2_pelt`):

| take | REST | REACH | GRASP | HOLD | RELEASE | RETRACT |
|---|---|---|---|---|---|---|
| vid1 | 1.00 | 0.95 | 0.84 | 0.98 | 0.97 | 0.85 |
| vid2 | 0.97 | 0.93 | 0.98 | 1.00 | 0.90 | 0.95 |
| vid3 | 0.98 | 1.00 | 0.95 | 0.95 | 0.98 | 0.91 |
| vid4 | 0.99 | 0.92 | 0.83 | 1.00 | **0.00** | 0.97 |
| vid5 | 0.99 | **0.59** | 0.80 | 0.97 | 0.82 | 0.82 |

vid5 per cycle (frame acc / boundaries true-pred-matched): cycle 1 hesitation 0.97 / 6-7-5; cycle 2 normal
0.87 / 6-5-3; cycle 3 failed grasp 0.78 / 6-7-4; cycle 4 occlusion 0.78 / 6-8-4. (v1: 0.73, 0.86, 0.46, 0.21.)

## Remaining failure modes

1. **RELEASE in the fast take is still missed (0.00; 0.23 with the grammar prior).** The true segments are 0.13 s
   (about 4 frames) and the frames go to HOLD and RETRACT. vid5's 0.07-0.23 s RELEASE segments are found much
   better (0.82).
2. **vid5 REACH accuracy 0.59:** 84 REACH frames are called GRASP. In cycles 2 and 3 of vid5 the hand approaches
   slowly and lingers near the object before grasping; the classifier calls the lingering GRASP, so GRASP starts
   about 0.7 s (cycle 2) and 1.8 s (cycle 3) before the ground truth. Plausible cause: clean takes never contain
   such a long open-hand pause at the object (not proven).
3. **Occlusion cycle (vid5 cycle 4):** aperture drops to ~0.2 (hand rotated), far outside the range of the
   training takes; the output has a spurious 0.36 s REST (24.37-24.73 s) and a HOLD/GRASP flip before settling.
4. **No hand data:** frames in the two out-of-frame intervals: 0.50 accuracy (10 of 20).
5. **Boundary timing:** matched-boundary error 0.03-0.05 s; boundary recall 0.67-0.83 on clean takes because
   some boundaries (GRASP/HOLD in particular) land more than 0.10 s from the hand-labelled frame.

## Disclosures: how optimistic are these numbers?

- **Design was informed by all five takes.** I chose the features, time-scale factors (1.5/2/3, motivated by
  vid4's phases being 2-4x shorter) and the decision to add a learned classifier after seeing v1 fail on vid1-5.
  Cross-validation keeps each take out of its own model, but not out of the design process, so the numbers are
  optimistic relative to a fresh recording. Only a newly recorded take (not yet done) would give an unbiased check.
- **Same-session clean takes:** vid1-3 are three takes from one session of the same motion, so leave-one-out on
  them measures within-session generalisation. Cross-session generalisation is untested.
- **Experiments run:** one feature set, one nested CV run, three decoders (all reported). Before the CV I looked
  at the raw frame-argmax accuracy of two classifiers trained on vid1-4 and applied to vid5 (0.82 / 0.83) to check
  that the direction was promising; nothing was tuned from it. No second iteration was done.
- **Labelling is no longer rule-based:** the project spec allows rule-based or learned labelling; v2 trades
  explainability for accuracy (gradient boosting). v1 remains as the explainable baseline.
- **Boundaries still come from ruptures PELT** (committed approach), now on classifier outputs. A per-frame
  argmax without PELT (`v2_argmax`) scores about the same or slightly worse, so PELT is not carrying the result.
- Small hyperparameter grids (2 classifier families x 5 penalties); tolerance unchanged at 0.10 s.


# Frozen model and exported tables

## Freeze (`scripts/freeze_model.py`, `models/segmenter_v2.{joblib,json}`)

- **Model:** gradient boosting on the 61 features, trained on all five takes plus their 1.5x/2x/3x time-compressed
  copies. File hash, library versions, feature columns, training takes and settings are recorded in
  `models/segmenter_v2.json`; `tests/test_frozen_model.py` fails if the file is altered or the features change.
- **PELT penalty = 1.0**, chosen as the best mean objective over leave-one-take-out predictions of all five takes.
  The objective is flat there (mean 0.850-0.857 for penalties 0.5-2.0), so the choice is not sensitive.
  Because all five takes took part in that choice, scores on them are mildly optimistic; a newly recorded take is
  a genuine hold-out.
- **Canonical events** (`data/segments/v2_frozen_oof/`): each take's events come from the leave-one-out model that
  never saw that take, decoded with the frozen penalty. Scored as `v2_frozen_oof` in `history.csv`: frame accuracy
  0.949 / 0.966 / 0.966 / 0.911 / 0.841 for vid1-5 (nested CV had 0.946 / 0.967 / 0.964 / 0.911 / 0.841).
- **New recordings:** `src/final.py::segment_new_take(take)` runs the frozen model on any take that has been
  converted and extracted into `data/raw/`. Nothing is retrained.

## Exported tables (`scripts/export_tables.py` -> `data/export/`)

`events` (101 rows), `signals` (3,960, one per 30 Hz sample, with predicted and ground-truth label), `frames`
(3,960, one per video frame: wrist and thumb/index tip positions, detection flag, handedness), `ground_truth`
(101), `takes` (5), `scores` (10) as CSV, and `raw_keypoints.parquet` (83,160 rows = 3,960 frames x 21 landmarks,
read straight from the Postgres landing table). `manifest.json` has row counts and SHA-256 per file for the
Spark and Snowflake row-count checks.

Checked by `tests/test_export.py`: events partition every video frame exactly once (`n_frames` sums to the take's
frame count), are contiguous and start at 0; `signals` labels agree with events and ground truth; raw keypoint
rows equal the source CSVs and the Postgres count; manifest hashes match the files.
Sanity check of the events against the recordings: the clean takes each yield exactly 19 events with the expected
durations (HOLD ~2.7 s, GRASP ~1.7 s, REACH ~1.4 s, RETRACT ~0.9 s); the fast take yields 16 events with no
RELEASE (known miss); vid5 yields 28 events vs 25 in the ground truth (includes the spurious 0.37 s REST).


# Phase 3: Spark (Databricks)

> **Retired 2026-10-08 (cloud cleanup).** The files and tests named in this section were removed from the working tree and are kept at the git tag `pre-cloud-cleanup` (commit `8dc35b5`). The results below were verified by those tests at the time; the cloud path (last section) now covers this stage and is checked on every job run.

Status: **complete.** Verified on local Spark 3.5.5 (Java 11) and **run on Databricks** (Free Edition workspace, serverless
compute, inputs in a Unity Catalog Volume): all 20 cells finished without error, all 12 checks passed. Evidence:
`evidence/databricks_phase3.html`, parsed and asserted by `tests/test_databricks_evidence.py` (checks text, 101-row events table
identical to the pandas events, correlation table, notebook code identical to the repo notebook). The Databricks runtime
version was not captured in the export (compute type only). Run steps: `databricks/README.md`.

## Scope decision (flagged)

Reimplemented in PySpark: (A) signal derivation from raw keypoints (wrist speed, hand aperture) and (B) event
construction and per-event aggregation. Not reimplemented: the gradient-boosting classifier and `ruptures` (no sensible
Spark equivalent); the frozen model's per-frame labels are an input table. So the events are *built* in Spark from
labels produced in Python; the segmentation itself is not run in Spark.

## Spark concepts actually used (all present in `databricks/spark_transforms.py`)

Window functions partitioned by take and ordered by frame (`lag`/`lead` central-difference velocity with the real,
variable frame timestamps; `avg` over `rowsBetween(-2, 2)` smoothing); `groupBy/agg` with an exact `percentile`
(per-take hand size) and a broadcast join; conditional aggregation to pivot 21 landmark rows into one row per frame;
gaps-and-islands (`lag` + cumulative-sum window + `groupBy/agg` + `lead`) for events; a range join for per-event stats.

## Local verification (`tests/test_spark.py`, notebook executed end to end)

- Row counts: raw 83,160 = 21 x 3,960 frames; Spark frames and signals = pandas frames = `takes.n_frames` for every take.
- Spark events = pandas events: 101 / 101, identical label, start and end for every event; `n_frames` identical; the events
  cover every frame exactly once.
- Undetected frames (vid1 1, vid2 30, vid3 21, vid4 16, vid5 58) are NULL in every signal column.
- **A bug was caught by these checks:** central differences skip the centre frame, so an undetected frame initially got a
  speed from its neighbours (silent interpolation), and the moving average did the same. Fixed by masking undetected frames.
- Spark and pandas signals agree on the same motion but are **not identical** (different methods: central difference + 5-frame
  moving average on native frames vs 30 Hz grid + Savitzky-Golay). Pearson r, speed / aperture: vid1 0.988 / 0.994, vid2
  0.988 / 0.996, vid3 0.994 / 0.994, vid4 0.989 / **0.874**, vid5 0.981 / 0.993. The lower vid4 aperture value is the fast
  take, where phases last only a few frames and a +/-2-frame moving average blurs them (a likely cause, not tested).
- Spark hand size for vid1 (282.28344492244173 px) equals the pandas value (282.28344492244184 px) to 13 digits.


# Phase 4: Snowflake

> **Retired 2026-10-08 (cloud cleanup).** The files and tests named in this section were removed from the working tree and are kept at the git tag `pre-cloud-cleanup` (commit `8dc35b5`). The results below were verified by those tests at the time; the cloud path (last section) now covers this stage and is checked on every job run.

Status: **complete.** Run on Snowflake on 2026-10-05: 46 of 46 verification checks passed and all five query results match the expected
results exactly. Evidence: `snowflake/actual_results/` (downloads), checked by `tests/test_snowflake_evidence.py`. Run steps and the
list of what is and is not proved: `snowflake/README.md`.

## What exists

- `scripts/build_snowflake_scripts.py` generates `snowflake/01_setup.sql` (database, schema, file formats, stage, 7 tables),
  `02_load.sql` (`COPY INTO` from the stage) and `03_verify.sql` (46 checks). Table columns come from the exported files, and the
  expected counts come from `data/export/manifest.json` and the per-take metadata, not from the loaded tables.
- Loaded tables: `takes` (5), `events` (101), `ground_truth` (101), `signals` (3,960), `frames` (3,960; the aggregated raw
  keypoints, one row per frame), `scores` (10), and the full `raw_keypoints` (83,160 rows, from Parquet; identical to the Postgres
  count). Reserved words renamed: `group` -> `take_group`, `false` -> `false_boundaries`.
- `queries.sql`: (1) average duration per event type by take kind, (2) most ambiguous takes, (3) time between consecutive REACH
  events (`LAG`), (4) phase transitions and whether they follow the protocol order (`LEAD`), (5) prediction agreement per phase.

## Local verification (`tests/test_snowflake_sql.py`, DuckDB)

- All 46 verification checks pass on the real export; **six deliberate corruptions** (a deleted event, frame or keypoint row, a
  shifted event start, a wrong frame count, a duplicated event) are each caught as `FAIL`.
- Table definitions match the files' column order (CSV loads map by position); no reserved words used as columns.
- The queries run and reproduce the saved expected results (`snowflake/expected_results/`), and results make sense against
  the recordings: clean-take HOLD averages 2.65 s; the fast take has no RELEASE events (the known miss); every clean transition
  follows the protocol order; the fast take's only off-protocol transition is HOLD to RETRACT (missing RELEASE); the hard take
  shows the occlusion-cycle flips; cycle time is about 10 s clean, 2.5 s fast, 6 to 9 s hard.
- Limit of this local check: DuckDB is not Snowflake. The real run is described in the next section.

## Reading query 2 correctly

"Ambiguous" means an event shorter than 0.3 s or detected with mean phase probability below 0.7 (thresholds fixed before looking at
results). vid5 ranks first (6 of 28 events). The fast take scores 0 ambiguous events, but only because its real 0.13 s RELEASE was
never detected, so the events table cannot show that ambiguity. The query measures the events as detected, not the ground truth.

## Snowflake run: what the saved files show

- `verify.csv` (46 rows): all `PASS`; `expected` equals `actual` everywhere; row counts equal the manifest (5, 101, 101, 3,960, 3,960, 10,
  83,160). Counts of events, frames, raw keypoints, signals and ground-truth segments also match per take; every video frame is in exactly
  one event; events are contiguous.
- Queries 1 to 5: identical to the locally produced results, and identical (maximum difference 0.00) to an independent pandas
  recomputation from the exported CSVs. Results make sense against the recordings: clean HOLD averages 2.65 s; the fast take has no
  RELEASE events; every clean-take transition follows the protocol order; the fast take's only off-protocol transition is HOLD to
  RETRACT; cycle time is about 10 to 12 s clean, 2.5 s fast, 6 to 9 s hard; vid5 ranks most ambiguous.
- The files carry upper-case column names (Snowflake's output) and differ byte-for-byte from the local expected files, so they are
  downloads, not copies.
- The comparison script was tightened during this review: it previously inherited numpy's default relative tolerance (about 1e-5);
  it now requires an exact match to 1e-6 absolute. All six files still match.

## Snowflake run: what is not proved

- Values in `raw_keypoints` and `frames` are verified only by row counts and a NULL check. `04_fingerprint.sql` (optional) adds
  value-level sums per table and take; locally it catches single-value corruption that row counts miss
  (`tests/test_snowflake_sql.py`), but it has not been run on Snowflake yet.
- The first load attempt left six of seven tables empty (only `takes` appears in `COPY_HISTORY`). The cause was not recorded. In the
  diagnostic script only statements 3 and 5 have run on Snowflake.
- Snowflake edition, region and warehouse size actually used were not captured.
- Human sanity check against the videos (spec requirement): done using `snowflake/sanity_check_checklist.md` (specific timestamps per
  query, generated from the data). The person who watched the videos reported in chat that everything matches the expected results,
  including the known explanations (missing fast-take RELEASE, vid5 occlusion-cycle flips, vid5 early GRASP). The filled-in
  checklist itself was not saved, so this is a recorded verbal sign-off, not a document.


# Phase 5: Tableau

> **Partly retired 2026-10-08.** The DuckDB table generator (`scripts/make_tableau_tables.py`), the comparison script and the matplotlib target image named below were removed (kept at the tag `pre-cloud-cleanup`). `data/tableau/` itself, the workbook and its verification stay. The cloud dashboard data now comes from the Snowflake `CLOUD` views (last section).

Status: **dashboard built by the project owner, saved (`tableau/hand-motion-phases.twbx`) and published on Tableau Public; workbook and export verified against the data; first-time-viewer test still open.** Steps and link: `tableau/README.md`. Export: `evidence/tableau_dashboard.png`.

- Tableau Public (the free edition) cannot connect to Snowflake, to the author's knowledge (not verified). The Tableau-ready tables are therefore CSVs produced
  by the SQL views in `snowflake/05_tableau_views.sql`, run locally in DuckDB (`scripts/make_tableau_tables.py`). The same views can be run in Snowflake and the
  exports compared with `scripts/compare_snowflake_results.py` (optional; not run yet).
- Tables: `tableau_phases.csv` (202 rows: 101 detected + 101 hand-labelled segments), `tableau_signals.csv` (3,960 samples), `tableau_accuracy.csv` (5).
  Plain-language phase names (At rest, Reaching, Grasping, Holding, Releasing, Returning), a fixed phase order and take labels (Take 1 (clean) ... Take 5 (hard)) are
  in the data. `tests/test_tableau_tables.py` checks: segments contiguous and covering each take for both sources, row counts equal the sources, name mapping complete,
  accuracy equals the scored frame accuracy, speed empty exactly where the hand was out of view, and the CSVs are not stale.
- `tableau/target_dashboard.png` was drawn with matplotlib from the same tables, as a visual target. It is not a Tableau output.
- Known data features to expect on the dashboard: a short speed spike in the first 0.1 s of takes 1 and 4 (cause not investigated: the table tap or a smoothing-filter
  edge effect), and two gaps in take 5's speed line where the hand left the frame.
- Still required by the spec: the first-time-viewer test (`tableau/user_test.md`); the dashboard is not done until someone new can describe it.

## Dashboard verification (`scripts/verify_dashboard_image.py`, `tests/test_tableau_dashboard_evidence.py`)

Re-done for the polished dashboard of 2026-10-06 (four charts). The first export's results are superseded.

- **Workbook** (`tableau/hand-motion-phases.twbx`): it packages exactly the three tested CSVs (byte-identical to `data/tableau/`); the timeline is restricted to detected phases by a data-source filter (`source = Detected`); each sheet uses the expected fields (timeline: `start_s` sized by `duration_s`, coloured by `phase_name`; speed: `speed` over `t_s` per take; scatter: one point per sample of `speed` against `aperture`, coloured by the detected phase; accuracy: `percent_frames_matching_human_labels`).
- **Picture** (`evidence/tableau_dashboard.png`): the timeline bands decode to exactly the detected events for all 5 takes (19, 19, 19, 16, 28 segments, same phases in the same order; starts within 0.046 s, about 1.6 pixels); the accuracy bars decode to 94.9, 96.6, 96.6, 91.1 and 84.1, equal to the scored values; the speed lines correlate 0.961 to 0.985 with the real speed signal at one consistent scale; the scatter draws all six phases and their vertical order matches the data (rank correlation 0.94). Its horizontal pixel check is weak (0.77) because overlapping circles hide the dense low-speed region, which is why its data binding is verified from the workbook. Each scale is calibrated on one take; the others are out-of-sample.
- **Published copy**: on 2026-10-06 the Tableau Public page loaded and showed the same four charts and the same accuracy labels as the export. This is a visual check only.
- Layout: the script finds each chart inside a box (`LAYOUT`) and reads the phase colours from the workbook. A re-export with a different arrangement needs `LAYOUT` updated; the tests then fail loudly rather than pass silently.

## Dashboard review (readability, not data)

Polished: axis titles with units on the scatter ("Wrist Speed (hand lengths/second)", "Hand opening (thumb-index gap / hand length)") and "Time (seconds)" under the speed panels; value labels on the accuracy bars; legend in phase order; the stray legend card removed; the timeline stacked above the speed panels on the same 0 to 34 s axis; a new speed-vs-hand-opening scatter.

Still worth fixing:

1. Colours are still Tableau's defaults, so "At rest" uses the same blue as the speed lines and the accuracy bars. The project palette is listed below.
2. The phase legend appears twice and is titled "Phase Name". Keep one legend and retitle it "What the hand is doing".
3. Row headers read "Take Label". Rename them to "Recording".
4. The speed panels' y-axis numbers are clipped ("1." instead of 10) and have no title or unit. Widen the axis and title it "Wrist speed (hand lengths per second)".
5. The timeline axis says "Start time (seconds)". It shows time, so "Time (seconds)" is accurate.
6. There is no caption explaining what the picture shows.
7. Accuracy labels show two decimals ("94.90%"), and the Take 2 and 3 labels are dark text on dark bars. Use one decimal and a light label colour, or put the labels outside the bars.
8. The scatter does not say its colours are the *detected* phase. Add "(colour = detected phase)" to its title.
9. Sheets are named "Sheet 1" to "Sheet 4" and the dashboard "Dashboard 2", which shows in the public URL. The accuracy table is attached twice as two data sources.

Whether a first-time viewer can read it is not known until `tableau/user_test.md` is run. Items 4 (speed axis with clipped numbers and no unit) and 6 (no caption) are the most likely causes of a failed answer to Q3 ("what does the line tell you").


# New recordings: hold-out test (run 2026-10-06)

Two new takes, vid6 (4 slow cycles) and vid7 (3 fast cycles), scored by the frozen model (`models/segmenter_v2.joblib`, trained on vid1-5).
No retraining, tuning or design change was made with them. Order kept and recorded: labels by the project owner, then locked
(`data/holdout/lock.json`, 12:07:00), then scored once (`data/holdout/runs.json`). `tests/test_holdout_results.py` checks the lock still
matches the label and model files, that scoring came after locking, and that every number is reproducible from the saved events.

## Results (tolerance 0.10 s; never pooled with vid1-5)

| take | kind | frames scored | frame accuracy | balanced | tolerant | boundary recall / precision | matched-boundary error (s) | majority baseline | chance recall |
|---|---|---|---|---|---|---|---|---|---|
| vid6 | slow, 4 cycles | 1,277 (+14 out of frame) | **0.839** | 0.825 | 0.879 | 0.50 / 0.40 | 0.042 | 0.31 | 0.13 |
| vid7 | fast, 3 cycles | 383 (+25 out of frame) | **0.862** | 0.892 | 0.950 | 0.56 / 0.59 | 0.050 | 0.29 | 0.22 |

Per cycle, frame accuracy: vid6 0.86, 0.85, 0.87, **0.72**; vid7 0.94, 0.83, **0.75**. Per phase: vid6 REST 0.99, REACH **0.70**, GRASP 0.74,
HOLD 0.89, RELEASE 0.86, RETRACT 0.77; vid7 REST **0.70**, REACH 1.00, GRASP 0.97, HOLD 0.83, RELEASE 0.91, RETRACT 0.94.
Frames with no hand data (scored separately): 0.43 (vid6, 14 frames) and 0.60 (vid7, 25 frames).

**Against the development numbers** (leave-one-take-out on vid1-5): clean takes 0.95 to 0.97, vid4 (fast) 0.91, vid5 (hard) 0.84. On new data
the frozen model scores 0.84 and 0.86: about 0.11 to 0.13 below the clean takes and 0.05 below vid4. Boundary recall drops from 0.61-0.89 to
0.50-0.56. The cross-validated numbers were optimistic, as stated beforehand; these are the project's unbiased figures. Both takes remain far above
the baselines (majority 0.29-0.31; random-boundary recall 0.13-0.22).

## Why the errors happen (checked against the plots in `evidence/holdout/` and the confusion matrices)

1. **A lingering approach is read as grasping (vid6).** In cycles 1 and 3 the hand reaches quickly, then waits open near the ball for 1 to 2 s; the
   model labels the wait GRASP, so REACH is split (REACH, GRASP, REACH, GRASP). 61 of 220 REACH frames go to GRASP. This is the failure first seen in
   vid5 cycles 2-3, now reproduced on unseen data. It matches the risk stated before scoring: slow motion is outside the training augmentation
   (which only sped takes up).
2. **A slow set-down is read as grasping (vid6).** In cycles 2 and 4 a false GRASP appears between HOLD and RELEASE (39 HOLD frames to GRASP).
3. **vid6 cycle 4 is the weakest (0.72):** the GRASP-to-HOLD boundary comes 0.5 s early, around the stretch where the hand was at the top edge of the frame.
4. **REST timing in the fast take (vid7).** Phases are mostly right (tolerant accuracy 0.95), but REST boundaries are 0.2 to 0.3 s off and the final
   REST is read as RETRACT (21 REST frames to RETRACT, 12 to REACH). That is why boundary recall falls to 0.33 and 0.17 in cycles 2 and 3 while frame
   accuracy stays at 0.83 and 0.75.
5. **vid7 does not test vid4's RELEASE problem.** RELEASE is found well in vid7 (0.91) because its releases last 0.43 to 0.60 s and its cycles about 4.5 s;
   vid4's releases lasted 0.13 s and its cycles about 2.5 s. The very-fast case remains untested on new data.

## Found and fixed during the run (before locking unless stated)

- **Pipeline bug: time base.** The conversion forced a 1/600 s time grid. vid1-5 and vid7 use 600, but vid6 was saved with a 90 kHz time base, so its
  frame times were rounded by up to 1.17 ms. `tests/test_extraction.py` (timestamps within 1 ms of the source) caught it. Fix: the conversion now
  keeps each source's own time base (`scripts/convert_videos.py`, new test). vid6 was re-extracted: same 1,291 frames, same 23 undetected frames.
- **vid6 labels carried to the corrected times.** The owner read the labels off the old overlay. Each boundary was a real frame time, so each was moved
  to the corrected time of the same frame: 30 of 52 times changed, by at most 1.0 ms (`data/holdout/vid6_label_time_remap.csv`, checked by a test).
- **Label files:** a stray first line (`take_6`, `take_7`) and an empty `source` column. Timings untouched; `source` set to `manual` (read off the overlay).
- **Out-of-frame intervals:** the file had the same stray first line. vid7's three intervals ended on the last frame without a hand; they were moved one
  frame later to the file's convention (first frame with the hand back). vid6's 14-frame dropout (38.55 to 38.98 s) was not declared; frames 1152-1172
  were checked (hand and ball at the top edge of the picture, partly out of view) and it was added as 38.546 to 39.013 s.
- No intermediate pipeline file (keypoints, signals, events, exports) was edited by hand.

## What this does and does not show

- Two takes by the same person, object and (as far as recorded) setup. The recording notes in `docs/new_takes.md` were not filled in, so whether this
  is a different session from vid1-5 is not documented.
- It is a fair test of the frozen model on unseen motion; it is not a test of other people, objects or camera positions.
- The end-to-end run (raw video to Tableau-ready tables, `data/holdout/tableau/`) worked through `scripts/run_new_take.py` with no hand edits to
  intermediate files; the 5-take export stayed unchanged.

## Framework notes

`src/holdout.py`, `scripts/run_new_take.py`, steps in `docs/new_takes.md`. Changes made while building it so the new takes could not leak into verified
results: `scripts/export_tables.py` now exports only the requested takes (it used to read every take in Postgres); `tests/test_export.py` counts only
the exported takes; scoring and plotting use `ALL_GROUPS`. Framework tests: `tests/test_holdout_framework.py` (sandbox); run checks:
`tests/test_holdout_results.py`.


# Cloud path: Databricks -> Snowflake -> Tableau (2026-10-07 to 2026-10-08)

Plan: `docs/cloud_pipeline_plan.md`. Run steps, results and evidence per milestone: `databricks/cloud/README.md`.

## What it reproduces, and how closely

Every comparison is with the verified local results, with tolerances fixed in `src/cloud/checks.py` before the runs.
- **Exact:** per-frame and per-sample phase labels, every event's label, start, end, duration and sample count (150 events,
  vid1-7), every count in the scores, every row of the Snowflake `CLOUD` tables against the verified `PIPELINE` tables (vid1-5),
  and the answers of the five queries for vid1-5.
- **Within floating-point noise:** motion signals differ from the local ones by at most 1.4e-13 (serverless has numpy 2.3.4,
  pandas 2.3.3, scipy 1.16.3; local numpy 1.26.4, pandas 3.0.6, scipy 1.17.1), phase probabilities by at most 3.5e-18, frame
  coordinates by one bit (2.2e-16: Spark and pandas parse CSV numbers differently), averaged event confidence by 5.6e-16.

## Failures and findings during the migration (all recorded with evidence)

1. **Serverless rejects the Spark connector's `sfURL` option** (`SERVERLESS_WRITE_OPTIONS_NOT_ALLOWED`); it accepts `host`.
   Found by the write check on 2026-10-07.
2. **Snowflake's Python connector disappeared after `%pip install` + restart** on serverless, although it was preinstalled
   before. M3 run 1 published the tables, then failed at `import snowflake.connector`; no Snowflake-side check ran. Fixed by
   installing it in the notebook's `%pip` cell (`evidence/cloud_m3_publish_snowflake_run1_failed.html`).
3. **A bug in a check, not in the data:** M3 run 2 failed "snowflake parity scores" (30 of 31 checks passed). Snowflake reports
   column names in lower case (`acc_rest`), the tolerance table said `acc_REST`, so those columns were compared exactly and
   last-bit rounding in the CSV-loaded `PIPELINE` values counted as differences. A direct comparison in Snowflake found no
   differing value. Fixed with case-insensitive lookups and a regression test (`evidence/cloud_m3_publish_snowflake_run2.html`).
   DuckDB keeps the original case, which is why the local test had passed.
4. **NaN versus NULL:** values the pandas code marks as NaN arrive in Spark as NaN, not NULL, so Spark averages became NaN where
   pandas skips missing values. They are converted to NULL after every `applyInPandas` step. Found by the local tests before any
   cloud run.
5. **The verified export files carry last-bit rounding** (up to 3.6e-15 in event times) because they were written after a
   default-precision CSV read. The cloud events match the original reference exactly; the comparison with the export files uses
   the stated 1e-12 tolerance for those columns.
6. **A check that could not fail its task:** the notebooks printed `N FAILED` but still finished successfully, which in a job
   would have let M3 publish after a failed M2 check. Both now raise after logging (found while building M4).

## Limits

- **Extraction is local by decision.** Converting and extracting the videos with MediaPipe runs on a laptop; moving it to
  Databricks (M5) was cancelled by the project owner. The cloud path starts from uploaded keypoint files, not from video.
- **Tableau Public cannot connect to Snowflake.** The three views are downloaded by hand and checked
  (`scripts/check_cloud_tableau_exports.py`); the dashboard is a copy, not a live connection.
- **The in-Snowflake comparison needs `PIPELINE`.** It was built by the retired Phase 4 scripts (tag `pre-cloud-cleanup`). If the
  trial account is replaced, those scripts are what rebuild it.
- **Same scores, same caveats:** the cloud path reproduces the model's results; it does not change how optimistic they are.
  vid1-5 are still scored by leave-one-take-out models, and vid6-7 remain the only unbiased figures (0.84 and 0.86 frame accuracy).
- **The serverless environment is not pinned.** The job's notebooks carry no `environment_version`, so jobs run in Databricks'
  default serverless environment. A diagnostic job without a pin ran on Python 3.11 with numpy 1.23, while the M1 check (run from
  the UI, pinned to environment 6) ran on Python 3.12 with numpy 2.3. Which one the recorded job runs used was not logged; their
  checks passed either way. Adding `environment_version = "6"` to the notebooks would fix the environment.
- **The Snowsight download rounds decimals** to about 10 significant digits (differences up to 5e-10 against the verified tables,
  2026-10-08), inside the export check's 1e-9 tolerance.

## M5 feasibility probe (2026-10-08, cancelled afterwards; nothing from it was kept in the pipeline)

Before M5 was cancelled, a scratch probe ran MediaPipe on serverless for vid4 and vid7. Recorded here because the findings explain
why cloud extraction would not reproduce the verified keypoints:
- Serverless runs on ARM (aarch64). MediaPipe 0.10.21, which made the verified keypoints, has no ARM build there; 0.10.18 does.
- Installing MediaPipe 0.10.18 into the notebook downgrades protobuf and numpy, and the notebook kernel then fails to start. It ran
  in an isolated subprocess instead (`pip install --target`).
- Conversion (tone mapping) worked, with every frame timestamp identical. Keypoints moved by a median of about 2 px (worst cases
  88-133 px), 12 of vid4's 281 frames switched between hand and no hand, and the phase sequence stayed identical. Frame accuracy:
  vid4 0.911 (unchanged), vid7 0.851 against the verified 0.862. On a Mac, MediaPipe 0.10.18 alone gave vid4 0.918, so both the
  version and the hardware change results.

# Open question: the vid5 failed-grasp label (cycle 3)

`CLAUDE.md` (the plan) and earlier versions of these notes describe the failed grasp as labelled "one long GRASP". The note written
by the person who performed it (`ground_truth/README.md`) says the label should stay REACH during the failed attempt. The labels file
has REACH until 16.037 s and GRASP from 16.037 to 17.270 s, which fits the recorded rule if the failed attempt happens before
16.037 s; that has not been checked against the video. The scores use the labels file as it is.

