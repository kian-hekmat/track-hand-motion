# Phase 2 roadmap: segmentation (keypoint time series → labelled events)

Status: executed (see NOTES.md for results). Changes from this plan: tolerance lowered to 0.10 s; PELT channels use fixed physical scales instead of per-take range scaling (range scaling amplified noise on flat signals). Facts below were computed from the current
`data/raw/*` and `ground_truth/*`.

## Goal and done-criteria (from CLAUDE.md)

Turn each take's keypoints into a contiguous sequence of events labelled REST / REACH / GRASP / HOLD /
RELEASE / RETRACT with start/end times, then score against `ground_truth/take_N.csv`.

Phase 2 is done only when all of these are shown, not claimed:
1. Segmentation function passes a synthetic-signal regression test (known inserted changepoints).
2. Scores exist for every take, reported **separately** for clean (vid1–3), Fast (vid4), Hard case (vid5):
   boundary MAE (s), boundary recall within 0.2 s, per-label frame accuracy.
3. vid5 has per-cycle results (hesitation c1, normal c2, failed grasp c3, occlusion c4).
4. Poor hard-case results are explained in `NOTES.md`, not excluded.
5. A human has looked at plots of detected vs ground-truth segments (saved in `evidence/phase2/`).

## Facts that shape the plan

- **The 0.2 s tolerance is large relative to some ground-truth segments.** Shortest segments:
  vid4 RELEASE 0.13 s (4 frames) with 7 of 19 segments under 0.4 s; vid5 RELEASE 0.07 s (2 frames), 5 of
  25 under 0.4 s. In vid4 a boundary can be "within tolerance" of its neighbour by chance. Recall alone
  will flatter the Fast take, so we also need precision and chance baselines (see tests).
- Clean takes are easy by comparison: shortest segment 0.27–0.53 s; median 1.1–1.5 s.
- **`ruptures` is not installed** (scipy and matplotlib are). It needs adding to `requirements.txt`.
- Wrist translation must come from **image** x/y. The `world_*` landmarks are centred on the hand, so the
  wrist's world coordinates do not move when the hand travels. Image x/y are normalized by frame width and
  height separately (1080p landscape), so they must be scaled to a common unit before computing distance.
- Timestamps are variable-rate (iPhone VFR), so speed must use real `dt`, and `ruptures` needs a uniform
  grid, so the signals get resampled.
- Gaps: isolated 1–2 frame NaNs are common in opening RESTs (vid2, vid3, vid5); vid5 has two real
  out-of-frame intervals (9.335–9.835, 17.103–17.270) inside GRASPs.

## Design decisions (to confirm before coding)

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | Approach | **`ruptures` PELT, L2 cost**, on a standardized 2-channel signal | Spec says pick one. PELT needs only a penalty; Binseg/Dynp need a known number of breakpoints, which would use ground-truth structure the hard take doesn't follow |
| D2 | Labelling | Per-segment rules on speed, aperture level, aperture trend, travel direction. Expected cycle order is **not** used to assign labels; it is only reported as a sanity check | If order were enforced, the hard take would be "fixed" by the grammar and the failures hidden |
| D3 | Same-label neighbours | Merge adjacent segments that receive the same label | This is how a hesitation inside REACH correctly becomes one REACH |
| D4 | Tuning protocol | Tune thresholds/penalty on **vid1 and vid2 only**; vid3 is held-out clean; vid4 and vid5 never tuned on. Record exactly what was tuned on what | Five videos is tiny. This stops "tuned until it looks right" from being reported as accuracy |
| D5 | Gaps | Linearly interpolate gaps of ≤2 frames in the derived signals only (flagged in an `interpolated` column); longer gaps stay NaN and are scored separately | Raw data is never altered; the vid5 out-of-frame intervals are not invented |

## Steps

### 1. Setup
- Add `ruptures` to `requirements.txt`; install; record the version.
- Create `src/signals.py`, `src/segment.py`, `src/score.py`, `scripts/run_segmentation.py`,
  `scripts/score_segmentation.py`, `scripts/plot_segmentation.py`. Output dir `data/segments/`.
- Ground-truth loader (`load_ground_truth(take)`) returning per-frame labels on the take's frame
  timestamps, using [start, end) semantics.

### 2. Signal derivation (`src/signals.py`)
- Per-frame table from `raw_keypoints` CSVs: one row per frame with wrist (id 0) image position, thumb tip
  (4), index tip (8), middle MCP (9).
- **Wrist speed:** convert x/y to a common length unit (x·width, y·height, or both scaled by the aspect
  ratio), then ‖Δposition‖/Δt using real timestamps. Normalize by hand size so it is scale-invariant.
- **Hand size:** wrist→middle-MCP distance per frame, using world coordinates (metric, stable), smoothed
  to a take-level robust median for the speed normalization.
- **Aperture:** thumb-tip→index-tip distance divided by hand size. Decide image vs world coordinates by
  testing which is more stable under hand rotation (HOLD in vid5 c4 rotates the wrist).
- **Direction:** signed wrist velocity along the A→B axis (camera is perpendicular to the path, so the
  image x component), needed to tell REACH from RETRACT.
- Gap handling per D5; resample to a uniform grid (30 Hz); zero-phase smoothing (Savitzky–Golay or
  Butterworth `filtfilt`) so boundaries are not shifted by filter lag. Window length is a parameter.
- Output columns: `t, speed, aperture, direction, aperture_slope, missing, interpolated`.

### 3. Segmentation (`src/segment.py`)
- Standardize each channel (robust: median/MAD), run PELT (`model="l2"`, `min_size` ≈ 3 frames, `jump` 1)
  on [speed, aperture] (direction used for labelling only, or added as a third channel if tests show
  it helps; decide on the tuning takes).
- Penalty selection rule defined up front (e.g. a fixed multiple of the noise variance estimated from
  REST), then frozen. No per-take penalty.
- NaN handling: segment on the interpolated signal, then mark segments with a high missing fraction.
- Return boundaries as times, not indices.

### 4. Label assignment
- For each segment compute mean speed, mean aperture, aperture slope, net displacement/direction,
  fraction missing.
- Rule set (levels set from tuning takes, expressed relative to each take's own REST level so it is
  scale-free):
  low speed + open aperture → REST; low speed + closed aperture → HOLD; moving toward B → REACH;
  moving back toward A → RETRACT; low travel + aperture falling → GRASP; low travel + aperture rising →
  RELEASE.
- Merge adjacent same-label segments (D3). Output the events table:
  `take, event_idx, label, start_s, end_s, duration_s, n_frames, mean_speed, mean_aperture, frac_missing`.
- Rules must be written down in this file / `NOTES.md` before looking at vid4/vid5 results.

### 5. Scoring (`src/score.py`)
- **Boundary match:** one-to-one greedy/Hungarian matching of predicted to true boundaries within 0.2 s.
  Report MAE over **matched** boundaries only, plus counts of unmatched true (misses) and unmatched
  predicted (false boundaries). MAE alone hides misses.
- **Recall and precision** within tolerance, plus the chance baselines: (a) equally spaced boundaries with
  the same count, (b) 1000 random boundary sets with the same count. Report model − baseline.
- **Per-label frame accuracy** over frames that have ground truth, plus a 6×6 confusion matrix.
- **Split frames by data availability:** frames inside declared out-of-frame intervals
  (`out_of_frame_intervals.csv`) reported separately.
- **Grouping:** clean takes, Fast, Hard case in separate tables. No pooled number. vid5 additionally by
  cycle (c1 hesitation, c2 normal, c3 failed grasp, c4 occlusion), cycle spans taken from the ground truth.
- Write results to `data/segments/scores.csv` and a short markdown results table.

### 6. Evidence and review
- Plot per take: speed and aperture with ground-truth bands and detected bands (matplotlib), saved to
  `evidence/phase2/`. A human looks at them (clean, vid4, vid5 at minimum) before any claim.

### 7. Documentation
- `NOTES.md`: results, the vid5 per-cycle failure analysis (expected: hesitation okay if merge works;
  failed grasp likely over-segmented into GRASP/RELEASE/GRASP; occlusion/out-of-frame low confidence),
  tuning protocol, what was tuned on what.
- `README.md` Phase 2 status updated only with numbers produced by saved output.

## Challenges and how the plan handles them

| Challenge | Mitigation |
|---|---|
| Tolerance 0.2 s vs 0.07–0.13 s segments (vid4, vid5 RELEASE) | Precision + chance baselines; report segment counts; per-label accuracy of RELEASE flagged as near-unmeasurable in vid4/vid5 |
| Piecewise-constant PELT on a speed *bell* curve (REACH rises then falls) will split one REACH into several pieces | `min_size`, penalty tuning, and same-label merge (D3); verified by a synthetic test with a bell-shaped speed profile |
| Speed alone cannot separate REACH from RETRACT, or GRASP from RELEASE | Signed direction channel and aperture slope; unit-tested label rules |
| REST vs HOLD both have near-zero speed | Aperture level (open vs closed), checked per take relative to its own REST |
| Smoothing blurs boundaries (hurts the Fast take most) | Zero-phase filter, window chosen on tuning takes only, test on a short-segment synthetic signal |
| Tuning on five takes leaks into reported accuracy | D4 split; vid4/vid5 never tuned; log the exact parameter values and which takes informed them |
| Hard-case failures (failed grasp, occlusion, out-of-frame) tempt "fixes" | Expected failures are documented, not tuned away (project rule); rules written before running vid4/5 |
| Ground truth has human timing noise (frame-level reading, judgement at GRASP/HOLD) | Tolerance 0.2 s is fixed in advance; report error distribution, not only the mean |
| Missing frames at rest (isolated NaN) could create false changepoints | D5 interpolation; test that injecting 1–2 frame NaN gaps does not add boundaries |
| `world_*` wrist coords do not encode translation | Use image x/y for speed (unit-tested with a synthetic translating hand) |
| VFR timestamps | Compute with real `dt`; resample to uniform grid; test on irregular timestamps |

## Tests to write

**Signals (`tests/test_signals.py`)**
- Synthetic keypoints with a wrist moving at constant known speed → derived speed equals it (units and
  aspect ratio handled; fails if x/y are not scaled to a common unit).
- Same motion with the hand twice as large (scaled keypoints) → normalized speed and aperture unchanged.
- Non-uniform timestamps (jittered dt) → speed still correct.
- Aperture of synthetic thumb/index at known separation, invariant to rotating the hand.
- NaN gaps: ≤2-frame gap is interpolated and flagged; a 15-frame gap stays NaN; raw table is never modified.
- Resampling output is uniform and covers the take duration.

**Segmentation (`tests/test_segment.py`)**
- **Required regression test:** synthetic 2-channel signal with programmatically inserted changepoints
  (known levels + noise, fixed seed); detected boundaries within a tolerance window of the true ones,
  none missing, none extra. Must keep passing as parameters change.
- Bell-shaped speed (REACH-like) yields one segment after label-merge, not several.
- Short-segment synthetic (0.13 s) at 30 Hz: documents the detection limit; the test pins which behaviour
  is expected rather than hiding it.
- Injected 1–2 frame NaN gaps do not add boundaries.
- Determinism: same input → identical output.
- Output is contiguous: first start = take start, last end = take end, no gaps/overlaps, boundaries
  strictly increasing.

**Labelling (`tests/test_labels.py`)**
- Synthetic segment feature sets for each of the six labels → correct label.
- Adjacent same-label segments merge; different labels do not.
- Scale-free: scaling all speeds by a constant leaves labels unchanged.

**Scoring (`tests/test_score.py`)**
- Prediction equal to ground truth → MAE 0, recall 1, precision 1, accuracy 1.
- Every boundary shifted by 0.1 s → MAE 0.1, recall 1; by 0.3 s → recall 0, MAE undefined (not 0.3).
- Extra predicted boundary lowers precision, not recall; missing boundary lowers recall, not MAE.
- One-to-one matching: two predictions near one true boundary count once.
- Tolerance edge case exactly at 0.2 s (inclusive/exclusive stated and tested).
- Per-label accuracy hand-computed on a tiny example, confusion matrix totals equal frame count.
- Chance baselines: ground truth scored against random boundaries gives low recall on the clean takes.
- Per-cycle slicing of vid5 matches hand-computed cycle spans.
- No pooled score function exists (guards the "never average" rule by construction).

**Integration (`tests/test_pipeline_phase2.py`, needs real data)**
- For all five takes: events cover the whole take, labels in the vocabulary, event count in a sane range,
  deterministic across two runs.
- Scores file contains clean, Fast and Hard-case rows separately and a per-cycle block for vid5.
- Frames in declared out-of-frame intervals are excluded from the primary accuracy and reported separately.

## Order of work

1. Confirm D1–D5. 2. Setup + ground-truth loader. 3. Signals + tests. 4. Scoring + tests (built before
the segmenter, so a segmenter cannot be shaped to a scorer written afterward). 5. Segmenter + synthetic
regression test. 6. Labelling rules + tests, written down before running vid4/vid5. 7. Tune on vid1–2
only. 8. Run all takes; scores; plots; human review. 9. NOTES.md and README.

## Risks to the claim "Phase 2 complete"

- Clean takes may segment well while vid4/vid5 do poorly. That is an acceptable, documented outcome.
- If a result looks wrong (e.g. a take with implausibly perfect recall), investigate cause first;
  do not adjust parameters until it looks better.
