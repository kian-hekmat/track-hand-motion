# New recordings: hold-out test of the frozen model

**Status: done on 2026-10-06.** Labels locked 12:07:00, scored once, exported. Results, error analysis and every fix made during the run: `NOTES.md`
(section "New recordings: hold-out test"). The recording notes at the bottom were not filled in.

Two new takes: **vid6** (4 slow cycles) and **vid7** (3 fast cycles). They are the only data the frozen model
(`models/segmenter_v2.joblib`, trained on vid1-5) and every design choice have never seen, so they give the project's only
unbiased score. They also serve as the end-to-end run in the Final Integration Check (raw video to Tableau-ready tables with
no manual patching).

The rules, enforced by `scripts/run_new_take.py`:
1. Ground truth is labelled **before** any model output exists.
2. The labels are **locked** (SHA-256 recorded) before scoring; scoring refuses if a label file or the model changes afterwards.
3. The frozen model runs **once, unchanged**: no retraining, no tuning, no threshold changes after seeing results.
4. Results are reported **as they come out**, as their own groups (`holdout_slow`, `holdout_fast`), never pooled with vid1-5.

## What you do

### 1. Put the videos in place
- Copy the slow take to `vids/vid6.mov` and the fast take to `vids/vid7.mov` (rename them; keep the `.mov` original format).
- **Do not trim the start.** If a clip has unwanted footage at the end only, keep the untrimmed original in `vids_original/` and trim
  the end, as was done for vid3 and vid4. Ask me if you need the trim done frame-accurately.
- Write a few lines in the "Recording notes" section at the bottom of this file: date, same or different session/day as vid1-5,
  anything changed in the setup (camera position, lighting, object, sleeves), and anything unusual during the takes.

### 2. Prepare (no model output is produced)
```
python scripts/run_new_take.py prepare
```
This converts each video, extracts the hand keypoints, writes the overlay clips (`evidence/vid6_overlay.mp4`, `vid7_overlay.mp4`),
loads Postgres, updates `data/raw/detection_report.csv`, and creates blank label files `ground_truth/take_6.csv` (25 rows) and
`take_7.csv` (19 rows). It never overwrites a label file that already exists.

### 3. Review the overlays
Watch both overlay clips. Check that the skeleton stays on the hand. If the hand leaves the frame for 3 or more frames, add the
interval to `data/raw/out_of_frame_intervals.csv` (as for vid5); the test suite fails until every sustained gap is declared.
Tell me if tracking looks wrong anywhere; that is a finding, not something to work around.

### 4. Label the ground truth (before anything else)
Fill in `start_s`, `end_s` and `source` (`manual`) in `ground_truth/take_6.csv` and `take_7.csv`, reading the `t=` stamp off the
overlay exactly as before. Same boundary rules as `ground_truth/README.md`; tolerance 0.10 s; first row starts at 0; the last row ends at
the take's last frame time; one REST row per still period. If a cycle went differently than planned, label what happened and write
it in the recording notes. **Do not look at any model output first; none exists yet, and keep it that way until step 5.**

### 5. Lock the labels
```
python scripts/run_new_take.py lock
```
It checks the labels (complete, contiguous, start 0, end = last frame, right number of cycles, valid labels) and refuses with a
list of problems if anything is off. When it passes it writes `data/holdout/lock.json` with the hashes of the two label files and
of the frozen model.

### 6. Score
```
python scripts/run_new_take.py score
```
Runs the frozen model on both takes, scores them against the locked labels (per take and per cycle; same metrics and 0.10 s
tolerance as vid1-5), writes `data/holdout/` (events, signals, `scores.csv`, `runs.json`), appends to `data/segments/history.csv`
as `v2_frozen_holdout`, and draws `evidence/holdout/vid6_segmentation.png` and `vid7_segmentation.png`.

### 7. End-to-end export
```
python scripts/run_new_take.py export
```
Builds the export tables for the new takes with the same code as vid1-5 (`data/holdout/export/`), the reference the cloud
checks compare against. Until the 2026-10-08 cloud cleanup this step also built Tableau tables locally with DuckDB; the
`data/holdout/tableau/` files it made for vid6-7 are kept as the reference for the Snowflake export check. To bring a new take
into the cloud path: upload its keypoint, meta and label files to the Volume and run the job (`databricks/cloud/README.md`).

`python scripts/run_new_take.py status` shows how far each take has got at any point.

## What the test suite does meanwhile
As soon as `vids/vid6.mov` and `vid7.mov` exist, the Phase 1 tests include them automatically. They fail until `prepare` has run
(no keypoints yet), and the ground-truth tests fail until the labels are filled in. That is intended: red means a step is not done.

## Fixed before seeing any result
Written down now so the reading cannot be adjusted afterwards:
- Model: `models/segmenter_v2.joblib` as frozen (hash in `models/segmenter_v2.json`), PELT penalty 1.0, tolerance 0.10 s.
- Reported per take and per cycle: frame accuracy, balanced accuracy, tolerant accuracy, boundary recall, precision and matched-boundary
  error, against the same chance and majority baselines as vid1-5.
- Known risk, stated in advance: training used vid1-5 plus copies sped up 1.5x, 2x and 3x, never slowed down. A take slower than the
  clean takes is partly outside what the model saw. The fast take is closer to vid4.
- Whatever the numbers are, they go into `NOTES.md` and the README unchanged. If they are poor, the cause is investigated and
  documented; the model is not re-tuned on these takes (that would turn them into training data).

## Recording notes (fill in)
- Date recorded:
- Same session as vid1-5? Anything changed in the setup?
- vid6 (slow, 4 cycles):
- vid7 (fast, 3 cycles):
