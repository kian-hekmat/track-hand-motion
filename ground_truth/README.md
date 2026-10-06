# Ground truth

New recordings vid6 and vid7 (`take_6.csv`, `take_7.csv`) are the hold-out test: label them with the same rules below, **before** any model output
exists, then lock them with `python scripts/run_new_take.py lock` (see `docs/new_takes.md`). vid6 has 4 cycles (25 rows), vid7 has 3 (19 rows). Both are locked (`data/holdout/lock.json`). vid6's times were carried to the corrected frame times
after the time-base fix (same frames; at most 1.0 ms; `data/holdout/vid6_label_time_remap.csv`).

One file per take: `take_N.csv` labels `vidN.mov` (confirmed mapping: vid1–3 clean, vid4 Fast,
vid5 Hard case). Columns: `take, cycle, label, start_s, end_s, source`.

- All timings were read off the video frames by hand (the overlay's `t=` stamp), so `source` is
  `manual` in every row. Audio cues are not used anywhere in this project.
- `start_s`/`end_s`: seconds on the same clock as `raw_keypoints.timestamp_ms / 1000`. Each file
  starts at 0 and ends at the take's last frame timestamp (checked by `tests/test_ground_truth.py`).
- Labels: `REST, REACH, GRASP, HOLD, RELEASE, RETRACT`.
- **One REST row per continuous still period.** The end of one cycle and the start of the next are
  the same physical stillness, so they are a single row, not two (a segmenter cannot see a boundary
  inside a still period). A take with three cycles has 19 rows: the opening REST, then for each
  cycle REACH, GRASP, HOLD, RELEASE, RETRACT, and the REST that follows it.
- Boundary rules (first frame where the event is visible; that frame's `t=` is both the ending
  phase's `end_s` and the next phase's `start_s`, no gaps):
  REST→REACH hand visibly leaves A; REACH→GRASP fingers start closing on the object;
  GRASP→HOLD lift finished and the lifted object stops moving, fingers closed; HOLD→RELEASE object
  touches the table and fingers start opening; RELEASE→RETRACT fingers open and hand starts back
  toward A; RETRACT→REST hand on A and stopped.
- Fast take (vid4): no stillness between phases; boundaries are marked at aperture/direction changes.

## vid5 notes

vid5 has four cycles (not three). What happened in each, as recorded by the person who performed it:

- Cycle 1 (hesitation): stopped halfway through the initial reach to the ball. the label should continue to be reach throughout the hesitation
- Cycle 2: normal cycle 
- Cycle 3 (failed grasp): grasped the ball and then released without picking it up, then grasped and picked it up and held it. Label should continue to be reach during the failed grasp.
- Cycle 4 (occlusion): grasped up the ball, turned hand to the side facing away from the camera so the ball is obscured from the camera, then rotated it back to the original position with the ball on the table and continued with the normaly cycle. The label should continue to be grasp throughout the occlusion

Frames with no hand data in vid5 (hand raised above the top of the frame): 9.335–9.835 s (inside
cycle 2 GRASP) and 17.103–17.270 s (the end of cycle 3 GRASP). See `data/raw/out_of_frame_intervals.csv`
and `NOTES.md`.

## Scoring tolerance

Boundary tolerance: 0.10 s (3 frames).

Originally 0.2 s. Lowered before any segmentation was run, because 0.2 s is wider than the shortest
ground-truth segments (vid4 RELEASE 0.13 s; vid5 RELEASE 0.07 s), which would inflate boundary recall by
chance. 0.10 s still exceeds the vid5 RELEASE, so precision and chance baselines are reported next to recall.
