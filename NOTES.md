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
