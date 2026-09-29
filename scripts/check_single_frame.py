"""Step 1 sanity check: run Hand Landmarker in IMAGE mode on one frame and draw it."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONVERTED_DIR, EVIDENCE_DIR
from mediapipe.tasks.python.vision import RunningMode
from src.landmarks import bgr_to_mp_image, draw_hand, make_landmarker

video = sys.argv[1] if len(sys.argv) > 1 else "vid1"
t_s = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

cap = cv2.VideoCapture(str(CONVERTED_DIR / f"{video}.mp4"))
cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
ok, frame = cap.read()
assert ok, "could not read frame"

with make_landmarker(RunningMode.IMAGE) as lm:
    res = lm.detect(bgr_to_mp_image(frame))

print(f"hands detected: {len(res.hand_landmarks)}")
for hand, hd in zip(res.hand_landmarks, res.handedness):
    print(f"handedness: {hd[0].category_name} score={hd[0].score:.3f}")
    draw_hand(frame, hand)

EVIDENCE_DIR.mkdir(exist_ok=True)
out = EVIDENCE_DIR / f"{video}_single_frame_{t_s:.1f}s.jpg"
cv2.imwrite(str(out), frame)
print(f"wrote {out}")
