"""Phase 1 data-quality metric: fraction of frames where no hand was detected, per take.

Dropouts are split by run length because they have different causes (see NOTES):
  isolated = 1-2 frame gaps (mostly tracker flicker on a visible, flat, edge-on hand)
  sustained = runs of >= 3 frames (e.g. hand physically leaving the frame)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR

SUSTAINED_MIN = 3

rows = []
for path in sorted(RAW_DIR.glob("*_keypoints.csv")):
    f = pd.read_csv(path).drop_duplicates("frame_idx").sort_values("frame_idx")
    missing = f.loc[~f["detected"], "frame_idx"].to_numpy()
    runs = np.split(missing, np.where(np.diff(missing) != 1)[0] + 1) if len(missing) else []
    sustained = [r for r in runs if len(r) >= SUSTAINED_MIN]
    n = len(f)
    rows.append({
        "take": f["take"].iloc[0],
        "frames": n,
        "frames_not_detected": len(missing),
        "fraction_not_detected": round(len(missing) / n, 4),
        "isolated_dropout_frames": len(missing) - sum(len(r) for r in sustained),
        "sustained_dropout_frames": sum(len(r) for r in sustained),
        "sustained_runs_s": "; ".join(
            f"{f.set_index('frame_idx').timestamp_ms[r[0]]/1000:.2f}-"
            f"{f.set_index('frame_idx').timestamp_ms[r[-1]]/1000:.2f}" for r in sustained),
        "handedness_left_frames": int((f["handedness"] == "Left").sum()),
    })

report = pd.DataFrame(rows)
report.to_csv(RAW_DIR / "detection_report.csv", index=False)
print(report.to_string(index=False))
