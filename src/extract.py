"""Phase 1: extract per-frame hand landmarks from a take in VIDEO running mode.

Output: one row per (frame, keypoint). Frames with no detected hand still get 21 rows
(coords NaN, detected=False) so the frame count in the output matches the video.

Schema note: the Hand Landmarker gives no per-landmark visibility score (unlike Pose).
Confidence is per frame: `detected` (hand found) and `handedness_score`.
"""
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from mediapipe.tasks.python.vision import RunningMode

from config import CONVERTED_DIR, EVIDENCE_DIR, RAW_DIR
from src.landmarks import NUM_LANDMARKS, bgr_to_mp_image, draw_hand, make_landmarker

COLUMNS = [
    "take", "frame_idx", "timestamp_ms", "keypoint_id",
    "x", "y", "z",                 # normalized image coords (x,y in [0,1]; z relative depth)
    "world_x", "world_y", "world_z",  # metric coords (m), origin at hand's geometric center
    "detected", "handedness", "handedness_score",
]


def ffmpeg_frame_count(path: Path) -> int:
    """Independent frame count by fully decoding with ffmpeg (not OpenCV)."""
    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path),
         "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    return int(out.rsplit("frame=", 1)[1].split()[0])


def extract_take(take: str, render_overlay: bool = False, overlay_scale: float = 0.5):
    video_path = CONVERTED_DIR / f"{take}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    nominal_fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    rows = []
    prev_ts = -1
    n_ts_bumped = 0
    frame_idx = 0

    with make_landmarker(RunningMode.VIDEO) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Real per-frame timestamp (iPhone video is variable frame rate).
            raw_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = int(round(raw_ms))
            if ts <= prev_ts:  # VIDEO mode requires strictly increasing timestamps
                ts = prev_ts + 1
                n_ts_bumped += 1
            prev_ts = ts

            res = landmarker.detect_for_video(bgr_to_mp_image(frame), ts)
            detected = len(res.hand_landmarks) > 0
            if detected:
                lms, wlms = res.hand_landmarks[0], res.hand_world_landmarks[0]
                hd = res.handedness[0][0]
                for k in range(NUM_LANDMARKS):
                    rows.append((take, frame_idx, ts, k,
                                 lms[k].x, lms[k].y, lms[k].z,
                                 wlms[k].x, wlms[k].y, wlms[k].z,
                                 True, hd.category_name, hd.score))
            else:
                for k in range(NUM_LANDMARKS):
                    rows.append((take, frame_idx, ts, k, *([np.nan] * 6),
                                 False, None, np.nan))

            if render_overlay:
                if detected:
                    draw_hand(frame, lms)
                label = (f"{take}  f={frame_idx}  t={ts/1000:.3f}s  "
                         + (f"{hd.category_name} {hd.score:.2f}" if detected else "NO HAND"))
                cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                            (0, 255, 0) if detected else (0, 0, 255), 4)
                small = cv2.resize(frame, None, fx=overlay_scale, fy=overlay_scale)
                if writer is None:
                    EVIDENCE_DIR.mkdir(exist_ok=True)
                    tmp_path = EVIDENCE_DIR / f"{take}_overlay_tmp.mp4"
                    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             nominal_fps, small.shape[1::-1])
                writer.write(small)
            frame_idx += 1
    cap.release()

    if writer is not None:
        writer.release()
        final = EVIDENCE_DIR / f"{take}_overlay.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", str(tmp_path), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", str(final)], check=True)
        tmp_path.unlink()

    df = pd.DataFrame(rows, columns=COLUMNS)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{take}_keypoints.csv", index=False)

    per_frame = df.drop_duplicates("frame_idx")
    duration_s = per_frame["timestamp_ms"].iloc[-1] / 1000.0
    meta = {
        "take": take,
        "frames_extracted": int(frame_idx),
        "frames_ffmpeg": ffmpeg_frame_count(video_path),
        "nominal_fps": nominal_fps,
        "last_timestamp_s": duration_s,
        "mean_fps_from_timestamps": (frame_idx - 1) / duration_s if duration_s else None,
        "timestamps_bumped_for_monotonicity": n_ts_bumped,
        "frames_not_detected": int((~per_frame["detected"]).sum()),
        "fraction_not_detected": float((~per_frame["detected"]).mean()),
        "handedness_counts": per_frame["handedness"].value_counts(dropna=True).to_dict(),
    }
    (RAW_DIR / f"{take}_meta.json").write_text(json.dumps(meta, indent=2))
    return df, meta
