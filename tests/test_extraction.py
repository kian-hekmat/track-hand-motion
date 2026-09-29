"""Phase 1 checks on the raw keypoint output (one parametrized run per take).

Hand Landmarker has no per-landmark confidence; the per-frame `detected` flag is the
confidence signal, so "frames above the confidence threshold" == frames with detected=True.
"""
import numpy as np

from config import FRAME_COUNT_TOLERANCE
from src.landmarks import NUM_LANDMARKS

COORDS = ["x", "y", "z", "world_x", "world_y", "world_z"]


def test_no_nan_coords_when_detected(take, keypoints):
    df = keypoints(take)
    det = df[df["detected"]]
    assert len(det) > 0
    assert not det[COORDS].isna().any().any()
    assert det["handedness_score"].between(0, 1).all()


def test_undetected_frames_have_no_coords(take, keypoints):
    """Undetected frames must be explicit NaN, never zero-filled or forward-filled."""
    df = keypoints(take)
    assert df.loc[~df["detected"], COORDS].isna().all().all()


def test_every_frame_has_all_landmarks(take, keypoints):
    df = keypoints(take)
    per_frame = df.groupby("frame_idx")["keypoint_id"].agg(["count", "min", "max", "nunique"])
    assert (per_frame["count"] == NUM_LANDMARKS).all()
    assert (per_frame["nunique"] == NUM_LANDMARKS).all()
    assert (per_frame["min"] == 0).all() and (per_frame["max"] == NUM_LANDMARKS - 1).all()


def test_frame_count_matches_source_video(take, keypoints, source_pts):
    """Output frames == frames decoded from the original .mov by ffmpeg (exact), and
    within tolerance of mean_fps x duration."""
    n_out = keypoints(take)["frame_idx"].nunique()
    pts = source_pts(take)
    assert n_out == len(pts)

    duration = pts[-1] - pts[0]
    mean_fps = (len(pts) - 1) / duration
    expected = mean_fps * duration + 1
    assert abs(n_out - expected) / expected <= FRAME_COUNT_TOLERANCE


def test_timestamps_strictly_increasing(take, keypoints):
    ts = keypoints(take).drop_duplicates("frame_idx").sort_values("frame_idx")["timestamp_ms"]
    assert (np.diff(ts.to_numpy()) > 0).all()


def test_timestamps_match_source_pts(take, keypoints, source_pts):
    """Timestamps are the real (variable-rate) frame times, not frame_idx / fps."""
    ts_s = (keypoints(take).drop_duplicates("frame_idx").sort_values("frame_idx")
            ["timestamp_ms"].to_numpy() / 1000.0)
    pts = source_pts(take)
    assert np.abs(ts_s - pts).max() <= 0.001  # within 1 ms (integer-ms rounding)
