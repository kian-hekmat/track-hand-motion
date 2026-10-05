"""Ground-truth loading. Segments are [start_s, end_s); the final segment also includes its end
(the last frame's timestamp equals the last end_s)."""
import numpy as np
import pandas as pd

from config import GROUND_TRUTH_DIR, RAW_DIR

OUT_OF_FRAME_CSV = RAW_DIR / "out_of_frame_intervals.csv"


def take_number(take: str) -> int:
    return int(take.removeprefix("vid"))


def load_ground_truth(take: str) -> pd.DataFrame:
    return pd.read_csv(GROUND_TRUTH_DIR / f"take_{take_number(take)}.csv")


def labels_at(segments: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """Label of the segment containing each time in `t` (object array; None outside all segments).
    `segments` needs columns label, start_s, end_s sorted by time, contiguous."""
    t = np.asarray(t, dtype=float)
    starts = segments["start_s"].to_numpy(float)
    ends = segments["end_s"].to_numpy(float)
    labels = segments["label"].to_numpy(object)
    idx = np.searchsorted(starts, t, side="right") - 1
    out = np.full(t.shape, None, dtype=object)
    ok = (idx >= 0) & (idx < len(labels))
    ii = idx[ok]
    inside = (t[ok] < ends[ii]) | np.isclose(t[ok], ends[-1], atol=1e-9) & (ii == len(labels) - 1)
    pos = np.flatnonzero(ok)[inside]
    out[pos] = labels[ii[inside]]
    return out


def boundaries(segments: pd.DataFrame) -> np.ndarray:
    """Internal boundary times: the start of every segment except the first."""
    return segments["start_s"].to_numpy(float)[1:]


def cycle_of_boundaries(gt: pd.DataFrame) -> np.ndarray:
    """Cycle id of each internal boundary = cycle of the segment that follows it."""
    return gt["cycle"].to_numpy()[1:]


def cycle_spans(gt: pd.DataFrame) -> dict[int, tuple[float, float]]:
    g = gt.groupby("cycle")
    return {int(c): (float(g.get_group(c)["start_s"].min()), float(g.get_group(c)["end_s"].max()))
            for c in sorted(gt["cycle"].unique())}


def out_of_frame_intervals(take: str) -> list[tuple[float, float]]:
    df = pd.read_csv(OUT_OF_FRAME_CSV)
    df = df[df["take"] == take]
    return [(float(r.start_s), float(r.end_s)) for r in df.itertuples()]


def in_intervals(t: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    mask = np.zeros(t.shape, dtype=bool)
    for a, b in intervals:
        mask |= (t >= a - 1e-9) & (t < b - 1e-9)
    return mask
