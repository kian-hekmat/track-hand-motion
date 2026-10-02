"""Known sustained data gaps are declared in data/raw/out_of_frame_intervals.csv and must match
the extracted data exactly. Any sustained dropout NOT declared there fails, so an unexplained
gap can't slip through."""
import numpy as np
import pandas as pd

from config import RAW_DIR

SUSTAINED_MIN = 3  # same definition as scripts/detection_report.py
INTERVALS = pd.read_csv(RAW_DIR / "out_of_frame_intervals.csv")


def _frames(keypoints, take):
    f = keypoints(take).drop_duplicates("frame_idx").sort_values("frame_idx")
    return f.assign(t=f["timestamp_ms"] / 1000.0)


def test_declared_intervals_are_fully_undetected_with_detection_at_both_edges(keypoints):
    for r in INTERVALS.itertuples():
        f = _frames(keypoints, r.take)
        inside = f[(f.t >= r.start_s - 1e-6) & (f.t < r.end_s - 1e-6)]
        assert len(inside) >= SUSTAINED_MIN
        assert not inside["detected"].any(), f"{r.take} {r.start_s}-{r.end_s} has detected frames"
        before, after = f[f.t < r.start_s - 1e-6].iloc[-1], f[f.t >= r.end_s - 1e-6].iloc[0]
        assert before["detected"] and after["detected"], "interval should be the full dropout run"


def test_every_sustained_dropout_is_declared(take, keypoints):
    f = _frames(keypoints, take)
    missing = f.index[~f["detected"]].to_numpy()
    runs = np.split(missing, np.where(np.diff(missing) != 1)[0] + 1) if len(missing) else []
    declared = INTERVALS[INTERVALS["take"] == take]
    for run in (r for r in runs if len(r) >= SUSTAINED_MIN):
        t0, t1 = f.loc[run[0], "t"], f.loc[run[-1], "t"]
        assert ((declared.start_s <= t0 + 1e-6) & (declared.end_s >= t1 - 1e-6)).any(), \
            f"undeclared sustained dropout {take} {t0:.3f}-{t1:.3f}"
