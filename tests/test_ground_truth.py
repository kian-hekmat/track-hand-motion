"""Schema checks for ground_truth/take_N.csv. Blank template rows are allowed; any row that has
times filled in must be well-formed."""
import pandas as pd
import pytest

from config import ROOT

LABELS = {"REST", "REACH", "GRASP", "HOLD", "RELEASE", "RETRACT"}
COLS = ["take", "cycle", "label", "start_s", "end_s", "source"]
FILES = sorted((ROOT / "ground_truth").glob("take_*.csv"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_ground_truth_schema(path):
    df = pd.read_csv(path)
    assert list(df.columns) == COLS
    assert set(df["label"]) <= LABELS
    assert df["take"].nunique() == 1

    filled = df.dropna(subset=["start_s", "end_s"])
    assert (filled["source"] == "manual").all(), "ground truth is read off the video frames: source must be manual"
    assert (filled["end_s"] > filled["start_s"]).all()
    assert (filled["start_s"].diff().dropna() >= 0).all(), "rows must be in time order"
    assert (filled["start_s"].iloc[1:].to_numpy() >= filled["end_s"].iloc[:-1].to_numpy() - 1e-9).all(), "overlap"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_ground_truth_spans_whole_take(path):
    """Convention: labels start at t=0 and the last label ends at the take's last frame timestamp
    (meta.last_timestamp_s), so every frame has exactly one ground-truth label."""
    import json
    from config import RAW_DIR

    df = pd.read_csv(path)
    take = df["take"].iloc[0]
    last_ts = json.loads((RAW_DIR / f"{take}_meta.json").read_text())["last_timestamp_s"]
    assert df["start_s"].iloc[0] == pytest.approx(0.0, abs=1e-3)
    assert df["end_s"].iloc[-1] == pytest.approx(last_ts, abs=1e-3)
