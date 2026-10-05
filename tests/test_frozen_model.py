"""Frozen-model integrity (models/segmenter_v2.*)."""
import numpy as np
import pandas as pd
import pytest

from config import LABELS, TAKE_GROUPS
from src import ground_truth as G
from src.features import take_data
from src.final import MODEL_PATH, load_meta, load_model, segment_new_take, sha256
from src.signals import load_frames

TAKES = sorted(TAKE_GROUPS)


def test_model_file_matches_recorded_hash():
    assert sha256(MODEL_PATH) == load_meta()["model_sha256"]
    load_model(verify=True)  # raises on mismatch


def test_metadata_records_training_takes_and_frozen_settings():
    m = load_meta()
    assert sorted(m["train_takes"]) == TAKES
    assert m["pelt_pen"] > 0 and m["boundary_tolerance_s"] == 0.1
    assert "all five" in m["note"]
    assert set(m["libraries"]) >= {"scikit-learn", "ruptures", "numpy", "pandas"}


def test_feature_columns_unchanged_since_freeze():
    _, feat, _, _ = take_data("vid1")
    assert list(feat.columns) == load_meta()["feature_columns"]


def test_tampered_model_is_detected(tmp_path, monkeypatch):
    import src.final as final
    bad = tmp_path / "model.joblib"
    bad.write_bytes(MODEL_PATH.read_bytes() + b"x")
    monkeypatch.setattr(final, "MODEL_PATH", bad)
    with pytest.raises(RuntimeError):
        final.load_model(verify=True)


def test_segment_new_take_is_valid_and_deterministic():
    a, sig, P = segment_new_take("vid3")
    b, _, _ = segment_new_take("vid3")
    pd.testing.assert_frame_equal(a, b)
    assert P.shape == (len(sig), len(LABELS)) and np.allclose(P.sum(1), 1, atol=1e-6)
    assert a["start_s"].iloc[0] == 0 and a["end_s"].iloc[-1] == pytest.approx(load_frames("vid3")["t"].iloc[-1])
    assert np.allclose(a["start_s"].to_numpy()[1:], a["end_s"].to_numpy()[:-1])
    assert set(a["label"]) <= set(LABELS)


def test_model_sanity_on_a_training_take_not_a_performance_claim():
    """vid2 is in the training set, so a high score here only shows the model loaded and works; the
    reported performance numbers come from leave-one-take-out predictions."""
    ev, sig, _ = segment_new_take("vid2")
    t = load_frames("vid2")["t"].to_numpy()
    acc = np.mean([a == b for a, b in zip(G.labels_at(G.load_ground_truth("vid2"), t), G.labels_at(ev, t))])
    assert acc > 0.9
