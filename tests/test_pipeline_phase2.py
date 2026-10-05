"""Phase 2 integration checks on the real takes. Need `python scripts/run_segmentation.py` and
`scripts/score_segmentation.py` to have been run (fail, not skip, if outputs are missing)."""
import json

import numpy as np
import pandas as pd
import pytest

from config import BOUNDARY_TOLERANCE_S, LABELS, SEGMENTS_DIR, TAKE_GROUPS, TUNING_TAKES
from src.evaluate import score_take
from src.ground_truth import in_intervals, load_ground_truth, out_of_frame_intervals
from src.segment import load_params, segment_take
from src.signals import load_frames

TAKES = sorted(TAKE_GROUPS)


@pytest.fixture(scope="module")
def scores():
    p = SEGMENTS_DIR / "scores.csv"
    if not p.exists():
        pytest.fail("run scripts/run_segmentation.py and scripts/score_segmentation.py")
    return pd.read_csv(p)


def events(take):
    p = SEGMENTS_DIR / f"{take}_events.csv"
    if not p.exists():
        pytest.fail(f"{p} missing: run scripts/run_segmentation.py")
    return pd.read_csv(p)


@pytest.mark.parametrize("take", TAKES)
def test_events_cover_whole_take_contiguously_with_valid_labels(take):
    ev, gt = events(take), load_ground_truth(take)
    assert ev["start_s"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert ev["end_s"].iloc[-1] == pytest.approx(gt["end_s"].iloc[-1], abs=1e-3)
    assert np.allclose(ev["start_s"].to_numpy()[1:], ev["end_s"].to_numpy()[:-1])
    assert (ev["duration_s"] > 0).all()
    assert set(ev["label"]) <= set(LABELS)
    assert (ev["label"].to_numpy()[1:] != ev["label"].to_numpy()[:-1]).all()
    assert 3 <= len(ev) <= 40  # sane count; ground truth has 19-25


def test_segmentation_is_deterministic():
    a, _ = segment_take("vid1", load_params())
    b, _ = segment_take("vid1", load_params())
    pd.testing.assert_frame_equal(a, b)
    saved = events("vid1")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), saved, check_dtype=False, rtol=1e-6)


def test_params_were_tuned_on_tuning_takes_only():
    meta = json.loads((SEGMENTS_DIR / "params.json").read_text())
    assert meta["tuned_on"] == list(TUNING_TAKES) == ["vid1", "vid2"]
    assert meta["boundary_tolerance_s"] == BOUNDARY_TOLERANCE_S


def test_scores_are_reported_per_group_and_never_pooled(scores):
    assert set(scores["group"]) == {"clean", "fast", "hard"}
    assert set(scores["take"]) == set(TAKES)
    assert not scores["take"].str.contains("all|pool|mean", case=False).any()
    assert not scores["scope"].str.contains("pool|overall|mean", case=False).any()
    v5 = scores[scores["take"] == "vid5"]["scope"].tolist()
    assert {"all", "out_of_frame", "cycle1", "cycle2", "cycle3", "cycle4"} <= set(v5)
    assert (scores["tolerance_s"] == BOUNDARY_TOLERANCE_S).all()


def test_scores_file_is_not_stale(scores):
    """Re-score from the saved events; must reproduce scores.csv."""
    for take in TAKES:
        t = load_frames(take)["t"].to_numpy()
        fresh = pd.DataFrame(score_take(take, events(take), t, with_cycles=(take == "vid5")))
        saved = scores[scores["take"] == take].reset_index(drop=True)
        for col in ("n_true", "n_pred", "n_matched", "frame_accuracy", "mae_matched_s", "recall"):
            np.testing.assert_allclose(fresh[col].to_numpy(float), saved[col].to_numpy(float), rtol=1e-6, equal_nan=True)


def test_out_of_frame_frames_are_excluded_from_primary_frame_accuracy(scores):
    t = load_frames("vid5")["t"].to_numpy()
    n_oof = int(in_intervals(t, out_of_frame_intervals("vid5")).sum())
    assert n_oof == 20  # 15 + 5 sustained undetected frames in vid5
    row = scores[(scores["take"] == "vid5") & (scores["scope"] == "all")].iloc[0]
    assert row["n_frames"] + row["n_unlabelled"] == len(t) - n_oof
    oof_row = scores[(scores["take"] == "vid5") & (scores["scope"] == "out_of_frame")].iloc[0]
    assert oof_row["n_frames"] == n_oof


def test_results_beat_chance_on_every_take(scores):
    """Sanity, not a quality claim: boundary recall exceeds what equally spaced or random boundary
    sets of the same count achieve, and frame accuracy exceeds always predicting the majority label."""
    for r in scores[scores["scope"] == "all"].itertuples():
        assert r.recall > max(r.chance_uniform_recall, r.chance_random_recall), r.take
        assert r.frame_accuracy > r.majority_baseline, r.take
