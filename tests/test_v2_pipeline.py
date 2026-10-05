"""Integration checks on the saved v2 cross-validated outputs (need scripts/cv_segmentation.py and
scripts/score_segmentation.py --events-dir data/segments/v2_* to have been run)."""
import json

import numpy as np
import pandas as pd
import pytest

from config import LABELS, SEGMENTS_DIR, TAKE_GROUPS
from src.evaluate import score_take
from src.ground_truth import load_ground_truth
from src.signals import load_frames

TAKES = sorted(TAKE_GROUPS)
VARIANTS = ("v2_pelt", "v2_grammar", "v2_argmax")


def events(variant, take):
    p = SEGMENTS_DIR / variant / f"{take}_events.csv"
    if not p.exists():
        pytest.fail(f"{p} missing: run scripts/cv_segmentation.py")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def history():
    p = SEGMENTS_DIR / "history.csv"
    if not p.exists():
        pytest.fail("history.csv missing")
    return pd.read_csv(p)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("take", TAKES)
def test_v2_events_cover_take_contiguously_with_valid_labels(variant, take):
    ev, gt = events(variant, take), load_ground_truth(take)
    assert ev["start_s"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert ev["end_s"].iloc[-1] == pytest.approx(gt["end_s"].iloc[-1], abs=1e-3)
    assert np.allclose(ev["start_s"].to_numpy()[1:], ev["end_s"].to_numpy()[:-1])
    assert set(ev["label"]) <= set(LABELS)
    assert (ev["label"].to_numpy()[1:] != ev["label"].to_numpy()[:-1]).all()


def test_cross_validation_never_trained_on_the_scored_take():
    sel = json.loads((SEGMENTS_DIR / "cv_selection.json").read_text())
    assert set(sel) == set(TAKES)
    for held, s in sel.items():
        assert held not in s["train_takes"] and len(s["train_takes"]) == 4


def test_history_is_not_stale_for_v2_pelt(history):
    for take in TAKES:
        t = load_frames(take)["t"].to_numpy()
        fresh = score_take(take, events("v2_pelt", take), t)[0]
        saved = history[(history["version"] == "v2_pelt") & (history["take"] == take) & (history["scope"] == "all")].iloc[0]
        for col in ("frame_accuracy", "balanced_accuracy", "tolerant_accuracy", "recall", "precision"):
            assert fresh[col] == pytest.approx(saved[col], rel=1e-6), (take, col)


def test_all_versions_are_tracked_in_history(history):
    assert {"v1_rules_pelt", *VARIANTS} <= set(history["version"])


def test_v2_headline_beats_v1_frame_accuracy_on_every_take(history):
    h = history[history["scope"] == "all"]
    for take in TAKES:
        v1 = h[(h["version"] == "v1_rules_pelt") & (h["take"] == take)]["frame_accuracy"].iloc[0]
        v2 = h[(h["version"] == "v2_pelt") & (h["take"] == take)]["frame_accuracy"].iloc[0]
        assert v2 > v1, take


def test_v2_frame_accuracy_floor_regression_guard(history):
    """Guard against silent regressions of the CURRENT results (not a target): measured values on
    2026-10-03 were >= 0.84 on every take."""
    h = history[(history["scope"] == "all") & (history["version"] == "v2_pelt")]
    assert (h["frame_accuracy"] >= 0.80).all(), h[["take", "frame_accuracy"]]


def test_classifier_training_is_deterministic():
    from src.features import take_data
    from src.learned import fit, predict_proba
    _, feat, _, _ = take_data("vid3")
    a = predict_proba(fit(["vid1", "vid2"], "logreg"), feat)
    b = predict_proba(fit(["vid1", "vid2"], "logreg"), feat)
    np.testing.assert_allclose(a, b)
