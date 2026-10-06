"""The hold-out run on vid6 and vid7 (data/holdout/): blind order kept, results reproducible from the frozen model, export complete.
Needs `scripts/run_new_take.py prepare/lock/score/export` to have been run."""
import datetime
import json

import numpy as np
import pandas as pd
import pytest

from config import GROUND_TRUTH_DIR, HOLDOUT_DIR, HOLDOUT_TAKES, LABELS, RAW_DIR, ROOT, SEGMENTS_DIR
from src import holdout as H
from src.evaluate import score_take
from src.final import MODEL_PATH, segment_new_take, sha256
from src.signals import load_frames

TAKES = sorted(HOLDOUT_TAKES)


def need(p):
    if not p.exists():
        pytest.fail(f"{p} missing: run scripts/run_new_take.py")
    return p


@pytest.fixture(scope="module")
def lock():
    return json.loads(need(HOLDOUT_DIR / "lock.json").read_text())


@pytest.fixture(scope="module")
def scores():
    return pd.read_csv(need(HOLDOUT_DIR / "scores.csv"))


def test_labels_and_model_are_unchanged_since_locking(lock):
    H.check_lock(TAKES, sha256(MODEL_PATH))                    # raises if a label file or the model changed
    assert lock["model_train_takes"] == ["vid1", "vid2", "vid3", "vid4", "vid5"]


def test_scoring_happened_after_locking(lock):
    runs = json.loads(need(HOLDOUT_DIR / "runs.json").read_text())
    assert len(runs) >= 1
    locked = datetime.datetime.fromisoformat(lock["locked_at"])
    for r in runs:
        assert datetime.datetime.fromisoformat(r["scored_at"]) >= locked
        assert r["lock"]["ground_truth"] == lock["ground_truth"]


@pytest.mark.parametrize("take", TAKES)
def test_saved_events_are_exactly_what_the_frozen_model_produces(take):
    saved = pd.read_csv(HOLDOUT_DIR / f"{take}_events.csv")
    fresh, _, _ = segment_new_take(take)
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), saved, check_dtype=False, rtol=1e-9)


@pytest.mark.parametrize("take", TAKES)
def test_events_cover_the_take_contiguously(take):
    ev = pd.read_csv(HOLDOUT_DIR / f"{take}_events.csv")
    last = json.loads((RAW_DIR / f"{take}_meta.json").read_text())["last_timestamp_s"]
    assert ev["start_s"].iloc[0] == 0 and ev["end_s"].iloc[-1] == pytest.approx(last)
    assert np.allclose(ev["start_s"].to_numpy()[1:], ev["end_s"].to_numpy()[:-1])
    assert set(ev["label"]) <= set(LABELS)


def test_scores_file_is_reproducible_from_the_saved_events(scores):
    for take in TAKES:
        fresh = pd.DataFrame(score_take(take, pd.read_csv(HOLDOUT_DIR / f"{take}_events.csv"), load_frames(take)["t"].to_numpy(), with_cycles=True))
        saved = scores[scores["take"] == take].reset_index(drop=True)
        for col in ("n_true", "n_pred", "n_matched", "recall", "precision", "frame_accuracy", "balanced_accuracy", "tolerant_accuracy"):
            np.testing.assert_allclose(fresh[col].to_numpy(float), saved[col].to_numpy(float), rtol=1e-9, equal_nan=True, err_msg=f"{take} {col}")


def test_holdout_results_are_reported_as_their_own_groups(scores):
    assert set(scores["group"]) == {"holdout_slow", "holdout_fast"}
    assert set(scores[scores["take"] == "vid6"]["scope"]) >= {"all", "cycle1", "cycle2", "cycle3", "cycle4", "out_of_frame"}
    hist = pd.read_csv(SEGMENTS_DIR / "history.csv")
    h = hist[hist["version"] == H.HISTORY_VERSION]
    assert set(h["take"]) == set(TAKES)
    assert not hist[hist["version"] != H.HISTORY_VERSION]["take"].isin(TAKES).any()   # never mixed into dev versions


def test_end_to_end_export_is_complete_and_separate_from_the_canonical_export():
    exp, tab = HOLDOUT_DIR / "export", HOLDOUT_DIR / "tableau"
    takes = pd.read_csv(need(exp / "takes.csv")).set_index("take")
    frames = int(takes["n_frames"].sum())
    assert frames == 1291 + 408
    raw = pd.read_parquet(exp / "raw_keypoints.parquet")
    assert len(raw) == 21 * frames and set(raw["take"]) == set(TAKES)
    ev = pd.read_csv(exp / "events.csv")
    for t, g in ev.groupby("take"):
        assert g["n_frames"].sum() == takes.loc[t, "n_frames"]
    ph = pd.read_csv(need(tab / "tableau_phases.csv"))
    assert len(ph) == len(ev) + len(pd.read_csv(exp / "ground_truth.csv"))
    assert len(pd.read_csv(tab / "tableau_signals.csv")) == frames
    canon = json.loads((ROOT / "data" / "export" / "manifest.json").read_text())
    assert canon["files"]["raw_keypoints.parquet"]["rows"] == 83160                  # 5-take export untouched


def test_vid6_label_time_remap_kept_every_boundary_on_the_same_frame():
    """vid6 was re-extracted after the time-base fix; the owner's boundaries were carried to the same frames' corrected times."""
    rm = pd.read_csv(need(HOLDOUT_DIR / "vid6_label_time_remap.csv"))
    new = pd.read_csv(RAW_DIR / "vid6_keypoints.csv").drop_duplicates("frame_idx").set_index("frame_idx")["timestamp_ms"] / 1000
    assert np.allclose(rm["new_s"].to_numpy(float), new.loc[rm["frame_idx"]].round(3).to_numpy(), atol=1e-9)
    assert (rm["new_s"] - rm["old_s"]).abs().max() <= 0.0012
    gt = pd.read_csv(GROUND_TRUTH_DIR / "take_6.csv")
    labelled = set(np.round(np.r_[gt["start_s"], gt["end_s"]], 3))
    assert labelled <= set(np.round(new.to_numpy(), 3))                                  # every boundary is a real frame time
