"""Phase 2 scoring tests on tiny hand-computed cases (no real data)."""
import numpy as np
import pandas as pd
import pytest

import src.score as score
from src.ground_truth import (boundaries, cycle_of_boundaries, cycle_spans, labels_at, load_ground_truth)

TRUE = np.array([2.0, 4.0, 6.0])


def test_perfect_prediction():
    m = score.boundary_metrics(TRUE, TRUE, 0.1)
    assert (m["recall"], m["precision"], m["mae_matched_s"], m["missed"], m["false"]) == (1, 1, 0, 0, 0)


def test_shift_within_tolerance_mae_is_shift():
    m = score.boundary_metrics(TRUE + 0.05, TRUE, 0.1)
    assert m["recall"] == 1 and m["mae_matched_s"] == pytest.approx(0.05)


def test_shift_beyond_tolerance_no_matches_and_mae_is_nan_not_zero():
    m = score.boundary_metrics(TRUE + 0.3, TRUE, 0.1)
    assert m["recall"] == 0 and m["precision"] == 0
    assert np.isnan(m["mae_matched_s"])


def test_extra_boundary_lowers_precision_not_recall():
    m = score.boundary_metrics(np.append(TRUE, 5.0), TRUE, 0.1)
    assert m["recall"] == 1 and m["precision"] == pytest.approx(0.75) and m["false"] == 1


def test_missing_boundary_lowers_recall_not_mae():
    m = score.boundary_metrics(TRUE[:2], TRUE, 0.1)
    assert m["recall"] == pytest.approx(2 / 3) and m["precision"] == 1 and m["mae_matched_s"] == 0


def test_one_to_one_matching_two_predictions_near_one_truth_count_once():
    m = score.boundary_metrics(np.array([3.95, 4.05]), np.array([4.0]), 0.1)
    assert m["n_matched"] == 1 and m["false"] == 1 and m["recall"] == 1


def test_matching_prefers_more_matches_over_closest_single_pair():
    # 4.0 is closest to truth 4.04, but taking it would leave truth 3.95 unmatched; both fit.
    m = score.boundary_metrics(np.array([4.0, 4.09]), np.array([3.95, 4.04]), 0.1)
    assert m["n_matched"] == 2


def test_tolerance_edge_is_inclusive():
    assert score.boundary_metrics(np.array([2.1]), np.array([2.0]), 0.1)["n_matched"] == 1
    assert score.boundary_metrics(np.array([2.1001]), np.array([2.0]), 0.1)["n_matched"] == 0


def test_empty_predictions():
    m = score.boundary_metrics(np.array([]), TRUE, 0.1)
    assert m["recall"] == 0 and m["missed"] == 3 and np.isnan(m["precision"])


def test_frame_metrics_hand_computed():
    t = ["REST"] * 4 + ["REACH"] * 2 + ["GRASP"] * 2
    p = ["REST", "REST", "REST", "REACH", "REACH", "REACH", "GRASP", "REST"]
    m = score.frame_metrics(t, p)
    assert m["n_frames"] == 8 and m["frame_accuracy"] == pytest.approx(6 / 8)
    assert m["acc_REST"] == pytest.approx(3 / 4) and m["acc_REACH"] == 1 and m["acc_GRASP"] == 0.5
    assert np.isnan(m["acc_HOLD"])
    assert m["majority_baseline"] == 0.5


def test_unlabelled_frames_are_excluded_and_counted():
    m = score.frame_metrics(["REST", None, "REACH"], ["REST", "REST", None])
    assert m["n_frames"] == 1 and m["n_unlabelled"] == 2


def test_confusion_matrix_totals_equal_frame_count():
    t = ["REST", "REST", "REACH", "HOLD"]
    p = ["REST", "REACH", "REACH", "HOLD"]
    c = score.confusion(t, p)
    assert c.to_numpy().sum() == 4 and c.loc["REST", "REACH"] == 1


def test_chance_baseline_is_low_on_real_clean_ground_truth():
    gt = load_ground_truth("vid1")
    true = boundaries(gt)
    b = score.chance_baselines(len(true), true, 0.0, gt["end_s"].iloc[-1], tol=0.1)
    assert b["random_recall"] < 0.4 and b["uniform_recall"] < 0.4


def test_chance_baseline_is_not_negligible_on_fast_take():
    """Documents WHY chance baselines are reported: the fast take has short segments."""
    gt = load_ground_truth("vid4")
    true = boundaries(gt)
    b = score.chance_baselines(len(true), true, 0.0, gt["end_s"].iloc[-1], tol=0.1)
    clean = load_ground_truth("vid1")
    bc = score.chance_baselines(len(boundaries(clean)), boundaries(clean), 0.0, clean["end_s"].iloc[-1], tol=0.1)
    assert b["random_recall"] > bc["random_recall"]


def test_no_pooling_function_exists():
    names = [n.lower() for n in dir(score)]
    assert not any("pool" in n or "overall" in n or "average_takes" in n for n in names)


# ---- ground-truth helpers ----
def test_labels_at_half_open_and_last_end_inclusive():
    seg = pd.DataFrame({"label": ["REST", "REACH"], "start_s": [0.0, 1.0], "end_s": [1.0, 2.0]})
    out = labels_at(seg, np.array([0.0, 0.99, 1.0, 1.99, 2.0, 2.5, -0.1]))
    assert list(out) == ["REST", "REST", "REACH", "REACH", "REACH", None, None]


def test_boundaries_and_cycle_assignment_on_vid5():
    gt = load_ground_truth("vid5")
    b, c = boundaries(gt), cycle_of_boundaries(gt)
    assert len(b) == len(gt) - 1 == len(c)
    spans = cycle_spans(gt)
    assert set(spans) == {1, 2, 3, 4}
    for bt, cy in zip(b, c):  # every boundary lies inside the span of the cycle it is assigned to
        lo, hi = spans[int(cy)]
        assert lo - 1e-9 <= bt <= hi + 1e-9


def test_every_take_boundary_count_matches_row_count():
    for n in range(1, 6):
        gt = load_ground_truth(f"vid{n}")
        assert len(boundaries(gt)) == len(gt) - 1


# ---- balanced + tolerant accuracy ----
def test_balanced_accuracy_is_mean_of_per_label_accuracy_not_frame_accuracy():
    t = ["REST"] * 8 + ["GRASP"] * 2
    p = ["REST"] * 8 + ["REST"] * 2          # all GRASP frames wrong
    m = score.frame_metrics(t, p)
    assert m["frame_accuracy"] == pytest.approx(0.8)
    assert m["balanced_accuracy"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2


def test_balanced_accuracy_ignores_labels_absent_from_ground_truth():
    m = score.frame_metrics(["REST", "REACH"], ["REST", "REACH"])
    assert m["balanced_accuracy"] == 1.0


def test_tolerant_accuracy_forgives_only_frames_near_a_true_boundary():
    t = np.arange(10) * 0.1                                   # 0.0 .. 0.9, boundary at 0.5
    true = ["A"] * 5 + ["B"] * 5
    shifted = ["A"] * 4 + ["B"] * 6                            # predicted boundary 0.1 s early
    strict = np.mean([a == b for a, b in zip(true, shifted)])
    tol = score.tolerant_frame_accuracy(t, true, shifted, [0.5], ["A"], ["B"], tol=0.1)
    assert strict == pytest.approx(0.9) and tol == 1.0
    wrong = ["A"] * 2 + ["C"] * 3 + ["B"] * 5                  # wrong label 0.2-0.4 s from the boundary
    assert score.tolerant_frame_accuracy(t, true, wrong, [0.5], ["A"], ["B"], tol=0.1) < 0.8


def test_tolerant_accuracy_never_below_strict_on_real_take():
    from src.signals import load_frames
    import pandas as pd
    from src.evaluate import score_take
    ev = pd.read_csv(__import__("config").SEGMENTS_DIR / "vid1_events.csv")
    r = score_take("vid1", ev, load_frames("vid1")["t"].to_numpy())[0]
    assert r["tolerant_accuracy"] >= r["frame_accuracy"] and r["balanced_accuracy"] <= 1.0
