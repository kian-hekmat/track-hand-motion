"""The exported Tableau dashboard image (evidence/tableau_dashboard.png) is decoded and compared with the data, so the claim
'the dashboard shows the right data' rests on the saved picture, not on a quick look. Layout assumptions are in
scripts/verify_dashboard_image.py; if the dashboard is re-exported with a different layout, update them."""
import sys

import pytest
from PIL import Image

from config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import verify_dashboard_image as V  # noqa: E402

TAKES = ["vid1", "vid2", "vid3", "vid4", "vid5"]


def test_image_exists_with_expected_size():
    assert V.IMG.exists()
    assert Image.open(V.IMG).size == (3348, 1475)


def test_timeline_bands_equal_the_detected_events_for_every_take():
    res = V.compare()
    for t in TAKES:
        r = res[t]
        assert r["labels_match"], t                                   # same phases, same order, same count
        assert r["n_segments_image"] == r["n_events"]
        assert r["max_start_error_s"] < 0.05, t                       # about one pixel (0.025 s); scale calibrated on vid1 only
        assert r["end_error_s"] < 0.05, t


def test_timeline_shows_the_known_fast_take_gap_and_hard_take_flips():
    res = V.compare()
    assert res["vid4"]["n_segments_image"] == 16                      # no RELEASE bands in the fast take (known miss)
    assert res["vid5"]["n_segments_image"] == 28                      # includes the occlusion-cycle flips


def test_accuracy_bars_equal_the_scored_accuracy():
    r = V.accuracy_bars()
    assert r["max_abs_error_points"] < 0.2, r
    assert r["actual_percent"] == [94.9, 96.6, 96.6, 91.1, 84.1]


def test_speed_lines_follow_the_real_speed_signal():
    res = V.speed_lines()
    for t in TAKES:
        assert res[t]["correlation"] > 0.95, (t, res[t])
        assert res[t]["start_x_error_px"] <= 3 and res[t]["end_x_error_s"] < 0.1, (t, res[t])
    scales = [res[t]["pixels_per_speed_unit"] for t in TAKES]
    assert max(scales) / min(scales) < 1.1                            # same fixed y-axis in all five panels
