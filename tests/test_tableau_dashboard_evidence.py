"""The saved Tableau workbook (tableau/hand-motion-phases.twbx) and its exported picture (evidence/tableau_dashboard.png) are
checked against the data, so 'the dashboard shows the right data' rests on saved files, not on a quick look.
Layout assumptions are in scripts/verify_dashboard_image.py (LAYOUT); if the dashboard is re-exported differently, update them."""
import sys

import pandas as pd
import pytest
from PIL import Image

from config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import verify_dashboard_image as V  # noqa: E402

TAKES = ["vid1", "vid2", "vid3", "vid4", "vid5"]


@pytest.fixture(scope="module")
def wb():
    assert V.TWBX.exists(), "tableau/hand-motion-phases.twbx missing"
    return V.workbook()


def _sheet(wb, mark):
    found = [s for s in wb["sheets"] if mark in s["marks"]]
    assert len(found) == 1, (mark, [s["name"] for s in wb["sheets"]])
    return found[0]


# ---------------------------------------------------------------- workbook
def test_workbook_packages_exactly_the_tested_tableau_csvs(wb):
    files = {p["file"] for p in wb["packaged_csvs"]}
    assert files == {"tableau_phases.csv", "tableau_signals.csv", "tableau_accuracy.csv"}
    assert all(p["identical_to_data_tableau"] for p in wb["packaged_csvs"])


def test_timeline_shows_detected_phases_only(wb):
    assert wb["datasource_filters"]["tableau_phases"] == [("[source]", ['"Detected"'])]
    g = _sheet(wb, "GanttBar")
    assert g["datasources"] == ["tableau_phases"]
    assert g["shelf_fields"] == ["start_s", "take_label"] and g["encodings"] == ["duration_s", "phase_name"]


def test_speed_sheet_plots_speed_over_time_per_take(wb):
    s = _sheet(wb, "Line")
    assert s["datasources"] == ["tableau_signals"] and s["shelf_fields"] == ["speed", "t_s", "take_label"]
    # the sheet aggregates SUM(speed) per (take, t_s); that equals the raw value only if t_s is unique within a take
    sig = pd.read_csv(ROOT / "data" / "tableau" / "tableau_signals.csv")
    assert not sig.duplicated(["take", "t_s"]).any()


def test_accuracy_and_scatter_sheets_use_the_right_fields(wb):
    acc = _sheet(wb, "Bar")
    assert acc["datasources"] == ["tableau_accuracy"] and "percent_frames_matching_human_labels" in acc["shelf_fields"]
    sc = [s for s in wb["sheets"] if s["shelf_fields"] == ["aperture", "speed"]]
    assert len(sc) == 1 and sc[0]["encodings"] == ["phase_name"] and sc[0]["datasources"] == ["tableau_signals"]


def test_workbook_colours_cover_all_six_phases():
    assert set(V.phase_colours()) == set(V.TO_LABEL)


# ---------------------------------------------------------------- picture
def test_image_exists_with_the_layout_size():
    assert Image.open(V.IMG).size == V.LAYOUT["size"]


def test_timeline_bands_equal_the_detected_events_for_every_take():
    res = V.timeline()
    for t in TAKES:
        r = res[t]
        assert r["labels_match"] and r["n_segments_image"] == r["n_events"], (t, r)
        assert r["max_start_error_s"] < 0.06 and r["end_error_s"] < 0.06, (t, r)     # about 2 pixels; scale from vid1 only
    assert [res[t]["n_segments_image"] for t in TAKES] == [19, 19, 19, 16, 28]      # vid4: no RELEASE (known miss)


def test_accuracy_bars_equal_the_scored_accuracy():
    r = V.accuracy_bars()
    assert r["actual_percent"] == [94.9, 96.6, 96.6, 91.1, 84.1]
    assert r["max_abs_error_points"] < 0.2, r


def test_speed_lines_follow_the_real_speed_signal():
    res = V.speed_lines()
    for t in TAKES:
        assert res[t]["correlation"] > 0.95, (t, res[t])
        assert res[t]["start_x_error_px"] <= 3 and res[t]["end_x_error_s"] < 0.1, (t, res[t])
    scales = [res[t]["pixels_per_speed_unit"] for t in TAKES]
    assert max(scales) / min(scales) < 1.1                                         # same fixed y-axis in all five panels


def test_scatter_draws_every_phase_in_the_right_vertical_order():
    """Weak pixel check (overlapping circles hide the dense low-speed region); the data binding is checked from the workbook."""
    r = V.scatter()
    assert all(n > 1000 for n in r["pixels_per_phase"].values()), r["pixels_per_phase"]
    assert r["rank_corr_aperture"] > 0.8, r
    assert r["rank_corr_speed"] > 0.5, r
