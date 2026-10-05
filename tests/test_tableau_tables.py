"""Phase 5: the Tableau-ready tables (data/tableau/*.csv), produced by the views in snowflake/05_tableau_views.sql."""
import sys

import numpy as np
import pandas as pd
import pytest

from config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import compare_snowflake_results as C  # noqa: E402
import local_sql as L  # noqa: E402
import make_tableau_tables as M  # noqa: E402

TAB, EXP = ROOT / "data" / "tableau", ROOT / "data" / "export"
NAMES = {"REST": "At rest", "REACH": "Reaching", "GRASP": "Grasping", "HOLD": "Holding", "RELEASE": "Releasing", "RETRACT": "Returning"}
TAKE_LABELS = {"vid1": "Take 1 (clean)", "vid2": "Take 2 (clean)", "vid3": "Take 3 (clean)", "vid4": "Take 4 (fast)", "vid5": "Take 5 (hard)"}


@pytest.fixture(scope="module")
def t():
    return {n: pd.read_csv(TAB / f"{n}.csv") for n in ("tableau_phases", "tableau_signals", "tableau_accuracy")}


def test_tableau_csvs_are_not_stale():
    fresh = M.build_tables()
    for name, df in fresh.items():
        saved = pd.read_csv(TAB / f"{name}.csv")
        assert C.compare_frames(saved, df) == [], name
        assert (L.SF / "expected_results" / f"{name}.csv").exists()


def test_row_counts_match_the_sources(t):
    ev, gt = pd.read_csv(EXP / "events.csv"), pd.read_csv(EXP / "ground_truth.csv")
    ph = t["tableau_phases"]
    assert len(ph) == len(ev) + len(gt) == 202
    assert (ph["source"] == "Detected").sum() == len(ev) and (ph["source"] == "Hand-labelled").sum() == len(gt)
    assert len(t["tableau_signals"]) == len(pd.read_csv(EXP / "signals.csv")) == 3960
    assert len(t["tableau_accuracy"]) == 5


def test_phases_are_contiguous_and_cover_each_take_for_both_sources(t):
    takes = pd.read_csv(EXP / "takes.csv").set_index("take")
    for (take, src), g in t["tableau_phases"].groupby(["take", "source"]):
        g = g.sort_values("start_s")
        assert g["start_s"].iloc[0] == 0 and g["end_s"].iloc[-1] == pytest.approx(takes.loc[take, "duration_s"], abs=1e-3)
        assert np.allclose(g["start_s"].to_numpy()[1:], g["end_s"].to_numpy()[:-1])
        assert (g["duration_s"] > 0).all()
        assert g["duration_s"].sum() == pytest.approx(takes.loc[take, "duration_s"], abs=1e-3)


def test_plain_language_names_are_complete_and_consistent(t):
    ph, sg = t["tableau_phases"], t["tableau_signals"]
    for df in (ph, sg):
        assert not df["phase_name"].isna().any()
        label_col = "label" if "label" in df else "predicted_label"
        assert {l: n for l, n in zip(df[label_col], df["phase_name"])} == NAMES
        order = dict(zip(df["phase_name"], df["phase_order"]))
        assert order == {"At rest": 1, "Reaching": 2, "Grasping": 3, "Holding": 4, "Releasing": 5, "Returning": 6}
    for df in (ph, sg, t["tableau_accuracy"]):
        assert {a: b for a, b in zip(df["take"], df["take_label"])} == TAKE_LABELS


def test_accuracy_table_equals_the_scored_frame_accuracy(t):
    sc = pd.read_csv(EXP / "scores.csv")
    sc = sc[sc["scope"] == "all"].set_index("take")
    for r in t["tableau_accuracy"].itertuples():
        assert r.percent_frames_matching_human_labels == pytest.approx(round(sc.loc[r.take, "frame_accuracy"] * 100, 1))


def test_signals_leave_gaps_where_the_hand_was_not_visible_and_nowhere_else(t):
    s = t["tableau_signals"]
    assert s.loc[s["hand_not_visible"], ["speed", "aperture"]].isna().all().all()
    assert not s.loc[~s["hand_not_visible"], ["speed", "aperture"]].isna().any().any()
    assert int(s["hand_not_visible"].sum()) == int(pd.read_csv(EXP / "signals.csv")["missing"].sum())


def test_headers_are_tableau_friendly_and_values_are_not_blank(t):
    for name, df in t.items():
        assert all(" " not in c and c == c.lower() for c in df.columns), name
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                assert (df[c].astype(str).str.strip() != "").all(), (name, c)


def test_snowflake_export_of_these_views_would_be_accepted_by_the_compare_script(t):
    """A Snowflake export has upper-case headers, shuffled rows and lower-case booleans; the compare script must accept it."""
    s = t["tableau_signals"].rename(columns=str.upper).sample(frac=1, random_state=3)
    assert C.compare_frames(s, t["tableau_signals"]) == []
    assert {"tableau_phases.csv", "tableau_signals.csv", "tableau_accuracy.csv"} <= C.OPTIONAL


def test_dashboard_preview_image_exists():
    img = ROOT / "tableau" / "target_dashboard.png"
    assert img.exists() and img.stat().st_size > 50_000
