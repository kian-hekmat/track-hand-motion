"""The sanity-check checklist is generated from the data; keep it in step and correct."""
import re
import sys

import pandas as pd

from config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import make_sanity_checklist as M  # noqa: E402

TEXT = (ROOT / "snowflake" / "sanity_check_checklist.md").read_text()


def test_checklist_is_up_to_date_with_the_generator():
    assert TEXT == M.build(), "run scripts/make_sanity_checklist.py"


def test_every_reach_time_in_the_checklist_comes_from_the_events():
    ev = pd.read_csv(ROOT / "data" / "export" / "events.csv")
    for take, g in ev[ev["label"] == "REACH"].groupby("take"):
        row = next(l for l in TEXT.splitlines() if l.startswith(f"| {take} |") and "," in l and "." in l.split("|")[2])
        times = [float(x) for x in row.split("|")[2].split(",")]
        assert times == [round(x, 2) for x in g["start_s"]], take


def test_off_protocol_transitions_listed_equal_the_query_4_results():
    q4 = pd.read_csv(ROOT / "snowflake" / "expected_results" / "q4_phase_transitions.csv")
    off = q4[q4["follows_protocol_order"] == "no"]
    listed = re.findall(r"\| (vid\d) \| (\w+) > (\w+) \| ", TEXT)
    assert len(listed) == int(off["n_transitions"].sum())
    assert sorted({(a, b) for _, a, b in listed}) == sorted(set(zip(off["from_label"], off["to_label"])))


def test_ambiguous_events_listed_equal_the_query_2_definition():
    ev = pd.read_csv(ROOT / "data" / "export" / "events.csv")
    n = int(((ev["duration_s"] < 0.3) | (ev["mean_confidence"] < 0.7)).sum())
    q2 = pd.read_csv(ROOT / "snowflake" / "expected_results" / "q2_most_ambiguous_takes.csv")
    assert n == int(q2["ambiguous_events"].sum())
    section = TEXT.split("## D.")[1].split("## E.")[0]
    assert len(re.findall(r"^\| vid\d \|", section, flags=re.M)) == n


def test_checklist_has_a_fill_in_column_for_every_check_and_a_signoff():
    for marker in ("## A.", "## B.", "## C.", "## D.", "## E.", "## F. Sign-off"):
        assert marker in TEXT
    assert "Results make sense" in TEXT and "do NOT match" in TEXT
