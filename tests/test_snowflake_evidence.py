"""The results downloaded from Snowflake (snowflake/actual_results/) are checked three ways:
 1. against the expected results produced locally (DuckDB);
 2. against an INDEPENDENT pandas recomputation from the exported CSVs (no SQL), so a mistake in the SQL logic
    would show up even though Snowflake and DuckDB run the same text;
 3. against the export manifest (row counts).
"""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import compare_snowflake_results as C  # noqa: E402

SF, EXP = ROOT / "snowflake", ROOT / "data" / "export"
NAMES = ["q1_avg_duration_by_event_type", "q2_most_ambiguous_takes", "q3_time_between_consecutive_reaches",
         "q4_phase_transitions", "q5_prediction_agreement_by_label"]


def actual(name):
    p = SF / "actual_results" / f"{name}.csv"
    if not p.exists():
        pytest.fail(f"{p} missing: download the result from Snowflake")
    return pd.read_csv(p)


def low(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def same(name, ref, keys):
    a = low(actual(name)).sort_values(keys).reset_index(drop=True)
    r = ref.sort_values(keys).reset_index(drop=True)
    assert list(a.columns) == list(r.columns) and len(a) == len(r), (name, len(a), len(r))
    for c in r.columns:
        if pd.api.types.is_numeric_dtype(r[c]):
            np.testing.assert_allclose(a[c].astype(float), r[c].astype(float), rtol=0, atol=1e-9, err_msg=f"{name}.{c}")
        else:
            assert (a[c].astype(str) == r[c].astype(str)).all(), f"{name}.{c}"


@pytest.fixture(scope="module")
def ev():
    return pd.read_csv(EXP / "events.csv").rename(columns={"group": "take_group"})


def test_actual_results_folder_contains_exactly_the_expected_files():
    got = {p.name for p in (SF / "actual_results").glob("*.csv")}
    assert {f"{n}.csv" for n in NAMES} | {"verify.csv"} <= got
    assert got <= {f"{n}.csv" for n in NAMES} | {"verify.csv", "fingerprint.csv"}, got


def test_downloads_look_like_snowflake_output_and_are_not_copies_of_the_local_files():
    """Snowflake returns upper-case column names; the local DuckDB files use lower case. A byte-identical file would
    mean the expected results were copied instead of downloaded."""
    for n in NAMES + ["verify"]:
        header = (SF / "actual_results" / f"{n}.csv").read_text().splitlines()[0]
        assert header == header.upper(), n
        assert (SF / "actual_results" / f"{n}.csv").read_bytes() != (SF / "expected_results" / f"{n}.csv").read_bytes(), n


def test_compare_script_reports_all_match_on_the_real_downloads():
    assert C.main() == 0


def test_verify_csv_46_checks_all_pass_with_expected_equal_to_actual():
    v = low(actual("verify"))
    exp = pd.read_csv(SF / "expected_results" / "verify.csv")
    assert len(v) == 46 and (v["status"] == "PASS").all()
    assert set(v["check_name"]) == set(exp["check_name"])
    assert (v["expected"].astype(float) == v["actual"].astype(float)).all()


def test_snowflake_row_counts_equal_the_export_manifest():
    man = json.loads((EXP / "manifest.json").read_text())["files"]
    v = low(actual("verify"))
    got = {r.check_name.removeprefix("row count: "): int(r.actual) for r in v.itertuples() if r.check_name.startswith("row count: ")}
    assert got == {k.split(".")[0]: m["rows"] for k, m in man.items()}
    assert got["raw_keypoints"] == 83160


def test_q1_matches_independent_pandas(ev):
    g = ev.groupby(["take_group", "label"])["duration_s"].agg(n_events="count", avg_duration_s="mean", min_duration_s="min", max_duration_s="max").reset_index()
    for c in ("avg_duration_s", "min_duration_s", "max_duration_s"):
        g[c] = g[c].round(3)
    same(NAMES[0], g, ["take_group", "label"])


def test_q2_matches_independent_pandas(ev):
    e = ev.assign(amb=((ev["duration_s"] < 0.3) | (ev["mean_confidence"] < 0.7)).astype(int))
    p = e.groupby(["take", "take_group"]).agg(n_events=("label", "size"), short_events=("duration_s", lambda s: int((s < 0.3).sum())),
                                              low_confidence_events=("mean_confidence", lambda s: int((s < 0.7).sum())),
                                              ambiguous_events=("amb", "sum"), avg_confidence=("mean_confidence", "mean")).reset_index()
    share = p["ambiguous_events"] / p["n_events"]
    p["ambiguous_share"], p["avg_confidence"] = share.round(3), p["avg_confidence"].round(3)
    p["ambiguity_rank"] = share.rank(method="min", ascending=False).astype(int)
    same(NAMES[1], p[["take", "take_group", "n_events", "short_events", "low_confidence_events", "ambiguous_events",
                      "ambiguous_share", "avg_confidence", "ambiguity_rank"]], ["take"])


def test_q3_matches_independent_pandas(ev):
    r = ev[ev["label"] == "REACH"].sort_values(["take", "start_s"]).copy()
    r["prev"] = r.groupby("take")["start_s"].shift(1)
    r = r.dropna(subset=["prev"])
    q = pd.DataFrame({"take": r["take"], "previous_reach_start_s": r["prev"].round(3), "reach_start_s": r["start_s"].round(3),
                      "seconds_between_reaches": (r["start_s"] - r["prev"]).round(3)})
    same(NAMES[2], q, ["take", "reach_start_s"])


def test_q4_matches_independent_pandas(ev):
    e = ev.sort_values(["take", "event_idx"]).copy()
    e["to_label"] = e.groupby("take")["label"].shift(-1)
    e = e.dropna(subset=["to_label"])
    order = {("REST", "REACH"), ("REACH", "GRASP"), ("GRASP", "HOLD"), ("HOLD", "RELEASE"), ("RELEASE", "RETRACT"), ("RETRACT", "REST")}
    q = e.groupby(["take_group", "label", "to_label"]).size().reset_index(name="n_transitions").rename(columns={"label": "from_label"})
    q["follows_protocol_order"] = ["yes" if (a, b) in order else "no" for a, b in zip(q["from_label"], q["to_label"])]
    same(NAMES[3], q[["take_group", "from_label", "to_label", "n_transitions", "follows_protocol_order"]], ["take_group", "from_label", "to_label"])


def test_q5_matches_independent_pandas():
    s = pd.read_csv(EXP / "signals.csv")
    s = s[(~s["missing"]) & s["ground_truth_label"].notna()].copy()
    s["ok"] = (s["predicted_label"] == s["ground_truth_label"]).astype(float)
    q = s.groupby(["take", "ground_truth_label"]).agg(n_samples=("ok", "size"), agreement=("ok", "mean")).reset_index().rename(columns={"ground_truth_label": "label"})
    q["agreement"] = q["agreement"].round(3)
    same(NAMES[4], q, ["take", "label"])


def test_snowflake_results_match_what_the_recordings_showed():
    """Plain-language sanity checks on the downloaded results (the 'does it make sense' step of Phase 4)."""
    q1 = low(actual(NAMES[0])).set_index(["take_group", "label"])
    assert 2.4 < q1.loc[("clean", "HOLD"), "avg_duration_s"] < 2.9                 # protocol asks for ~1 s; the takes held ~2.7 s
    assert ("fast", "RELEASE") not in q1.index                                     # known miss in the fast take
    assert q1.loc[("hard", "RELEASE"), "avg_duration_s"] < 0.25
    q2 = low(actual(NAMES[1]))
    assert q2.iloc[0]["take"] == "vid5" and q2["ambiguity_rank"].min() == 1
    q3 = low(actual(NAMES[2])).groupby("take")["seconds_between_reaches"].mean()
    assert q3["vid4"] < 3 < q3["vid5"] < 10 < q3["vid1"] < q3["vid2"]
    q4 = low(actual(NAMES[3]))
    assert (q4[q4["take_group"] == "clean"]["follows_protocol_order"] == "yes").all()
    assert set(map(tuple, q4[(q4["take_group"] == "fast") & (q4["follows_protocol_order"] == "no")][["from_label", "to_label"]].values)) == {("HOLD", "RETRACT")}


def test_optional_fingerprint_if_it_was_run_in_snowflake():
    p = SF / "actual_results" / "fingerprint.csv"
    if not p.exists():
        pytest.skip("optional: 04_fingerprint.sql has not been run in Snowflake yet")
    assert C.compare_frames(pd.read_csv(p), pd.read_csv(SF / "expected_results" / "fingerprint.csv"), tol=C.TOLERANCE["fingerprint.csv"]) == []
