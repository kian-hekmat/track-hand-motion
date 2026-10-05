"""Phase 4: the Snowflake scripts and queries, executed locally in DuckDB (same table definitions, positional
CSV load, portable SQL). This tests the SQL logic and the generated scripts; it cannot test Snowflake itself."""
import re
import sys

import pandas as pd
import pytest

from config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import build_snowflake_scripts as B  # noqa: E402
import compare_snowflake_results as C  # noqa: E402
import local_sql as L  # noqa: E402

SF = ROOT / "snowflake"


@pytest.fixture()
def con():
    return L.connect()


def test_generated_scripts_are_up_to_date():
    for name, text in B.build().items():
        assert (SF / name).read_text() == text, f"run scripts/build_snowflake_scripts.py ({name} is stale)"


def test_table_definitions_match_file_columns_in_order():
    """CSV loads map by position, so DDL order must equal the file's column order (after the two renames)."""
    setup = (SF / "01_setup.sql").read_text()
    for t in B.CSV_TABLES + ["raw_keypoints"]:
        block = re.search(rf"CREATE OR REPLACE TABLE {t} \(\n(.*?)\n\);", setup, re.S).group(1)
        ddl_cols = [l.strip().split()[0] for l in block.splitlines()]
        file_cols = (list(pd.read_parquet(B.EXP / "raw_keypoints.parquet").columns) if t == "raw_keypoints"
                     else list(pd.read_csv(B.EXP / f"{t}.csv").columns))
        assert ddl_cols == [B.RENAME.get(c, c) for c in file_cols], t
    assert not any(c in B.RESERVED for t in B.CSV_TABLES + ["raw_keypoints"] for c, _ in B.columns(t))


def test_load_script_covers_every_exported_file_exactly_once():
    load = (SF / "02_load.sql").read_text()
    staged = re.findall(r"PATTERN = '\.\*(\w+)\[\.\](csv|parquet)", load)
    assert sorted(f"{n}.{ext}" for n, ext in staged) == sorted([f"{t}.csv" for t in B.CSV_TABLES] + ["raw_keypoints.parquet"])
    assert load.count("TRUNCATE TABLE") == 7 and "ABORT_STATEMENT" in load


def test_every_verification_check_passes_on_the_real_export(con):
    ver = L.run_file(con, SF / "03_verify.sql")
    assert len(ver) == 46 and (ver["status"] == "PASS").all(), ver[ver["status"] != "PASS"]


@pytest.mark.parametrize("damage,expect_fail", [
    ("DELETE FROM events WHERE take = 'vid2' AND event_idx = 3", "events"),
    ("DELETE FROM frames WHERE take = 'vid1' AND frame_idx = 10", "frames"),
    ("DELETE FROM raw_keypoints WHERE take = 'vid4' AND frame_idx = 5 AND keypoint_id = 0", "raw"),
    ("UPDATE events SET start_s = start_s + 0.5 WHERE take = 'vid3' AND event_idx = 4", "gaps"),
    ("UPDATE events SET n_frames = n_frames + 1 WHERE take = 'vid5' AND event_idx = 0", "n_frames"),
    ("INSERT INTO events SELECT * FROM events WHERE take = 'vid1' AND event_idx = 0", "events"),
])
def test_verification_script_catches_deliberate_damage(con, damage, expect_fail):
    con.execute(damage)
    ver = L.run_file(con, SF / "03_verify.sql")
    failed = ver[ver["status"] == "FAIL"]
    assert len(failed) >= 1, "the verify script did not notice the damage"
    assert failed["check_name"].str.contains(expect_fail, case=False).any(), failed


def test_queries_file_has_five_named_queries_and_no_reserved_identifier_use():
    qs = L.named_queries(ROOT / "queries.sql")
    assert list(qs) == ["q1_avg_duration_by_event_type", "q2_most_ambiguous_takes", "q3_time_between_consecutive_reaches",
                        "q4_phase_transitions", "q5_prediction_agreement_by_label"]
    text = (ROOT / "queries.sql").read_text()
    assert re.search(r"\bLAG\(", text) and re.search(r"\bLEAD\(", text)
    assert not re.search(r"\bgroup\b(?!\s+by)", re.sub(r"--.*", "", text), re.I)


def test_queries_reproduce_the_saved_expected_results(con):
    exp_dir = SF / "expected_results"
    for name, q in L.named_queries(ROOT / "queries.sql").items():
        q = "\n".join(l for l in q.splitlines() if not l.strip().upper().startswith("USE "))
        problems = C.compare_frames(con.execute(q).fetchdf(), pd.read_csv(exp_dir / f"{name}.csv"))
        assert not problems, (name, problems)
    assert (exp_dir / "verify.csv").exists()


def _q(con, name):
    q = L.named_queries(ROOT / "queries.sql")[name]
    return con.execute("\n".join(l for l in q.splitlines() if not l.strip().upper().startswith("USE "))).fetchdf()


def test_query_results_make_sense_against_the_recordings(con):
    q1 = _q(con, "q1_avg_duration_by_event_type")
    clean = q1[q1["take_group"] == "clean"].set_index("label")
    assert clean.loc["HOLD", "avg_duration_s"] > clean.loc["REACH", "avg_duration_s"] > clean.loc["RETRACT", "avg_duration_s"]
    assert (clean["n_events"].loc[["REACH", "GRASP", "HOLD", "RELEASE", "RETRACT"]] == 9).all()      # 3 takes x 3 cycles
    fast = q1[q1["take_group"] == "fast"]
    assert "RELEASE" not in set(fast["label"])                                  # the known fast-take miss
    assert fast["avg_duration_s"].max() < 1.5 and clean["avg_duration_s"].max() > 2

    q2 = _q(con, "q2_most_ambiguous_takes")
    assert q2.iloc[0]["take"] == "vid5" and int(q2.iloc[0]["ambiguity_rank"]) == 1

    q3 = _q(con, "q3_time_between_consecutive_reaches")
    assert len(q3) == 2 + 2 + 2 + 2 + 3                                         # cycles - 1 per take
    per = q3.groupby("take")["seconds_between_reaches"].mean()
    assert per["vid4"] < per["vid5"] < per["vid1"]                              # fast < hard < clean cycle time

    q4 = _q(con, "q4_phase_transitions")
    assert (q4[q4["take_group"] == "clean"]["follows_protocol_order"] == "yes").all()
    off = q4[(q4["take_group"] == "fast") & (q4["follows_protocol_order"] == "no")]
    assert set(zip(off["from_label"], off["to_label"])) == {("HOLD", "RETRACT")}   # RELEASE missing
    assert (q4[q4["take_group"] == "hard"]["follows_protocol_order"] == "no").any()

    q5 = _q(con, "q5_prediction_agreement_by_label")
    assert len(q5) == 30 and q5["agreement"].between(0, 1).all()
    assert float(q5[(q5["take"] == "vid4") & (q5["label"] == "RELEASE")]["agreement"].iloc[0]) == 0.0


# ---- comparison script ----
def test_compare_accepts_snowflake_style_output_and_rejects_changes():
    exp = pd.read_csv(SF / "expected_results" / "q1_avg_duration_by_event_type.csv")
    snow = exp.rename(columns=str.upper).sample(frac=1, random_state=1)           # upper-case headers, shuffled rows
    assert C.compare_frames(snow, exp) == []
    bad = snow.copy(); bad.loc[bad.index[0], "AVG_DURATION_S"] += 0.01
    assert C.compare_frames(bad, exp) and C.compare_frames(snow.iloc[1:], exp)
    assert C.compare_frames(snow.rename(columns={"LABEL": "KIND"}), exp)


def test_compare_main_reports_missing_and_all_match(tmp_path, monkeypatch):
    sf = tmp_path / "snowflake"
    (sf / "expected_results").mkdir(parents=True); (sf / "actual_results").mkdir()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(sf / "expected_results" / "q.csv", index=False)
    monkeypatch.setattr(C, "SF", sf)
    assert C.main() == 1                                                          # actual file missing
    pd.DataFrame({"A": [2, 1], "B": ["y", "x"]}).to_csv(sf / "actual_results" / "q.csv", index=False)
    assert C.main() == 0


# ---- value-level fingerprint (04_fingerprint.sql) ----
def _fingerprint(con):
    return L.run_file(con, SF / "04_fingerprint.sql")


def test_fingerprint_reproduces_saved_expected_values_and_matches_pandas_on_the_source_files(con):
    fp = _fingerprint(con)
    assert C.compare_frames(fp, pd.read_csv(SF / "expected_results" / "fingerprint.csv"), tol=C.TOLERANCE["fingerprint.csv"]) == []
    raw = pd.read_parquet(B.EXP / "raw_keypoints.parquet")                         # independent of SQL
    v = fp[(fp["table_name"] == "raw_keypoints") & (fp["metric"] == "sum_x")].set_index("take")["value"]
    for take, g in raw.groupby("take"):
        assert float(v[take]) == pytest.approx(round(g["x"].sum(), 4), abs=2e-4)
    sig = pd.read_csv(B.EXP / "signals.csv")
    m = fp[(fp["table_name"] == "signals") & (fp["metric"] == "missing_samples")].set_index("take")["value"]
    for take, g in sig.groupby("take"):
        assert int(m[take]) == int(g["missing"].sum())


@pytest.mark.parametrize("damage", [
    "UPDATE raw_keypoints SET x = x + 0.01 WHERE take = 'vid2' AND frame_idx = 7 AND keypoint_id = 4 AND detected",
    "UPDATE raw_keypoints SET detected = NOT detected WHERE take = 'vid1' AND frame_idx = 100 AND keypoint_id = 0",
    "UPDATE frames SET wrist_y = wrist_y - 0.01 WHERE take = 'vid3' AND frame_idx = 50 AND detected",
    "UPDATE signals SET missing = NOT missing WHERE take = 'vid5' AND t_s < 0.04",
    "UPDATE signals SET predicted_label = 'HOLD' WHERE take = 'vid1' AND t_s < 0.04",
    "UPDATE events SET mean_speed = mean_speed + 0.01 WHERE take = 'vid4' AND event_idx = 2",
])
def test_fingerprint_catches_single_value_corruption_that_row_counts_miss(con, damage):
    assert (L.run_file(con, SF / "03_verify.sql")["status"] == "PASS").all()      # still passes before damage
    con.execute(damage)
    ver = L.run_file(con, SF / "03_verify.sql")
    fp_problems = C.compare_frames(_fingerprint(con), pd.read_csv(SF / "expected_results" / "fingerprint.csv"), tol=C.TOLERANCE["fingerprint.csv"])
    row_counts_still_pass = (ver[ver["check_name"].str.startswith("row count")]["status"] == "PASS").all()
    assert row_counts_still_pass                                                  # damage is invisible to row counts...
    if "predicted_label" in damage:
        # labels are covered by query 5 rather than the numeric fingerprint
        expected = pd.read_csv(SF / "expected_results" / "q5_prediction_agreement_by_label.csv")
        got = con.execute("SELECT take, ground_truth_label AS label, COUNT(*) AS n_samples, ROUND(AVG(CASE WHEN predicted_label = ground_truth_label THEN 1.0 ELSE 0.0 END), 3) AS agreement "
                          "FROM signals WHERE NOT missing AND ground_truth_label IS NOT NULL GROUP BY take, ground_truth_label").fetchdf()
        assert C.compare_frames(got, expected)
    else:
        assert fp_problems, "...but the fingerprint must notice it"


def test_compare_treats_missing_fingerprint_as_optional_and_other_files_as_required(tmp_path, monkeypatch):
    sf = tmp_path / "snowflake"
    (sf / "expected_results").mkdir(parents=True); (sf / "actual_results").mkdir()
    for n in ("q.csv", "fingerprint.csv"):
        pd.DataFrame({"a": [1.0]}).to_csv(sf / "expected_results" / n, index=False)
    pd.DataFrame({"A": [1.0]}).to_csv(sf / "actual_results" / "q.csv", index=False)
    monkeypatch.setattr(C, "SF", sf)
    assert C.main() == 0                                             # fingerprint skipped, q matches
    (sf / "actual_results" / "q.csv").unlink()
    assert C.main() == 1                                             # a required file is missing


def test_compare_is_exact_not_relatively_lenient():
    exp = pd.DataFrame({"v": [10000.0]})
    assert C.compare_frames(pd.DataFrame({"V": [10000.5]}), exp)     # 5e-5 relative would have passed numpy's default rtol
