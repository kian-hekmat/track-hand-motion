"""Cloud path M2: the bronze -> silver -> gold chain (src/cloud/), run on local Spark from a landing folder laid out exactly
as the Databricks Volume will be. Every parity check in src.cloud.checks must pass here before the chain runs on Databricks."""
import shutil

import pytest

from config import GROUND_TRUTH_DIR, RAW_DIR
from src.cloud import checks


@pytest.fixture(scope="module")
def tables(spark, tmp_path_factory):
    landing = tmp_path_factory.mktemp("landing")
    (landing / "raw").mkdir()
    (landing / "ground_truth").mkdir()
    for take in checks.TAKES:
        for suffix in ("_keypoints.csv", "_meta.json"):
            shutil.copy(RAW_DIR / f"{take}{suffix}", landing / "raw")
    shutil.copy(RAW_DIR / "out_of_frame_intervals.csv", landing / "raw")
    for p in GROUND_TRUTH_DIR.glob("take_*.csv"):
        shutil.copy(p, landing / "ground_truth")
    return {k: v.toPandas() for k, v in checks.build(spark, str(landing)).items()}


def test_row_counts_through_the_chain(tables):
    assert len(tables["bronze.raw_keypoints"]) == 118_839          # 5,659 frames x 21 landmarks
    assert len(tables["silver.frames"]) == len(tables["gold.frame_labels"]) == 5_659
    assert len(tables["silver.signals"]) == 5_659                  # one 30 Hz sample per frame for these takes
    assert len(tables["gold.events"]) == 101 + 31 + 18             # vid1-5 canonical + vid6 + vid7


def test_every_parity_check_passes(tables):
    results = checks.compare(tables)
    failed = [(c, d) for c, s, d in results if s != "PASS"]
    assert not failed, failed
    assert len(results) == 7 + 2 + 7 * 3 + 1  # bronze per take, bronze files, (frames, signals, events) per take, scores


def test_a_wrong_model_is_caught(tables):
    """The check is not vacuous: relabel one take's model and its signals check must fail."""
    broken = dict(tables)
    s = tables["silver.signals"].copy()
    s.loc[s["take"] == "vid1", "model"] = "segmenter_v2.joblib"
    broken["silver.signals"] = s
    status = {c: st for c, st, _ in checks.compare(broken)}
    assert status["silver vid1: signals"] == "FAIL" and status["silver vid2: signals"] == "PASS"


def test_a_shifted_event_boundary_is_caught(tables):
    broken = dict(tables)
    e = tables["gold.events"].copy()
    e.loc[(e["take"] == "vid4") & (e["event_idx"] == 3), "start_s"] += 1e-9
    broken["gold.events"] = e
    assert {c: st for c, st, _ in checks.compare(broken)}["gold vid4: events"] == "FAIL"


def test_notebook_is_valid_and_uses_the_tested_code():
    import ast
    import re

    from config import ROOT

    src = (ROOT / "databricks" / "cloud" / "m2_build_tables.py").read_text()
    assert src.startswith("# Databricks notebook source\n")
    ast.parse(src)
    pins = re.search(r"^# MAGIC %pip install -q (.+)$", src, re.M).group(1).split()
    req = [l for l in (ROOT / "databricks" / "cloud" / "requirements-cloud.txt").read_text().splitlines()
           if l and not l.startswith("#")]
    assert pins == req
    # the notebook runs the same transforms and the same checks the tests above run
    for call in ("C.read_landing(", "C.build_frames(", "C.predict_signals(", "C.build_events(", "C.label_frames(",
                 "C.frame_scores(", "checks.compare(", "load_model_for_take("):
        assert call in src, call
    assert 'mode("overwrite")' in src and 'mode("append").saveAsTable(f"{CATALOG}.{SCHEMAS[\'gold\']}.run_log")' in src
