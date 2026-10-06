"""Integrity of data/export (the tables handed to Databricks, Snowflake and Tableau). Needs
scripts/export_tables.py to have been run (and Postgres up for the raw-keypoint count check)."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from config import LABELS, RAW_DIR, ROOT, TAKE_GROUPS
from src import ground_truth as G
from src.db import connect
from src.final import load_meta

EXP = ROOT / "data" / "export"
TAKES = sorted(TAKE_GROUPS)


def tbl(name):
    p = EXP / f"{name}.csv"
    if not p.exists():
        pytest.fail(f"{p} missing: run scripts/export_tables.py")
    return pd.read_csv(p)


def test_manifest_matches_files_and_model():
    m = json.loads((EXP / "manifest.json").read_text())
    assert m["model_sha256"] == load_meta()["model_sha256"]
    for fn, info in m["files"].items():
        data = (EXP / fn).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"], fn
        n = len(pd.read_parquet(EXP / fn)) if fn.endswith(".parquet") else len(pd.read_csv(EXP / fn))
        assert n == info["rows"], fn


def test_raw_keypoints_export_matches_csvs_and_postgres_exactly():
    raw = pd.read_parquet(EXP / "raw_keypoints.parquet")
    src = {t: len(pd.read_csv(RAW_DIR / f"{t}_keypoints.csv")) for t in TAKES}
    assert len(raw) == sum(src.values()) == 83160
    for t in TAKES:
        assert (raw["take"] == t).sum() == src[t]
    try:
        conn = connect()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Postgres unreachable ({e})")
    with conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM raw_keypoints WHERE take = ANY(%s)", (TAKES,))   # new takes may be loaded too
        assert cur.fetchone()[0] == len(raw)
    assert not raw.loc[raw["detected"], ["x", "y", "z"]].isna().any().any()


def test_events_partition_every_frame_exactly_once_and_are_contiguous():
    ev, takes = tbl("events"), tbl("takes").set_index("take")
    for t, e in ev.groupby("take"):
        e = e.sort_values("event_idx")
        assert e["start_s"].iloc[0] == 0
        assert e["end_s"].iloc[-1] == pytest.approx(takes.loc[t, "duration_s"], abs=1e-3)
        assert np.allclose(e["start_s"].to_numpy()[1:], e["end_s"].to_numpy()[:-1])
        assert (e["duration_s"] > 0).all()
        assert e["n_frames"].sum() == takes.loc[t, "n_frames"]          # no frame dropped or double counted
        assert len(e) == takes.loc[t, "n_events"]
    assert set(ev["label"]) <= set(LABELS) and set(ev["take"]) == set(TAKES)
    assert (ev["group"] == ev["take"].map(TAKE_GROUPS)).all()
    assert not ev[["mean_speed", "mean_aperture", "mean_confidence", "frac_missing"]].isna().any().any()
    assert ev["frac_missing"].between(0, 1).all() and ev["mean_confidence"].between(0, 1).all()


def test_signals_are_consistent_with_events_and_ground_truth():
    sig, ev = tbl("signals"), tbl("events")
    for t, s in sig.groupby("take"):
        e = ev[ev["take"] == t].sort_values("event_idx")
        pred = G.labels_at(e, s["t_s"].to_numpy())
        assert list(pred) == list(s["predicted_label"].where(s["predicted_label"].notna(), None))
        gt = G.labels_at(G.load_ground_truth(t), s["t_s"].to_numpy())
        assert list(gt) == list(s["ground_truth_label"].where(s["ground_truth_label"].notna(), None))
    assert sig["predicted_label"].notna().all()
    assert set(sig["predicted_label"]) <= set(LABELS)
    # missing samples carry no signal values; non-missing ones do
    assert sig.loc[sig["missing"], ["speed", "aperture"]].isna().all().all()
    assert not sig.loc[~sig["missing"], ["speed", "aperture"]].isna().any().any()


def test_frames_table_matches_takes_metadata():
    fr, takes = tbl("frames"), tbl("takes").set_index("take")
    for t, f in fr.groupby("take"):
        assert len(f) == takes.loc[t, "n_frames"]
        assert (~f["detected"]).sum() == takes.loc[t, "frames_not_detected"]
        assert (f["handedness"] == "Left").sum() == takes.loc[t, "left_handedness_frames"]
        assert f["t_s"].is_monotonic_increasing and f["frame_idx"].is_unique


def test_ground_truth_and_scores_tables_are_complete():
    gt, sc = tbl("ground_truth"), tbl("scores")
    for t in TAKES:
        assert len(gt[gt["take"] == t]) == len(G.load_ground_truth(t))
    assert set(sc["take"]) == set(TAKES) and set(sc["group"]) == {"clean", "fast", "hard"}
    assert (sc["scope"] == "all").sum() == 5
