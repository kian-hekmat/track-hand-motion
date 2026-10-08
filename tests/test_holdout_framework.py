"""Mechanics of the hold-out framework (src/holdout.py, scripts/run_new_take.py), exercised in a sandbox with an existing take
(vid3) standing in for a new recording. vid3 was used to train the frozen model, so nothing here is a performance result."""
import json
import shutil
import sys

import pandas as pd
import pytest

from config import ALL_GROUPS, GROUND_TRUTH_DIR, HOLDOUT_CYCLES, HOLDOUT_TAKES, ROOT, TAKE_GROUPS
from src import holdout as H
from src.final import load_meta

sys.path.insert(0, str(ROOT / "scripts"))

FAKE_META = {"model_sha256": "abc", "train_takes": ["vid1", "vid2"]}   # a model that did NOT see vid3


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    gt = tmp_path / "ground_truth"
    gt.mkdir()
    shutil.copy(GROUND_TRUTH_DIR / "take_3.csv", gt / "take_3.csv")
    monkeypatch.setattr(H, "GROUND_TRUTH_DIR", gt)
    monkeypatch.setattr(H, "HOLDOUT_DIR", tmp_path / "holdout")
    monkeypatch.setattr(H, "SEGMENTS_DIR", tmp_path / "segments")
    monkeypatch.setattr(H, "HOLDOUT_CYCLES", {"vid3": 3, "vid9": 3})
    (tmp_path / "holdout").mkdir()
    (tmp_path / "segments").mkdir()
    return tmp_path


# ---------------------------------------------------------------- configuration
def test_holdout_takes_are_new_and_never_in_training_or_the_dev_groups():
    assert set(HOLDOUT_TAKES) == {"vid6", "vid7"} and set(HOLDOUT_CYCLES) == set(HOLDOUT_TAKES)
    assert not set(HOLDOUT_TAKES) & set(TAKE_GROUPS)
    assert not set(HOLDOUT_TAKES) & set(load_meta()["train_takes"])
    assert all(ALL_GROUPS[t].startswith("holdout") for t in HOLDOUT_TAKES)


# ---------------------------------------------------------------- templates
def test_template_has_one_opening_rest_then_six_rows_per_cycle():
    df = H.ground_truth_template("vid6", 4)
    assert len(df) == 1 + 6 * 4 and list(df.columns) == H.GT_COLUMNS
    assert list(df["label"][:7]) == ["REST", "REACH", "GRASP", "HOLD", "RELEASE", "RETRACT", "REST"]
    assert df["cycle"].max() == 4 and (df["start_s"] == "").all()
    assert len(H.ground_truth_template("vid7", 3)) == 19


def test_template_never_overwrites_existing_labels(sandbox):
    before = (sandbox / "ground_truth" / "take_3.csv").read_bytes()
    assert H.write_template("vid3", 3) is False
    assert (sandbox / "ground_truth" / "take_3.csv").read_bytes() == before


# ---------------------------------------------------------------- validation
def test_real_labels_validate_and_blank_or_broken_labels_do_not(sandbox):
    assert H.validate_ground_truth("vid3") == []
    p = sandbox / "ground_truth" / "take_3.csv"
    good = pd.read_csv(p)
    H.ground_truth_template("vid3", 3).to_csv(p, index=False)
    assert any("no start time" in x for x in H.validate_ground_truth("vid3"))
    for change, expect in [(lambda d: d.assign(start_s=d["start_s"].where(d.index != 5, d["start_s"] + 0.2)), "contiguous"),
                           (lambda d: d.assign(end_s=d["end_s"].where(d.index != len(d) - 1, 99.0)), "last frame"),
                           (lambda d: d.assign(source="audio"), "manual"),
                           (lambda d: d.assign(label=d["label"].replace("HOLD", "PAUSE")), "unknown labels")]:
        change(good).to_csv(p, index=False)
        assert any(expect in x for x in H.validate_ground_truth("vid3")), expect


def test_wrong_number_of_cycles_is_reported(sandbox, monkeypatch):
    monkeypatch.setattr(H, "HOLDOUT_CYCLES", {"vid3": 4})
    assert any("cycles" in x for x in H.validate_ground_truth("vid3"))


# ---------------------------------------------------------------- lock
def test_lock_refuses_a_take_the_frozen_model_was_trained_on(sandbox):
    from src.final import MODEL_PATH, sha256
    with pytest.raises(H.HoldoutError, match="used to train"):
        H.lock(["vid3"], load_meta(), sha256(MODEL_PATH))


def test_lock_refuses_incomplete_labels_and_existing_model_output(sandbox):
    p = sandbox / "ground_truth" / "take_3.csv"
    good = p.read_bytes()
    H.ground_truth_template("vid3", 3).to_csv(p, index=False)
    with pytest.raises(H.HoldoutError, match="not ready"):
        H.lock(["vid3"], FAKE_META, "abc")
    p.write_bytes(good)
    (sandbox / "holdout" / "vid3_events.csv").write_text("x")
    with pytest.raises(H.HoldoutError, match="BEFORE scoring"):
        H.lock(["vid3"], FAKE_META, "abc")


def test_lock_records_hashes_of_the_labels_and_model(sandbox):
    rec = H.lock(["vid3"], FAKE_META, "abc")
    saved = json.loads((sandbox / "holdout" / "lock.json").read_text())
    assert saved == rec and rec["model_sha256"] == "abc"
    assert rec["ground_truth"]["vid3"]["sha256"] == H._sha(sandbox / "ground_truth" / "take_3.csv")


# ---------------------------------------------------------------- score
def _fake_segment(take):
    """Ground truth used as the 'prediction', so the expected score is known: perfect boundaries."""
    gt = pd.read_csv(H.gt_path(take))
    ev = gt[["label", "start_s", "end_s"]].copy()
    ev.insert(0, "take", take)
    ev["event_idx"] = range(len(ev))
    return ev, pd.DataFrame({"t": [0.0]}), None


def _score(takes, sha):
    from src.evaluate import score_take
    from src.signals import load_frames
    return H.score(takes, sha, segment=_fake_segment, score_take=score_take, load_frames=load_frames)


def test_score_refuses_without_a_lock(sandbox):
    with pytest.raises(H.HoldoutError, match="not locked"):
        _score(["vid3"], "abc")


def test_score_refuses_if_labels_or_model_changed_after_locking(sandbox):
    H.lock(["vid3"], FAKE_META, "abc")
    with pytest.raises(H.HoldoutError, match="model changed"):
        _score(["vid3"], "different")
    p = sandbox / "ground_truth" / "take_3.csv"
    d = pd.read_csv(p)
    d.loc[3, "end_s"] += 0.01
    d.loc[4, "start_s"] += 0.01
    d.to_csv(p, index=False)
    with pytest.raises(H.HoldoutError, match="changed after locking"):
        _score(["vid3"], "abc")


def test_score_writes_events_scores_run_record_and_history(sandbox):
    H.lock(["vid3"], FAKE_META, "abc")
    df = _score(["vid3"], "abc")
    allrow = df[df["scope"] == "all"].iloc[0]
    assert allrow["recall"] == 1 and allrow["precision"] == 1 and allrow["mae_matched_s"] == 0   # GT scored against itself
    assert {"cycle1", "cycle2", "cycle3"} <= set(df["scope"])                                     # per-cycle rows
    for f in ("vid3_events.csv", "vid3_signals.csv", "scores.csv", "runs.json"):
        assert (sandbox / "holdout" / f).exists(), f
    runs = json.loads((sandbox / "holdout" / "runs.json").read_text())
    assert len(runs) == 1 and runs[0]["lock"]["model_sha256"] == "abc"
    hist = pd.read_csv(sandbox / "segments" / "history.csv")
    assert set(hist["version"]) == {H.HISTORY_VERSION} and set(hist["take"]) == {"vid3"}


# ---------------------------------------------------------------- prepare
def test_prepare_refuses_without_a_video_and_after_scoring(sandbox):
    noop = lambda *a, **k: None  # noqa: E731
    with pytest.raises(H.HoldoutError, match="not found"):
        H.prepare("vid9", noop, noop, noop)
    (sandbox / "holdout" / "vid3_events.csv").write_text("x")
    with pytest.raises(H.HoldoutError, match="before scoring"):
        H.prepare("vid3", noop, noop, noop)


def test_prepare_runs_the_stages_in_order_and_keeps_existing_labels(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(H, "CONVERTED_DIR", sandbox / "converted")
    r = H.prepare("vid3",
                  convert=lambda s, d: calls.append(("convert", s.name, d.name)),
                  extract=lambda t: calls.append(("extract", t)) or (None, {"frames_extracted": 775, "frames_not_detected": 21}),
                  load_postgres=lambda t: calls.append(("load", t)) or 16275)
    assert [c[0] for c in calls] == ["convert", "extract", "load"]
    assert calls[0][1:] == ("vid3.mov", "vid3.mp4")
    assert r["template_written"] is False and r["postgres_rows"] == 16275


# ---------------------------------------------------------------- export and CLI
def test_export_of_selected_takes_contains_only_those_takes(tmp_path):
    """The parametrised export used for the hold-out reproduces the same tables for one take and leaves the 5-take export alone."""
    import export_tables
    from config import SEGMENTS_DIR
    canon = (ROOT / "data" / "export" / "manifest.json").read_text()
    out = tmp_path / "export"
    export_tables.build(takes_to_export=["vid3"], src=SEGMENTS_DIR / "v2_frozen_oof", out=out, model_version="test",
                        scores_path=SEGMENTS_DIR / "v2_frozen_oof" / "scores.csv")
    raw = pd.read_parquet(out / "raw_keypoints.parquet")
    assert set(raw["take"]) == {"vid3"} and len(raw) == 21 * 775
    ev = pd.read_csv(out / "events.csv")
    assert set(ev["take"]) == {"vid3"} and ev["n_frames"].sum() == 775
    assert (ROOT / "data" / "export" / "manifest.json").read_text() == canon     # canonical export untouched


def test_cli_rejects_unknown_takes_and_reports_status(capsys):
    import run_new_take
    assert run_new_take.main(["prepare", "vid3"]) == 2
    capsys.readouterr()
    assert run_new_take.main(["status"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert set(st) == {"vid6", "vid7"} and all(set(v) >= {"video", "locked", "scored"} for v in st.values())
    assert run_new_take.main([]) == 2
