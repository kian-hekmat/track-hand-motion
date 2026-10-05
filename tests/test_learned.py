"""Tests for the v2 pipeline (features, augmentation, decoders, leakage guards)."""
import numpy as np
import pandas as pd
import pytest

from config import GRID_HZ, LABELS
from src import ground_truth as G
from src.features import build_features, scaled_ground_truth, take_data, time_scaled_frames
from src.learned import LAB_IDX, decode_argmax, decode_grammar, decode_pelt, fit, predict_proba
from src.score import boundary_metrics
from src.signals import SignalParams, derive_signals, load_frames
from tests.test_signals import make_frames

DT = 1 / GRID_HZ


def test_features_have_no_nan_and_fixed_columns_even_with_gaps():
    f = make_frames(vx_px=300.0)
    f.loc[100:114, "detected"] = False
    feat = build_features(derive_signals(f))
    assert not feat.isna().any().any()
    assert feat["missing"].sum() > 0
    assert list(feat.columns) == list(build_features(derive_signals(make_frames(vx_px=300.0))).columns)


def test_features_do_not_depend_on_other_takes():
    """Same input -> same features; features of a take are a pure function of that take's signals."""
    a = build_features(derive_signals(make_frames(vx_px=300.0)))
    b = build_features(derive_signals(make_frames(vx_px=300.0)))
    pd.testing.assert_frame_equal(a, b)


def test_time_scaling_doubles_speed_and_halves_ground_truth_times():
    base = derive_signals(make_frames(vx_px=300.0))
    fast = derive_signals(time_scaled_frames(make_frames(vx_px=300.0), 2.0))
    mid = lambda s: s[(s.t > 0.8) & (s.t < base.t.iloc[-1] / 2 - 0.5)]["speed"].mean()
    assert mid(fast) == pytest.approx(2 * mid(base), rel=0.05)
    gt = G.load_ground_truth("vid1")
    g2 = scaled_ground_truth(gt, 2.0)
    assert g2["end_s"].iloc[-1] == pytest.approx(gt["end_s"].iloc[-1] / 2)
    assert (g2["end_s"] - g2["start_s"]).to_numpy() == pytest.approx((gt["end_s"] - gt["start_s"]).to_numpy() / 2)


def test_take_data_labels_cover_every_sample_and_use_scaled_clock():
    sig, feat, lab, t_end = take_data("vid1", 2.0)
    assert len(sig) == len(feat) == len(lab)
    assert t_end == pytest.approx(load_frames("vid1")["t"].iloc[-1] / 2)
    assert sum(l is None for l in lab) <= 1  # at most the final sample


def test_heldout_take_is_never_in_training_set():
    clf = fit(["vid1", "vid2"], "logreg")
    assert set(clf.train_takes_) == {"vid1", "vid2"}
    assert "vid3" not in clf.train_takes_


def test_prediction_changes_when_a_take_is_removed_from_training_proving_it_is_used():
    """Sanity for the leakage guard: a model trained WITH vid1 predicts vid1 better than one trained
    without it (if both were equal the 'train_takes_' bookkeeping would prove nothing)."""
    _, feat, lab, _ = take_data("vid1")
    keep = np.array([l is not None for l in lab])
    y = np.array([LAB_IDX[l] for l in lab[keep]])
    acc = lambda takes: (predict_proba(fit(takes, "hgb"), feat)[keep].argmax(1) == y).mean()
    assert acc(["vid1", "vid2"]) > acc(["vid2", "vid3"])


def _post(seq):
    """One-hot-ish posteriors from a list of (label, n_samples)."""
    rows = []
    for lab, n in seq:
        r = np.full(len(LABELS), 0.02); r[LAB_IDX[lab]] = 0.9
        rows += [r] * n
    return np.array(rows)


def test_pelt_decoder_finds_known_label_changes_within_tolerance():
    seq = [("REST", 45), ("REACH", 36), ("GRASP", 45), ("HOLD", 45), ("RELEASE", 24), ("RETRACT", 36), ("REST", 45)]
    P = _post(seq)
    rng = np.random.default_rng(0)
    P = np.clip(P + 0.05 * rng.standard_normal(P.shape), 0, 1)
    t = np.arange(len(P)) * DT
    ev = decode_pelt(P, t, t[-1], "x", pen=2.0)
    true_b = np.cumsum([n for _, n in seq])[:-1] * DT
    m = boundary_metrics(ev["start_s"].to_numpy()[1:], true_b, 0.1)
    assert m["recall"] == 1 and m["false"] == 0
    assert list(ev["label"]) == [l for l, _ in seq]


def test_decoders_return_contiguous_events():
    P = _post([("REST", 60), ("REACH", 40), ("GRASP", 50)])
    t = np.arange(len(P)) * DT
    for ev in (decode_pelt(P, t, t[-1], "x", pen=2.0), decode_argmax(P, t, t[-1], "x"),
               decode_grammar(P, t, t[-1], "x")):
        assert ev["start_s"].iloc[0] == 0 and ev["end_s"].iloc[-1] == pytest.approx(t[-1])
        assert np.allclose(ev["start_s"].to_numpy()[1:], ev["end_s"].to_numpy()[:-1])


def test_grammar_decoder_enforces_protocol_order_but_pelt_decoder_does_not():
    """A GRASP directly after REST is off-grammar: Viterbi with a strong order prior and weak evidence
    must not output it; the headline (pelt) decoder reports what the classifier says."""
    seq = [("REST", 40), ("HOLD", 6), ("REST", 40)]   # brief, off-grammar HOLD blip
    P = _post(seq)
    t = np.arange(len(P)) * DT
    pelt = decode_pelt(P, t, t[-1], "x", pen=0.5)
    gram = decode_grammar(P, t, t[-1], "x", switch_penalty=8.0, off_grammar_penalty=30.0)
    assert "HOLD" in set(pelt["label"])
    assert set(gram["label"]) == {"REST"}


def test_grammar_decoder_follows_the_cycle_order_on_clean_input():
    seq = [("REST", 30), ("REACH", 30), ("GRASP", 30), ("HOLD", 30), ("RELEASE", 30), ("RETRACT", 30), ("REST", 30)]
    P = _post(seq)
    t = np.arange(len(P)) * DT
    assert list(decode_grammar(P, t, t[-1], "x")["label"]) == [l for l, _ in seq]
