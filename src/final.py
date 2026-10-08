"""Frozen v2 segmenter: load the saved model and segment ANY take (including new recordings).

models/segmenter_v2.joblib + models/segmenter_v2.json are written by scripts/freeze_model.py. The metadata
records the training takes, the frozen PELT penalty, the feature columns and a SHA-256 of the model file, so
that a modified or swapped model is detected (tests/test_frozen_model.py).
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from config import ROOT
from src.features import build_features
from src.learned import decode_pelt, predict_proba
from src.signals import SignalParams, derive_signals, load_frames

MODEL_PATH = ROOT / "models" / "segmenter_v2.joblib"
META_PATH = ROOT / "models" / "segmenter_v2.json"
FOLDS_DIR = ROOT / "models" / "folds"                 # leave-one-take-out models (scripts/save_fold_models.py)
FOLDS_META_PATH = FOLDS_DIR / "folds.json"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_meta() -> dict:
    return json.loads(META_PATH.read_text())


def load_model(verify: bool = True):
    meta = load_meta()
    if verify and sha256(MODEL_PATH) != meta["model_sha256"]:
        raise RuntimeError("segmenter_v2.joblib does not match the hash recorded in segmenter_v2.json")
    return joblib.load(MODEL_PATH), meta


def fold_model_path(take: str):
    return FOLDS_DIR / f"segmenter_v2_fold_{take}.joblib"


def load_model_for_take(take: str, verify: bool = True):
    """The model that may label `take` honestly: for a development take, its leave-one-take-out fold model (never
    trained on it); for any other take, the full frozen model. Returns (model, meta, model_name)."""
    model, meta = load_model(verify)
    if take not in meta["train_takes"]:
        return model, meta, MODEL_PATH.name
    fold = json.loads(FOLDS_META_PATH.read_text())["folds"][take]
    path = fold_model_path(take)
    if verify and sha256(path) != fold["sha256"]:
        raise RuntimeError(f"{path.name} does not match the hash recorded in folds.json")
    clf = joblib.load(path)
    assert take not in clf.train_takes_, f"fold model for {take} was trained on it"
    return clf, meta, path.name


def segment_frames(frames: pd.DataFrame, take: str, model=None, meta=None):
    """Frozen-model segmentation of a frames table (see src.signals.load_frames).
    Returns (events, signals, posteriors)."""
    if model is None:
        model, meta = load_model()
    t_end = float(frames["t"].iloc[-1])
    sig = derive_signals(frames, SignalParams(smooth_win=meta["smooth_win"]), t_end=t_end)
    feat = build_features(sig)
    assert list(feat.columns) == meta["feature_columns"], "feature columns changed since the model was frozen"
    P = predict_proba(model, feat)
    ev = decode_pelt(P, sig["t"].to_numpy(), t_end, take, pen=meta["pelt_pen"], smooth=meta["posterior_smooth"],
                     min_size_s=meta["min_size_s"])
    return ev, sig, P


def segment_new_take(take: str):
    """Segment a take whose keypoint CSV is already in data/raw (after convert + extract)."""
    return segment_frames(load_frames(take), take)
