"""Cloud-path milestone M0: the saved leave-one-take-out fold models (models/folds/, scripts/save_fold_models.py).

The cloud pipeline labels each development take with the fold model that never saw it. These tests pin that the saved
files are the ones recorded, that no fold model was trained on the take it labels, and that the loader the cloud
pipeline calls (src.final.load_model_for_take + segment_frames) reproduces the verified canonical events byte for byte."""
import json

import pytest

from config import HOLDOUT_DIR, HOLDOUT_TAKES, SEGMENTS_DIR, TAKE_GROUPS
from src.final import (FOLDS_META_PATH, MODEL_PATH, fold_model_path, load_meta, load_model_for_take,
                       segment_frames, sha256)
from src.signals import load_frames

TAKES = sorted(TAKE_GROUPS)


@pytest.fixture(scope="module")
def folds():
    return json.loads(FOLDS_META_PATH.read_text())["folds"]


def test_one_fold_model_per_development_take_with_matching_hash(folds):
    assert sorted(folds) == TAKES == sorted(load_meta()["train_takes"])
    for take, f in folds.items():
        assert fold_model_path(take).name == f["file"]
        assert sha256(fold_model_path(take)) == f["sha256"], take


def test_no_fold_model_was_trained_on_the_take_it_labels(folds):
    for take, f in folds.items():
        assert f["labels_take"] == take
        assert sorted(f["train_takes"]) == [t for t in TAKES if t != take]
        clf, _, name = load_model_for_take(take)
        assert take not in clf.train_takes_ and name == f["file"]


@pytest.mark.parametrize("take", TAKES)
def test_cloud_loader_reproduces_canonical_events_exactly(take):
    clf, meta, _ = load_model_for_take(take)
    ev, _, _ = segment_frames(load_frames(take), take, model=clf, meta=meta)
    assert ev.to_csv(index=False) == (SEGMENTS_DIR / "v2_frozen_oof" / f"{take}_events.csv").read_text()


@pytest.mark.parametrize("take", sorted(HOLDOUT_TAKES))
def test_takes_outside_training_use_the_full_model_and_reproduce_holdout_events(take):
    model, meta, name = load_model_for_take(take)
    assert name == MODEL_PATH.name and take not in model.train_takes_
    ev, _, _ = segment_frames(load_frames(take), take, model=model, meta=meta)
    assert ev.to_csv(index=False) == (HOLDOUT_DIR / f"{take}_events.csv").read_text()
