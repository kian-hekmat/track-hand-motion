"""Negative control for the learned segmenter: if the classifier learns real phase signatures, then
training on time-shifted (misaligned) labels must destroy held-out accuracy. If it did not, the good
numbers would be suspect (leak / shortcut)."""
import numpy as np

from src.features import take_data
from src.learned import LAB_IDX, make_classifier, predict_proba
from sklearn.utils.class_weight import compute_sample_weight


def _xy(take, shift=0):
    _, feat, lab, _ = take_data(take)
    keep = np.array([l is not None for l in lab])
    y = np.array([LAB_IDX[l] for l in lab[keep]])
    X = feat[keep].reset_index(drop=True)
    if shift:
        y = np.roll(y, shift)  # same label counts, wrong timing
    return X, y


def _heldout_accuracy(train_takes, test_take, shift):
    parts = [_xy(t, shift) for t in train_takes]
    X = np.concatenate([p[0].to_numpy() for p in parts]); y = np.concatenate([p[1] for p in parts])
    clf = make_classifier("hgb")
    clf.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    Xt, yt = _xy(test_take)
    P = np.zeros((len(Xt), 6)); P[:, clf.classes_] = clf.predict_proba(Xt.to_numpy())
    return float((P.argmax(1) == yt).mean())


def test_aligned_labels_generalise_but_shifted_labels_do_not():
    real = _heldout_accuracy(["vid1", "vid2"], "vid3", shift=0)
    control = _heldout_accuracy(["vid1", "vid2"], "vid3", shift=200)
    assert real > 0.85, real
    assert control < 0.5, control
    assert real - control > 0.4
