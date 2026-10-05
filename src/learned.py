"""Phase 2 v2: frame classifier + ruptures PELT on its phase probabilities + decoders.

Pipeline for a take:  features -> class probabilities (T x 6) -> decoder -> events.
Decoders:
  pelt     (headline)  PELT on smoothed probabilities -> boundaries; each segment takes the label with the
                       highest mean probability; adjacent equal labels merge. Phase order is NOT used.
  argmax   (diagnostic) per-frame argmax with a minimum-duration filter; no changepoint step.
  grammar  (variant)   Viterbi over the protocol's phase order (REST>REACH>GRASP>HOLD>RELEASE>RETRACT>REST,
                       self-loops free, other transitions penalised). Uses protocol knowledge; reported
                       separately from the headline.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import GRID_HZ, LABELS
from src.features import take_data

AUG_FACTORS = (1.0, 1.5, 2.0, 3.0)
LAB_IDX = {l: i for i, l in enumerate(LABELS)}
NEXT = {"REST": "REACH", "REACH": "GRASP", "GRASP": "HOLD", "HOLD": "RELEASE", "RELEASE": "RETRACT", "RETRACT": "REST"}


def make_classifier(kind: str, seed: int = 0):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, random_state=seed))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08,
                                              l2_regularization=1.0, random_state=seed)
    raise ValueError(kind)


def training_set(takes, factors=AUG_FACTORS):
    X, y = [], []
    for t in takes:
        for f in factors:
            _, feat, lab, _ = take_data(t, f)
            keep = np.array([l is not None for l in lab])
            X.append(feat[keep]); y.append(np.array([LAB_IDX[l] for l in lab[keep]]))
    return pd.concat(X, ignore_index=True), np.concatenate(y)


def fit(takes, kind: str, seed: int = 0, factors=AUG_FACTORS):
    """Train on `takes` only (plus their time-scaled copies)."""
    X, y = training_set(takes, factors)
    clf = make_classifier(kind, seed)
    w = compute_sample_weight("balanced", y)
    if kind == "logreg":
        clf.fit(X, y, logisticregression__sample_weight=w)
    else:
        clf.fit(X, y, sample_weight=w)
    clf.train_takes_ = tuple(takes)
    return clf


def predict_proba(clf, feat: pd.DataFrame) -> np.ndarray:
    P = np.zeros((len(feat), len(LABELS)))
    P[:, clf.classes_ if hasattr(clf, "classes_") else clf[-1].classes_] = clf.predict_proba(feat)
    return P


# ---------------- decoders ----------------
def _smooth(P: np.ndarray, n: int) -> np.ndarray:
    return pd.DataFrame(P).rolling(n, center=True, min_periods=1).mean().to_numpy() if n > 1 else P


def _events(labels_idx: list[int], edges: list[int], t: np.ndarray, t_end: float, P: np.ndarray, take: str):
    merged = [[labels_idx[0], edges[0], edges[1]]]
    for lab, a, b in zip(labels_idx[1:], edges[1:-1], edges[2:]):
        if lab == merged[-1][0]:
            merged[-1][2] = b
        else:
            merged.append([lab, a, b])
    rows = []
    for i, (lab, a, b) in enumerate(merged):
        s = float(t[a]); e = float(t[b]) if b < len(t) else float(t_end)
        rows.append({"take": take, "event_idx": i, "label": LABELS[lab], "start_s": s, "end_s": e,
                     "duration_s": e - s, "n_samples": b - a, "mean_confidence": float(P[a:b, lab].mean())})
    return pd.DataFrame(rows)


def decode_pelt(P, t, t_end, take, pen: float, smooth: int = 5, min_size_s: float = 0.1):
    Ps = _smooth(P, smooth)
    bk = rpt.Pelt(model="l2", min_size=max(2, int(round(min_size_s * GRID_HZ))), jump=1).fit(Ps).predict(pen=pen)
    edges = [0] + [b for b in bk if b < len(Ps)] + [len(Ps)]
    labs = [int(np.argmax(Ps[a:b].mean(0))) for a, b in zip(edges[:-1], edges[1:])]
    return _events(labs, edges, t, t_end, Ps, take)


def decode_argmax(P, t, t_end, take, smooth: int = 5, min_len_s: float = 0.2):
    Ps = _smooth(P, smooth)
    lab = Ps.argmax(1)
    # drop runs shorter than min_len by absorbing them into the previous run
    runs = []
    for i, l in enumerate(lab):
        if runs and runs[-1][0] == l:
            runs[-1][2] = i + 1
        else:
            runs.append([l, i, i + 1])
    min_n = int(round(min_len_s * GRID_HZ))
    fixed = []
    for l, a, b in runs:
        if fixed and (b - a) < min_n:
            fixed[-1][2] = b
        else:
            fixed.append([l, a, b])
    edges = [fixed[0][1]] + [r[2] for r in fixed]
    return _events([r[0] for r in fixed], edges, t, t_end, Ps, take)


def decode_grammar(P, t, t_end, take, switch_penalty: float = 4.0, off_grammar_penalty: float = 12.0, smooth: int = 3):
    """Viterbi with free self-loops, cheap transitions along the protocol order, expensive others."""
    Ps = np.clip(_smooth(P, smooth), 1e-4, 1.0)
    logp = np.log(Ps)
    K = len(LABELS)
    trans = np.full((K, K), -off_grammar_penalty)
    for a in range(K):
        trans[a, a] = 0.0
        trans[a, LAB_IDX[NEXT[LABELS[a]]]] = -switch_penalty
    T = len(Ps)
    score = np.zeros((T, K)); back = np.zeros((T, K), int)
    score[0] = logp[0]
    for i in range(1, T):
        cand = score[i - 1][:, None] + trans
        back[i] = cand.argmax(0)
        score[i] = cand.max(0) + logp[i]
    path = [int(score[-1].argmax())]
    for i in range(T - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path = path[::-1]
    edges = [0] + [i for i in range(1, T) if path[i] != path[i - 1]] + [T]
    labs = [path[a] for a in edges[:-1]]
    return _events(labs, edges, t, t_end, Ps, take)
