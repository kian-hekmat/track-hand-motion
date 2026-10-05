"""Scoring of predicted events against ground truth. There is deliberately no function that pools
takes: clean / fast / hard results are always reported separately."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import BOUNDARY_TOLERANCE_S, LABELS

_BIG = 1e6
_EPS = 1e-9


def match_boundaries(pred: np.ndarray, true: np.ndarray, tol: float = BOUNDARY_TOLERANCE_S):
    """One-to-one matching of predicted to true boundary times, maximising the number of matches
    within `tol` (inclusive), then minimising total error. Returns (pred_idx, true_idx, abs_err)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if len(pred) == 0 or len(true) == 0:
        return np.array([], int), np.array([], int), np.array([], float)
    d = np.abs(pred[:, None] - true[None, :])
    cost = np.where(d <= tol + _EPS, d, _BIG)
    pi, ti = linear_sum_assignment(cost)
    keep = cost[pi, ti] < _BIG
    return pi[keep], ti[keep], d[pi[keep], ti[keep]]


def boundary_metrics(pred, true, tol: float = BOUNDARY_TOLERANCE_S) -> dict:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    pi, ti, err = match_boundaries(pred, true, tol)
    n_m = len(pi)
    return {
        "n_true": len(true), "n_pred": len(pred), "n_matched": n_m,
        "recall": n_m / len(true) if len(true) else np.nan,
        "precision": n_m / len(pred) if len(pred) else np.nan,
        "mae_matched_s": float(err.mean()) if n_m else np.nan,  # NaN, not 0, when nothing matched
        "missed": len(true) - n_m, "false": len(pred) - n_m,
    }


def chance_baselines(n_pred: int, true, t_start: float, t_end: float,
                     tol: float = BOUNDARY_TOLERANCE_S, n_random: int = 1000, seed: int = 0) -> dict:
    """Recall/precision that boundary sets with the same COUNT as the prediction get by chance:
    equally spaced, and the mean over random uniform sets."""
    true = np.asarray(true, float)
    if n_pred == 0:
        return {"uniform_recall": 0.0, "uniform_precision": np.nan,
                "random_recall": 0.0, "random_precision": np.nan}
    uni = np.linspace(t_start, t_end, n_pred + 2)[1:-1]
    u = boundary_metrics(uni, true, tol)
    rng = np.random.default_rng(seed)
    rr, rp = [], []
    for _ in range(n_random):
        m = boundary_metrics(np.sort(rng.uniform(t_start, t_end, n_pred)), true, tol)
        rr.append(m["recall"]); rp.append(m["precision"])
    return {"uniform_recall": u["recall"], "uniform_precision": u["precision"],
            "random_recall": float(np.mean(rr)), "random_precision": float(np.mean(rp))}


def events_to_boundaries(events: pd.DataFrame) -> np.ndarray:
    return events["start_s"].to_numpy(float)[1:]


def confusion(true_labels, pred_labels) -> pd.DataFrame:
    t = pd.Categorical(true_labels, categories=LABELS)
    p = pd.Categorical(pred_labels, categories=LABELS)
    return pd.crosstab(t, p, dropna=False).reindex(index=LABELS, columns=LABELS, fill_value=0)


def frame_metrics(true_labels, pred_labels) -> dict:
    """Overall frame accuracy and per-label accuracy (fraction of that label's frames predicted
    correctly). Frames with an unknown label on either side are excluded and counted."""
    true_labels, pred_labels = np.asarray(true_labels, object), np.asarray(pred_labels, object)
    ok = np.array([a is not None and b is not None for a, b in zip(true_labels, pred_labels)], bool)
    tl, pl = true_labels[ok], pred_labels[ok]
    out = {"n_frames": int(ok.sum()), "n_unlabelled": int((~ok).sum()),
           "frame_accuracy": float((tl == pl).mean()) if len(tl) else np.nan}
    for lab in LABELS:
        m = tl == lab
        out[f"acc_{lab}"] = float((pl[m] == lab).mean()) if m.any() else np.nan
        out[f"n_{lab}"] = int(m.sum())
    accs = [out[f"acc_{l}"] for l in LABELS if out[f"n_{l}"] > 0]
    # mean of per-label accuracies: not dominated by frequent easy labels (REST, REACH)
    out["balanced_accuracy"] = float(np.mean(accs)) if accs else np.nan
    if len(tl):  # accuracy of always predicting the most common true label
        out["majority_baseline"] = float(pd.Series(tl).value_counts(normalize=True).iloc[0])
    else:
        out["majority_baseline"] = np.nan
    return out


def tolerant_frame_accuracy(t, true_labels, pred_labels, bound_t, bound_before, bound_after,
                            tol: float = BOUNDARY_TOLERANCE_S) -> float:
    """Frame accuracy where frames within `tol` seconds of a true boundary are also counted correct
    if the prediction equals the label on EITHER side of that boundary. Separates boundary-timing
    jitter (human labels are good to ~1-2 frames) from genuinely wrong labels. Always >= strict accuracy."""
    t = np.asarray(t, float)
    true_labels, pred_labels = np.asarray(true_labels, object), np.asarray(pred_labels, object)
    ok = np.array([a is not None and b is not None for a, b in zip(true_labels, pred_labels)], bool)
    correct = np.array([a == b for a, b in zip(true_labels, pred_labels)], bool)
    for bt, lb, la in zip(bound_t, bound_before, bound_after):
        near = np.abs(t - bt) <= tol + _EPS
        correct |= near & np.array([p in (lb, la) for p in pred_labels], bool)
    return float(correct[ok].mean()) if ok.any() else np.nan
