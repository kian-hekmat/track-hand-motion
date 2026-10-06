"""Glue: score one take's events against its ground truth (per take, never pooled)."""
import numpy as np
import pandas as pd

from config import ALL_GROUPS, BOUNDARY_TOLERANCE_S
from src import ground_truth as G
from src import score as S


def predicted_labels_at(events: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    return G.labels_at(events, t)


def score_take(take: str, events: pd.DataFrame, frame_t: np.ndarray, tol: float = BOUNDARY_TOLERANCE_S,
               with_cycles: bool = False) -> list[dict]:
    """Rows of metrics: one 'all' row (and one per cycle if with_cycles). Frame metrics use the native
    frame timestamps; frames inside declared out-of-frame intervals are excluded from the primary
    accuracy and reported on their own row ('out_of_frame')."""
    gt = G.load_ground_truth(take)
    true_b, pred_b = G.boundaries(gt), S.events_to_boundaries(events)
    t0, t1 = float(gt["start_s"].iloc[0]), float(gt["end_s"].iloc[-1])
    tl, pl = G.labels_at(gt, frame_t), predicted_labels_at(events, frame_t)
    oof = G.in_intervals(frame_t, G.out_of_frame_intervals(take))
    base = {"take": take, "group": ALL_GROUPS[take], "tolerance_s": tol}

    before = G.labels_at(gt, true_b - 1e-6)
    after = G.labels_at(gt, true_b + 1e-6)

    def row(scope, tmask, bmask_true, bmask_pred):
        bm = S.boundary_metrics(pred_b[bmask_pred], true_b[bmask_true], tol)
        m = tmask & ~oof
        fm = S.frame_metrics(tl[m], pl[m])
        fm["tolerant_accuracy"] = S.tolerant_frame_accuracy(
            frame_t[m], tl[m], pl[m], true_b[bmask_true], before[bmask_true], after[bmask_true], tol)
        ch = S.chance_baselines(int(bmask_pred.sum()), true_b[bmask_true], t0, t1, tol)
        return {**base, "scope": scope, **bm, **{f"chance_{k}": v for k, v in ch.items()}, **fm}

    rows = [row("all", np.ones(len(frame_t), bool), np.ones(len(true_b), bool), np.ones(len(pred_b), bool))]
    if oof.any():
        fm = S.frame_metrics(tl[oof], pl[oof])
        rows.append({**base, "scope": "out_of_frame", **fm})
    if with_cycles:
        cyc_b = G.cycle_of_boundaries(gt)
        for c, (a, b) in G.cycle_spans(gt).items():
            rows.append(row(f"cycle{c}", (frame_t >= a) & (frame_t < b + 1e-9),
                            cyc_b == c, (pred_b >= a) & (pred_b < b)))
    return rows
