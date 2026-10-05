"""Tune segmentation parameters by seeded random search on TUNING_TAKES ONLY (vid1, vid2).

vid3 (held-out clean), vid4 (fast) and vid5 (hard) are never loaded here. Objective per take =
0.5 * boundary F1 (tolerance from config) + 0.5 * frame accuracy; the search maximises the mean over
the tuning takes. The chosen parameters are frozen in data/segments/params.json together with a
record of what they were tuned on.
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BOUNDARY_TOLERANCE_S, SEGMENTS_DIR, TUNING_TAKES
from src import ground_truth as G
from src import score as S
from src.evaluate import predicted_labels_at
from src.segment import PARAMS_FILE, SegParams, segment_signals
from src.signals import SignalParams, derive_signals, load_frames

N_TRIALS, SEED = 800, 20251002
assert set(TUNING_TAKES) == {"vid1", "vid2"}

SPACE = {
    "smooth_win": lambda r: int(r.choice([5, 7, 9, 11])),
    "pen": lambda r: float(np.exp(r.uniform(np.log(0.3), np.log(30)))),
    "ap_scale": lambda r: float(r.choice([0.3, 0.5, 0.8])),
    "still_thr": lambda r: float(r.uniform(0.15, 0.8)),
    "moving_thr": lambda r: float(r.uniform(0.6, 2.0)),
    "p_home": lambda r: float(r.uniform(0.15, 0.35)),
    "p_far": lambda r: float(r.uniform(0.45, 0.75)),
    "ap_closed": lambda r: float(r.uniform(0.8, 1.0)),
    "ap_delta": lambda r: float(r.uniform(0.05, 0.25)),
    "dp_min": lambda r: float(r.uniform(0.02, 0.15)),
}


def objective(params, data):
    scores = []
    for take, (frames, t_end, sig_cache) in data.items():
        sig = sig_cache(params.smooth_win)
        ev = segment_signals(sig, params, t_end)
        gt = G.load_ground_truth(take)
        bm = S.boundary_metrics(S.events_to_boundaries(ev), G.boundaries(gt), BOUNDARY_TOLERANCE_S)
        p, r = bm["precision"], bm["recall"]
        f1 = 0.0 if (not np.isfinite(p)) or p + r == 0 else 2 * p * r / (p + r)
        ft = frames["t"].to_numpy()
        fm = S.frame_metrics(G.labels_at(gt, ft), predicted_labels_at(ev, ft))
        scores.append(0.5 * f1 + 0.5 * fm["frame_accuracy"])
    return float(np.mean(scores)), scores


def main():
    data = {}
    for take in TUNING_TAKES:
        frames = load_frames(take)
        cache = {}
        def sig_cache(win, frames=frames, cache=cache):
            if win not in cache:
                cache[win] = derive_signals(frames, SignalParams(smooth_win=win), t_end=float(frames["t"].iloc[-1]))
            return cache[win]
        data[take] = (frames, float(frames["t"].iloc[-1]), sig_cache)

    rng = np.random.default_rng(SEED)
    base, trials = SegParams(), []
    trials.append((objective(base, data), base))
    for _ in range(N_TRIALS):
        P = SegParams(**{k: f(rng) for k, f in SPACE.items()})
        trials.append((objective(P, data), P))
    trials.sort(key=lambda x: -x[0][0])
    print(f"default params objective: {objective(base, data)[0]:.3f}")
    for (obj, per), P in trials[:5]:
        print(f"{obj:.3f} per-take={[round(x, 3) for x in per]} {P.to_dict()}")
    (best, per), P = trials[0]
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps({
        "params": P.to_dict(), "tuned_on": list(TUNING_TAKES), "objective": best,
        "per_take_objective": dict(zip(TUNING_TAKES, per)), "n_trials": N_TRIALS, "seed": SEED,
        "boundary_tolerance_s": BOUNDARY_TOLERANCE_S, "date": datetime.date.today().isoformat(),
        "note": "vid3, vid4, vid5 were not used in any way during tuning.",
    }, indent=2))
    print("frozen ->", PARAMS_FILE)


if __name__ == "__main__":
    main()
