"""Plot speed, aperture and progress with ground-truth bands (top strip) and detected bands
(bottom strip) for human review. Writes evidence/phase2/<take>_segmentation.png."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EVIDENCE_DIR, SEGMENTS_DIR, TAKE_GROUPS
from src.ground_truth import load_ground_truth, out_of_frame_intervals

COL = {"REST": "#bbbbbb", "REACH": "#4c78a8", "GRASP": "#f58518", "HOLD": "#54a24b",
       "RELEASE": "#e45756", "RETRACT": "#b279a2"}


def bands(ax, seg, y0, h):
    for r in seg.itertuples():
        ax.axvspan(r.start_s, r.end_s, ymin=y0, ymax=y0 + h, color=COL[r.label], alpha=0.9)


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--events-dir", default=str(SEGMENTS_DIR))
ap.add_argument("--out-subdir", default="phase2")
ap.add_argument("takes", nargs="*")
args = ap.parse_args()
EVD = Path(args.events_dir)
out = EVIDENCE_DIR / args.out_subdir
out.mkdir(parents=True, exist_ok=True)
for take in args.takes or sorted(TAKE_GROUPS):
    sig = pd.read_csv(EVD / f"{take}_signals.csv")
    ev = pd.read_csv(EVD / f"{take}_events.csv")
    gt = load_ground_truth(take)
    fig, ax = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    for a, c, lab in zip(ax, ["speed", "aperture", "p"], ["wrist speed (hand-lengths/s)", "aperture", "progress p"]):
        a.plot(sig["t"], sig[c], color="k", lw=0.9)
        a.set_ylabel(lab)
        for b in gt["start_s"].iloc[1:]:
            a.axvline(b, color="r", lw=0.5, alpha=0.6)
        for b in ev["start_s"].iloc[1:]:
            a.axvline(b, color="b", lw=0.5, ls="--", alpha=0.6)
        for s, e in out_of_frame_intervals(take):
            a.axvspan(s, e, color="yellow", alpha=0.4)
    bands(ax[0], gt, 0.93, 0.07)
    bands(ax[0], ev, 0.86, 0.07)
    ax[0].set_title(f"{take} ({TAKE_GROUPS[take]}): top strip = ground truth, second strip = detected; "
                    "red solid = true boundary, blue dashed = detected; yellow = hand out of frame")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()], labels=list(COL), ncol=6, loc="upper right", fontsize=7)
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / f"{take}_segmentation.png", dpi=80)
    plt.close(fig)
    print("wrote", out / f"{take}_segmentation.png")
