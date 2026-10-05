"""Score <events-dir>/<take>_events.csv against ground truth, one table per group (clean / fast /
hard); nothing is pooled across groups. Writes scores.csv + results.md into the events dir and appends
every row to data/segments/history.csv (the run log used to track progress across versions).

Usage: score_segmentation.py [--events-dir DIR] [--version NAME] [--note TEXT]
"""
import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BOUNDARY_TOLERANCE_S, LABELS, SEGMENTS_DIR, TAKE_GROUPS
from src.evaluate import score_take
from src.signals import load_frames

ap = argparse.ArgumentParser()
ap.add_argument("--events-dir", default=str(SEGMENTS_DIR))
ap.add_argument("--version", default="v1_rules_pelt")
ap.add_argument("--note", default="")
args = ap.parse_args()
EVD = Path(args.events_dir)

rows = []
for take in sorted(TAKE_GROUPS):
    ev = pd.read_csv(EVD / f"{take}_events.csv")
    t = load_frames(take)["t"].to_numpy()
    rows += score_take(take, ev, t, with_cycles=(take == "vid5"))
df = pd.DataFrame(rows)
df.to_csv(EVD / "scores.csv", index=False)

# ---- run history (append) ----
hist_cols = ["n_true", "n_pred", "n_matched", "recall", "precision", "mae_matched_s",
             "frame_accuracy", "balanced_accuracy", "tolerant_accuracy", "majority_baseline", "n_frames"]
h = df[["take", "group", "scope", *hist_cols]].copy()
h.insert(0, "version", args.version)
h.insert(1, "date", datetime.date.today().isoformat())
h["note"] = args.note
hp = SEGMENTS_DIR / "history.csv"
if hp.exists():
    old = pd.read_csv(hp)
    old = old[old["version"] != args.version]  # re-scoring a version replaces its rows
    h = pd.concat([old, h], ignore_index=True)
h.to_csv(hp, index=False)

f = lambda v, n=3: "n/a" if pd.isna(v) else f"{v:.{n}f}"
i = lambda v: "" if pd.isna(v) else str(int(v))
out = [f"# Phase 2 results: {args.version} (boundary tolerance {BOUNDARY_TOLERANCE_S:.2f} s)\n",
       "Reported per group; never pooled. Frame accuracy excludes frames inside declared out-of-frame "
       "intervals (separate rows). `balanced` = mean of per-label accuracies; `tolerant` = frames within "
       "the tolerance of a true boundary also count if they match either neighbouring label. `chance` = "
       "recall of boundary sets with the same count as predicted (uniform / mean of 1000 random).\n"]
for group in ("clean", "fast", "hard"):
    out.append(f"\n## {group}\n")
    out.append("| take | scope | true | pred | matched | recall | precision | MAE matched (s) | chance recall | frame acc | balanced | tolerant | majority baseline |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in df[df["group"] == group].itertuples():
        if r.scope == "out_of_frame":
            out.append(f"| {r.take} | out_of_frame | | | | | | | | {f(r.frame_accuracy)} ({i(r.n_frames)} frames) | | | |")
            continue
        out.append(f"| {r.take} | {r.scope} | {i(r.n_true)} | {i(r.n_pred)} | {i(r.n_matched)} | {f(r.recall)} | {f(r.precision)} | "
                   f"{f(r.mae_matched_s)} | {f(r.chance_uniform_recall,2)}/{f(r.chance_random_recall,2)} | "
                   f"{f(r.frame_accuracy)} | {f(r.balanced_accuracy)} | {f(r.tolerant_accuracy)} | {f(r.majority_baseline)} |")
    out.append("\nPer-label frame accuracy (scope=all):\n")
    out.append("| take | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(LABELS))
    for r in df[(df["group"] == group) & (df["scope"] == "all")].to_dict("records"):
        out.append(f"| {r['take']} | " + " | ".join(f"{f(r[f'acc_{l}'],2)} (n={r[f'n_{l}']})" for l in LABELS) + " |")
(EVD / "results.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
