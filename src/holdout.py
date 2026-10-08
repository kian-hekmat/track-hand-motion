"""Hold-out test of the frozen model on newly recorded takes (vid6, vid7).

The order is enforced so the result stays unbiased:
  1. prepare  convert + extract keypoints + overlay + Postgres + a blank ground-truth template. NO model output is produced.
  2. (human)  review the overlay, then label the ground truth from the overlay's t= stamps.
  3. lock     validate the labels and record their SHA-256 (and the frozen model's) BEFORE any model output exists.
  4. score    run the frozen model once per take (no retraining, no tuning), score against the locked labels, report the
              hold-out takes as their own groups. Refuses if the labels or the model changed since locking.
  5. export   Tableau-ready tables for the new takes through the same export and SQL-view code as vid1-5 (end-to-end run).
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (BOUNDARY_TOLERANCE_S, CONVERTED_DIR, GROUND_TRUTH_DIR, HOLDOUT_CYCLES, HOLDOUT_DIR, HOLDOUT_TAKES,
                    LABELS, RAW_DIR, SEGMENTS_DIR, VIDS_DIR)
from src import ground_truth as G

GT_COLUMNS = ["take", "cycle", "label", "start_s", "end_s", "source"]
CYCLE = ["REACH", "GRASP", "HOLD", "RELEASE", "RETRACT", "REST"]
HISTORY_VERSION = "v2_frozen_holdout"


class HoldoutError(RuntimeError):
    """Raised when a step is run out of order or its inputs are not in the required state."""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gt_path(take: str) -> Path:
    return GROUND_TRUTH_DIR / f"take_{G.take_number(take)}.csv"


# ------------------------------------------------------------------ 1. prepare
def ground_truth_template(take: str, n_cycles: int) -> pd.DataFrame:
    """Opening REST, then per cycle REACH, GRASP, HOLD, RELEASE, RETRACT and the REST that follows (19 rows for 3 cycles)."""
    rows = [(take, 1, "REST")] + [(take, c, lab) for c in range(1, n_cycles + 1) for lab in CYCLE]
    df = pd.DataFrame(rows, columns=["take", "cycle", "label"])
    for c in ("start_s", "end_s", "source"):
        df[c] = ""
    return df[GT_COLUMNS]


def write_template(take: str, n_cycles: int) -> bool:
    """Write the blank template; never overwrite an existing file (it may hold labels). Returns True if written."""
    p = gt_path(take)
    if p.exists():
        return False
    ground_truth_template(take, n_cycles).to_csv(p, index=False)
    return True


def prepare(take: str, convert, extract, load_postgres) -> dict:
    """Stage 1 for one take. The three callables are injected so tests can run without video files."""
    src = VIDS_DIR / f"{take}.mov"
    if not src.exists():
        raise HoldoutError(f"{src} not found: copy the recording there first")
    if (HOLDOUT_DIR / f"{take}_events.csv").exists():
        raise HoldoutError(f"{take} already has model output; prepare must run before scoring")
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    convert(src, CONVERTED_DIR / f"{take}.mp4")
    _, meta = extract(take)
    loaded = load_postgres(take)
    written = write_template(take, HOLDOUT_CYCLES[take])
    return {"take": take, "frames": meta["frames_extracted"], "frames_not_detected": meta["frames_not_detected"],
            "postgres_rows": loaded, "template_written": written}


# ------------------------------------------------------------------ 3. lock
def validate_ground_truth(take: str) -> list[str]:
    """Problems that stop the labels being locked. Phase order is NOT checked (a hard take may deviate)."""
    p = gt_path(take)
    if not p.exists():
        return [f"{p.name} missing"]
    gt = pd.read_csv(p)
    problems = []
    if list(gt.columns) != GT_COLUMNS:
        return [f"{p.name}: columns must be {GT_COLUMNS}"]
    if gt[["start_s", "end_s", "source"]].isna().any().any():
        problems.append(f"{p.name}: {int(gt['start_s'].isna().sum())} rows have no start time; every row needs start_s, end_s, source")
        return problems
    if not (gt["take"] == take).all():
        problems.append(f"{p.name}: column take must be {take} on every row")
    bad = set(gt["label"]) - set(LABELS)
    if bad:
        problems.append(f"{p.name}: unknown labels {sorted(bad)}")
    if not (gt["source"] == "manual").all():
        problems.append(f"{p.name}: source must be 'manual' (read off the video frames)")
    if not (gt["end_s"] > gt["start_s"]).all():
        problems.append(f"{p.name}: every row needs end_s > start_s")
    gaps = np.abs(gt["start_s"].to_numpy()[1:] - gt["end_s"].to_numpy()[:-1])
    if (gaps > 1e-6).any():
        problems.append(f"{p.name}: rows must be contiguous (next start = previous end); first break at row {int(np.argmax(gaps > 1e-6)) + 2}")
    if abs(gt["start_s"].iloc[0]) > 1e-3:
        problems.append(f"{p.name}: first row must start at 0")
    meta_p = RAW_DIR / f"{take}_meta.json"
    if not meta_p.exists():
        problems.append(f"{meta_p.name} missing: run prepare first")
    else:
        last = json.loads(meta_p.read_text())["last_timestamp_s"]
        if abs(gt["end_s"].iloc[-1] - last) > 1e-3:
            problems.append(f"{p.name}: last end_s must equal the take's last frame time ({last})")
    if int(gt["cycle"].max()) != HOLDOUT_CYCLES.get(take, int(gt["cycle"].max())):
        problems.append(f"{p.name}: {int(gt['cycle'].max())} cycles labelled but HOLDOUT_CYCLES says {HOLDOUT_CYCLES[take]} "
                        "(fix the labels, or update config.HOLDOUT_CYCLES if the recording really differs)")
    return problems


def _lock_file() -> Path:
    return HOLDOUT_DIR / "lock.json"


def lock(takes: list[str], model_meta: dict, model_sha: str) -> dict:
    if any((HOLDOUT_DIR / f"{t}_events.csv").exists() for t in takes):
        raise HoldoutError("model output already exists for these takes; labels must be locked BEFORE scoring")
    problems = [p for t in takes for p in validate_ground_truth(t)]
    if problems:
        raise HoldoutError("ground truth not ready:\n  " + "\n  ".join(problems))
    if model_sha != model_meta["model_sha256"]:
        raise HoldoutError("frozen model file does not match its recorded hash")
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"locked_at": datetime.datetime.now().isoformat(timespec="seconds"), "model_sha256": model_sha,
           "model_train_takes": model_meta["train_takes"],
           "ground_truth": {t: {"file": gt_path(t).name, "sha256": _sha(gt_path(t)), "rows": len(pd.read_csv(gt_path(t)))} for t in takes}}
    overlap = set(takes) & set(model_meta["train_takes"])
    if overlap:
        raise HoldoutError(f"{sorted(overlap)} were used to train the frozen model; they cannot be a hold-out")
    _lock_file().write_text(json.dumps(rec, indent=2))
    return rec


def check_lock(takes: list[str], model_sha: str) -> dict:
    if not _lock_file().exists():
        raise HoldoutError("labels are not locked: run the lock step first")
    rec = json.loads(_lock_file().read_text())
    missing = [t for t in takes if t not in rec["ground_truth"]]
    if missing:
        raise HoldoutError(f"{missing} not in the lock")
    for t in takes:
        if _sha(gt_path(t)) != rec["ground_truth"][t]["sha256"]:
            raise HoldoutError(f"{gt_path(t).name} changed after locking; the hold-out score would no longer be blind")
    if model_sha != rec["model_sha256"]:
        raise HoldoutError("the frozen model changed after locking")
    return rec


# ------------------------------------------------------------------ 4. score
def score(takes: list[str], model_sha: str, segment, score_take, load_frames) -> pd.DataFrame:
    """Run the frozen model once per take and score it. `segment(take) -> (events, signals, P)`."""
    rec = check_lock(takes, model_sha)
    rows = []
    for t in takes:
        ev, sig, _ = segment(t)
        ev.to_csv(HOLDOUT_DIR / f"{t}_events.csv", index=False)
        sig.to_csv(HOLDOUT_DIR / f"{t}_signals.csv", index=False)
        rows += score_take(t, ev, load_frames(t)["t"].to_numpy(), with_cycles=True)
    df = pd.DataFrame(rows)
    df.to_csv(HOLDOUT_DIR / "scores.csv", index=False)
    check_lock(takes, model_sha)            # labels and model still unchanged after scoring
    run = {"scored_at": datetime.datetime.now().isoformat(timespec="seconds"), "lock": rec, "takes": takes,
           "boundary_tolerance_s": BOUNDARY_TOLERANCE_S,
           "summary": df[df["scope"] == "all"][["take", "group", "frame_accuracy", "balanced_accuracy", "recall", "precision"]].to_dict("records")}
    runs_p = HOLDOUT_DIR / "runs.json"
    runs = json.loads(runs_p.read_text()) if runs_p.exists() else []
    runs.append(run)
    runs_p.write_text(json.dumps(runs, indent=2, default=float))
    _append_history(df)
    return df


def _append_history(df: pd.DataFrame) -> None:
    cols = ["n_true", "n_pred", "n_matched", "recall", "precision", "mae_matched_s", "frame_accuracy", "balanced_accuracy",
            "tolerant_accuracy", "majority_baseline", "n_frames"]
    h = df[["take", "group", "scope", *[c for c in cols if c in df]]].copy()
    h.insert(0, "version", HISTORY_VERSION)
    h.insert(1, "date", datetime.date.today().isoformat())
    h["note"] = "frozen model on new recordings; labels locked before scoring"
    hp = SEGMENTS_DIR / "history.csv"
    if hp.exists():
        old = pd.read_csv(hp)
        old = old[~((old["version"] == HISTORY_VERSION) & old["take"].isin(df["take"].unique()))]
        h = pd.concat([old, h], ignore_index=True)
    h.to_csv(hp, index=False)


def status(takes: list[str]) -> dict:
    out = {}
    for t in takes:
        out[t] = {
            "video": (VIDS_DIR / f"{t}.mov").exists(),
            "keypoints": (RAW_DIR / f"{t}_keypoints.csv").exists(),
            "ground_truth_file": gt_path(t).exists(),
            "ground_truth_problems": validate_ground_truth(t) if (RAW_DIR / f"{t}_meta.json").exists() else ["run prepare first"],
            "locked": _lock_file().exists() and t in json.loads(_lock_file().read_text()).get("ground_truth", {}),
            "scored": (HOLDOUT_DIR / f"{t}_events.csv").exists(),
            "exported": (HOLDOUT_DIR / "export" / "events.csv").exists(),
        }
    return out
