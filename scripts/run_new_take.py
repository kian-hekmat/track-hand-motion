"""Run the new recordings (vid6, vid7) through the pipeline as a hold-out test of the frozen model.

Usage (in this order; see docs/new_takes.md):
  python scripts/run_new_take.py status
  python scripts/run_new_take.py prepare            # convert, extract keypoints + overlay, Postgres, ground-truth template
  ... review evidence/vidN_overlay.mp4 and fill in ground_truth/take_6.csv and take_7.csv ...
  python scripts/run_new_take.py lock               # validate and hash the labels BEFORE any model output exists
  python scripts/run_new_take.py score              # frozen model, once; scores, plots, history
  python scripts/run_new_take.py export             # verified export tables for the new takes (data/holdout/export), the
                                                    # reference the cloud checks compare against; Tableau tables now come from
                                                    # the Snowflake CLOUD views (see databricks/cloud/README.md, M7)
Add take names to limit a step to some takes, e.g. `prepare vid6`.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HOLDOUT_DIR, HOLDOUT_TAKES, ROOT  # noqa: E402
from src import holdout as H  # noqa: E402

PY = sys.executable


def _prepare(takes):
    from convert_videos import convert
    from src.db import connect, init_schema, load_take
    from src.extract import extract_take

    conn = connect()
    init_schema(conn)
    for t in takes:
        r = H.prepare(t, convert=convert, extract=lambda take: extract_take(take, render_overlay=True),
                      load_postgres=lambda take: load_take(conn, take))
        print(json.dumps(r))
    subprocess.run([PY, str(ROOT / "scripts" / "detection_report.py")], check=True)
    print("\nNext: watch evidence/<take>_overlay.mp4, declare any hand-out-of-frame stretch in data/raw/out_of_frame_intervals.csv,"
          "\nthen label ground_truth/take_6.csv and take_7.csv (see docs/new_takes.md). Do not run 'score' before 'lock'.")


def _lock(takes):
    from src.final import MODEL_PATH, load_meta, sha256
    rec = H.lock(takes, load_meta(), sha256(MODEL_PATH))
    print(json.dumps(rec, indent=2))


def _score(takes):
    from src.evaluate import score_take
    from src.final import MODEL_PATH, segment_new_take, sha256
    from src.signals import load_frames
    df = H.score(takes, sha256(MODEL_PATH), segment=segment_new_take, score_take=score_take, load_frames=load_frames)
    show = df[["take", "group", "scope", "n_true", "n_pred", "recall", "precision", "mae_matched_s", "frame_accuracy", "balanced_accuracy"]]
    print(show.round(3).to_string(index=False))
    subprocess.run([PY, str(ROOT / "scripts" / "plot_segmentation.py"), "--events-dir", str(HOLDOUT_DIR), "--out-subdir", "holdout", *takes], check=True)


def _export(takes):
    import export_tables
    out = HOLDOUT_DIR / "export"
    export_tables.build(takes_to_export=takes, src=HOLDOUT_DIR, out=out, model_version="v2_frozen_holdout",
                        scores_path=HOLDOUT_DIR / "scores.csv")


def main(argv):
    if not argv or argv[0] not in {"status", "prepare", "lock", "score", "export"}:
        print(__doc__)
        return 2
    step, takes = argv[0], argv[1:] or sorted(HOLDOUT_TAKES)
    unknown = set(takes) - set(HOLDOUT_TAKES)
    if unknown:
        print(f"not hold-out takes: {sorted(unknown)} (configured: {sorted(HOLDOUT_TAKES)})")
        return 2
    try:
        if step == "status":
            print(json.dumps(H.status(takes), indent=2))
        else:
            {"prepare": _prepare, "lock": _lock, "score": _score, "export": _export}[step](takes)
    except H.HoldoutError as e:
        print(f"STOPPED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
