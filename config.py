"""Central paths and thresholds for the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDS_DIR = ROOT / "vids"                      # original iPhone .mov files
CONVERTED_DIR = ROOT / "data" / "converted"   # H.264 SDR .mp4 copies
RAW_DIR = ROOT / "data" / "raw"               # per-take keypoint CSVs
EVIDENCE_DIR = ROOT / "evidence"              # overlay clips / screenshots
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

NUM_HANDS = 1
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Frame-count test tolerance (fraction of expected frames)
FRAME_COUNT_TOLERANCE = 0.02

# ---- Phase 2 (segmentation) ----
GROUND_TRUTH_DIR = ROOT / "ground_truth"
SEGMENTS_DIR = ROOT / "data" / "segments"
FRAME_W, FRAME_H = 1920, 1080        # all takes verified 1920x1080 (landscape)
GRID_HZ = 30.0                       # uniform grid for segmentation (nominal iPhone fps)
MAX_INTERP_FRAMES = 2                # gaps <= this many frames are interpolated (flagged)

# Boundary-match tolerance. Fixed BEFORE any segmentation was run. 0.10 s = 3 frames: wide enough
# to cover frame-level human labelling error (~1-2 frames), narrow enough that it is not larger
# than most ground-truth segments (0.2 s was larger than the vid4/vid5 RELEASE segments).
BOUNDARY_TOLERANCE_S = 0.10

# Reporting groups: never pooled.
TAKE_GROUPS = {"vid1": "clean", "vid2": "clean", "vid3": "clean", "vid4": "fast", "vid5": "hard"}
TUNING_TAKES = ("vid1", "vid2")      # the ONLY takes thresholds/penalty may be tuned on
LABELS = ["REST", "REACH", "GRASP", "HOLD", "RELEASE", "RETRACT"]

# ---- New recordings: the hold-out test of the frozen model (never used for training, tuning or design) ----
HOLDOUT_TAKES = {"vid6": "holdout_slow", "vid7": "holdout_fast"}   # vid6: 4 slow cycles, vid7: 3 fast cycles
HOLDOUT_CYCLES = {"vid6": 4, "vid7": 3}
HOLDOUT_DIR = ROOT / "data" / "holdout"
ALL_GROUPS = {**TAKE_GROUPS, **HOLDOUT_TAKES}
