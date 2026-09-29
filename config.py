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
