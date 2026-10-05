"""Run segmentation with the frozen params on takes (default: all) and write
data/segments/<take>_events.csv and <take>_signals.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEGMENTS_DIR, TAKE_GROUPS
from src.segment import load_params, segment_take

if __name__ == "__main__":
    P = load_params()
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for take in sys.argv[1:] or sorted(TAKE_GROUPS):
        ev, sig = segment_take(take, P)
        ev.to_csv(SEGMENTS_DIR / f"{take}_events.csv", index=False)
        sig.to_csv(SEGMENTS_DIR / f"{take}_signals.csv", index=False)
        print(f"{take}: {len(ev)} events")
