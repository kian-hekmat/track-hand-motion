"""Run extraction on one or more takes. Usage: extract_take.py vid1 [vid2 ...] [--overlay]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.extract import extract_take

args = [a for a in sys.argv[1:] if not a.startswith("--")]
overlay = "--overlay" in sys.argv
for take in args:
    _, meta = extract_take(take, render_overlay=overlay)
    print(json.dumps(meta, indent=2))
