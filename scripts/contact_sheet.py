"""Tile overlay-video frames into a contact sheet for visual review.
Usage: contact_sheet.py <take> <out.jpg> [frame_idx ...]  (default: 24 evenly spaced frames)"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EVIDENCE_DIR

take, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(str(EVIDENCE_DIR / f"{take}_overlay.mp4"))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = [int(a) for a in sys.argv[3:]] or list(np.linspace(0, n - 1, 24).astype(int))
tiles = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    if ok:
        tiles.append(cv2.resize(f, (480, 270)))
cols = 4
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
cv2.imwrite(out, grid)
print(f"{len(idxs)} frames -> {out}")
