"""Leave-one-take-out cross-validation of the v2 segmenter (frame classifier + PELT / grammar).

For each held-out take H:
  * the 4 other takes (plus their time-scaled copies) are the training set; H is never used for
    training or for any choice;
  * hyperparameters (classifier family, PELT penalty, grammar switch penalty) are selected by an INNER
    leave-one-take-out loop over those 4 takes, objective = 0.5*boundary F1 + 0.5*frame accuracy;
  * the chosen config is trained on all 4 and applied to H.
Events for each take therefore always come from a model that never saw that take.

Outputs: data/segments/v2_pelt|v2_grammar|v2_argmax/<take>_events.csv (+ signals), cv_selection.json.
Score them with scripts/score_segmentation.py --events-dir ... --version ...
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BOUNDARY_TOLERANCE_S, SEGMENTS_DIR, TAKE_GROUPS
from src import ground_truth as G
from src import score as S
from src.features import take_data
from src.learned import decode_argmax, decode_grammar, decode_pelt, fit, predict_proba
from src.signals import load_frames

TAKES = sorted(TAKE_GROUPS)
KINDS = ("logreg", "hgb")
PELT_PENS = (0.5, 1.0, 2.0, 4.0, 8.0)
GRAMMAR_SWITCH = (2.0, 4.0, 8.0)
CACHE = {}


def td(take):
    if take not in CACHE:
        sig, feat, _, t_end = take_data(take)
        CACHE[take] = (sig, feat, t_end, load_frames(take)["t"].to_numpy())
    return CACHE[take]


def objective(take, ev):
    gt = G.load_ground_truth(take)
    bm = S.boundary_metrics(S.events_to_boundaries(ev), G.boundaries(gt), BOUNDARY_TOLERANCE_S)
    p, r = bm["precision"], bm["recall"]
    f1 = 0.0 if (not np.isfinite(p)) or p + r == 0 else 2 * p * r / (p + r)
    ft = td(take)[3]
    fm = S.frame_metrics(G.labels_at(gt, ft), G.labels_at(ev, ft))
    return 0.5 * f1 + 0.5 * fm["frame_accuracy"]


def main():
    out = {k: SEGMENTS_DIR / k for k in ("v2_pelt", "v2_grammar", "v2_argmax")}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    selection = {}
    for held in TAKES:
        train = [t for t in TAKES if t != held]
        # ---- inner LOTO over the 4 training takes ----
        inner = {}  # (kind) -> {inner_take: P}
        for kind in KINDS:
            inner[kind] = {}
            for t_in in train:
                clf = fit([t for t in train if t != t_in], kind)
                assert t_in not in clf.train_takes_ and held not in clf.train_takes_
                inner[kind][t_in] = predict_proba(clf, td(t_in)[1])
        def inner_score(kind, decoder):
            vals = []
            for t_in in train:
                sig, _, t_end, _ = td(t_in)
                vals.append(objective(t_in, decoder(inner[kind][t_in], sig["t"].to_numpy(), t_end, t_in)))
            return float(np.mean(vals))
        grid_pelt = {(k, p): inner_score(k, lambda P, t, te, tk, p=p: decode_pelt(P, t, te, tk, pen=p))
                     for k in KINDS for p in PELT_PENS}
        grid_gram = {(k, s): inner_score(k, lambda P, t, te, tk, s=s: decode_grammar(P, t, te, tk, switch_penalty=s))
                     for k in KINDS for s in GRAMMAR_SWITCH}
        (kp, pen), sp = max(grid_pelt.items(), key=lambda x: x[1])
        (kg, sw), sg = max(grid_gram.items(), key=lambda x: x[1])
        # ---- final models on all 4 training takes, applied to the held-out take ----
        sig, feat, t_end, _ = td(held)
        t = sig["t"].to_numpy()
        models = {}
        for kind in {kp, kg}:
            models[kind] = fit(train, kind)
            assert held not in models[kind].train_takes_
        P_p, P_g = predict_proba(models[kp], feat), predict_proba(models[kg], feat)
        decode_pelt(P_p, t, t_end, held, pen=pen).to_csv(out["v2_pelt"] / f"{held}_events.csv", index=False)
        decode_grammar(P_g, t, t_end, held, switch_penalty=sw).to_csv(out["v2_grammar"] / f"{held}_events.csv", index=False)
        decode_argmax(P_p, t, t_end, held).to_csv(out["v2_argmax"] / f"{held}_events.csv", index=False)
        for d in out.values():
            sig.to_csv(d / f"{held}_signals.csv", index=False)
        selection[held] = {"train_takes": train, "pelt": {"classifier": kp, "pen": pen, "inner_objective": sp},
                           "grammar": {"classifier": kg, "switch_penalty": sw, "inner_objective": sg},
                           "inner_grid_pelt": {f"{k}|{p}": v for (k, p), v in grid_pelt.items()},
                           "inner_grid_grammar": {f"{k}|{s}": v for (k, s), v in grid_gram.items()}}
        print(f"held-out {held}: pelt -> {kp}, pen={pen} (inner obj {sp:.3f}); grammar -> {kg}, switch={sw} (inner obj {sg:.3f})", flush=True)
    (SEGMENTS_DIR / "cv_selection.json").write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
