"""Rule-based label assignment on synthetic segment features (explicit thresholds, not tuned ones)."""
import pytest

from src.segment import SegParams, _label

P = SegParams(still_thr=0.35, moving_thr=1.0, p_home=0.25, p_far=0.6, ap_closed=0.9, ap_delta=0.12, dp_min=0.05)


def ft(speed, p, dp=0.0, ap=1.0, dap=0.0):
    return {"mean_speed": speed, "p_mean": p, "dp": dp, "mean_aperture": ap, "d_ap": dap, "frac_missing": 0.0}


@pytest.mark.parametrize("features,prev,expected", [
    (ft(0.1, 0.0), None, "REST"),
    (ft(2.5, 0.5, dp=+0.7), "REST", "REACH"),
    (ft(2.5, 0.5, dp=-0.7), "RELEASE", "RETRACT"),
    (ft(0.5, 0.9, ap=0.8, dap=-0.3), "REACH", "GRASP"),        # fingers closing at the object
    (ft(0.1, 0.9, ap=0.8), "GRASP", "HOLD"),                   # closed and still
    (ft(0.6, 0.9, ap=0.8), "REACH", "GRASP"),                  # closed and slowly moving = lift
    (ft(0.3, 0.9, ap=1.3, dap=+0.4), "HOLD", "RELEASE"),       # fingers opening
    (ft(0.1, 0.9, ap=1.2), "HOLD", "RELEASE"),                 # open and stable after a hold
    (ft(0.1, 0.9, ap=1.2), "REACH", "REACH"),                  # open and stable before grasping
    (ft(0.2, 0.45), "REACH", "REACH"),                         # pause mid-reach keeps REACH
    (ft(0.2, 0.45), "RETRACT", "RETRACT"),                     # pause mid-retract keeps RETRACT
])
def test_label_rules(features, prev, expected):
    assert _label(features, prev, P) == expected


def test_label_ignores_cycle_order_for_unusual_sequences():
    """The expected cycle order is not used: HOLD-like features after REST still give HOLD."""
    assert _label(ft(0.1, 0.9, ap=0.8), "REST", P) == "HOLD"
