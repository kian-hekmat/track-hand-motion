"""Signal derivation on synthetic keypoints with known motion (no real data)."""
import numpy as np
import pandas as pd
import pytest

from config import FRAME_H, FRAME_W, GRID_HZ
from src.signals import derive_signals

N = 300  # 10 s at 30 fps


def make_frames(*, vx_px=0.0, vy_px=0.0, hand_px=150.0, aperture=0.5, rot=0.0, t=None,
                world_wrist_noise=0.0, seed=0):
    """Wrist moves at constant pixel velocity; hand of `hand_px` pixels (wrist->middle MCP)."""
    t = np.arange(N) / 30.0 if t is None else t
    rng = np.random.default_rng(seed)
    X, Y = 400 + vx_px * t, 500 + vy_px * t
    df = pd.DataFrame({"t": t, "detected": True})
    df["i0_x"], df["i0_y"] = X / FRAME_W, Y / FRAME_H
    df["i9_x"], df["i9_y"] = X / FRAME_W, (Y - hand_px) / FRAME_H
    for k in (4, 8):
        df[f"i{k}_x"], df[f"i{k}_y"] = df["i0_x"], df["i0_y"]
    # world coords, metres: hand length 0.09; thumb-index separation = aperture * 0.09
    sep = aperture * 0.09
    ang = rot
    d = np.array([np.cos(ang), np.sin(ang), 0.0]) * sep / 2
    wr = world_wrist_noise * rng.standard_normal((len(t), 3))
    pts = {0: np.zeros(3), 9: np.array([0, 0.09, 0]), 4: d, 8: -d}
    for k, v in pts.items():
        for j, c in enumerate("xyz"):
            df[f"w{k}_{c}"] = v[j] + (wr[:, j] if k == 0 else 0.0)
    return df


def interior(s, a=1.0, b=9.0):
    return s[(s.t > a) & (s.t < b)]


def test_constant_speed_in_hand_lengths_per_second():
    s = derive_signals(make_frames(vx_px=600.0, hand_px=150.0))
    assert interior(s)["speed"].mean() == pytest.approx(600 / 150, rel=0.02)


def test_speed_is_scale_invariant_when_hand_is_twice_as_big():
    a = derive_signals(make_frames(vx_px=600.0, hand_px=150.0))
    b = derive_signals(make_frames(vx_px=1200.0, hand_px=300.0))
    assert interior(a)["speed"].mean() == pytest.approx(interior(b)["speed"].mean(), rel=0.02)


def test_aspect_ratio_handled_same_pixel_speed_in_x_and_y():
    a = derive_signals(make_frames(vx_px=300.0))
    b = derive_signals(make_frames(vy_px=300.0))
    assert interior(a)["speed"].mean() == pytest.approx(interior(b)["speed"].mean(), rel=0.02)


def test_irregular_timestamps_use_real_dt():
    rng = np.random.default_rng(1)
    t = np.arange(N) / 30.0 + rng.uniform(-0.006, 0.006, N)
    t.sort()
    s = derive_signals(make_frames(vx_px=600.0, t=t))
    assert interior(s)["speed"].mean() == pytest.approx(4.0, rel=0.03)


def test_grid_is_uniform_and_covers_take():
    s = derive_signals(make_frames(vx_px=100.0))
    assert np.allclose(np.diff(s["t"]), 1 / GRID_HZ)
    assert s["t"].iloc[0] == pytest.approx(0.0) and s["t"].iloc[-1] == pytest.approx((N - 1) / 30.0, abs=1 / 30)


def test_aperture_known_value_and_rotation_invariant():
    a = derive_signals(make_frames(aperture=0.5, rot=0.0))
    b = derive_signals(make_frames(aperture=0.5, rot=1.2))
    assert interior(a)["aperture"].mean() == pytest.approx(0.5, rel=0.01)
    assert interior(b)["aperture"].mean() == pytest.approx(0.5, rel=0.01)


def test_speed_comes_from_image_not_world_wrist():
    """World coords are hand-centred: a wrist that is still in the image has ~0 speed even if its
    world coordinates jump around."""
    s = derive_signals(make_frames(vx_px=0.0, world_wrist_noise=0.02))
    assert interior(s)["speed"].max() < 0.05


def test_short_gap_interpolated_long_gap_left_nan_and_input_untouched():
    f = make_frames(vx_px=300.0)
    f.loc[50:51, "detected"] = False      # 2 frames -> interpolated
    f.loc[100:114, "detected"] = False    # 15 frames -> real gap
    before = f.copy()
    s = derive_signals(f)
    pd.testing.assert_frame_equal(f, before)  # raw frames not modified
    short = s[(s.t >= 50 / 30) & (s.t <= 51 / 30)]
    assert not short["missing"].any() and short["interpolated"].any() and short["speed"].notna().all()
    long = s[(s.t > 101 / 30) & (s.t < 113 / 30)]
    assert long["missing"].all() and long["speed"].isna().all() and long["aperture"].isna().all()
    # and the interpolated stretch still has the right speed
    assert short["speed"].mean() == pytest.approx(2.0, rel=0.1)


def test_three_frame_gap_is_not_interpolated():
    f = make_frames(vx_px=300.0)
    f.loc[60:62, "detected"] = False
    s = derive_signals(f)
    assert s[(s.t > 60.5 / 30) & (s.t < 61.5 / 30)]["missing"].all()


def test_progress_axis_found_from_data_and_signed():
    """Wrist goes home -> far -> home: p rises to ~1 then returns; dp changes sign."""
    t = np.arange(N) / 30.0
    x = 400 + 800 * np.clip(np.sin(np.pi * t / 10) ** 2, 0, 1)  # out and back
    f = make_frames(t=t)
    f["i0_x"] = x / FRAME_W
    s = derive_signals(f)
    assert interior(s, 4.5, 5.5)["p"].min() > 0.8
    assert s["p"].iloc[2] < 0.1 and s["dp"][(s.t > 2) & (s.t < 3)].mean() > 0
    assert s["dp"][(s.t > 6) & (s.t < 8)].mean() < 0
