"""Scoring boundaries, every partial level, zero gate, two valid solutions, blind words. Run: python -m pytest tests -q (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def quat(yaw=0.0, tilt=0.0):
    """yaw about z (deg) after a tilt about x (deg)."""
    a, b = math.radians(yaw) / 2, math.radians(tilt) / 2
    qz = np.array([0, 0, math.sin(a), math.cos(a)])
    qx = np.array([math.sin(b), 0, 0, math.cos(b)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def world(dx=0.0, dy=0.0, z=T.SEAT_Z + 0.003, yaw=0.0, tilt=0.0):
    z0 = np.zeros(3)
    return {"frame": PartState(pos=np.array([T.POST_X + dx, dy, z]), quat_xyzw=quat(yaw, tilt), lin_vel=z0, ang_vel=z0,
                               low=z0, high=z0)}


def lv(w):
    return T.classify(w)[0]


def test_start_zero():
    w = world(dx=T.X0 - T.POST_X, z=T.START_Z)
    assert lv(w) == 0.0


def test_full_and_valid_variants():
    l, ok, m = T.classify(world())
    assert l == 1.0 and ok, m
    for kw in (dict(dx=0.010), dict(dy=-0.010), dict(dx=-0.008, dy=0.008), dict(yaw=5), dict(tilt=6), dict(z=T.SEAT_Z + 0.001)):
        assert lv(world(**kw)) == 1.0, kw


def test_two_solutions_different_approach_same_end():
    assert lv(world(dx=0.01, dy=0.01, yaw=3)) == lv(world(dx=-0.01, dy=-0.01, yaw=-3)) == 1.0


def test_1_0_boundaries():
    assert lv(world(dx=0.020)) == 0.75              # centre 20 mm off, clearance 30 mm
    assert lv(world(z=T.SEAT_Z + 0.016)) == 0.75
    assert lv(world(tilt=10)) == 0.75
    assert lv(world(dy=0.016)) == 0.75


def test_clearance_gate_of_one():
    w = world(yaw=0.0)
    assert T.fit(w)["clear"] > 0.03                  # nominal 50 mm


def test_075_and_boundaries():
    assert lv(world(dx=0.024, z=T.SEAT_Z + 0.018)) == 0.75
    assert lv(world(dx=0.030, z=T.SEAT_Z + 0.018)) == 0.5
    assert lv(world(z=T.SEAT_Z + 0.025)) == 0.5


def test_05_level():
    assert lv(world(dx=0.035, z=T.POST_TOP - 0.010)) == 0.5
    assert lv(world(z=T.POST_TOP - 0.004)) == 0.25


def test_025_above_post():
    assert lv(world(z=T.POST_TOP + 0.03)) == 0.25
    assert lv(world(dx=0.06, z=T.POST_TOP + 0.03)) == 0.25


def test_beside_post_zero():
    w = world(dx=0.0, dy=0.22, z=T.POST_TOP + 0.03)
    assert lv(w) == 0.0
    assert lv(world(dy=0.30, z=T.SEAT_Z + 0.003)) == 0.0


def test_gates():
    assert lv(world(tilt=13)) == 0.0
    assert lv(world(dx=0.07, z=T.SEAT_Z + 0.003)) == 0.0   # post under a bar / frame fallen aside
    assert lv(world(dx=0.06, z=T.POST_TOP - 0.01)) == 0.0   # bar at the post
    assert lv(world(dx=0.03, dy=0.0, yaw=30, z=T.POST_TOP - 0.01)) in (0.0,) or T.fit(world(dx=0.03, yaw=30))["clear"] >= 0   # yaw changes the post position in the frame axes


def test_hold_counter():
    T._reset_state()
    for i in range(T.HOLD_STEPS + 2):
        T._STATE["step"] += 1
        s = T.success(world())
    assert s
    T._reset_state()
    for i in range(T.HOLD_STEPS - 3):
        T._STATE["step"] += 1
        s = T.success(world())
    assert not s


def test_blind_words():
    here = pathlib.Path(T.__file__).parent
    txt = T.__doc__ + "".join(p.read_text() for p in (here / "task.py", here / "README.md") if p.exists())
    assert not re.search(r"(?i)robocurve|/home/|team|\.brev", txt.replace("Dream-1", "").replace("--policy dream", "").replace("dream_run", ""))
