"""Scoring boundaries, every partial level, zero gate, two valid solutions, blind words. Run: python -m pytest tests -q (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402

LZ = T.TOP + 0.0605


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def qx(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def st(pos, quat):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=quat, lin_vel=z, ang_vel=z, low=z, high=z)


def world(gap=0.015, mid=0.0, step=0.0, dyaw=0.0, dx=0.0, zb=LZ, tilt=0.0):
    """Plates with the inner edges `gap` apart centred on y=mid; blue is `step` higher, orange yawed by -dyaw."""
    h = T.PL_Y / 2
    yb, yo = mid + gap / 2 + h, mid - gap / 2 - h
    qb = qx(tilt) if tilt else qz(0)
    return {"blue": st((T.X_C + dx, yb, zb + step / 2), qb), "orange": st((T.X_C, yo, zb - step / 2), qz(-dyaw))}


def lv(w):
    return T.classify(w)[0]


def test_start_zero():
    w = {"blue": st((T.X_C, 0.125, T.REST_Z), qz(0)), "orange": st((T.X_C, -0.125, T.REST_Z), qz(0))}
    assert lv(w) == 0.0


def test_full():
    l, ok, m = T.classify(world())
    assert l == 1.0 and ok, m


def test_two_valid_gaps():
    for g in (0.010, 0.020):
        assert lv(world(gap=g)) == 1.0


def test_one_plate_down_zero():
    w = world()
    w["orange"] = st((T.X_C, -0.06, T.REST_Z), qz(0))
    assert lv(w) == 0.0


def test_gate_overlap_and_tilt_and_step():
    assert lv(world(gap=-0.010)) == 0.0
    assert lv(world(gap=-0.005)) == 0.5 or lv(world(gap=-0.005)) <= 0.5
    assert lv(world(tilt=13)) == 0.0
    assert lv(world(step=0.05)) == 0.0


def test_075_level_and_boundaries():
    assert lv(world(gap=0.030)) == 0.75
    assert lv(world(gap=0.005)) == 0.75
    assert lv(world(step=0.018)) == 0.75
    assert lv(world(dyaw=10)) == 0.75
    assert lv(world(mid=0.02)) == 0.75              # midpoint off the datum: only 0.75


def test_05_level():
    assert lv(world(gap=0.050)) == 0.5
    assert lv(world(dyaw=15)) == 0.5
    assert lv(world(step=0.03, zb=LZ + 0.02)) == 0.5


def test_025_level():
    assert lv(world(gap=0.12)) == 0.25
    assert lv(world(dyaw=25)) == 0.25


def test_1_0_tolerances():
    assert lv(world(step=0.010)) == 1.0
    assert lv(world(step=0.015)) == 0.75
    assert lv(world(dyaw=7)) == 1.0
    assert lv(world(dyaw=9)) == 0.75
    assert lv(world(mid=0.010)) == 1.0
    assert lv(world(gap=0.026)) == 0.75
    assert lv(world(gap=0.007)) == 0.75
    assert lv(world(dx=0.03)) == 0.75


def test_hold_counter():
    T._reset_state()
    for i in range(T.HOLD_STEPS + 2):
        T._STATE["step"] += 1
        s = T.success(world())
    assert s


def test_blind_words():
    txt = (pathlib.Path(T.__file__).read_text() + (pathlib.Path(T.__file__).parents[0] / "README.md").read_text()
           if (pathlib.Path(T.__file__).parents[0] / "README.md").exists() else pathlib.Path(T.__file__).read_text())
    assert not re.search(r"(?i)robocurve|dream|/home/|team|\.brev", txt.replace("Dream-1", "").replace("--policy dream", "").replace("dream_run", ""))
