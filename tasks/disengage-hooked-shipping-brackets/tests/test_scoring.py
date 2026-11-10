"""Scoring boundaries, every level reachable, zero gate, two valid solutions, geometry (straight pull blocked, rotate-then-slide path
exists with clearance), blind words. Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import geom  # noqa: E402
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402

LZ = T.TOP + 0.04


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def qx(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def st(pos, quat):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=quat, lin_vel=z, ang_vel=z, low=z, high=z)


def world(rx=0.0, ry=-0.28, turn=0.0, ax=None, ay=None, za=LZ, zb=LZ, tilt_b=0.0, tilt_a=0.0, yaw_a=0.0):
    ax = T.A_X if ax is None else ax
    ay = T.A_Y if ay is None else ay
    a = st((ax, ay, za), qx(tilt_a) if tilt_a else qz(yaw_a))
    b = st((T.A_X + T.B_REL_OUT[0] + (rx - T.B_REL_OUT[0]) + (ax - T.A_X), ay + ry, zb), qx(tilt_b) if tilt_b else qz(turn))
    return {"blue": a, "orange": b}


def lv(w, peak=80.0):
    return T.classify(w, peak)[0]


def test_start_zero():
    w = {"blue": st((T.A_X, T.A_Y, T.A_Z), qz(0)), "orange": st((T.A_X, T.A_Y, T.B_Z), qz(0))}
    assert lv(w, 0.0) == 0.0 and lv(w, 80.0) == 0.0


def test_full_and_two_valid_solutions():
    for turn in (0.0, 80.0, 40.0, -30.0):      # leave the bracket turned or turn it back: both finish
        assert lv(world(turn=turn), peak=max(abs(turn), 20.0)) == 1.0, turn
    assert lv(world(rx=0.02, ry=-0.26)) == 1.0


def test_lifted_gate():
    assert lv(world(za=T.TOP + 0.015)) == 0.0
    assert lv(world(zb=T.TOP + 0.015)) == 0.0
    assert lv(world(zb=T.TOP + 0.021)) == 1.0
    assert lv(world(tilt_b=21)) == 0.0 and lv(world(tilt_a=21)) == 0.0
    assert lv(world(ax=0.6)) == 0.0


def test_075_level():
    assert lv(world(tilt_b=10)) == 0.75
    assert lv(world(tilt_b=7)) == 1.0
    assert lv(world(ry=-0.19)) == 0.75          # head out but only 0.19 apart
    assert lv(world(rx=0.04, ry=-0.28)) == 0.75    # head 40 mm off its outline
    assert lv(world(yaw_a=10)) == 0.75
    assert lv(world(ax=T.A_X + 0.04)) == 0.75   # housing 40 mm off its outline


def test_05_level():
    assert lv(world(ry=-0.13, turn=70)) == 0.5   # head 0.13 away but still inside the front face line
    assert lv(world(ry=-0.11, turn=60), peak=60) == 0.25
    assert lv(world(ry=-0.13, turn=70), peak=70) == 0.5
    assert lv(world(ry=-0.13, turn=70), peak=29) == 0.25
    assert lv(world(ry=-0.19), peak=10) == 0.0    # unhooked pose without a recorded turn of 15 deg: not scored


def test_025_level():
    assert lv(world(ry=0.0), peak=16) == 0.25
    assert lv(world(ry=0.0), peak=14) == 0.0


def test_hooked_start_not_scored():
    # lifted together without turning: stays at 0
    assert lv(world(rx=0.0, ry=0.0, turn=0.0), peak=0.0) == 0.0


def test_peak_counter_and_hold():
    T._reset_state()
    w = world(turn=80.0)
    T._STATE["step"] += 1
    T.score(world(ry=-0.1, turn=60.0))
    assert T._STATE["peak"] >= 59.9
    for _ in range(T.HOLD_STEPS + 2):
        T._STATE["step"] += 1
        s = T.success(world(turn=0.0))
    assert s and T._STATE["peak"] >= 59.9
    T._reset_state()
    for _ in range(T.HOLD_STEPS + 2):       # never turned: unhooked pose without a turn is unreachable, scores 0.0 on peak
        T._STATE["step"] += 1
        s = T.success(world(turn=0.0))
    assert not s


def test_geom_start_free_and_pull_blocked():
    assert not geom.collides((0.0, 0.0, 0.0), 0.0)
    for y in (-0.075, -0.085, -0.095):
        assert geom.collides((0.0, y, 0.0), 0.0), y          # straight pull hits the lips
    assert not geom.collides((0.0, -0.30, 0.0), 0.0)         # free once out


def test_geom_rotate_then_slide_path_clearance():
    p, why = geom.plan(margin=0.010)
    assert p is not None, why
    assert not geom.collides((p[:, 0], p[:, 1], p[:, 2]), 0.010).any()
    assert math.degrees(abs(p[:, 2]).max()) >= 50
    assert geom.head_outside((p[-1, 0], p[-1, 1], p[-1, 2]))
    # the head cannot pass the slot unturned: is blocked at 0 and 10 deg turn (checked at the lips)
    assert all(geom.collides((x, -0.085, math.radians(t)), 0.0) for x in (-0.02, 0.0, 0.02) for t in (0, 10))


def test_blind_words():
    here = pathlib.Path(T.__file__).parent
    txt = "".join(p.read_text() for p in (here / "task.py", here / "geom.py", here / "demo.py") if p.exists())
    if (here / "README.md").exists():
        txt += (here / "README.md").read_text()
    clean = txt.replace("Dream-1", "").replace("--policy dream", "").replace("dream_run", "").replace("dream", "")
    assert not re.search(r"(?i)robocurve|/home/|team|\.brev", clean)
