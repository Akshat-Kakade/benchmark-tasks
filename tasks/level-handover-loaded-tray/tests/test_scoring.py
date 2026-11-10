"""Scoring boundaries, every partial level, zero gate, two solutions, blind words. Run: python -m pytest tests -q (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def qx(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def st(pos, quat):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=quat, lin_vel=z, ang_vel=z, low=z, high=z)


def world(y=T.PAD_L[1], x=T.PAD_L[0], lift=0.0, tilt=0.0, yaw=0.0, cup_dy=0.0, cup_tilt=(0, 0, 0), cup_off=(False, False, False)):
    qr = qz(yaw) if not tilt else qx(tilt)
    o = {"pad_left": st((0, 0, 0), qz(0)), "pad_right": st((0, 0, 0), qz(0)),
         "rack": st((x, y, T.RACK_Z + lift), qr)}
    rack = o["rack"]
    for i, (n, cy) in enumerate(zip(T.CUP_NAMES, T.CUP_Y)):
        lp = (0.0, cy + cup_dy, 0.012 + 0.0005)
        if cup_off[i]:
            lp = (0.0, cy, -0.05)
        o[n] = st(rack.to_world(lp), qx(cup_tilt[i]))
    return o


def run(seq):
    memo = {"step": 0, "scored": None, "last": (0.0, False), "lifted": False, "crossed": False, "far": False, "max_tilt": 0.0,
            "gate": False, "held": 0}
    out = None
    for w in seq:
        out = T.classify(w, memo)
    return out, memo


def path_to_far(tilt=0.0):
    seq = [world(lift=0.0), world(lift=0.07, y=0.4, tilt=0), world(lift=0.07, y=0.10, tilt=tilt),
           world(lift=0.07, y=-0.10, tilt=tilt), world(lift=0.07, y=-0.35)]
    return seq


def test_start_zero():
    (lv, ok, _), _ = run([world()])
    assert lv == 0.0 and not ok


def test_025_and_not_before_cross():
    (lv, _, _), _ = run([world(lift=0.07, y=0.30)])
    assert lv == 0.0
    (lv, _, _), _ = run([world(lift=0.07, y=0.10)])
    assert lv == 0.25
    (lv, _, _), _ = run([world(lift=0.03, y=0.10)])
    assert lv == 0.0                               # not lifted 40 mm


def test_05_handed_over_level():
    (lv, _, _), _ = run(path_to_far(tilt=5.0))
    assert lv == 0.5
    (lv, _, _), _ = run(path_to_far(tilt=12.0))    # tilted > 8 while carried: stays at 0.25
    assert lv == 0.25


def test_placed_after_unlevel_carry_stays_025():
    seq = path_to_far(tilt=12.0) + [world(y=T.PAD_R[1], lift=0.0)]
    (lv, ok, _), _ = run(seq)
    assert lv == 0.25 and not ok


def test_075_placed_but_cups_shifted():
    seq = path_to_far() + [world(y=T.PAD_R[1] + 0.005, lift=0.0, cup_dy=0.03)]
    (lv, ok, _), _ = run(seq)
    assert lv == 0.75 and not ok


def test_075_yaw_boundary():
    for yaw, want in ((6.0, 1.0), (10.0, 0.5)):
        seq = path_to_far() + [world(y=T.PAD_R[1], lift=0.0, yaw=yaw)]
        (lv, _, _), _ = run(seq)
        assert lv == want, (yaw, lv)


def test_1_0_and_tolerances():
    seq = path_to_far() + [world(y=T.PAD_R[1] + 0.012, lift=0.0)]
    (lv, ok, _), _ = run(seq)
    assert lv == 1.0 and ok
    seq = path_to_far() + [world(y=T.PAD_R[1] + 0.02, lift=0.0)]
    (lv, ok, _), _ = run(seq)
    assert lv == 0.5 and not ok                    # 20 mm off: carried but not placed


def test_gate_cup_fallen_or_off():
    seq = path_to_far() + [world(y=T.PAD_R[1], lift=0.0, cup_tilt=(0, 90, 0))]
    (lv, _, _), _ = run(seq)
    assert lv == 0.0
    seq = path_to_far() + [world(y=T.PAD_R[1], lift=0.0, cup_off=(False, True, False))]
    (lv, _, _), _ = run(seq)
    assert lv == 0.0


def test_gate_tilt_20_during_carry_latches():
    seq = [world(lift=0.07, y=0.4), world(lift=0.07, y=0.1, tilt=22)] + path_to_far()[2:] + [world(y=T.PAD_R[1])]
    (lv, ok, _), _ = run(seq)
    assert lv == 0.0 and not ok


def test_gate_dropped_off_pad():
    seq = [world(lift=0.07, y=0.4), world(lift=0.07, y=0.10), world(lift=0.0, y=0.0)]
    (lv, _, _), _ = run(seq)
    assert lv == 0.0


def test_two_solutions():
    for mid in (0.0, 0.06):                        # handover at y = 0 or y = 0.06: same path through the far zone
        seq = [world(lift=0.0), world(lift=0.07, y=0.2), world(lift=0.07, y=mid), world(lift=0.07, y=-0.35),
               world(y=T.PAD_R[1] - 0.004, x=T.PAD_R[0] + 0.006, lift=0.0)]
        (lv, ok, _), _ = run(seq)
        assert lv == 1.0 and ok
