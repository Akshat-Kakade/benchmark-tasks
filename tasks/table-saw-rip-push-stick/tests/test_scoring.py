"""Scoring boundaries, every partial level, zero gate, kerf rules (monotonic cut front, lift-over, shuffle), two valid solutions, blind words.
Run on the VM: python -m pytest tests -q"""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def st(pos, sx, sy, sz, quat=(0, 0, 0, 1), v=0.0):
    z = np.zeros(3)
    pos = np.array(pos, float)
    half = np.array([sx, sy, sz]) / 2
    return PartState(pos=pos, quat_xyzw=np.array(quat, float), lin_vel=np.array([v, 0, 0.0]), ang_vel=z, low=pos - half, high=pos + half)


def quat_z(deg):
    return (0, 0, math.sin(math.radians(deg) / 2), math.cos(math.radians(deg) / 2))


def objs(front=None, rear=None, dz=0.0, tilt=0.0, v=0.0, dy=0.0, yaw=0.0, dy_half=0.0):
    """Plank (intact) and halves with the given front edge (or rear edge) x."""
    if rear is None:
        rear = front - T.PLANK_B
    cx = rear + T.PLANK_B / 2
    qx = (math.sin(math.radians(tilt) / 2), 0, 0, math.cos(math.radians(tilt) / 2))
    q = quat_z(yaw) if yaw else qx
    b = st((cx, dy, T.PLANK_CZ + dz), T.PLANK_B, T.PLANK_L, T.PLANK_T, q, v)
    hl = st((cx, dy + T.HALF_DY + dy_half, T.PLANK_CZ + dz), T.PLANK_B, T.HALF_L, T.PLANK_T, q, v)
    hr = st((cx, dy - T.HALF_DY - dy_half, T.PLANK_CZ + dz), T.PLANK_B, T.HALF_L, T.PLANK_T, q, v)
    stick = st((0.1, 0.15, T.TOP + 0.05), 0.3, 0.03, 0.05)
    return {"board": b, "half_l": hl, "half_r": hr, "stick_l": stick, "stick_r": st((0.1, -0.15, T.TOP + 0.05), 0.3, 0.03, 0.05),
            "saw": st((0.3, 0.0, T.TOP), 0.8, 0.5, 0.03)}


def grip(l=(0.10, 0.20), r=(0.10, -0.20), z=0.10):
    def g(p):
        p = np.array([p[0], p[1], T.SAW_TOP + z])
        return dict(tcp=p, pts=[p, p + [0, 0, 0.13], p + [0.02, 0, 0], p - [0.02, 0, 0]], gap=0.05)
    return {"L": g(l), "R": g(r)}


def fresh(**kw):
    return T._fresh() | {"step": 10} | kw


CUT = dict(cut=True, kerf=T.S_FULL)
DONE = T.CUT_DONE + 0.001                     # rear x when complete


def lv(o, gr, memo=None):
    memo = memo if memo is not None else fresh(**CUT)
    return T.classify(o, gr, memo)[:2]


def test_levels():
    assert lv(objs(rear=T.PLANK_REAR0), grip(), fresh())[0] == 0.0
    f = T.BLADE_X0 + 0.1
    assert lv(objs(front=f), grip(), fresh(kerf=T.KERF_MIN - 0.001))[0] == 0.0
    assert lv(objs(front=f), grip(), fresh(kerf=T.KERF_MIN))[0] == 0.25
    assert lv(objs(rear=DONE - 0.003), grip())[0] == 0.25            # separated but not cleared
    assert lv(objs(rear=DONE), grip())[0] == 1.0
    assert lv(objs(rear=T.BLADE_X1 + 0.05, tilt=12.0), grip())[0] == 0.5      # tipped but clear of the blade
    assert lv(objs(rear=DONE, v=0.05), grip())[0] == 0.75


def test_no_kerf_no_credit():
    assert lv(objs(rear=DONE), grip(), fresh())[0] == 0.0
    assert lv(objs(rear=DONE), grip(), fresh(kerf=0.02))[0] == 0.0


def test_hand_piece_clearance():
    g = grip()
    p = np.array([DONE + 0.07, 0.15, T.SAW_TOP + 0.01])
    g["R"] = dict(tcp=p, pts=[p], gap=0.05)
    r = lv(objs(rear=DONE), g)
    assert r[0] <= 0.75 and not r[1] or r == (1.0, True)
    p2 = np.array([DONE + 0.30, 0.15, T.SAW_TOP + 0.01])
    g["R"] = dict(tcp=p2, pts=[p2], gap=0.05)
    assert lv(objs(rear=DONE), g) == (1.0, True)


def test_zero_gate():
    near = grip(l=(T.BLADE_X0 + 0.05, 0.05), z=0.05)                # inside 100 mm of the blade, low
    memo = fresh(**CUT)
    assert T.classify(objs(rear=DONE), near, memo)[0] == 0.0 and memo["gate"]
    high = grip(l=(T.BLADE_X0 + 0.05, 0.05), z=0.20)                # above 150 mm: no event
    memo = fresh(**CUT)
    assert T.classify(objs(rear=DONE), high, memo)[0] == 1.0 and not memo["gate"]
    memo = fresh(**CUT)
    assert T.classify(objs(rear=DONE, dz=-0.05), grip(), memo)[0] == 0.0 and memo["gate"]    # piece fell
    memo = fresh(**CUT)
    T.classify(objs(rear=DONE), near, memo)
    assert T.classify(objs(rear=DONE), grip(), memo)[0] == 0.0     # latched


def test_lift_over_scores_zero():
    memo = fresh()
    T.classify(objs(front=T.BLADE_X0 + 0.02, dz=0.012), grip(), memo)
    assert memo["lifted"]
    assert T.classify(objs(rear=DONE), grip(), memo)[0] == 0.0
    assert T.classify(objs(rear=DONE), grip(), fresh(**CUT, lifted=True))[0] == 0.0
    # the limit is one constant: just above it counts, just below does not
    m2 = fresh()
    T.classify(objs(front=T.BLADE_X0 + 0.02, dz=T.LIFT_OVER + 0.001), grip(), m2)
    assert m2["lifted"]
    m3 = fresh()
    T.classify(objs(front=T.BLADE_X0 + 0.02, dz=T.LIFT_OVER - 0.001), grip(), m3)
    assert not m3["lifted"]


def test_tipped_plank_counts_as_lifted():
    memo = fresh()
    T.accrue_kerf(objs(front=T.BLADE_X0 + 0.01), memo)
    memo["step"] = 11
    T.accrue_kerf(objs(front=T.BLADE_X0 + 0.03, tilt=12.0), memo)
    assert memo["lifted"] and memo["kerf"] == 0.0


def feed(fronts, memo=None, **kw):
    memo = memo or fresh()
    for i, f in enumerate(fronts):
        memo["step"] = 11 + i
        T.accrue_kerf(objs(front=f, **kw), memo)
    return memo


def test_kerf_is_forward_travel_when_engaged():
    fs = [T.BLADE_X0 - 0.005 + 0.002 * i for i in range(40)]       # 2 mm per step, front goes from 5 mm before the blade to 73 mm past
    assert abs(feed(fs)["kerf"] - (fs[-1] - T.BLADE_X0)) < 1e-9


def test_kerf_monotonic_back_and_forth_adds_nothing():
    base = [T.BLADE_X0 + 0.002 * i for i in range(11)]              # to 20 mm
    m = feed(base)
    assert abs(m["kerf"] - 0.020) < 1e-9
    shuffle = []
    for _ in range(50):                                             # 50 strokes over the same 20 mm
        shuffle += [T.BLADE_X0 + 0.020 - 0.002 * i for i in range(11)] + [T.BLADE_X0 + 0.002 * i for i in range(11)]
    m = feed(shuffle, memo=m)
    assert abs(m["kerf"] - 0.020) < 1e-9                            # summed travel would be about 2 m
    assert lv(objs(front=T.BLADE_X0 + 0.02), grip(), m)[0] == 0.0   # < KERF_MIN: no credit


def test_kerf_ignores_travel_made_when_not_engaged():
    memo = fresh()
    T.accrue_kerf(objs(front=T.BLADE_X0 + 0.005), memo)
    memo["step"] = 11
    T.accrue_kerf(objs(front=T.BLADE_X0 + 0.015, dy=0.02), memo)    # 20 mm off the blade plane: not engaged
    assert memo["kerf"] == 0.0
    memo["step"] = 12
    T.accrue_kerf(objs(front=T.BLADE_X0 + 0.025), memo)              # engaged again: only this step's travel counts
    assert abs(memo["kerf"] - 0.010) < 1e-9
    m2 = feed([T.BLADE_X0 + 0.005, T.BLADE_X0 + 0.015], yaw=8.0)      # turned 8 degrees: not engaged
    assert m2["kerf"] == 0.0


def test_two_valid_solutions():
    a = objs(rear=DONE)
    b = objs(rear=DONE + 0.03, dy_half=0.02)                         # halves ended 20 mm apart and 30 mm further on
    assert lv(a, grip()) == (1.0, True) and lv(b, grip()) == (1.0, True)


def test_cut_offset_and_wander_bands():
    memo = fresh(**CUT, wmax=0.003)
    assert lv(objs(rear=DONE), grip(), memo) == (1.0, True)
    memo = fresh(**CUT, wmax=0.005)
    assert lv(objs(rear=DONE), grip(), memo)[0] == 0.75
    memo = fresh(**CUT, wmax=0.007)
    assert lv(objs(rear=DONE), grip(), memo)[0] == 0.5
    o = objs(front=T.BLADE_X0 + 0.05, dy=0.004)
    assert abs(T.cut_offset(o["board"]) + 0.004) < 1e-9


def test_feed_resistance_derivation():
    assert abs(T.FEED_N - 13.0 * 0.019 / 0.030) < 1e-9 and T.FEED_N < 14.0     # both arms together; each stalls near 7 N


def test_geometry_and_guard():
    assert abs(2 * T.HALF_L + T.KERF - T.PLANK_L) < 1e-9
    assert T.BLADE_T < T.KNIFE_T < T.KERF and T.KNIFE_X0 > T.BLADE_X1 and T.KNIFE_H < T.BLADE_H
    # the hood covers the blade (roof above its top, walls beside it) and the plank passes under the walls with clearance
    assert T.HOOD_X0 < T.BLADE_X0 and T.HOOD_X1 > T.BLADE_X1 and T.HOOD_Z1 - T.HOOD_ROOF > T.BLADE_H and T.HOOD_Z0 > T.PLANK_T + 0.003
    assert T.HOOD_HW > T.BLADE_T
    # the push lines are outside the hood and 100 mm+ from the blade plane
    assert T.STICK_Y - T.HEEL_W / 2 > T.HOOD_HW + 0.01 and T.STICK_Y > T.ZONE_R
    assert T.COL_Y0 > T.HALF_L + T.KERF / 2 + 0.005                  # the column clears the half's end
    assert T.PLANK_REAR0 + T.PLANK_B < T.BLADE_X0
    assert T.SAW_X0 <= T.STICK_FACE_REST - T.BAR_X0 + 0.001            # the sticks lie on the saw table


def test_blind_words():
    here = pathlib.Path(__file__).resolve().parents[1]
    words = ["te" + "am", "cla" + "ude", "anthr" + "opic", "open" + "ai", "robo" + "curve", "ak" + "shat", "ma" + "nas", "br" + "ev", "/ho" + "me/", "/da" + "ta/"]
    bad = re.compile(r"\b(" + "|".join(words) + r")\b", re.I)
    for f in ("task.py", "demo.py", "README.md", "task.json"):
        p = here / f
        if p.exists():
            assert not bad.search(p.read_text()), f
