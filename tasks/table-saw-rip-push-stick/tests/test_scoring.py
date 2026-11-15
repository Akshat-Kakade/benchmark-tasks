"""Scoring boundaries, every partial level, zero gate, two valid solutions, blind words. Run on the VM: python -m pytest tests -q"""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def st(pos, w, l, h, quat=(0, 0, 0, 1), v=0.0):
    z = np.zeros(3)
    pos = np.array(pos, float)
    half = np.array([w, l, h]) / 2
    return PartState(pos=pos, quat_xyzw=np.array(quat, float), lin_vel=np.array([0, v, 0.0]), ang_vel=z, low=pos - half, high=pos + half)


def board(rear, dz=0.0, tilt=0.0, v=0.0):
    cy = rear + T.BOARD_L / 2
    qx = (math.sin(math.radians(tilt) / 2), 0, 0, math.cos(math.radians(tilt) / 2))
    s = st((T.STRIP_CX, cy, T.BOARD_CZ + dz), T.STRIP_W, T.BOARD_L, T.BOARD_T, qx, v)
    o = st((T.OFF_CX, cy, T.BOARD_CZ + dz), T.OFF_W, T.BOARD_L, T.BOARD_T, qx, v)
    b = st((T.BOARD_CX, cy, T.BOARD_CZ + dz), T.BOARD_W, T.BOARD_L, T.BOARD_T, qx, v)
    stick = st((T.BLADE_X, 0.0, T.TOP + 0.05), 0.09, 0.3, 0.05)
    return {"board": b, "strip": s, "offcut": o, "stick": stick, "saw": st((0.25, 0.0, T.TOP), 0.2, 0.7, 0.03)}


def fresh(**kw):
    return T._fresh() | {"step": 10} | kw


CUT = dict(cut=True, kerf=T.KERF_FULL)


def grip(l=(0.30, -0.20, 0.0), r=(0.2275, -0.30, 0.0), z=0.10):
    def g(p):
        p = np.array([p[0], p[1], T.SAW_TOP + z])
        return dict(tcp=p, pts=[p, p + [0, 0, 0.13], p + [0, 0.02, 0], p - [0, 0.02, 0]], gap=0.05)
    return {"L": g(l), "R": g(r)}


def lv(objs, gr, memo=None):
    memo = memo if memo is not None else fresh(**CUT)
    return T.classify(objs, gr, memo)[:2]


DONE = T.CUT_DONE + 0.001                     # rear y when complete


def test_levels():
    # intact board, no kerf yet: 0
    assert lv(board(T.BOARD_REAR0), grip(), fresh())[0] == 0.0
    # front past the blade but no engaged feed counted: still 0 (position alone gives nothing)
    assert lv(board(T.BLADE_Y0 + 0.1 - T.BOARD_L), grip(), fresh())[0] == 0.0
    assert lv(board(T.BLADE_Y0 + 0.1 - T.BOARD_L), grip(), fresh(kerf=T.KERF_MIN - 0.001))[0] == 0.0
    assert lv(board(T.BLADE_Y0 + 0.1 - T.BOARD_L), grip(), fresh(kerf=T.KERF_MIN))[0] == 0.25
    # separated (cut) but the rear ends have not cleared the blade: 0.25
    assert lv(board(DONE - 0.003), grip())[0] == 0.25
    assert lv(board(DONE), grip(), )[0] == 1.0
    assert lv(board(DONE, tilt=12.0), grip())[0] == 0.5
    assert lv(board(DONE, v=0.05), grip())[0] == 0.75


def test_no_kerf_no_credit_even_if_board_ends_past_blade():
    # the intact board slid past the blade without engaging (kerf 0): zero
    assert lv(board(DONE), grip(), fresh())[0] == 0.0
    # ... or with a short kerf only
    assert lv(board(DONE), grip(), fresh(kerf=0.02))[0] == 0.0


def test_kerf_accrual_rules():
    memo = fresh()
    b = board(T.BLADE_Y0 + 0.01 - T.BOARD_L)
    T.accrue_kerf(b, memo)                                   # first sample: no accrual
    b2 = board(T.BLADE_Y0 + 0.03 - T.BOARD_L)
    memo["step"] = 11
    T.accrue_kerf(b2, memo)
    assert abs(memo["kerf"] - 0.02) < 1e-9                   # engaged feed counted
    # lifted 6 mm: the feed does not count and the lift latches
    b3 = board(T.BLADE_Y0 + 0.05 - T.BOARD_L, dz=0.006)
    memo["step"] = 12
    T.accrue_kerf(b3, memo)
    assert abs(memo["kerf"] - 0.02) < 1e-9 and memo["lifted"]
    # off the fence by 10 mm: does not count
    memo = fresh()
    T.accrue_kerf(board(T.BLADE_Y0 + 0.01 - T.BOARD_L), memo)
    b4 = board(T.BLADE_Y0 + 0.03 - T.BOARD_L)
    b4["board"] = st((T.BOARD_CX + 0.010, b4["board"].pos[1], T.BOARD_CZ), T.BOARD_W, T.BOARD_L, T.BOARD_T)
    memo["step"] = 11
    T.accrue_kerf(b4, memo)
    assert memo["kerf"] == 0.0
    # kerf is capped (cannot grow past the full length by more than 50 mm)
    memo = fresh(kerf=T.KERF_FULL + 0.049, prev_front=T.BLADE_Y0 + 0.1)
    memo["step"] = 12
    T.accrue_kerf(board(T.BLADE_Y0 + 0.5 - T.BOARD_L), memo)
    assert memo["kerf"] <= T.KERF_FULL + 0.05 + 1e-9


def test_hand_piece_clearance():
    g = grip()
    side = T.OFF_CX + T.OFF_W / 2                                                       # offcut's far edge x
    p = np.array([side + 0.020, DONE + 0.13, T.SAW_TOP + 0.01])                         # 20 mm beside the offcut, far from the blade
    g["R"] = dict(tcp=p, pts=[p], gap=0.05)
    assert lv(board(DONE), g) == (0.75, False)
    p2 = p + [0.02, 0, 0]                                                                # 40 mm beside the offcut
    g["R"] = dict(tcp=p2, pts=[p2], gap=0.05)
    assert lv(board(DONE), g) == (1.0, True)


def test_zero_gate():
    near = grip(l=(0.30, T.BLADE_Y0 - 0.06, 0.0), z=0.05)                              # inside 100 mm of the blade, low
    memo = fresh(**CUT)
    assert T.classify(board(DONE), near, memo)[0] == 0.0 and memo["gate"]
    memo = fresh(**CUT)
    high = grip(l=(0.30, T.BLADE_Y0 - 0.06, 0.0), z=0.20)                              # the arm passes ABOVE 150 mm: no event
    assert T.classify(board(DONE), high, memo)[0] == 1.0 and not memo["gate"]   # above 150 mm: no event, no end condition
    memo = fresh(**CUT)
    far = grip(l=(0.30, T.BLADE_Y0 - 0.12, 0.0), z=0.05)                               # 120 mm: no event
    assert T.classify(board(DONE), far, memo)[0] == 1.0
    memo = fresh(**CUT)
    assert T.classify(board(DONE, dz=-0.05), grip(), memo)[0] == 0.0 and memo["gate"]  # piece fell
    memo = fresh(**CUT)
    T.classify(board(DONE), near, memo)
    assert T.classify(board(DONE), grip(), memo)[0] == 0.0                              # latched hazard


def test_lifted_over_blade_scores_zero():
    memo = fresh()
    o = board(T.BLADE_Y0 + 0.01 - T.BOARD_L / 2, dz=0.012)                              # board overlaps the blade 12 mm above the table
    T.classify(o, grip(), memo)
    assert memo["lifted"]
    assert memo["lifted"] and T.classify(board(DONE), grip(), memo)[0] == 0.0
    assert T.classify(board(DONE), grip(), fresh(**CUT, lifted=True))[0] == 0.0


def test_hold_one_second():
    T._reset_state()
    T._STATE["step"] = 10
    T._STATE.update(CUT)
    results = []
    for i in range(T.HOLD_STEPS + 2):
        T._STATE["step"] += 1
        T._STATE["scored"] = None
        T._grip = None
        orig = T.grip_state
        T.grip_state = lambda: grip()
        try:
            results.append(T._update(board(DONE)))
        finally:
            T.grip_state = orig
    assert results[0][0] == 0.75 and not results[0][1]
    assert results[T.HOLD_STEPS - 2][1] is False
    assert results[T.HOLD_STEPS - 1] == (1.0, True)


def test_two_valid_solutions():
    # solution A: the cut finishes with the pieces still against the fence; solution B: pieces ended 20 mm apart sideways and 30 mm further on
    a = board(DONE)
    b = board(DONE + 0.03)
    b["offcut"] = st((T.OFF_CX + 0.02, b["offcut"].pos[1], T.BOARD_CZ), T.OFF_W, T.BOARD_L, T.BOARD_T)
    assert lv(a, grip()) == (1.0, True) and lv(b, grip()) == (1.0, True)


def test_blind_words():
    here = pathlib.Path(__file__).resolve().parents[1]
    words = ["te" + "am", "cla" + "ude", "anthr" + "opic", "open" + "ai", "robo" + "curve", "ak" + "shat", "ma" + "nas", "br" + "ev", "/ho" + "me/", "/da" + "ta/"]
    bad = re.compile(r"\b(" + "|".join(words) + r")\b", re.I)
    for f in ("task.py", "demo.py", "README.md", "task.json"):
        p = here / f
        if p.exists():
            assert not bad.search(p.read_text()), f


def test_geometry():
    assert abs(T.STRIP_W + T.KERF + T.OFF_W - T.BOARD_W) < 1e-9
    assert T.STRIP_W == 0.040
    assert T.BOARD_REAR0 + T.BOARD_L < T.BLADE_Y0                                       # board starts clear of the blade
    assert T.SAW_Y0 < T.STICK_START[1] - T.BAR_Y0                                       # the whole stick lies on the saw table
    # the fork gap straddles the blade with >= 5 mm each side
    assert T.GAP / 2 - T.BLADE_T / 2 >= 0.005
    # the bar rides at least 3 mm above the blade top when the heel is 3 mm above the table
    assert (T.BAR_Z0 - T.HEEL_Z0) + 0.007 >= T.BLADE_H + 0.003


def test_strip_width():
    memo = fresh(**CUT)
    o = board(T.BLADE_YC - T.BOARD_L / 2 + 0.05)                 # blade centre is inside the pieces
    T.classify(o, grip(), memo)
    assert abs(memo["wmin"] - T.STRIP_W) < 1e-6
    o["strip"] = st((T.STRIP_CX + 0.007, o["strip"].pos[1], T.BOARD_CZ), T.STRIP_W, T.BOARD_L, T.BOARD_T)   # 4 mm off the fence
    memo = fresh(**CUT)
    T.classify(o, grip(), memo)
    assert abs(memo["wmin"] - (T.STRIP_W - 0.007)) < 1e-6
    memo["wmin"] = memo["wmax"] = T.STRIP_W - 0.007
    assert T.classify(board(DONE), grip(), memo)[0] == 0.5      # complete but strip 7 mm narrow: 0.5 (> 6.4 mm)
    memo["wmin"] = memo["wmax"] = T.STRIP_W - 0.005
    assert T.classify(board(DONE), grip(), memo)[0] == 0.75     # within 6.4 mm, not within 3.2 mm
    memo["wmin"] = memo["wmax"] = T.STRIP_W - 0.003
    assert T.classify(board(DONE), grip(), memo)[0] == 1.0


def test_guard_and_knife_clear_of_stick_path():
    # riving knife thicker than the blade plate, thinner than the kerf, just behind the blade, below the blade top
    assert T.BLADE_T < T.KNIFE_T < T.KERF and T.KNIFE_Y0 > T.BLADE_Y1 and T.KNIFE_H < T.BLADE_H
    # guard plate bottom is above the stick bar top (3 mm clearance at least) and above the board top
    assert T.GUARD_Z0 >= T.BAR_Z1 + 0.007 + 0.003 and T.GUARD_Z0 > T.BOARD_T + 0.0005
    # the column stands beside the stock path (stock far edge 0.274), the heel never goes beyond CUT_DONE + 6 mm (the fork gap covers the knife)
    assert T.POST_X0 > T.OFF_CX + T.OFF_W / 2 + 0.02 and T.GAP > T.KNIFE_T + 0.010
    assert T.GUARD_Y0 < T.BLADE_Y0 and T.GUARD_Y1 > T.BLADE_Y1 and T.BLADE_X + T.GUARD_HW < T.POST_X1


def test_tipped_board_over_blade_counts_as_lifted():
    memo = fresh()
    T.accrue_kerf(board(T.BLADE_Y0 + 0.01 - T.BOARD_L), memo)
    memo["step"] = 11
    T.accrue_kerf(board(T.BLADE_Y0 + 0.03 - T.BOARD_L, tilt=12.0), memo)       # flat on the table by height but tipped 12 deg
    assert memo["lifted"] and memo["kerf"] == 0.0
