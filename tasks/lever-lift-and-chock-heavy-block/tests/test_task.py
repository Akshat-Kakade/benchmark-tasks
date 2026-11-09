"""Scoring tests for the lever-and-chock task. Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q"""
import math
import pathlib
import re
import sys

import numpy as np
import pytest

pytest.importorskip("yam_arena")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402


class S:
    def __init__(self, pos, rx_deg=0.0, ry_deg=0.0):
        self.pos = np.array(pos, float)
        self.rx, self.ry = math.radians(rx_deg), math.radians(ry_deg)

    def rotation(self):
        c, s = math.cos(self.rx), math.sin(self.rx)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        c, s = math.cos(self.ry), math.sin(self.ry)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return Ry @ Rx


B0 = np.array(T.BLOCK_START)
C0 = np.array(T.CHOCK_START)
C_IN = np.array([T.X, B0[1] + T.CHOCK_FINAL_DY, C0[2]])


def states(lift=0.0, lever=0.0, chock=C0, tilt=0.0, block_xy=(0.0, 0.0)):
    return {"block": S(B0 + [block_xy[0], block_xy[1], lift], rx_deg=tilt), "lever": S(T.PIVOT, rx_deg=lever),
            "chock": S(chock)}


def run(seq, hold=25):
    """Feed a sequence of states one per control step (the last one held), return (score, success)."""
    T._reset_state()
    out = (0.0, False)
    seq = list(seq) + [seq[-1]] * hold
    for st in seq:
        T._STATE["step"] += 1
        out = (T.score(st), T.success(st))
    return out


def test_settle_scores_zero():
    assert run([states()]) == (0.0, False)


def test_025_lever_used():
    assert run([states(), states(0.007, 5.0)]) == (0.25, False)


def test_below_025_threshold():
    assert run([states(), states(0.005, 4.0)])[0] == 0.0


def test_05_cleared_and_chock_moved():
    seq = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C0 - [0, 0.09, 0])]
    assert run(seq)[0] == 0.5


def test_05_needs_both():
    assert run([states(), states(0.011, 9.5)])[0] == 0.25


def test_075_chock_under_lever_pressed():
    seq = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C_IN)]
    assert run(seq) == (0.75, False)


def test_075_when_block_tilts():
    seq = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C_IN), states(0.004, 0.0, chock=C_IN, tilt=7.0)]
    assert run(seq)[0] == 0.75


def test_10_full():
    seq = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C_IN), states(0.004, 0.0, chock=C_IN)]
    assert run(seq) == (1.0, True)


def test_10_not_held():
    seq = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C_IN), states(0.004, 0.0, chock=C_IN)]
    assert run(seq, hold=5) == (1.0, False)


def test_lever_still_pressed_blocks_full():
    seq = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C_IN), states(0.011, 6.0, chock=C_IN)]
    assert run(seq)[0] == 0.75


def test_block_too_high_or_low_on_chock():
    base = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C_IN)]
    assert run(base + [states(0.012, 0.0, chock=C_IN)])[0] == 0.75
    assert run(base + [states(0.0, 0.0, chock=C_IN)])[0] == 0.75


def test_gate_block_tipped():
    seq = [states(), states(0.011, 9.5), states(0.0, 0.0, tilt=40.0)]
    assert run(seq) == (0.0, False)


def test_gate_lifted_by_hand():
    # block raised 8 mm with the bar at rest and no chock: gripper lift, zero even if a chock then goes in
    seq = [states(), states(0.008, 0.0), states(0.011, 9.5), states(0.011, 9.5, chock=C_IN),
           states(0.004, 0.0, chock=C_IN)]
    assert run(seq) == (0.0, False)


def test_gate_chock_thrown():
    seq = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C0 + [0.0, 0.0, 0.0] - [0.35, 0, 0])]
    assert run(seq)[0] == 0.0


def test_gate_block_carried_away():
    seq = [states(), states(0.011, 9.5), states(0.011, 9.5, block_xy=(0.0, 0.25))]
    assert run(seq)[0] == 0.0


def test_two_solutions_both_full():
    """Chock slid in from a slightly different approach (x off by 8 mm, y 10 mm short) and the bar released slowly."""
    a = [states(), states(0.011, 9.5), states(0.011, 9.5, chock=C_IN), states(0.004, 0.0, chock=C_IN)]
    c2 = C_IN + [0.008, 0.010, 0.0]
    b = [states(), states(0.009, 7.5), states(0.010, 8.0, chock=C0 - [0, 0.1, 0]), states(0.010, 8.0, chock=c2),
         states(0.006, 3.0, chock=c2), states(0.0045, 1.0, chock=c2)]
    assert run(a) == (1.0, True)
    assert run(b) == (1.0, True)


def test_geometry_constants():
    assert T.CHOCK_H - T.BLOCK_BOTTOM >= 0.004 - 1e-9          # chock cannot go under the unlifted block
    assert T.LIFT_CLEAR - (T.CHOCK_H - T.BLOCK_BOTTOM) >= 0.004   # clearance when lifted
    assert FINAL_OK()


def FINAL_OK():
    lift_final = T.CHOCK_H - T.BLOCK_BOTTOM
    return T.FINAL_LIFT[0] <= lift_final <= T.FINAL_LIFT[1]
