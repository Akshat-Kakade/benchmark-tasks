"""Scoring, gate and hygiene checks without the simulator:  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests"""
import math
import pathlib
import re

import numpy as np
import pytest

import task as M
from partkit import PartState

ROOT = pathlib.Path(M.__file__).parent
P0 = np.array([M.GUIDE_XY[0], M.GUIDE_XY[1], M.GUIDE_Z])


def _quat_from_matrix_rot_x(extra_deg):
    a = M.THETA + math.radians(extra_deg)
    return np.array([math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)])


def guide(s=0.0, shift=0.0, tilt=0.0):
    """Guide state: rod pulled s mm, guide moved shift mm along y, tilted tilt degrees about x."""
    z = np.zeros(3)
    return {"guide": PartState(pos=P0 + [0.0, shift / 1000, 0.0], quat_xyzw=_quat_from_matrix_rot_x(tilt), lin_vel=z,
                               ang_vel=z, low=z, high=z, sliders={"rod": -s / 1000})}


@pytest.fixture(autouse=True)
def fresh():
    M._reset_state()


def run(seq):
    out = None
    for s in seq:
        M._STATE["step"] += 1
        out = (M.score(s), M.success(s))
    return out


def rest(n=45):
    return [guide()] * n


def hold(s, n=25, **kw):
    return [guide(s, **kw)] * n


def ramp(a, b, n):
    return [guide(a + (b - a) * k / (n - 1)) for k in range(n)]


def test_start_scores_zero():
    assert run(rest(60)) == (0.0, False)


def test_nominal_tilt_is_zero_and_axis_geometry():
    assert M._tilt_deg(guide()["guide"]) < 1e-6
    assert abs(M._tilt_deg(guide(tilt=7)["guide"]) - 7) < 1e-6
    ax = M.R_GUIDE @ np.array([0, 0, -1.0])
    assert np.allclose(ax, [0, M.CA, M.SA])
    assert np.allclose(M.Lw(M.axis_point(0.3) - P0), [0, 0, -0.3])


def test_spawned_at_top_scores_nothing_until_rolled_down():
    assert run([guide(185.0)] * 60)[0] == 0.0
    assert run(rest(5) + hold(100.0))[0] == 0.5


def test_green_band_edge_matches_target():
    # rod tip u = 5 mm + s: s = 172 puts the tip on the band's leading edge u = 177
    assert abs(1000 * M.BAND_U[0] - (1000 * M.ROD_U0 + M.TARGET_LO)) < 1e-9


def test_level_025_boundary():
    assert run(rest() + hold(29.0))[0] == 0.0
    M._reset_state()
    assert run(rest() + hold(31.0))[0] == 0.25


def test_level_050_boundary():
    assert run(rest() + hold(89.0))[0] == 0.25
    M._reset_state()
    assert run(rest() + hold(91.0))[0] == 0.5


def test_level_075_boundary():
    assert run(rest() + hold(159.0))[0] == 0.5
    M._reset_state()
    assert run(rest() + hold(161.0))[0] == 0.75


def test_level_100_boundary():
    assert run(rest() + hold(171.0))[0] == 0.75
    M._reset_state()
    assert run(rest() + hold(173.0)) == (1.0, True)
    M._reset_state()
    assert run(rest() + hold(185.0)) == (1.0, True)


def test_upper_limit_not_scored_above_band():
    assert run(rest() + hold(190.0))[0] == 0.75


def test_short_hold_does_not_count():
    assert run(rest() + hold(175.0, n=15) + hold(0.0, n=30))[0] == 0.0


def test_rollback_after_hold_keeps_credit_but_no_more():
    assert run(rest() + hold(100.0) + ramp(100, 0, 60))[0] == 0.5


def test_flick_through_target_scores_nothing():
    # the rod shoots up and rolls back at ~48 mm/s: never still for 1 s
    seq = rest() + ramp(0, 180, 6) + ramp(180, 0, 80)
    assert run(seq)[0] == 0.0


def test_slow_drift_is_not_hold():
    assert run(rest() + ramp(40, 100, 20))[0] == 0.0


def test_hold_while_creeping_up_within_8mm():
    assert run(rest() + ramp(165, 172.5, 20) + hold(172.5, n=1)) [0] in (0.75, 1.0)


def test_guide_shift_gate_zero_sticky():
    seq = rest() + hold(175.0, shift=13.0) + hold(175.0)
    assert run(seq) == (0.0, False)


def test_guide_tilt_gate_zero():
    assert run(rest() + hold(175.0, tilt=7.0))[0] == 0.0


def test_gate_ignored_first_40_steps():
    assert run([guide(0, shift=20)] * 5 + rest(45) + hold(175.0))[0] == 1.0


def test_open_hands_rod_rolls_back_scores_zero():
    assert run(rest() + ramp(0, 50, 40) + ramp(50, 0, 21))[0] == 0.0


def test_two_valid_solutions_both_full():
    # right-left-right and left-right-left exchange traces: stationary overlaps, strokes between, final hold
    def strokes(order):
        seq, s = rest(), 0.0
        for _ in order:
            seq += hold(s, 25) + ramp(s, s + 58, 60)
            s += 58
        return seq + hold(s + 0.0, 40)
    for order in ("RLR", "LRL"):
        M._reset_state()
        seq = strokes(order)
        assert max(M._travel_mm(g["guide"]) for g in seq) > 172
        assert run(seq) == (1.0, True)


def test_every_level_reachable():
    got = set()
    for tail in ([], hold(40), hold(100), hold(165), hold(175)):
        M._reset_state()
        got.add(run(rest() + tail)[0])
    assert got == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_one_arm_span_below_full():
    # the geometry that keeps one hand from finishing alone: finger centre limits inside each opening
    g = 0.0125 + 0.002
    right = M.BLUE_OPEN[1] - g - (M.BLUE_OPEN[0] + g + 0.010 + 0.015)
    left = M.ORANGE_OPEN[1] - g - (M.ORANGE_OPEN[0] + g + 0.012)
    assert 1000 * right < 135 and 1000 * left < 135
    assert M.TARGET_LO > 1000 * max(right, left)


def test_exposed_rod_for_every_stroke():
    # RLR strokes of 58 mm: rod span in each opening at the moment of each grip is >= 60 mm clear of the shrouds
    s = 0.0
    for hand in "RLR":
        lo, hi = M.BLUE_OPEN if hand == "R" else M.ORANGE_OPEN
        tip, end = M.ROD_U0 + s / 1000, M.ROD_U0 + M.ROD_LEN + s / 1000
        free = min(hi, end) - max(lo, tip)
        assert free > 0.06
        s += 58


def test_rod_never_leaves_guide():
    assert M.ROD_U0 + M.ROD_LEN + M.TRAVEL < M.SHROUDS[2][1] + 0.05
    assert M.ROD_U0 + M.TRAVEL > M.SHROUDS[0][0]


def test_blind_words():
    banned = re.compile(r"(/home/|/data/|team|universit|institut)", re.I)
    assert not banned.search(M.INSTRUCTION)
    for f in ("task.json", "README.md"):
        p = ROOT / f
        if p.exists():
            assert not banned.search(p.read_text()), f


def test_no_function_named_metrics():
    assert not hasattr(M, "metrics")
