"""Scoring, gate and hygiene checks without the simulator:  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests"""
import pathlib
import re

import numpy as np
import pytest

import task as M
from partkit import PartState

ROOT = pathlib.Path(M.__file__).parent
Z = np.zeros(3)
J0 = np.array([M.JX, M.JY, M.ORIGIN_Z])
Q = np.array([0.0, 0.0, 0.0, 1.0])


def st(gap=3.0, plate=(0.0, -220.0, None), jshift=(0.0, 0.0), tilt=0.0, vel=0.0):
    """Joint with the flange open gap mm; plate offset (mm from the green mark, plate-centre z in the gap by default)."""
    px, py, pz = plate
    if pz is None:
        pz = M.PLATE_START[2] if py < -150 else M.UPPER_UNDER - gap / 2000.0 if gap > 14 else M.UPPER_UNDER - 0.006
    q = Q
    if tilt:
        a = np.radians(tilt) / 2
        q = np.array([np.sin(a), 0.0, 0.0, np.cos(a)])
    joint = PartState(pos=J0 + [jshift[0] / 1000, jshift[1] / 1000, 0], quat_xyzw=Q, lin_vel=Z, ang_vel=Z, low=Z, high=Z,
                      sliders={"flange": -gap / 1000})
    pl = PartState(pos=np.array([M.JX + px / 1000, M.JY + py / 1000, pz]), quat_xyzw=q, lin_vel=np.array([vel, 0, 0]),
                   ang_vel=Z, low=Z, high=Z)
    return {"joint": joint, "plate": pl}


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
    return [st()] * n


def lifted(gap=32, n=5):
    return [st(gap, (0, -220, M.PLATE_START[2] + 0.05))] * n


def entering(gap=32, y=-40, x=0, n=5):
    return [st(gap, (x, y, M.UPPER_UNDER - gap / 2000))] * n


def seated(gap=32, x=0, y=0, n=12):
    return [st(gap, (x, y, M.UPPER_UNDER - gap / 2000))] * n


def clamped(gap=12, x=0, y=0, n=25, **kw):
    return [st(gap, (x, y, M.UPPER_UNDER - 0.006), **kw)] * n


def test_start_zero():
    assert run(rest(60)) == (0.0, False)


def test_025_open_with_plate_held():
    assert run(rest() + lifted()) == (0.25, False)


def test_open_without_plate_lifted_is_zero():
    assert run(rest() + [st(32)] * 10)[0] == 0.0


def test_lifted_but_closed_is_zero():
    assert run(rest() + [st(10, (0, -220, M.PLATE_START[2] + 0.05))] * 10)[0] == 0.0


def test_gap_boundary_025():
    assert run(rest() + lifted(gap=21))[0] == 0.0
    M._reset_state()
    assert run(rest() + lifted(gap=23))[0] == 0.25


def test_050_entered():
    assert run(rest() + lifted() + entering())[0] == 0.5


def test_entry_depth_boundary():
    assert run(rest() + lifted() + entering(y=-70))[0] == 0.25
    M._reset_state()
    assert run(rest() + lifted() + entering(y=-60))[0] == 0.5


def test_entry_needs_gap_14():
    assert run(rest() + lifted() + entering(gap=12))[0] == 0.25


def test_plate_on_post_top_is_not_in_gap():
    seq = rest() + lifted() + [st(32, (0, -40, M.UPPER_UNDER + 0.03))] * 5
    assert run(seq)[0] == 0.25


def test_pushing_closed_joint_blocked():
    seq = rest() + [st(3, (0, -80, M.PLATE_START[2]))] * 20
    assert run(seq)[0] == 0.0


def test_075_centred_open():
    assert run(rest() + lifted() + entering() + seated())[0] == 0.75


def test_off_centre_stays_050():
    assert run(rest() + lifted() + entering() + seated(x=8))[0] == 0.5
    M._reset_state()
    assert run(rest() + lifted() + entering() + seated(y=7))[0] == 0.5


def test_centre_boundary():
    assert run(rest() + lifted() + entering() + seated(x=5, y=3))[0] == 0.75


def test_tilted_plate_not_centred():
    seq = rest() + lifted() + entering() + [st(32, (0, 0, M.UPPER_UNDER - 0.016), tilt=15)] * 12
    assert run(seq)[0] == 0.5


def test_seat_needs_half_second():
    assert run(rest() + lifted() + entering() + seated(n=6))[0] == 0.5


def test_full_released():
    assert run(rest() + lifted() + entering() + seated() + [st(20, (0, 0, M.UPPER_UNDER - 0.01))] * 3 + clamped()) == (1.0, True)


def test_full_needs_one_second():
    seq = rest() + lifted() + entering() + seated() + clamped(n=15)
    assert run(seq) == (0.75, False)


def test_gap_range_for_full():
    for g, ok in ((9.5, False), (10.5, True), (13.5, True), (14.5, False)):
        M._reset_state()
        seq = rest() + lifted() + entering() + seated() + [st(g, (0, 0, M.UPPER_UNDER - 0.006))] * 25
        assert (run(seq)[0] == 1.0) is ok, g


def test_full_needs_still():
    seq = rest() + lifted() + entering() + seated() + clamped(vel=0.05)
    assert run(seq)[1] is False


def test_clamped_off_centre_not_full():
    seq = rest() + lifted() + entering() + seated() + clamped(x=8)
    assert run(seq)[0] == 0.75


def test_two_valid_solutions_full():
    for xy in ((2.0, -3.0), (-4.0, 4.0)):
        M._reset_state()
        seq = rest() + lifted() + entering() + seated(x=xy[0], y=xy[1]) + clamped(x=xy[0], y=xy[1])
        assert run(seq) == (1.0, True)


def test_gate_plate_dropped():
    seq = rest() + lifted() + entering() + [st(32, (0, 0, M.TOP + 0.006))] * 3 + seated() + clamped()
    assert run(seq) == (0.0, False)


def test_gate_joint_shift():
    seq = rest() + lifted() + entering() + seated() + clamped(jshift=(12, 0))
    assert run(seq) == (0.0, False)


def test_joint_shift_boundary():
    seq = rest() + lifted() + entering() + seated() + clamped(jshift=(8, 0))
    assert run(seq) == (1.0, True)


def test_gate_flange_stall():
    seq = rest() + [st(44.8)] * 25 + lifted() + entering() + seated() + clamped()
    assert run(seq) == (0.0, False)


def test_short_touch_of_limit_ok():
    seq = rest() + [st(44.8)] * 10 + lifted() + entering() + seated() + clamped()
    assert run(seq) == (1.0, True)


def test_gate_ignored_first_steps():
    assert run([st(44.8)] * 5 + rest() + lifted() + entering() + seated() + clamped()) == (1.0, True)


def test_every_level_reachable():
    got = set()
    for tail in ([], lifted(), lifted() + entering(), lifted() + entering() + seated(),
                 lifted() + entering() + seated() + clamped()):
        M._reset_state()
        got.add(run(rest() + tail)[0])
    assert got == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_tilt_zero_when_flat():
    assert M._tilt_deg(st()["plate"]) < 1e-6


def test_blind_words():
    banned = re.compile(r"(/home/|/data/|team|universit|institut)", re.I)
    assert not banned.search(M.INSTRUCTION)
    for f in ("task.json", "README.md"):
        p = ROOT / f
        if p.exists():
            assert not banned.search(p.read_text()), f


def test_no_function_named_metrics():
    assert not hasattr(M, "metrics")
