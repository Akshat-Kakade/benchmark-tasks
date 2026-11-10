"""Scoring, gate and hygiene checks without the simulator:  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests"""
import math
import pathlib
import re

import numpy as np
import pytest

import task as M
from partkit import PartState

ROOT = pathlib.Path(M.__file__).parent


def _quat(tilt_deg):
    a = math.radians(tilt_deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def ps(pos, tilt=0.0, sliders=None):
    z = np.zeros(3)
    return PartState(pos=np.asarray(pos, float), quat_xyzw=_quat(tilt), lin_vel=z, ang_vel=z, low=z, high=z,
                     sliders=sliders or {})


def scene(sag=0.0, press=0.0, paddle="away", shift=0.0, tilt=0.0):
    pz = M.PANEL_REST_Z - sag / 1000
    panel = ps((M.CX + shift / 1000, 0.0, pz), tilt, {"clip": -press / 1000})
    if paddle == "under":
        pd = ps((M.CX, M.HOLE_Y, pz - 0.004), 0.0)
    elif paddle == "far":                                  # fork under the panel edge, 80 mm from the clip
        pd = ps((M.CX, M.HOLE_Y + 0.08, pz - 0.004), 0.0)
    else:
        pd = ps((M.CX, 0.18, M.TOP + 0.03))
    return {"panel": panel, "base": ps((M.CX, 0.0, M.TOP + 0.02)), "paddle": pd}


@pytest.fixture(autouse=True)
def fresh():
    M._reset_state()


def run(seq):
    out = None
    for s in seq:
        M._STATE["step"] += 1
        out = (M.score(s), M.success(s))
    return out


def hold(*parts, n=25):
    return [scene()] * 45 + [s for s in parts for _ in range(n)]


def test_start_scores_zero():
    assert run(hold(scene())) == (0.0, False)


def test_025_press_alone_sags():
    assert run(hold(scene(sag=12, press=5)))[0] == 0.25


def test_050_paddle_support_no_press():
    assert run(hold(scene(paddle="under")))[0] == 0.5


def test_paddle_far_from_hole_is_not_support():
    assert run(hold(scene(paddle="far")))[0] == 0.0


def test_050_boundary_sag_8():
    assert run(hold(scene(paddle="under", sag=7.5)))[0] == 0.5
    M._reset_state()
    assert run(hold(scene(paddle="under", sag=9.0)))[0] == 0.0


def test_075_seated_under_support_paddle_left_in():
    assert run(hold(scene(paddle="under", press=9.5, sag=1.0)))[0] == 0.75


def test_075_needs_sag_le_3():
    assert run(hold(scene(paddle="under", press=9.5, sag=4.0)))[0] == 0.5


def test_full_and_hold():
    seq = hold(scene(paddle="under", press=9.8, sag=1.0), scene(paddle="under", sag=0.5), scene(sag=0.0), n=25)
    assert run(seq) == (1.0, True)


def test_two_valid_solutions_both_full():
    for sag in (0.0, 2.0, -2.0):                     # paddle left slightly up or down: panel back at rest +-4 mm
        M._reset_state()
        seq = hold(scene(paddle="under", press=9.8, sag=1.0), scene(sag=sag), n=25)
        assert run(seq) == (1.0, True)


def test_full_needs_the_hold_second():
    seq = [scene()] * 45 + [scene(paddle="under", press=9.8, sag=1.0)] * 12 + [scene(sag=0.0)] * 5
    assert run(seq) == (0.75, False) or run(seq)[1] is False


def test_panel_not_back_at_rest_is_075():
    assert run(hold(scene(paddle="under", press=9.8, sag=1.0), scene(sag=6.0)))[0] == 0.75


def test_unsupported_bottomed_press_is_zero_gate():
    assert run(hold(scene(sag=20, press=9.5)))[0] == 0.0


def test_gate_sticky_and_ignored_early():
    M._reset_state()
    assert run([scene(sag=20, press=9.5)] * 30)[0] == 0.25
    M._reset_state()
    assert run(hold(scene(sag=20, press=5), scene(paddle="under", press=9.8, sag=1.0), scene(sag=0)))[0] == 0.0


@pytest.mark.parametrize("kw", [dict(shift=60), dict(tilt=30)])
def test_gates_shift_and_tilt(kw):
    assert run(hold(scene(**kw), scene(paddle="under", press=9.8, sag=1.0), scene()))[0] == 0.0


def test_every_level_reachable():
    got = set()
    for seq in ([scene()], [scene(sag=12, press=5)], [scene(paddle="under")], [scene(paddle="under", press=9.8, sag=1)],
                [scene(paddle="under", press=9.8, sag=1), scene(sag=0)]):
        M._reset_state()
        got.add(run(hold(*seq))[0])
    assert got == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_blind_words():
    banned = re.compile(r"(/home/|/data/|team|universit|institut)", re.I)
    assert not banned.search(M.INSTRUCTION)
    for f in ("task.json", "README.md"):
        p = ROOT / f
        if p.exists():
            assert not banned.search(p.read_text()), f
