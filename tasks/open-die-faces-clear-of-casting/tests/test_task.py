"""Scoring, gate and hygiene checks without the simulator:  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests"""
import math
import pathlib
import re

import numpy as np
import pytest

import task as M
from partkit import PartState

ROOT = pathlib.Path(M.__file__).parent
Z = np.zeros(3)


def states(tl=0.0, tr=0.0, cast=(0.0, 0.0), tilt=0.0, base=(0.0, 0.0), cross=0.0):
    """Travels in mm of left/right face, casting moved (x, y) mm, casting tilted degrees, left base moved (x, y) mm."""
    out = {}
    for name, sign, t in (("die_l", 1, tl), ("die_r", -1, tr)):
        q = M.QUAT_L if sign > 0 else M.QUAT_R
        p = np.array([M.CX, sign * M.BASE_OUT, M.ORIGIN_Z])
        if sign > 0:
            p = p + [base[0] / 1000, base[1] / 1000, 0.0]
        out[name] = PartState(pos=p, quat_xyzw=np.array(q), lin_vel=Z, ang_vel=Z, low=Z, high=Z,
                              sliders={"face": -t / 1000})
    a = math.radians(tilt) / 2
    q = np.array([math.sin(a), 0.0, 0.0, math.cos(a)])
    out["casting"] = PartState(pos=np.array([M.CX + cast[0] / 1000, cast[1] / 1000, M.TOP + 0.048]), quat_xyzw=q,
                               lin_vel=Z, ang_vel=Z, low=Z, high=Z)
    return out


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
    return [states()] * n


def test_rest_scores_zero():
    assert run(rest()) == (0.0, False)


@pytest.mark.parametrize("tl,tr,exp", [
    (19, 30, 0.0), (20, 20, 0.25), (45, 20, 0.25), (40, 40, 0.5), (40, 70, 0.5), (45, 75, 0.5),
    (55, 55, 0.75), (55, 74, 0.75), (54, 54, 0.5), (55, 76, 0.5), (64, 64, 0.75), (65, 65, 1.0), (65, 77, 1.0), (65, 77.5, 0.75),
    (70, 70, 1.0), (75, 63, 0.75), (75, 75, 1.0), (50, 20, 0.25), (40, 70, 0.5), (40, 70.5, 0.25)])
def test_tiers(tl, tr, exp):
    M._reset_state()
    got = run(rest() + [states(tl, tr)])[0]
    assert got == exp


def test_one_arm_only_is_zero():
    assert run(rest() + [states(70, 0)] * 30)[0] == 0.0


def test_diff_boundaries_for_full():
    assert run(rest() + [states(77, 65)])[0] == 1.0      # diff 12 mm
    M._reset_state()
    assert run(rest() + [states(77.5, 65)])[0] == 0.75   # diff 12.5 mm


def test_success_needs_one_second():
    seq = rest() + [states(70, 70)] * 19
    assert run(seq) == (1.0, False)
    M._reset_state()
    assert run(rest() + [states(70, 70)] * 20) == (1.0, True)


def test_hold_must_be_continuous():
    seq = rest() + [states(70, 70)] * 15 + [states(30, 30)] + [states(70, 70)] * 15
    assert run(seq)[1] is False


def test_clearance_is_measured_not_assumed():
    # faces at rest sit 10.5 mm from the casting; 65 mm travel gives 75.5 mm clearance, so the clearance clause only bites
    # when the casting itself has been dragged toward a face
    cl, cr = M._clearances_mm(states(0, 0))
    assert cl == pytest.approx(10.5, abs=0.01) and cr == pytest.approx(10.5, abs=0.01)
    cl, cr = M._clearances_mm(states(65, 70))
    assert cl == pytest.approx(75.5, abs=0.01) and cr == pytest.approx(80.5, abs=0.01)
    # casting shifted 11 mm toward the left face: clearance 64.5 mm still fine; the 12 mm gate stops larger shifts
    assert M._level(70, 70, 34.0, 80) == 0.75


@pytest.mark.parametrize("kw", [dict(cast=(13, 0)), dict(cast=(0, 13)), dict(tilt=9), dict(base=(16, 0))])
def test_gates_zero(kw):
    seq = rest() + [states(70, 70, **kw)] * 3 + [states(70, 70)] * 30
    assert run(seq) == (0.0, False)


def test_gate_boundaries_pass():
    assert run(rest() + [states(70, 70, cast=(8, 6), tilt=7, base=(10, 0))] * 25) == (1.0, True)


def test_gate_ignored_first_40_steps():
    assert run([states(70, 70, cast=(20, 0))] * 5 + rest() + [states(70, 70)] * 25) == (1.0, True)


def test_two_valid_solutions_both_full():
    # (a) both faces at their outward stops; (b) both held in the 65-75 mm band, slightly unequal
    for tl, tr in ((75, 75), (66, 71)):
        M._reset_state()
        assert run(rest() + [states(tl, tr)] * 25) == (1.0, True)


def test_every_level_reachable():
    got = set()
    for tl, tr in ((0, 0), (25, 25), (45, 45), (58, 58), (70, 70)):
        M._reset_state()
        got.add(run(rest() + [states(tl, tr)] * 25)[0])
    assert got == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_blind_words():
    banned = re.compile(r"(/home/|/data/|team|universit|institut)", re.I)
    assert not banned.search(M.INSTRUCTION)
    for f in ("task.json", "README.md"):
        p = ROOT / f
        if p.exists():
            assert not banned.search(p.read_text()), f


def test_no_function_named_metrics():
    assert not hasattr(M, "metrics")


def test_frames_map_slider_travel_outward():
    from partkit import _quat_matrix
    for sign, q in ((1, M.QUAT_L), (-1, M.QUAT_R)):
        R = _quat_matrix((q[3], q[0], q[1], q[2]))
        _, L, _ = M.side_frame(sign)
        assert np.allclose(R @ np.array(L(0.1, 0.2, 0.3)), (0.1, sign * 0.2, 0.3))
