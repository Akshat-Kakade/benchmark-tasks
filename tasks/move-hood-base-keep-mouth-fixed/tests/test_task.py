"""Scoring, gate and hygiene checks without the simulator:  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests"""
import math
import pathlib
import re

import numpy as np
import pytest

import task as M
from partkit import PartState

ROOT = pathlib.Path(M.__file__).parent
P0 = np.array([M.FOOT_X0, M.FOOT_Y0, M.ORIGIN_Z])


def quat(tilt=0.0, yaw=0.0):
    """Flat foot quaternion (xyzw), then an extra tilt about world y and a yaw about world z (degrees)."""
    def mul(a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    q = np.array(M.QUAT_FLAT)
    if tilt:
        a = math.radians(tilt) / 2
        q = mul([0.0, math.sin(a), 0.0, math.cos(a)], q)
    if yaw:
        a = math.radians(yaw) / 2
        q = mul([0.0, 0.0, math.sin(a), math.cos(a)], q)
    return q


def hood(D=0.0, d=0.0, ext=None, y=0.0, lift=0.0, tilt=0.0, yaw=0.0):
    """State with the foot D mm toward -x and y mm off line; the mouth ends d mm (along x) from its start.
    ext defaults to D + d (mouth = foot x + 0.13 + ext)."""
    ext = D + d if ext is None else ext
    z = np.zeros(3)
    return {"foot": PartState(pos=P0 + [-D / 1000, y / 1000, lift / 1000], quat_xyzw=quat(tilt, yaw), lin_vel=z, ang_vel=z,
                              low=z, high=z, sliders={"arm": -ext / 1000})}


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
    return [hood()] * n


def test_start_scores_zero():
    assert run(rest(60)) == (0.0, False)


def test_flat_has_zero_tilt_yaw():
    s = hood()["foot"]
    assert M._tilt_deg(s) < 1e-6 and abs(M._yaw_deg(s)) < 1e-6


def test_mouth_position_follows_extension():
    s = hood(D=70, ext=70)["foot"]
    m = M.mouth_pos(s)
    assert abs(m[0] - M.MOUTH_X) < 1e-9 and abs(m[1]) < 1e-9 and abs(m[2] - M.MOUTH_ZC) < 1e-9


def test_tilt_and_yaw_measure():
    assert abs(M._tilt_deg(hood(tilt=7)["foot"]) - 7) < 0.01
    assert abs(abs(M._yaw_deg(hood(yaw=6)["foot"])) - 6) < 0.01


def test_mouth_moves_with_foot_when_not_extended():
    # whole assembly moved, no extension: mouth distance == D
    assert run(rest() + [hood(70, 0, ext=0)] * 30)[0] == 0.0


def test_025():
    assert run(rest() + [hood(25, 10)] * 25)[0] == 0.25


def test_025_boundary():
    assert run(rest() + [hood(19, 0)] * 25)[0] == 0.0
    M._reset_state()
    assert run(rest() + [hood(21, 0)] * 25)[0] == 0.25


def test_05():
    assert run(rest() + [hood(45, 25)] * 25)[0] == 0.5


def test_05_when_d_between_20_and_30():
    assert run(rest() + [hood(60, 25)] * 25)[0] == 0.5


def test_075():
    assert run(rest() + [hood(60, 10, y=15)] * 25)[0] == 0.75


def test_075_needs_small_mouth_drift():
    assert run(rest() + [hood(60, 21)] * 25)[0] == 0.5


def test_overshoot_is_075():
    assert run(rest() + [hood(85, 5)] * 25)[0] == 0.75


def test_full():
    assert run(rest() + [hood(70, 5)] * 25) == (1.0, True)


def test_full_needs_one_second():
    assert run(rest() + [hood(70, 5)] * 19)[0] == 0.75
    M._reset_state()
    assert run(rest() + [hood(70, 5)] * 20) == (1.0, True)


def test_full_boundaries():
    assert run(rest() + [hood(64, 0)] * 25)[0] == 0.75
    for kw in (dict(D=79, d=0), dict(D=70, ext=57), dict(D=70, d=0, y=13), dict(D=70, d=0, ext=59),
               dict(D=70, d=0, tilt=9), dict(D=70, d=0, yaw=9)):
        M._reset_state()
        assert run(rest() + [hood(**kw)] * 25)[0] < 1.0, kw


def test_full_edges_inside():
    for kw in (dict(D=65.5, d=0, ext=65.5), dict(D=77.5, d=0), dict(D=70, ext=61), dict(D=70, d=0, y=11),
               dict(D=62, d=0, ext=60.5)):
        M._reset_state()
        if kw["D"] < 65:
            continue
        assert run(rest() + [hood(**kw)] * 25) == (1.0, True), kw


@pytest.mark.parametrize("kw", [dict(lift=22), dict(tilt=11), dict(y=125)])
def test_gates_zero_sticky(kw):
    seq = rest() + [hood(70, 0, **kw)] * 3 + [hood(70, 0)] * 30
    assert run(seq) == (0.0, False)


def test_gate_ignored_first_40_steps():
    assert run([hood(tilt=20)] * 3 + rest() + [hood(70, 0)] * 30)[0] == 1.0


def test_no_move_zero():
    assert run(rest() + [hood(0, 30)] * 40)[0] == 0.0


def test_two_valid_solutions_both_full():
    # foot taken straight back with the mouth dead still, or with a few mm of slow drift and extension a little larger
    for kw in (dict(D=70, d=0), dict(D=74, ext=76, y=-6, tilt=3, yaw=2)):
        M._reset_state()
        assert run(rest() + [hood(**kw)] * 30) == (1.0, True), kw


def test_every_level_reachable():
    got = set()
    for s in (hood(), hood(25, 10), hood(45, 25), hood(60, 10, y=15), hood(70, 5)):
        M._reset_state()
        got.add(run(rest() + [s] * 25)[0])
    assert got == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_geometry_hygiene():
    # 1.0 band is reachable physically: extension at D=65..78 stays under the hard stop with the mouth held
    assert 0.078 < M.STROKE
    assert 0.065 < M.TARGET_D < 0.078
    # green line sits just behind the foot's rear edge at the target
    assert abs((M.FOOT_REAR0 - M.TARGET_D) - M.GREEN_X) < 0.006


def test_blind_words():
    banned = re.compile(r"(/home/|/data/|team|universit|institut)", re.I)
    assert not banned.search(M.INSTRUCTION)
    for f in ("task.json", "README.md"):
        p = ROOT / f
        if p.exists():
            assert not banned.search(p.read_text()), f


def test_no_function_named_metrics():
    assert not hasattr(M, "metrics")
