"""Scoring, gate and hygiene checks without the simulator:  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests"""
import math
import pathlib
import re

import numpy as np
import pytest

import task as M
from partkit import PartState

ROOT = pathlib.Path(M.__file__).parent
P0 = np.array([M.PLATE_XY[0], M.PLATE_XY[1], M.ORIGIN_Z])


def plate(travel=0.0, shift=(0.0, 0.0), lift=0.0, tilt=0.0):
    """Plate state: carriage pulled travel mm, plate moved shift mm (x, y), lifted lift mm, tilted tilt degrees."""
    h = math.sqrt(0.5)
    q = np.array([h, 0.0, 0.0, h])
    if tilt:
        a = math.radians(tilt) / 2          # extra turn about the world y axis
        qx = np.array([0.0, math.sin(a), 0.0, math.cos(a)])
        x1, y1, z1, w1 = qx
        x2, y2, z2, w2 = q
        q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    z = np.zeros(3)
    return {"plate": PartState(pos=P0 + [shift[0] / 1000, shift[1] / 1000, lift / 1000], quat_xyzw=q, lin_vel=z,
                               ang_vel=z, low=z, high=z, sliders={"carriage": -travel / 1000})}


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
    return [plate()] * n


def band_hold(travel=62, shift=(0, 0), n=25):
    return [plate(travel, shift)] * n


def test_start_scores_zero():
    assert run(rest(60)) == (0.0, False)


def test_flat_plate_has_zero_tilt():
    assert M._tilt_deg(plate()["plate"]) < 1e-6


def test_engaged_025():
    assert run(rest() + [plate(25)] * 20 + [plate(2)] * 30)[0] == 0.25


def test_engaged_boundary():
    assert run(rest() + [plate(19)] * 5)[0] == 0.0
    M._reset_state()
    assert run(rest() + [plate(21)] * 5)[0] == 0.25


def test_band_entered_050():
    assert run(rest() + [plate(56)] * 10 + [plate(5)] * 30)[0] == 0.5


def test_band_short_hold_is_050():
    assert run(rest() + band_hold(n=15) + [plate(5)] * 30)[0] == 0.5


def test_band_hold_075_when_not_released():
    assert run(rest() + band_hold(n=25) + [plate(40)] * 5)[0] == 0.75


def test_band_hold_needs_continuous_second():
    seq = rest() + band_hold(n=12) + [plate(45)] + band_hold(n=12) + [plate(5)] * 30
    assert run(seq)[0] == 0.5



def test_full_score_released():
    assert run(rest() + band_hold() + [plate(30)] * 5 + [plate(3)] * 20) == (1.0, True)


def test_full_needs_still():
    assert run(rest() + band_hold() + [plate(9.9)] * 2)[0] == 1.0
    M._reset_state()
    assert run(rest() + band_hold() + [plate(9.9)] * 2)[1] is False


def test_slow_release_stays_075():
    assert run(rest() + band_hold() + [plate(30)] * 70 + [plate(3)] * 20)[0] == 0.75


def test_release_not_to_10mm_is_075():
    assert run(rest() + band_hold() + [plate(15)] * 40)[0] == 0.75


def test_overload_gate_sticky():
    seq = rest() + band_hold() + [plate(85)] * 2 + [plate(3)] * 30
    assert run(seq) == (0.0, False)


def test_overload_boundary():
    assert run(rest() + band_hold() + [plate(79)] * 3 + [plate(3)] * 30)[0] == 1.0


def test_plate_slide_gate():
    seq = rest() + band_hold(shift=(16, 0)) + [plate(3)] * 30
    assert run(seq) == (0.0, False)


def test_plate_shift_between_5_and_15_blocks_075():
    assert run(rest() + band_hold(shift=(8, 0)) + [plate(3, (8, 0))] * 30)[0] == 0.5


def test_small_shift_still_full():
    assert run(rest() + band_hold(shift=(3, 1)) + [plate(3, (3, 1))] * 30) == (1.0, True)


@pytest.mark.parametrize("kw", [dict(lift=12), dict(tilt=25), dict(shift=(0, 20))])
def test_gates_zero(kw):
    seq = rest() + band_hold() + [plate(3, **kw)] * 3 + [plate(3)] * 30
    assert run(seq) == (0.0, False)


def test_gate_ignored_first_40_steps():
    assert run([plate(85)] * 5 + [plate()] * 40 + band_hold() + [plate(3)] * 30)[0] == 1.0


def test_zero_gate_no_pull_with_slide():
    assert run(rest() + [plate(0, (20, 0))] * 30)[0] == 0.0


def test_two_valid_solutions_both_full():
    # post pinched (plate ends ~1 mm forward) or post held with a fist (plate ends ~3 mm back)
    for shift in ((1.0, 0.0), (-3.0, 1.0)):
        M._reset_state()
        assert run(rest() + band_hold(shift=shift) + [plate(3, shift)] * 30) == (1.0, True)


def test_every_level_reachable():
    got = set()
    for tail in ([], [plate(25)] * 10, [plate(58)] * 5, band_hold() + [plate(30)] * 5,
                 band_hold() + [plate(3)] * 30):
        M._reset_state()
        got.add(run(rest() + tail)[0] if tail else run(rest())[0])
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
