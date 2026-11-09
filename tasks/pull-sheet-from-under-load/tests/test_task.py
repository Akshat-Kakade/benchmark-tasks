"""Scoring, gate and hygiene checks without the simulator:  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests"""
import math
import pathlib
import re

import numpy as np
import pytest

import task as M
from partkit import PartState

ROOT = pathlib.Path(M.__file__).parent
B0 = np.array([M.BLOCK_XY[0], M.BLOCK_XY[1], M.TOP + M.SHEET_T + M.BLOCK_SIZE / 2])
S0 = np.array([M.SHEET_XY[0], M.SHEET_XY[1], M.TOP + M.SHEET_T / 2])


def _quat(tilt_deg):
    a = math.radians(tilt_deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def ps(pos, tilt=0.0):
    z = np.zeros(3)
    return PartState(pos=np.asarray(pos, float), quat_xyzw=_quat(tilt), lin_vel=z, ang_vel=z, low=z, high=z)


def scene(dy=0.0, shift=(0.0, 0.0), tilt=0.0, lift=0.0):
    """Sheet pulled dy mm in +y; block moved shift (mm, x y), tilted, lifted lift mm."""
    return {"sheet": ps(S0 + [0, dy / 1000, 0]),
            "load": ps(B0 + [shift[0] / 1000, shift[1] / 1000, lift / 1000], tilt)}


@pytest.fixture(autouse=True)
def fresh():
    M._reset_state()


def run(seq):
    """Feed states one per control step; returns the last (score, success)."""
    out = None
    for s in seq:
        M._STATE["step"] += 1
        out = (M.score(s), M.success(s))
    return out


def hold(s, n=25):
    return [scene()] * 45 + [s] * n


def test_start_scores_zero():
    assert run(hold(scene())) == (0.0, False)


def test_engaged_025():
    assert run(hold(scene(dy=70)))[0] == 0.25


def test_partial_050_by_pull_distance():
    assert run(hold(scene(dy=110)))[0] == 0.5          # 110 pulled: still under the block (clear needs 135)


def test_partial_050_clear_but_block_dragged():
    assert run(hold(scene(dy=150, shift=(0, 40))))[0] == 0.5


def test_075_block_shifted_12_to_25():
    assert run(hold(scene(dy=170, shift=(0, 18))))[0] == 0.75


def test_075_block_tilted_5_to_20():
    assert run(hold(scene(dy=150, tilt=10)))[0] == 0.75


def test_full_score_and_hold():
    assert run(hold(scene(dy=150, shift=(0, 6)), n=25)) == (1.0, True)


def test_full_needs_the_hold_second():
    assert run(hold(scene(dy=150), n=10)) == (1.0, False)


def test_clear_boundary_is_5mm():
    # sheet slab's -y edge is 130 mm below its origin (y 40 -> -90); block +y face is at +40: gap = dy - 130 + ... = dy - 135 + 5
    # edge after pull: -90 + dy; gap = (-90 + dy) - 40 = dy - 130 mm; clear needs gap >= 5 -> dy >= 135
    assert run(hold(scene(dy=134)))[0] == 0.5
    M._reset_state()
    assert run(hold(scene(dy=136)))[0] == 1.0


@pytest.mark.parametrize("kw", [dict(tilt=25), dict(shift=(0, 85)), dict(lift=12)])
def test_zero_gate_is_sticky(kw):
    seq = [scene()] * 45 + [scene(dy=150, **kw)] * 3 + [scene(dy=150)] * 30
    assert run(seq) == (0.0, False)


def test_gate_ignored_during_first_40_steps():
    seq = [scene()] + [scene(lift=15)] * 10 + [scene(dy=150)] * 60
    assert run(seq)[0] == 1.0


def test_two_valid_solutions_both_full():
    # fist brace on the +y face: block ends ~4 mm toward -y; pinch on the x sides: ~2 mm toward +y
    for shift in ((0.0, -4.0), (0.5, 2.0)):
        M._reset_state()
        assert run(hold(scene(dy=160, shift=shift), n=30)) == (1.0, True)


def test_lifting_the_block_off_scores_zero():
    assert run([scene()] * 45 + [scene(lift=30)] * 5 + [scene(dy=160)] * 30)[0] == 0.0


def test_shoving_block_off_sheet_scores_zero():
    assert run([scene()] * 45 + [scene(shift=(0, 150))] * 30)[0] == 0.0


def test_every_level_reachable():
    got = set()
    for s in [scene(), scene(dy=70), scene(dy=110), scene(dy=170, shift=(0, 18)), scene(dy=150)]:
        M._reset_state()
        got.add(run(hold(s))[0])
    assert got == {0.0, 0.25, 0.5, 0.75, 1.0}


def test_blind_words():
    banned = re.compile(r"(/home/|/data/|team|universit|institut)", re.I)
    assert not banned.search(M.INSTRUCTION)
    for f in ("task.json", "README.md"):
        p = ROOT / f
        if p.exists():
            assert not banned.search(p.read_text()), f
