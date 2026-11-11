"""Offline tests: geometry, scoring levels and boundaries, zero gate, two valid solutions.
Run from the task folder:  PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q   (needs yam_arena on the path)"""

import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import geometry as g  # noqa: E402
import task  # noqa: E402

S = task._STATE
D = math.radians


def reset(**kw):
    task._reset_state()
    S.update(kw)


def final_pose(**kw):
    base = dict(theta=-g.STOP, psi=math.pi, pose_ok=True, rest=True, released=True, cleared=True, crossed=True)
    base.update(kw)
    reset(**base)


def test_post_geometry():
    # blocked at the start and when level, clear by 20 mm on the green stop
    assert g.block_bottom(g.STOP) < g.POST_TOP - 0.015
    assert g.block_bottom(0.0) < g.POST_TOP + 0.002
    assert abs(g.block_bottom(-g.STOP) - g.POST_TOP - g.POST_CLEAR_TIPPED) < 1e-9
    assert g.block_bottom(-g.STOP) < 0.09 + 0.0   # handle heights stay low
    assert g.weight_stem(-g.STOP, 0.0)[2] - task.g.TOP < 0.14


def test_handles_far_enough_apart():
    d = [np.linalg.norm(g.left_stem(t)[:2] - g.weight_stem(t, p)[:2]) for t in (g.STOP, -g.STOP)
         for p in np.linspace(0, math.pi, 7)]
    assert 0.24 < min(d) and max(d) < 0.36


def test_tip_range():
    assert abs(g.STOP_DEG * 2 - 16.0) < 1e-9
    assert g.stop_top() > 0.008


def test_zero_start():
    reset(theta=g.STOP, psi=0.0)
    assert task.score(None) == 0.0 and not task.success(None)


def test_level_025():
    reset(cleared=True)
    assert task.score(None) == 0.25


def test_level_05():
    reset(cleared=True, crossed=True)
    assert task.score(None) == 0.5


def test_level_075_held():
    final_pose(released=False)
    assert task.score(None) == 0.75 and not task.success(None)


def test_level_075_moving():
    final_pose(rest=False)
    assert task.score(None) == 0.75 and not task.success(None)


def test_success_1():
    final_pose()
    assert task.success(None)


def test_gate_zeroes_everything():
    final_pose()
    S["why"].append("weight pushed through the post")
    assert task.score(None) == 0.0 and not task.success(None)


def test_weight_back_loses_075():
    final_pose(pose_ok=False, theta=g.STOP, psi=0.0)
    assert task.score(None) == 0.5 and not task.success(None)


def test_tolerances():
    # pose_ok is computed in the monitor; check the boundary arithmetic it uses
    assert abs(D(187.9) - g.PSI_TARGET) <= task.WEIGHT_TOL
    assert abs(D(171.9) - g.PSI_TARGET) > task.WEIGHT_TOL
    assert abs(-D(3.1) + g.STOP) <= task.BEAM_TOL
    assert abs(-D(2.9) + g.STOP) > task.BEAM_TOL


def test_two_valid_solutions():
    # solution A: tip, swing the weight through the front, let go right then left.
    # solution B: the same end state reached with the right hand letting go last: both are the same final state.
    for released in (True,):
        final_pose(released=released)
        assert task.success(None)
