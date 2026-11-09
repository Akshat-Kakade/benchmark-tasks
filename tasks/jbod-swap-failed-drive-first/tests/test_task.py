"""Offline tests of the drive-drawer task's state machine and scoring (no simulator).

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
"""

import json
import math
import pathlib
import re

import numpy as np
import pytest

import task as T
from yam_arena.task_file import ObjectState

HERE = pathlib.Path(__file__).resolve().parents[1]
UP = (0.0, 0.0, 0.0, 1.0)
CLEAR = [(0.0, 0.4, 1.0), (0.0, -0.4, 1.0)]
NEAR = [(T.DRAWER_X, 0.0, T.FLOOR_Z + T.DRIVE_H + 0.02)]


class World:
    def __init__(self):
        T._reset()
        T.HANDS_OVERRIDE = CLEAR
        self.p = {}
        self.q = {n: UP for n in T.DRIVES}
        for k, n in enumerate(T.ORIGINALS):
            self.p[n] = (T.DRAWER_X, T.SLOT_Y[k], T.FLOOR_Z + T.DRIVE_H / 2 + 0.001)
        for i, n in enumerate(T.SPARES):
            self.p[n] = (*T.TRAY_XY[i], T.TOP + T.DRIVE_H / 2 + 0.001)
        self.last = 0.0

    def objs(self):
        z = np.zeros(3)
        return {n: ObjectState(pos=np.array(self.p[n], float), quat_xyzw=np.array(self.q[n], float), lin_vel=z,
                                 ang_vel=z, low=z, high=z) for n in T.DRIVES}

    def run(self, seconds=0.05):
        for _ in range(max(1, round(seconds * T.CONTROL_HZ))):
            self.last = T.score(self.objs())
        return self.last

    def lift(self, n):
        x, y, _ = self.p[n]
        self.p[n] = (x, y, T.FLOOR_Z + T.DRIVE_H / 2 + 0.15)
        return self.run()

    def to_bin(self, n, dy=0.0):
        self.p[n] = (T.BIN_XY[0], T.BIN_XY[1] + dy, T.BIN_FLOOR_Z + T.DRIVE_H / 2)
        return self.run()

    def seat(self, n, k):
        self.p[n] = (T.DRAWER_X, T.SLOT_Y[k], T.FLOOR_Z + T.DRIVE_H / 2 + 0.001)
        return self.run()

    def full(self, first="spare1", second="spare2", bin_dy=(0.04, -0.04)):
        self.lift("drive2"); self.to_bin("drive2", bin_dy[0])
        self.lift(first); self.seat(first, 1); self.run(T.REBUILD_S + 0.5)
        self.lift("drive3"); self.to_bin("drive3", bin_dy[1])
        self.lift(second); self.seat(second, 2)
        return self.run(T.REBUILD_S + 1.5)


@pytest.fixture(autouse=True)
def _clean():
    yield
    T.HANDS_OVERRIDE = None


def test_settle_scores_zero_and_no_gate():
    w = World()
    assert w.run(5.0) == 0.0
    a = T.array()
    assert not a.data_loss and not a.ok_pulled and a.lamps() == ["green", "red", "amber", "green"]


def test_every_level_reachable():
    w = World()
    w.lift("drive2")
    assert w.last == 0.0            # held in the air: not stowed
    assert w.to_bin("drive2") == 0.25
    w.lift("spare1"); w.seat("spare1", 1)
    assert w.last == 0.25 and T.array().lamps()[1] in ("white", None)
    assert w.run(T.REBUILD_S + 0.5) == 0.5
    w.lift("drive3"); assert not T.array().data_loss
    assert w.run() == 0.0            # drive3 held in the air: not stowed
    assert w.to_bin("drive3", -0.05) == 0.75
    w.lift("spare2"); w.seat("spare2", 2)
    assert w.run(1.0) == 0.75
    assert w.run(T.REBUILD_S) == 1.0
    assert T.success(w.objs())


def test_two_valid_solutions_both_full():
    assert World().full("spare1", "spare2", (0.04, -0.04)) == 1.0
    assert World().full("spare2", "spare1", (-0.06, 0.06)) == 1.0


def test_hands_near_drawer_block_full_score():
    w = World()
    w.full()
    T.HANDS_OVERRIDE = NEAR
    assert w.run() == 0.75


def test_failed_only_is_half():
    w = World()
    w.lift("drive2"); w.to_bin("drive2"); w.lift("spare1"); w.seat("spare1", 1)
    assert w.run(T.REBUILD_S + 2) == 0.5


def test_parallel_pull_is_data_loss():
    w = World()
    w.p["drive2"] = (T.DRAWER_X, T.SLOT_Y[1], T.FLOOR_Z + 0.2)
    w.p["drive3"] = (T.DRAWER_X, T.SLOT_Y[2], T.FLOOR_Z + 0.2)
    w.run()
    assert T.array().data_loss and T.array().lamps() == ["red"] * 4
    assert World().full() == 1.0  # sanity: a fresh world still works
    w = World()
    w.p["drive2"] = (T.DRAWER_X, T.SLOT_Y[1], T.FLOOR_Z + 0.2); w.p["drive3"] = (T.DRAWER_X, T.SLOT_Y[2], T.FLOOR_Z + 0.2)
    w.run()
    w.to_bin("drive2", 0.05); w.seat("spare1", 1); w.run(T.REBUILD_S + 1)
    w.to_bin("drive3", -0.05); w.seat("spare2", 2)
    assert w.run(T.REBUILD_S + 1) == 0.0


def test_amber_first_is_data_loss():
    w = World()
    w.lift("drive3")
    assert T.array().data_loss


def test_no_wait_is_data_loss():
    w = World()
    w.lift("drive2"); w.to_bin("drive2"); w.seat("spare1", 1); w.run(2.0)
    w.lift("drive3")
    assert T.array().data_loss
    w.to_bin("drive3", -0.05); w.seat("spare2", 2)
    assert w.run(T.REBUILD_S + 2) == 0.0


def test_ok_drive_pulled_is_zero_even_if_put_back():
    w = World()
    w.lift("drive1"); w.seat("drive1", 0)
    assert T.array().ok_pulled
    assert w.full() == 0.0


def test_drive_left_on_table_is_zero():
    w = World()
    w.full()
    w.p["drive2"] = (0.40, -0.30, T.TOP + T.DRIVE_Y / 2)  # lying on the table
    assert w.run() == 0.0


def test_seating_thresholds():
    w = World()
    o = w.objs()["drive1"]
    assert T.seated(o, 0)
    raised = ObjectState(pos=o.pos + [0, 0, 0.006], quat_xyzw=o.quat_xyzw, lin_vel=o.lin_vel, ang_vel=o.ang_vel,
                           low=o.low, high=o.high)
    assert not T.seated(raised, 0)
    a = math.radians(12)
    tilted = ObjectState(pos=o.pos, quat_xyzw=np.array([math.sin(a / 2), 0, 0, math.cos(a / 2)]),
                           lin_vel=o.lin_vel, ang_vel=o.ang_vel, low=o.low, high=o.high)
    assert not T.seated(tilted, 0)


def test_rebuild_resets_if_spare_unseated():
    w = World()
    w.lift("drive2"); w.to_bin("drive2"); w.seat("spare1", 1); w.run(6.0)
    w.lift("spare1"); w.seat("spare1", 1); w.run(6.0)
    assert T.array().status[1] == "rebuilding"


def _blind_words():
    """Identifying words from the environment (BLIND_WORDS, comma-separated) or a config file found above this folder."""
    import os
    raw = os.environ.get("BLIND_WORDS", "")
    for parent in HERE.parents:
        cfg = parent / "pipeline" / "config.env"
        if not raw and cfg.exists():
            for line in cfg.read_text().splitlines():
                if line.startswith("TEAM_WORDS="):
                    raw = line.split("=", 1)[1]
    return [w.strip().lower() for w in raw.split(",") if w.strip()]


def test_blind_words():
    words = _blind_words()
    for f in ["task.py", "demo.py", "README.md", "task.json", "partkit.py"]:
        p = HERE / f
        if p.exists():
            text = p.read_text().lower()
            for wd in words:
                assert not re.search(rf"\b{re.escape(wd)}\b", text), (f, wd)
    j = HERE / "task.json"
    if j.exists():
        json.loads(j.read_text())
