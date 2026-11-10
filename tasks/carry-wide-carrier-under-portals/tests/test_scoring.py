"""Scoring boundaries, every partial level, zero gate, two solutions, blind words. Run: python -m pytest tests -q (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def qx(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def world(x, y=0.0, dz=0.0, roll=0.0, yaw=0.0):
    z = np.zeros(3)
    q = qx(roll) if roll else qz(yaw)
    return {"carrier": PartState(pos=np.array([x, y, T.REST_Z + dz], float), quat_xyzw=q, lin_vel=z, ang_vel=z, low=z, high=z)}


def run(seq):
    memo = {"step": 0, "scored": None, "last": (0.0, False), "past1": False, "past2": False, "gate": False, "held": 0, "max_tilt": 0.0}
    out = None
    for w in seq:
        out = T.classify(w, memo)
    return out


GX, GY = T.TARGET


def through(**end):
    return [world(0.21), world(0.21, 0.025, 0.015), world(0.34, 0.0, 0.015), world(0.43, -0.025, 0.015), world(**end)]


def test_start_zero():
    assert run([world(0.21)])[0] == 0.0


def test_025_boundary():
    assert run([world(0.334, 0.025, 0.015)])[0] == 0.0
    assert run([world(0.336, 0.025, 0.015)])[0] == 0.25


def test_05_boundary_and_needs_past1():
    assert run([world(0.21), world(0.34, 0.0, 0.015), world(0.421, -0.025, 0.015)])[0] == 0.5
    assert run([world(0.21), world(0.34, 0.0, 0.015), world(0.419, -0.025, 0.015)])[0] == 0.25


def test_075_levels():
    lv, ok, _ = run(through(x=GX, y=0.02, dz=0.0, yaw=10.0))
    assert lv == 0.75 and not ok
    lv, ok, _ = run(through(x=GX, y=0.0, dz=0.012))
    assert lv == 0.75 and not ok                      # not resting: height 12 mm
    lv, ok, _ = run(through(x=GX, y=0.0, dz=0.0, roll=10.0))
    assert lv == 0.75 and not ok


def test_05_far_from_target():
    assert run(through(x=0.43, y=-0.025, dz=0.0))[0] == 0.5
    assert run(through(x=GX, y=0.0, dz=0.0, roll=14.0))[0] == 0.5
    assert run(through(x=GX, y=0.0, dz=0.0, yaw=14.0))[0] == 0.5


def test_1_0_and_boundaries():
    lv, ok, _ = run(through(x=GX, y=0.0, dz=0.0))
    assert lv == 1.0 and ok
    assert run(through(x=GX + 0.011, y=0.0, dz=0.005, yaw=7.0))[0] == 1.0
    assert run(through(x=GX + 0.013, y=0.0, dz=0.0))[0] == 0.75
    assert run(through(x=GX, y=0.0, dz=0.007))[0] == 0.75
    assert run(through(x=GX, y=0.0, dz=0.0, roll=9.0))[0] == 0.75


def test_dragging_does_not_score():
    seq = [world(0.21), world(0.34, 0.0, 0.005), world(0.43, -0.02, 0.005), world(GX, 0.0, 0.0)]
    assert run(seq)[0] == 0.0
    assert run([world(0.336, 0.02, 0.0095)])[0] == 0.0 and run([world(0.336, 0.02, 0.0105)])[0] == 0.25


def test_gate_tilt_latches():
    seq = [world(0.21), world(0.34, 0.0, 0.015, roll=22.0)] + through(x=GX, y=0.0, dz=0.0)
    lv, ok, _ = run(seq)
    assert lv == 0.0 and not ok


def test_gate_fell_off_table():
    seq = through(x=GX, y=0.0, dz=0.0)[:-1] + [world(0.3, 0.5, -0.30)]
    assert run(seq)[0] == 0.0


def test_two_solutions():
    a = through(x=GX, y=0.0, dz=0.0)
    b = [world(0.21), world(0.21, 0.025, 0.020), world(0.34, 0.0, 0.020), world(0.43, -0.025, 0.020),
         world(GX + 0.008, 0.008, 0.002, yaw=4.0)]
    for s in (a, b):
        lv, ok, _ = run(s)
        assert lv == 1.0 and ok


def test_geometry_slots_fit():
    c = T.CARRIER
    bx, by = T.BLOCK[0], T.BLOCK[1]
    assert T.SLOT_W >= by + 0.03                       # 15 mm per side
    for (x, cy) in T.PORTALS:
        assert abs(cy) + by / 2 <= T.SLOT_W / 2 + 0.06
    # stems and open fingers stay outside the guard plates
    assert T.STEM_Y - 0.047 > T.GUARD_HALF_Y + 0.02
    # plate top at rest is below the guard, and the block top is above it
    assert T.TOP + T.PLATE[2] < T.GUARD_Z < T.TOP + T.PLATE[2] + T.BLOCK[2]
    # lateral shift between slots fits between the guard faces
    assert (T.PORTALS[1][0] - T.GUARD_T / 2 - bx / 2) - (T.PORTALS[0][0] + T.GUARD_T / 2 + bx / 2) >= 0.025


def test_no_paths_or_urls():
    words = re.compile(r"/home/|/Users/|https?://|@[a-z]+\.", re.I)
    for f in ("task.py", "demo.py", "README.md", "task.json"):
        p = pathlib.Path(__file__).resolve().parents[1] / f
        if p.exists():
            assert not words.search(p.read_text()), f
