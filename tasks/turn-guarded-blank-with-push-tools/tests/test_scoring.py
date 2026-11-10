"""Scoring boundaries, every partial level, zero gate, two valid solutions, geometry sanity, blind words.
Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def qx(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def st(pos, quat):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=quat, lin_vel=z, ang_vel=z, low=z, high=z)


def tool(blank_yaw, sign, back=0.09):
    """Tool whose paddle sits `back` m from the blank centre along the blank's own -x (blue, sign +1) / +x (orange)."""
    a = math.radians(blank_yaw)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = R @ np.array([-sign * back, sign * T.PAD_A]) + np.array([T.CX, T.CY])
    return st((p[0], p[1], T.TOOL_Z), qz(blank_yaw + (0 if sign > 0 else 180)))


def world(yaw=-45.0, dx=0.0, dy=0.0, tilt=0.0, back=0.09, far=True):
    q = qz(yaw)
    if tilt:
        q = qx(tilt)
    w = {"blank": st((T.CX + dx, T.CY + dy, T.REST_B), q), "blue_tool": tool(yaw, 1, back), "orange_tool": tool(yaw, -1, back)}
    if far:  # handles out of the guard: push the tools out along their own axes (stems are at |y| ~ 0.2 already)
        pass
    return w


def lv(w):
    return T.classify(w)[0]


def test_start_zero():
    w = world(yaw=0.0, back=0.099)
    assert lv(w) == 0.0 and not T.classify(w)[1]


def test_full_and_hold():
    l, ok, m = T.classify(world())
    assert l == 1.0 and ok, m
    T._reset_state()
    for _ in range(T.HOLD_STEPS + 2):
        T._STATE["step"] += 1
        s = T.success(world())
    assert s


def test_two_valid_solutions():
    assert lv(world(yaw=-40.0, dx=0.008, dy=-0.006)) == 1.0
    assert lv(world(yaw=-51.0, dx=-0.005, dy=0.010)) == 1.0


def test_1_0_boundaries():
    assert lv(world(yaw=-36.0)) == 0.75 and lv(world(yaw=-54.0)) == 0.75
    assert lv(world(yaw=-37.5)) == 1.0 and lv(world(yaw=-52.5)) == 1.0
    assert lv(world(dx=0.017)) == 0.75 and lv(world(dx=0.013)) == 1.0
    assert lv(world(back=0.07)) == 0.75          # paddle still on the blank: tips not clear


def test_075_boundaries():
    assert lv(world(yaw=-33.5, dx=0.028)) == 0.75 and lv(world(yaw=-56.5, dx=0.028)) == 0.75
    assert lv(world(yaw=-32.0, dx=0.028)) == 0.5 and lv(world(yaw=-58.0)) == 0.5
    assert lv(world(dx=0.032)) == 0.5


def test_05_boundaries():
    assert lv(world(yaw=-26.0, dx=0.040)) == 0.5
    assert lv(world(yaw=-24.0, dx=0.040)) == 0.25
    assert lv(world(yaw=-30.0, dx=0.036, dy=0.036)) == 0.25      # drift 51 mm: only 0.25
    assert lv(world(yaw=-45.0, dx=0.047)) == 0.0                 # corner reaches the guard


def test_025_boundaries():
    assert lv(world(yaw=-13.0)) == 0.25
    assert lv(world(yaw=-11.0)) == 0.0
    assert lv(world(yaw=+30.0)) == 0.0           # turned the wrong way


def test_zero_gates():
    assert lv(world(tilt=9.0)) == 0.0
    assert lv(world(dx=0.09)) == 0.0             # centre far off (and corner at the guard)
    w = world()
    w["blue_tool"] = st((0.30, -0.10, T.TOOL_Z), qz(0))   # a handle stem inside the guard footprint
    assert lv(w) == 0.0


def test_corner_gate():
    # at yaw -45 the corner is 85 mm out: ok; a 35 mm shift pushes a corner to the rail line
    assert lv(world(yaw=-45.0, dx=0.020)) == 0.75
    assert lv(world(yaw=-45.0, dx=0.045)) == 0.0


def test_geometry_sanity():
    # tool initial: paddle face 20 mm from the blank face, handle stem outside the guard
    for n, o in (("blue_tool", T.BLUE_T), ("orange_tool", T.ORANGE_T)):
        pass
    w = {"blank": st((T.CX, T.CY, T.REST_B), qz(0)), "blue_tool": st(T.BLUE_T.pos, qz(0)),
         "orange_tool": st(T.ORANGE_T.pos, qz(180))}
    m = T._metrics(w)
    assert m["stem_in"] > T.HANDLE_KEEP_OUT + 0.04, m
    face_gap = (np.array(w["blank"].to_local(w["blue_tool"].to_world((T.PAD_FACE, 0, 0)))))[0] + T.H
    assert abs(abs(face_gap) - T.GAP0) < 1e-6
    # paddle z overlaps the blank's thickness and clears the table
    assert T.TOOL_Z - T.PAD[2] / 2 >= T.TOP and T.TOOL_Z + T.PAD[2] / 2 > T.TOP + 0.015
    # guard: rail underside above the tool paddles, blank corners at -45 deg inside the clear area
    assert T.ROOF_Z > T.TOP + 0.08
    assert T.H * math.sqrt(2) < T.CLEAR_HALF - 0.02


def test_tool_balance():
    """COM of the tool (uniform density) near the pinch axis in y (weight torque on the pinch is small)."""
    boxes = T.BLUE_T.boxes
    vol = np.array([b[1][0] * b[1][1] * b[1][2] for b in boxes])
    cy = sum(v * b[0][1] for v, b in zip(vol, boxes)) / vol.sum()
    assert abs(cy - T.STEM_Y) < 0.03, cy


def test_blind_words():
    here = pathlib.Path(T.__file__).resolve().parent
    txt = T.__doc__ + T.INSTRUCTION + (here / "task.py").read_text() + (here / "demo.py").read_text()
    if (here / "README.md").exists():
        txt += (here / "README.md").read_text()
    if (here / "task.json").exists():
        txt += (here / "task.json").read_text()
    txt = txt.replace("Dream-1", "").replace("--policy dream", "").replace("dream_run", "").replace("Dream", "")
    assert not re.search(r"(?i)robocurve|/home/|\bteam\b|\.brev|claude|anthropic", txt)
