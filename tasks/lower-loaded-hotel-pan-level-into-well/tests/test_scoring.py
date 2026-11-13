"""Scoring boundaries, every partial level, zero gates, two valid solutions, hold, blind words. Run: python -m pytest tests -q (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def quat_rx(deg):
    b = math.radians(deg) / 2
    return np.array([math.sin(b), 0, 0, math.cos(b)])


def quat_ry(deg):
    b = math.radians(deg) / 2
    return np.array([0, math.sin(b), 0, math.cos(b)])


def pan(dx=0.0, dy=0.0, z=T.SEAT_Z, tilt=0.0, v=0.0, w=0.0, axis="x"):
    z0 = np.zeros(3)
    q = quat_rx(tilt) if axis == "x" else quat_ry(tilt)
    return PartState(pos=np.array([T.WELL_X + dx, dy, z]), quat_xyzw=q, lin_vel=np.array([0, 0, -v]), ang_vel=np.array([w, 0, 0]),
                     low=z0, high=z0)


def grips(p, open_=True, near=False, q=0.0):
    """Synthetic grip readings: TCPs 0.1 m beside the handles (released) or on them (holding)."""
    out = {}
    for tag, s in (("A", 1), ("B", -1)):
        tcp = p.to_world((0.0, s * T.HANDLE_Y, T.BAR_Z))
        if not near:
            tcp = tcp + np.array([0.0, s * 0.10, 0.10])
        out[tag] = (tcp, 0.048 if open_ else 0.084, q)
    return out


def memo():
    m = T._fresh()
    m["step"] = 10
    return m


def lv(p, g=None, m=None):
    return T.classify(p, m or memo(), g if g is not None else grips(p))[0]


def test_geometry_numbers():
    assert abs(T.SEAT_Z - (T.TOP + 0.003)) < 1e-9
    assert abs(T.OPEN_X - T.PAN_X - 0.012) < 1e-9 and T.FL_X > T.OPEN_X and T.FL_Y > T.OPEN_Y     # flange wider than the pit


def test_start_zero():
    p = pan(dx=T.PAN_X0 - T.WELL_X, z=T.START_Z)
    assert lv(p) == 0.0


def test_full_and_variants():
    p = pan()
    l, ok, m = T.classify(p, memo(), grips(p))
    assert l == 1.0 and ok, m
    for kw in (dict(dx=0.005), dict(dy=-0.006), dict(dx=-0.003, dy=0.004), dict(z=T.SEAT_Z + 0.0015), dict(tilt=0.3)):
        q = pan(**kw)
        assert lv(q, grips(q)) == 1.0, kw


def test_two_solutions_either_handle_roles_and_side():
    a, b = pan(dx=0.004, dy=0.005), pan(dx=-0.004, dy=-0.005)
    assert lv(a, grips(a)) == lv(b, grips(b)) == 1.0
    g = grips(a)
    swapped = {"A": g["B"], "B": g["A"]}                    # the left arm holding the right handle is also fine
    assert lv(a, swapped) == 1.0


def test_boundaries_1_0():
    assert lv(pan(z=T.SEAT_Z + 0.0025)) == 0.5              # flange 2.5 mm proud: not seated
    assert lv(pan(tilt=1.0)) == 0.5                         # a corner > 2 mm proud: not seated
    assert lv(pan(dx=0.009)) == 0.5                         # offset over 8 mm: not seated (jam)


def test_075_still_held():
    p = pan()
    assert lv(p, grips(p, near=True, open_=False)) == 0.75   # gripper still on the handles
    assert lv(p, grips(p, near=True, open_=True)) == 0.75    # open but within 40 mm
    assert lv(p, grips(p, q=0.2)) == 0.75                    # arms still moving
    assert lv(pan(w=0.5)) == 0.75


def test_05_level():
    p = pan(z=T.SEAT_Z + 0.010)                             # underside 6.. mm... body in the pit, flange 10 mm proud
    assert lv(p) == 0.5
    assert lv(pan(z=T.TOP + T.COUNTER_H - 0.003)) == 0.25    # underside only 3 mm below the counter top: not in
    assert lv(pan(dx=0.019, z=T.SEAT_Z + 0.015)) == 0.5
    assert lv(pan(dx=0.025, z=T.SEAT_Z + 0.015)) == 0.0


def test_025_over_well():
    assert lv(pan(z=T.TOP + T.COUNTER_H + T.PAN_H - T.FL_T + 0.02)) == 0.25
    assert lv(pan(dx=0.025, z=T.TOP + 0.15)) == 0.0          # not over the well
    assert lv(pan(z=T.TOP + T.COUNTER_H + 0.037 - 0.001 + 0.0)) in (0.0, 0.25)


def test_gates():
    m = memo()
    assert T.classify(pan(tilt=9), m, grips(pan()))[0] == 0.0          # tilt
    assert m["gate"]
    assert T.classify(pan(), m, grips(pan()))[0] == 0.0                # latched: seated afterwards still 0
    m = memo()
    assert T.classify(pan(v=0.6), m, grips(pan()))[0] == 0.0           # dropped on the counter
    assert lv(pan(z=T.TOP - 0.05)) == 0.0                              # off the table


def test_tilt_8_boundary_and_pitch_axis():
    m = memo()
    assert T.classify(pan(tilt=8.5, axis="y"), m, grips(pan()))[0] == 0.0
    assert lv(pan(tilt=7.5, z=T.SEAT_Z + 0.03)) != 0.0


def test_hold_counter():
    T._reset_state()
    for i in range(T.HOLD_STEPS + 2):
        T._STATE["step"] += 1
        s = T.classify(pan(), T._STATE, grips(pan()))
        T._STATE["held"] = T._STATE["held"] + 1 if s[1] else 0
    assert T._STATE["held"] >= T.HOLD_STEPS
    T._reset_state()
    for i in range(T.HOLD_STEPS - 3):
        T._STATE["step"] += 1
        s = T.classify(pan(), T._STATE, grips(pan()))
        T._STATE["held"] = T._STATE["held"] + 1 if s[1] else 0
    assert T._STATE["held"] < T.HOLD_STEPS


def test_released_geometry():
    p = pan()
    assert T.released(p, grips(p)) and not T.released(p, grips(p, near=True)) and not T.released(p, grips(p, open_=False))
    assert not T.released(p, None)


def test_flange_corner_heights():
    h = T.flange_heights(pan())
    assert max(abs(x - T.LEDGE_Z) for x in h) < 1e-9
    h = T.flange_heights(pan(tilt=3.0))
    assert max(h) - min(h) > 0.005                                     # a 3 deg tilt shows in the corners


def test_blind_words():
    here = pathlib.Path(T.__file__).parent
    txt = T.__doc__ + "".join(p.read_text() for p in (here / "task.py", here / "README.md") if p.exists())
    assert not re.search(r"(?i)robocurve|/home/|\bteam\b|\.brev", txt.replace("Dream-1", "").replace("--policy dream", "").replace("dream_run", ""))


def test_instruction_gives_no_method():
    for w in ("level", "both", "two hands", "tilt", "handle"):
        assert w not in T.INSTRUCTION.lower(), w


def test_demo_reads_no_simulator_state():
    src = (pathlib.Path(__file__).resolve().parents[1] / "demo.py").read_text()
    for bad in ("states(", "object_states", "_env", "body_pos", "root_pos"):
        assert bad not in src, bad
