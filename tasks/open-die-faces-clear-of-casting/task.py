"""On the station table: two die halves stand either side of a grey casting. Each half is a heavy grey base carrying a die
face (blue on the left, orange on the right) on a short slider that a spring (60 N/m, modelled in the physics engine) pulls
back toward the casting; a long bar carries the face out to a grip tab. Both tabs must be pulled outward at the same
time, to the clearance marks, and held, without disturbing the casting. One arm cannot do it: the other face springs back.

partkit sliders run along the part's local z axis, so each base's body frame is turned: local z = world -y (left base) or
world +y (right base), and on_reset writes that pose. The slider's travel is therefore the face's outward travel.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Slider, on_reset, on_step, part, write_pose  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Pull the blue left die face left and the orange right die face right until both clear the grey casting "
               "by the marked distance; keep the casting in place.")

GREY = (0.55, 0.57, 0.60)
DARK = (0.30, 0.31, 0.33)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)
GREEN = (0.10, 0.70, 0.20)
RED = (0.85, 0.10, 0.08)
WHITE = (0.95, 0.95, 0.95)

CX = 0.29                       # x of the die, casting on the centreline y = 0
BASE_T = 0.015
BASE_W = 0.155                  # each base covers 0 .. 0.155 outward of the centreline
BASE_OUT = BASE_W / 2           # body origin, outward distance from the centreline
ORIGIN_Z = TOP + BASE_T / 2
FACE_OUT0 = 0.063               # face centre, outward distance, at rest
FACE_W, FACE_L, FACE_H = 0.045, 0.14, 0.055      # outward thickness, x length, height
FACE_Z = TOP + BASE_T + 0.001 + FACE_H / 2
TAB_OUT0 = 0.205
TAB_Z = 0.835                   # tab (post) centre; the pinch is at GRIP_Z
GRIP_Z = 0.85
TAB = (0.040, 0.020, 0.060)     # x, outward length (pinch width), z
CAST_R, CAST_H = 0.030, 0.060
K_SPRING = 60.0                 # N/m
STROKE = 0.075                  # hard stop
MARK_OUT = FACE_OUT0 - FACE_W / 2 + 0.065 - 0.002     # witness mark just short of the inner edge at 65 mm travel


def side_frame(sign):
    """Quaternion xyzw of a base frame, and the maps world (dx, out, dz) -> local and world size -> local size."""
    c = math.sqrt(0.5)
    quat = (c, 0.0, 0.0, c) if sign > 0 else (0.0, c, c, 0.0)
    xs = 1.0 if sign > 0 else -1.0

    def L(dx, out, dz):
        return (xs * dx, dz, -out)

    def S(sx, so, sz):
        return (sx, sz, so)

    return quat, L, S


def _die(name, sign, color):
    quat, L, S = side_frame(sign)
    o = BASE_OUT
    boxes = [
        (L(0, 0, 0), S(0.230, BASE_W, BASE_T), GREY),
        (L(0, MARK_OUT - o, BASE_T / 2 + 0.0005), S(0.200, 0.006, 0.001), GREEN),         # 65 mm clearance witness mark
    ]
    fz = FACE_Z - ORIGIN_Z
    face = L(0, FACE_OUT0 - o, fz)
    bar_z = 0.8125 - ORIGIN_Z
    sl = Slider(
        name="face", rest=face, travel=STROKE, rate=K_SPRING, damping=5.0, mass=0.16, friction=0.1,
        boxes=((face, S(FACE_L, FACE_W, FACE_H), color),
               (L(0, (FACE_OUT0 + TAB_OUT0) / 2 - 0.0025 - o, bar_z), S(0.030, TAB_OUT0 - FACE_OUT0 - 0.005, 0.015), color),
               (L(0, TAB_OUT0 - o, TAB_Z - ORIGIN_Z), S(*TAB[:1], TAB[1], TAB[2]), color)))
    return part(name, pos=(CX, sign * BASE_OUT, ORIGIN_Z + 0.16), mass=3.0, color=GREY, friction=1.0,
                boxes=boxes, sliders=(sl,)), quat


_DL, QUAT_L = _die("die_l", +1, BLUE)
_DR, QUAT_R = _die("die_r", -1, ORANGE)
CASTING = part("casting", pos=(CX, 0.0, TOP + BASE_T + CAST_H / 2 + 0.001), mass=0.12, color=DARK, friction=0.8,
               boxes=[((0, 0, -CAST_H / 2 + 0.002), (0.03, 0.03, 0.004), DARK)],
               cylinders=[((0, 0, 0), CAST_R, CAST_H, DARK)])
OBJECTS = [_DL, _DR, CASTING]

# -- scoring thresholds ---------------------------------------------------------------------------------------------------
GATE_FROM_STEP = 40
SETTLE_STEPS = 10         # a teleported part pops a little in its first steps: baselines are taken at step 10
GATE_CAST_MM = 12.0
GATE_TILT_DEG = 8.0
GATE_BASE_MM = 15.0
LEVELS = (  # min travel (mm), max travel difference (mm)
    (20.0, None), (40.0, 30.0), (55.0, 20.0), (65.0, 12.0))
CLEAR_MM = 35.0
HOLD_STEPS = 20           # 1 s of the full pose ends the run with success

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), tripped=False, start=None, streak=0, latched=False)
    if environment is not None:
        write_pose(environment, "die_l", (CX, float(BASE_OUT), float(ORIGIN_Z)), tuple(float(v) for v in QUAT_L))
        write_pose(environment, "die_r", (CX, float(-BASE_OUT), float(ORIGIN_Z)), tuple(float(v) for v in QUAT_R))
        write_pose(environment, "casting", (CX, 0.0, float(TOP + BASE_T + CAST_H / 2 + 0.001)), (0.0, 0.0, 0.0, 1.0))


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _travel_mm(state) -> float:
    return -1000.0 * float(state.sliders.get("face", 0.0))


def _face_y(state, sign) -> float:
    """World y of the face centre from the base pose and the slider reading."""
    _, L, _ = side_frame(sign)
    rest = np.array(L(0, FACE_OUT0 - BASE_OUT, FACE_Z - ORIGIN_Z))
    rest[2] += float(state.sliders.get("face", 0.0))
    return float(state.to_world(rest)[1])


def _clearances_mm(objects) -> tuple[float, float]:
    """Gap from each face's inner edge to the casting's outer surface on its side (mm)."""
    cy = float(objects["casting"].pos[1])
    yl, yr = _face_y(objects["die_l"], +1), _face_y(objects["die_r"], -1)
    return 1000 * ((yl - FACE_W / 2) - (cy + CAST_R)), 1000 * ((cy - CAST_R) - (yr + FACE_W / 2))


def _tilt_deg(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def _level(tl, tr, cl, cr) -> float:
    lo, diff = min(tl, tr), abs(tl - tr)
    if lo >= 65 and diff <= 12 and min(cl, cr) >= CLEAR_MM:
        return 1.0
    if lo >= 55 and diff <= 20:
        return 0.75
    if lo >= 40 and diff <= 30:
        return 0.5
    if lo >= 20:
        return 0.25
    return 0.0


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    step = _STATE["step"]
    if step < SETTLE_STEPS:
        _STATE["last"] = (0.0, False)
        return _STATE["last"]
    dl, dr, c = objects["die_l"], objects["die_r"], objects["casting"]
    if _STATE["start"] is None:
        _STATE["start"] = (np.asarray(c.pos, float).copy(), np.asarray(dl.pos, float).copy(),
                           np.asarray(dr.pos, float).copy())
    c0, l0, r0 = _STATE["start"]
    cast_mm = 1000 * float(np.linalg.norm(np.asarray(c.pos, float)[:2] - c0[:2]))
    base_mm = 1000 * max(float(np.linalg.norm(np.asarray(d.pos, float)[:2] - p0[:2])) for d, p0 in ((dl, l0), (dr, r0)))
    tl, tr = _travel_mm(dl), _travel_mm(dr)
    cl, cr = _clearances_mm(objects)
    if step > GATE_FROM_STEP and (cast_mm > GATE_CAST_MM or _tilt_deg(c) > GATE_TILT_DEG or base_mm > GATE_BASE_MM
                                  or min(cl, cr) < 0.0):
        _STATE["tripped"] = True
    lvl = _level(tl, tr, cl, cr)
    _STATE["streak"] = _STATE["streak"] + 1 if lvl >= 1.0 else 0
    if _STATE["streak"] >= HOLD_STEPS:
        _STATE["latched"] = True
    if _STATE["tripped"]:
        result = (0.0, False)
    elif _STATE["latched"]:
        result = (1.0, True)
    else:
        result = (lvl, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Both faces pulled outward >= 65 mm, travel difference <= 12 mm, each inner face >= 35 mm clear of the casting,
    held for 1 s; casting not moved > 12 mm or tilted > 8 deg, neither base moved > 15 mm, no face across the casting."""
    return _update(objects)[1]


def score(objects) -> float:
    """Highest tier satisfied by the current pose: 0.25 both faces out >= 20 mm; 0.5 both >= 40 mm, difference <= 30 mm;
    0.75 both >= 55 mm, difference <= 20 mm; 1.0 both >= 65 mm, difference <= 12 mm, each inner face >= 35 mm from the
    casting (held 1 s). Zero if, after step 40, the casting moves > 12 mm or tilts > 8 deg, a base moves > 15 mm, or a
    face touches the casting."""
    return _update(objects)[0]
