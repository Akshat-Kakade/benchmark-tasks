"""On the station table: a pipe joint made of two grey flanges. The upper flange is fixed on a column; the lower flange
hangs under it on a spring (100 N/m, modelled in the physics engine) so the joint is normally closed (gap about 0).
An orange blind plate (a 120 mm square, 12 mm thick, with a stem and a knob) lies on a white tray in front of the joint.
One hand must pull the lower flange down by its yellow lug to open a gap of at least 22 mm and hold it; the other hand
must lift the plate by its knob, slide it into that gap and centre it on the green mark; then the lower flange is eased
back up so the spring closes the joint on the plate (gap = plate thickness, 10-14 mm).

The joint is one partkit part (heavy, wide base plate): the lower flange is a prismatic slider along the part's z axis
(partkit sliders travel down), so no re-orientation is needed. The plate is a second, loose part.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Slider, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Isolate the pipe before the pump is opened: hold the lower grey flange down and slide the orange blind "
               "plate into the gap, stem out front, then ease the flange back so it grips the plate.")

GREY = (0.55, 0.57, 0.60)
DARK = (0.25, 0.26, 0.28)
YELLOW = (0.95, 0.80, 0.05)
ORANGE = (1.00, 0.50, 0.00)
WHITE = (0.95, 0.95, 0.95)
GREEN = (0.10, 0.70, 0.20)

# -- the joint (articulation root; origin = centre of the base plate, 5 mm above the table) -------------------------
JX, JY = 0.32, 0.0                      # flange centre (the green mark)
ORIGIN_Z = TOP + 0.005
UPPER_UNDER = TOP + 0.120               # underside of the upper flange (world z) = top of the closed lower flange
PLATE_T = 0.012
FLANGE_T = 0.020
FLANGE_HALF = 0.065                     # flanges are 130 mm in y; lower one 124 mm in x
K_SPRING = 100.0                        # N/m: 3.2 N at 32 mm
STROKE = 0.045


def L(dx, dy, dz_world):
    """World offset from the joint origin (world z given as height above the table top) -> part-frame vector."""
    return (dx, dy, dz_world - 0.005)


_LOW_Z = 0.120 - FLANGE_T / 2           # lower flange centre height above table, closed
JOINT_BOXES = [
    (L(0, 0, 0.005), (0.19, 0.16, 0.010), DARK),                                   # base plate
    (L(0.095, 0, 0.065), (0.04, 0.06, 0.130), GREY),                               # column behind the flanges
    (L(0.082, 0, 0.130), (0.036, 0.06, FLANGE_T), GREY),                           # bridge to the upper flange
    (L(0, 0, 0.130), (0.130, 0.130, FLANGE_T), GREY),                              # upper flange
    (L(0, 0, 0.1415), (0.030, 0.030, 0.003), GREEN),                               # centre mark on the upper flange
]
FLANGE = Slider(
    name="flange", rest=L(0, 0, _LOW_Z), travel=STROKE, rate=K_SPRING, damping=4.0, mass=0.03, friction=0.2,
    boxes=((L(0, 0, _LOW_Z), (0.124, 0.130, FLANGE_T), GREY),
           (L(0, 0.110, _LOW_Z), (0.036, 0.090, FLANGE_T), YELLOW)))              # lug sticking out +y (y 0.065-0.155)
JOINT = part("joint", pos=(JX, JY, ORIGIN_Z), mass=8.0, color=GREY, friction=0.8, boxes=JOINT_BOXES, sliders=(FLANGE,))

# -- the blind plate (origin = centre of the square body) ----------------------------------------------------------------
TRAY_TOP = TOP + 0.020
PLATE_START = (JX, -0.22, TRAY_TOP + PLATE_T / 2)
KNOB_DY = -0.14
PLATE_BOXES = [
    ((0, 0, 0), (0.120, 0.120, PLATE_T), ORANGE),
    ((0, -0.11, 0), (0.036, 0.100, PLATE_T), ORANGE),                              # stem, out front (-y)
    ((0, KNOB_DY, 0.020), (0.030, 0.016, 0.028), ORANGE),                          # knob to pinch
]
PLATE = part("plate", pos=PLATE_START, mass=0.10, color=ORANGE, friction=0.4, boxes=PLATE_BOXES)
TRAY = Box("tray", pos=(JX, -0.22 - 0.03, TOP + 0.010), size=(0.14, 0.14, 0.020), color=WHITE)   # y -0.32..-0.18 under the body
OBJECTS = [TRAY, JOINT, PLATE]

# -- scoring thresholds ---------------------------------------------------------------------------------------------------
GATE_FROM_STEP = 40
SETTLE_STEPS = 10
GATE_SHIFT_MM = 10.0        # joint moved
GATE_STALL_STEPS = 20       # flange at its limit for 1 s
OPEN_MM = 22.0
HOLD_OPEN_MM = 14.0
LIFT_MM = 12.0              # plate lifted off the tray (0.25 needs it held up while the gap is open)
ENTER_Y = -0.065            # plate centre y at which the plate edge is 60 mm past the flange edge
CENTRE_MM = 6.0
TILT_DEG = 12.0
SEAT_STEPS = 10
CLAMP_MM = (10.0, 14.0)
HOLD_STEPS = 20
DROP_Z = TOP + 0.012

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), j0=None, tripped=False, stall=0, level=0.0, seat=0,
                  clamp=0)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _tilt_deg(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def _gap_mm(joint) -> float:
    return -1000.0 * float(joint.sliders.get("flange", 0.0))


def _in_gap(pz, gap_mm) -> bool:
    low = UPPER_UNDER - gap_mm / 1000.0
    return pz - PLATE_T / 2 >= low - 0.003 and pz + PLATE_T / 2 <= UPPER_UNDER + 0.003


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    if _STATE["step"] < SETTLE_STEPS:
        _STATE["last"] = (0.0, False)
        return _STATE["last"]
    joint, plate = objects["joint"], objects["plate"]
    jp, pp = np.asarray(joint.pos, float), np.asarray(plate.pos, float)
    if _STATE["j0"] is None:
        _STATE["j0"] = jp.copy()
    step = _STATE["step"]
    gap = _gap_mm(joint)
    shift = 1000.0 * float(np.linalg.norm(jp[:2] - _STATE["j0"][:2]))
    _STATE["stall"] = _STATE["stall"] + 1 if gap >= 1000 * STROKE - 1.0 else 0
    if step > GATE_FROM_STEP and (pp[2] < DROP_Z or shift > GATE_SHIFT_MM or _STATE["stall"] > GATE_STALL_STEPS):
        _STATE["tripped"] = True
    cx, cy = 1000 * (pp[0] - JX), 1000 * (pp[1] - JY)
    centred = math.hypot(cx, cy) <= CENTRE_MM and _tilt_deg(plate) <= TILT_DEG
    in_gap = _in_gap(pp[2], gap)
    lifted = 1000 * (pp[2] - PLATE_START[2]) >= LIFT_MM
    entered = in_gap and pp[1] >= ENTER_Y and abs(cx) <= 20.0 and gap >= HOLD_OPEN_MM
    seated_open = centred and in_gap and gap >= HOLD_OPEN_MM
    clamped = centred and in_gap and CLAMP_MM[0] <= gap <= CLAMP_MM[1]
    _STATE["seat"] = _STATE["seat"] + 1 if seated_open else 0
    speed = float(np.linalg.norm(np.asarray(plate.lin_vel, float)))
    _STATE["clamp"] = _STATE["clamp"] + 1 if (clamped and speed < 0.02) else 0
    lvl = 0.0
    if gap >= OPEN_MM and lifted:
        lvl = 0.25
    if entered:
        lvl = 0.5
    if _STATE["seat"] >= SEAT_STEPS:
        lvl = 0.75
    if _STATE["clamp"] >= HOLD_STEPS:
        lvl = 1.0
    _STATE["level"] = max(_STATE["level"], min(lvl, 0.75))
    if _STATE["tripped"]:
        result = (0.0, False)
    elif _STATE["clamp"] >= HOLD_STEPS:
        result = (1.0, True)
    else:
        result = (_STATE["level"], False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Plate centred within 6 mm of the green mark, level, inside the joint, the joint closed on it (gap 10-14 mm),
    plate still for 1 s; no zero-gate."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 gap >= 22 mm while the plate is held >= 12 mm above the tray; 0.5 plate inside the joint (edge 60 mm past
    the flange edge, within the gap height) while gap >= 14 mm; 0.75 plate centred within 6 mm and level in the gap for
    0.5 s with the flange still held open (>= 14 mm); 1.0 joint closed on the plate (10-14 mm), centred, still for 1 s.
    Zero if the plate drops to the table, the joint shifts > 10 mm, or the flange sits on its limit for > 1 s."""
    return _update(objects)[0]
