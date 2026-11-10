"""On the station table: a grey anchor plate (120 x 300 x 15 mm, low friction) carries a yellow gauge carriage on a short
rail along y. A spring (80 N/m, modelled in the physics engine) pulls the carriage back; its travel is the gauge reading,
and a green band on the scale marks the proof load (55-70 mm, about 4.4-5.6 N). An orange fin stands at the plate's
+y end. One hand must hold the plate by the fin while the other pulls the yellow handle into the green band, holds it
for one second and lets it ease back, without the plate sliding.

The carriage is a prismatic slider of the plate (partkit sliders run along the part's local z axis), so the plate's body
frame is turned 90 degrees about x: local (x, y, z) = world (x, up, -y) and the carriage's local travel is world +y.
Hold point, rail and handle lie on one line (x = 0.30), so the pull reaction gives no yaw torque.
The plate is spawned upright and turned flat by a write_pose in on_reset.

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

INSTRUCTION = ("Proof-test the grey anchor plate: pull its yellow handle until the gauge carriage reaches the green "
               "band, hold it for one second, then let it ease back, without the plate sliding.")

GREY = (0.55, 0.57, 0.60)
YELLOW = (0.95, 0.80, 0.05)
ORANGE = (1.00, 0.50, 0.00)
WHITE = (0.95, 0.95, 0.95)
GREEN = (0.10, 0.70, 0.20)
RED = (0.85, 0.10, 0.08)
BLACK = (0.12, 0.12, 0.13)

# -- frame helpers: plate body frame is the world frame turned +90 degrees about x ---------------------------------------
PLATE_XY = (0.30, -0.02)
PLATE_T = 0.015
ORIGIN_Z = TOP + PLATE_T / 2              # body origin: plate centre
QUAT_FLAT = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))      # xyzw: local z -> world -y, local y -> world up


def L(dx, dy, dz):
    """World offset from the plate origin -> plate-frame vector."""
    return (dx, dz, -dy)


def S(sx, sy, sz):
    """World box size -> plate-frame size."""
    return (sx, sz, sy)


K_SPRING = 80.0           # N/m
STROKE = 0.090             # carriage travel limit (m)
BAND = (0.055, 0.070)
OVERLOAD = 0.080
CAR_Y0 = -0.11             # carriage centre at rest, y offset from the plate origin (it moves toward +y)
CAR_Z = PLATE_T / 2 + 0.001 + 0.006
POST_Y = 0.12              # orange fin at the +y end of the plate, on the carriage's line
HANDLE = (0.040, 0.020, 0.040)     # x, y (pinch width), z

_top = PLATE_T / 2
PLATE_BOXES = [
    (L(0, 0, 0), S(0.120, 0.300, PLATE_T), GREY),
    (L(0, POST_Y, _top + 0.020), S(0.070, 0.020, 0.040), ORANGE),                      # anchor fin
    (L(-0.0255, -0.05, _top + 0.006), S(0.005, 0.180, 0.012), BLACK),                  # rails along y
    (L(0.0255, -0.05, _top + 0.006), S(0.005, 0.180, 0.012), BLACK),
    (L(0.045, -0.0375, _top + 0.001), S(0.020, 0.095, 0.002), WHITE),                  # scale strip
    (L(0.045, -0.0225, _top + 0.0025), S(0.020, 0.015, 0.002), GREEN),                 # band 55-70 mm
    (L(0.045, 0.0, _top + 0.0025), S(0.020, 0.010, 0.002), RED),                       # overload zone 80-90 mm
]
CARRIAGE = Slider(
    name="carriage", rest=L(0, CAR_Y0, CAR_Z), travel=STROKE, rate=K_SPRING, damping=5.0, mass=0.10, friction=0.1,
    boxes=((L(0, CAR_Y0, CAR_Z), S(0.040, 0.050, 0.012), YELLOW),
           (L(0, CAR_Y0, CAR_Z + 0.026), S(*HANDLE), YELLOW)))
PLATE = part("plate", pos=(PLATE_XY[0], PLATE_XY[1], ORIGIN_Z + 0.16), mass=0.8, color=GREY, friction=0.1,
             boxes=PLATE_BOXES, sliders=(CARRIAGE,))
OBJECTS = [PLATE]

# -- scoring thresholds ---------------------------------------------------------------------------------------------------
GATE_FROM_STEP = 40
SETTLE_STEPS = 10         # the freshly placed plate pops about 15 mm / 13 mm and settles in the first steps: no baseline before
GATE_SHIFT_MM = 15.0
GATE_LIFT_MM = 10.0
GATE_TILT_DEG = 20.0
ENGAGED_MM = 20.0          # 0.25: carriage pulled this far
SHIFT_FULL_MM = 5.0
HOLD_STEPS = 20            # 1.0 s inside the band
EASE_STEPS = 60            # 3 s to come back after the hold
REST_MM = 10.0
STILL_STEPS = 10

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(start=None, step=0, scored=None, last=(0.0, False), tripped=False, max_travel=0.0, streak=0,
                  held_at=None, held_ok=False, hist=[], last_band=0, eased=False)
    if environment is not None:
        write_pose(environment, "plate", (PLATE_XY[0], PLATE_XY[1], float(ORIGIN_Z)), QUAT_FLAT)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _tilt_deg(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 1])))))


def _travel_mm(state) -> float:
    return -1000.0 * float(state.sliders.get("carriage", 0.0))


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    plate = objects["plate"]
    if _STATE["step"] < SETTLE_STEPS:
        _STATE["last"] = (0.0, False)
        return _STATE["last"]
    pos = np.asarray(plate.pos, float)
    if _STATE["start"] is None:
        _STATE["start"] = pos.copy()
    shift = 1000.0 * float(np.linalg.norm(pos[:2] - _STATE["start"][:2]))
    lift = 1000.0 * float(pos[2] - _STATE["start"][2])
    travel = _travel_mm(plate)
    tilt = _tilt_deg(plate)
    step = _STATE["step"]
    if step > GATE_FROM_STEP and (travel > 1000 * OVERLOAD or shift > GATE_SHIFT_MM or lift > GATE_LIFT_MM
                                  or tilt > GATE_TILT_DEG):
        _STATE["tripped"] = True
    _STATE["max_travel"] = max(_STATE["max_travel"], travel)
    in_band = 1000 * BAND[0] <= travel <= 1000 * BAND[1]
    _STATE["streak"] = _STATE["streak"] + 1 if in_band else 0
    if in_band:
        _STATE["last_band"] = step
    if _STATE["held_at"] is None and _STATE["streak"] >= HOLD_STEPS:
        _STATE["held_at"], _STATE["held_ok"] = step, shift <= SHIFT_FULL_MM
    h = _STATE["hist"]
    h.append(travel)
    del h[:-STILL_STEPS]
    still = len(h) == STILL_STEPS and max(h) - min(h) < 1.0
    held, lb = _STATE["held_ok"], _STATE["last_band"]
    if held and travel <= REST_MM and step - lb <= EASE_STEPS:
        _STATE["eased"] = True                 # back at rest within 3 s of leaving the band (latched)
    if _STATE["tripped"]:
        result = (0.0, False)
    elif _STATE["eased"] and travel <= REST_MM and shift <= SHIFT_FULL_MM:
        result = (1.0, bool(still))
    elif held:
        result = (0.75, False)
    elif _STATE["max_travel"] >= 1000 * BAND[0]:
        result = (0.5, False)
    elif _STATE["max_travel"] >= ENGAGED_MM:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Carriage held 1 s in the 55-70 mm band with the plate moved <= 5 mm, then back at <= 10 mm within 3 s of leaving the band, plate
    moved <= 5 mm, carriage still; never past 80 mm, plate never moved > 15 mm, lifted > 10 mm or tilted > 20 degrees."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 carriage pulled >= 20 mm; 0.5 reached 55 mm (band entered); 0.75 held 1 s in 55-70 mm with the plate moved
    <= 5 mm; 1.0 also back to <= 10 mm within 3 s of leaving the band with the plate moved <= 5 mm. Zero if the carriage ever passed 80 mm,
    or the plate moved > 15 mm, lifted > 10 mm or tilted > 20 degrees after step 40."""
    return _update(objects)[0]
