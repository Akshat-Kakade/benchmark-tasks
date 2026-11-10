"""On the station table: a blue plate (left) and an orange plate (right), each resting on a low pedestal and each with a
handle block at its outer edge, stand either side of a marked bench outline. Both arms lift their plate and bring the two
straight inner edges together above the outline, level, with a small positive gap and no overlap, and hold them there.
The plates are free rigid boxes (0.12 kg); nothing is pinned, fastened or touching.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Hold the blue and orange plates above the marked bench area and bring their straight inner edges "
               "together, level and without overlap.")

BLUE = (0.10, 0.30, 0.90)
ORANGE = (0.95, 0.50, 0.08)
WHITE = (0.95, 0.95, 0.95)
GREY = (0.62, 0.62, 0.65)

PED_H = 0.020                  # pedestal height
PL_X, PL_Y, PL_T = 0.14, 0.105, 0.012   # plate size (x along the edge, y across, thickness)
HANDLE = (0.06, 0.04, 0.050)   # handle block (x, y, z), joined to the plate's outer edge
HC = PL_Y / 2 + HANDLE[1] / 2  # handle centre offset from the plate centre (0.0725)
START_Y = 0.125                # plate centres at +-START_Y
X_C = 0.31
REST_Z = TOP + PED_H + 0.0005  # plate origin (underside centre) at rest


def _boxes(sign, color):
    return [((0.0, 0.0, PL_T / 2), (PL_X, PL_Y, PL_T), color),
            ((0.0, sign * HC, HANDLE[2] / 2), HANDLE, color)]


BLUE_P = part("blue", pos=(X_C, START_Y, REST_Z), mass=0.12, boxes=_boxes(+1, BLUE), color=BLUE, friction=0.6)
ORANGE_P = part("orange", pos=(X_C, -START_Y, REST_Z), mass=0.12, boxes=_boxes(-1, ORANGE), color=ORANGE, friction=0.6)

OUTLINE = Box("bench_outline", pos=(X_C, 0.0, TOP + 0.001), size=(0.19, 0.29, 0.002), color=WHITE)
PED_L = Box("ped_left", pos=(X_C, START_Y, TOP + PED_H / 2), size=(0.10, 0.06, PED_H), color=GREY)
PED_R = Box("ped_right", pos=(X_C, -START_Y, TOP + PED_H / 2), size=(0.10, 0.06, PED_H), color=GREY)
OBJECTS = [OUTLINE, PED_L, PED_R, BLUE_P, ORANGE_P]

# -- scoring thresholds ------------------------------------------------------------------------------------------------
LIFT_Z = TOP + 0.050          # plate underside centre at least this high (rest height is TOP + 0.0205): "held in the air"
GATE_TILT, GATE_OVERLAP = 12.0, 0.008
EDGE_Z = PL_T / 2             # height of the inner-edge midpoint above the plate origin
HOLD_STEPS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), held=0)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def tilt_deg(state: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def yaw_deg(state: PartState) -> float:
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def seam(objects) -> dict:
    """Measured fit of the two plates: seam gap (inner-edge y distance, negative = overlap), seam midpoint y, height step,
    relative yaw, along-seam (x) offset of the edge midpoints, tilts and plate heights."""
    b, o = objects["blue"], objects["orange"]
    eb = np.asarray(b.to_world((0.0, -PL_Y / 2, EDGE_Z)), float)   # blue inner edge midpoint
    eo = np.asarray(o.to_world((0.0, +PL_Y / 2, EDGE_Z)), float)   # orange inner edge midpoint
    dyaw = (yaw_deg(b) - yaw_deg(o) + 180.0) % 360.0 - 180.0
    return dict(gap=float(eb[1] - eo[1]), mid=float((eb[1] + eo[1]) / 2), step=float(abs(b.pos[2] - o.pos[2])),
                dyaw=abs(dyaw), dx=float(abs(eb[0] - eo[0])), tilt_b=tilt_deg(b), tilt_o=tilt_deg(o),
                zb=float(b.pos[2]), zo=float(o.pos[2]))


def classify(objects):
    """(level, all-1.0-conditions-hold, info) from the current poses only."""
    m = seam(objects)
    lifted = m["zb"] >= LIFT_Z and m["zo"] >= LIFT_Z
    if (not lifted or max(m["tilt_b"], m["tilt_o"]) > GATE_TILT or m["gap"] < -GATE_OVERLAP):
        return 0.0, False, m
    if m["step"] > 0.040:
        return 0.0, False, m
    if (0.008 <= m["gap"] <= 0.025 and m["step"] <= 0.012 and m["dyaw"] <= 8.0 and abs(m["mid"]) <= 0.012
            and m["dx"] <= 0.020 and max(m["tilt_b"], m["tilt_o"]) <= 8.0):
        return 1.0, True, m
    if 0.0 <= m["gap"] <= 0.035 and m["step"] <= 0.020 and m["dyaw"] <= 12.0 and m["dx"] <= 0.040:
        return 0.75, False, m
    if m["gap"] <= 0.065 and m["dyaw"] <= 20.0:
        return 0.5, False, m
    return 0.25, False, m


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    level, perfect, _ = classify(objects)
    _STATE["held"] = _STATE["held"] + 1 if perfect else 0
    result = (level, _STATE["held"] >= HOLD_STEPS)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Both plates held in the air (underside >= 30 mm above the pedestal tops), tilt <= 8 deg, inner edges parallel
    (relative yaw <= 8 deg), seam gap 8-25 mm, height step <= 12 mm, seam midpoint within 12 mm of y = 0, edge midpoints
    within 20 mm along the seam; held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if a plate is not held in the air, tilts over 12 deg, the plates overlap by more than 8 mm, or their heights differ
    by more than 40 mm. 0.25 both plates held, otherwise; 0.5 gap <= 65 mm and relative yaw <= 20 deg; 0.75 gap 0-35 mm,
    height step <= 20 mm, yaw <= 12 deg; 1.0 as in success()."""
    return _update(objects)[0]
