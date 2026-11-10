"""On the station table: a blue vertical shaft (a 45 mm cylinder with an upright handle stem on top, a black stripe facing
forward) stands inside an orange open shell (a U of three joined boxes, one side wall white, a handle stem on a tab at its
outer edge) that surrounds it with 20 mm clearance all round. A green bar lies on the table in front of the pair. The left arm
pinches the shaft's stem and lifts it; the right arm pinches the shell's handle, lifts the shell and carries it round the shaft
through 90 deg clockwise (seen from above) until the white wall lies parallel to the green bar, while the shaft stays where
it is, in the same pose, in the air. Both bodies are free rigid bodies: nothing is pinned, hinged or fastened.

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

INSTRUCTION = ("Hold the blue shaft by its upper handle and turn the orange open shell around it, with the other hand, until "
               "the shell's white wall is parallel to the green bar; keep the shaft's black stripe facing forward.")

BLUE = (0.10, 0.30, 0.90)
ORANGE = (0.95, 0.50, 0.08)
WHITE = (0.95, 0.95, 0.95)
BLACK = (0.04, 0.04, 0.04)
GREEN = (0.15, 0.65, 0.25)

SH_X, SH_Y = 0.34, 0.0           # shaft axis
SH_R, SH_H = 0.0225, 0.070       # shaft cylinder: 45 mm diameter, 70 mm tall
STEM_S = (0.022, 0.022, 0.065)   # shaft handle stem (x, y, z) on the shaft axis, standing on the shaft top: top at 135 mm
PINCH_S = 0.095                  # left pinch height above the shaft underside
CAV = 0.085                      # shell cavity (inner opening, square); the open side faces +x
WALL = 0.0275                    # shell wall (outer 0.14 x 0.14)
SHELL_H = 0.035
R_H = 0.16                       # shell handle distance from the shaft axis
STEM_H = (0.035, 0.040, 0.085)   # shell handle stem (x, y, z), 50 mm above the shell top
PINCH_H = 0.060                  # right pinch height above the shell underside
CLEAR0 = CAV / 2 - SH_R          # 0.020 radial clearance when concentric

_HW = CAV / 2 + WALL / 2         # wall centre offset 0.05625
SHAFT_BOXES = [
    ((0.0, 0.0, SH_H + STEM_S[2] / 2), STEM_S, BLUE),
    ((SH_R - 0.003, 0.0, SH_H / 2), (0.007, 0.014, SH_H - 0.004), BLACK),     # stripe, faces forward (+x)
]
SHELL_BOXES = [
    ((-_HW, 0.0, SHELL_H / 2), (WALL, 0.14, SHELL_H), ORANGE),                # back wall (-x)
    ((0.0125, -_HW, SHELL_H / 2), (0.115, WALL, SHELL_H), ORANGE),            # side wall (-y), where the handle tab is
    ((0.0125, +_HW, SHELL_H / 2), (0.115, WALL, SHELL_H), WHITE),             # side wall (+y): the white marked wall
    ((0.0, -0.105, SHELL_H / 2), (0.040, 0.070, SHELL_H), ORANGE),    # tab to the handle
    ((0.0, -R_H, STEM_H[2] / 2), STEM_H, ORANGE),                             # handle stem
]

REST = TOP + 0.0005
SHAFT = part("shaft", pos=(SH_X, SH_Y, REST), mass=0.12, boxes=SHAFT_BOXES, color=BLUE, friction=0.6,
             cylinders=(((0.0, 0.0, SH_H / 2), SH_R, SH_H, BLUE),))
SHELL = part("shell", pos=(SH_X, SH_Y, REST), mass=0.10, boxes=SHELL_BOXES, color=ORANGE, friction=0.6)
MARK_X = SH_X + 0.085
MARK = Box("angle_mark", pos=(MARK_X, 0.0, TOP + 0.0015), size=(0.010, 0.140, 0.003), color=GREEN)
OBJECTS = [MARK, SHAFT, SHELL]

# -- scoring thresholds ------------------------------------------------------------------------------------------------
LIFT_MIN = 0.015               # both origins must be this far above their rest height (held in the air)
GATE_DRIFT, GATE_TILT = 0.030, 12.0
L1_D = 20.0
L2_D, L2_YAW = 55.0, 20.0
L3_D, L3_DRIFT, L3_YAW = (70.0, 110.0), 0.025, 12.0
L4_D, L4_DRIFT, L4_YAW, L4_CONC, L4_TILT, L4_CLEAR = (78.0, 102.0), 0.015, 8.0, 0.015, 8.0, 0.008
HOLD_STEPS = 20
BASE_STEP = 10                  # shaft baseline is taken after the first steps (the teleported parts settle)

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), held=0, base=None)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def yaw_deg(state: PartState) -> float:
    r = state.rotation()
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def tilt_deg(state: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def measure(objects, base=None) -> dict:
    """Measured relation of shell and shaft. ``turn`` = clockwise turn of the shell relative to the shaft (deg, positive =
    instructed direction); drift = shaft xy shift from its baseline; yaw_drift = shaft yaw change; conc = horizontal offset
    of the shell cavity centre from the shaft axis; clear = smallest gap between shaft and shell wall (negative: contact)."""
    sh, sl = objects["shaft"], objects["shell"]
    bx, by, byaw = base if base is not None else (SH_X, SH_Y, 0.0)
    ys, yl = yaw_deg(sh), yaw_deg(sl)
    p = sl.to_local((float(sh.pos[0]), float(sh.pos[1]), float(sl.pos[2])))      # shaft axis in the shell frame
    clear = min(CAV / 2 - abs(float(p[1])) - SH_R, float(p[0]) + CAV / 2 - SH_R)
    return dict(turn=-wrap(yl - ys), drift=float(math.hypot(sh.pos[0] - bx, sh.pos[1] - by)), yaw_drift=abs(wrap(ys - byaw)),
                conc=float(math.hypot(sh.pos[0] - sl.pos[0], sh.pos[1] - sl.pos[1])), clear=float(clear),
                zs=float(sh.pos[2] - REST), zl=float(sl.pos[2] - REST), tilt_s=tilt_deg(sh), tilt_l=tilt_deg(sl))


def classify(objects, base=None):
    """(level, all-1.0-conditions-hold, info) from the current poses only."""
    m = measure(objects, base)
    if (min(m["zs"], m["zl"]) < LIFT_MIN or m["clear"] < 0.0 or m["drift"] > GATE_DRIFT
            or max(m["tilt_s"], m["tilt_l"]) > GATE_TILT):
        return 0.0, False, m
    d = m["turn"]
    if (L4_D[0] <= d <= L4_D[1] and m["drift"] <= L4_DRIFT and m["yaw_drift"] <= L4_YAW and m["conc"] <= L4_CONC
            and max(m["tilt_s"], m["tilt_l"]) <= L4_TILT and m["clear"] >= L4_CLEAR):
        return 1.0, True, m
    if L3_D[0] <= d <= L3_D[1] and m["drift"] <= L3_DRIFT and m["yaw_drift"] <= L3_YAW:
        return 0.75, False, m
    if d >= L2_D and m["yaw_drift"] <= L2_YAW:
        return 0.5, False, m
    if d >= L1_D:
        return 0.25, False, m
    return 0.0, False, m


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    if _STATE["base"] is None and _STATE["step"] >= BASE_STEP:
        sh = objects["shaft"]
        _STATE["base"] = (float(sh.pos[0]), float(sh.pos[1]), yaw_deg(sh))
    level, perfect, _ = classify(objects, _STATE["base"])
    _STATE["held"] = _STATE["held"] + 1 if perfect else 0
    result = (level, _STATE["held"] >= HOLD_STEPS)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Both bodies held in the air, the shell turned 78-102 deg clockwise relative to the shaft, the shaft within 15 mm and
    8 deg of where it started, the shell cavity within 15 mm of the shaft axis, no contact (gap >= 8 mm); held for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if either body is not lifted at least 15 mm, the shell touches the shaft, the shaft moves over 30 mm or either body
    tilts over 12 deg. Else 0.25 for a shell turn >= 20 deg relative to the shaft; 0.5 for >= 55 deg with the shaft yaw
    within 20 deg; 0.75 for 70-110 deg, shaft within 25 mm and 12 deg; 1.0 as in success()."""
    return _update(objects)[0]
