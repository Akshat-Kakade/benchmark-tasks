"""On the station table a grey sample rod (540 x 20 x 20 mm, 0.30 kg) runs through three fixed box shrouds of a guide that
rises 7.5 degrees toward +y. Between the shrouds the rod is bare in two openings: a blue one on the right (u 110-260 mm
along the guide) and an orange one on the left (u 490-640 mm). The rod is one slider of the guide (travel 185 mm, no spring,
damping 8 N.s/m), so it rolls back down the guide at about 48 mm/s as soon as no hand holds it. The rod must be withdrawn
uphill until its lower (red) tip reaches the green band on the blue opening, which needs 172 mm of travel while one
hand can only follow the rod for about 110 mm inside one opening: the hands must exchange grips, one holding while the
other lets go and re-grips, so the rod never runs back.

partkit sliders run along the part's local z axis and move toward -z, so the guide's body frame is turned by 97.5 degrees
about x in on_reset: local -z = the uphill guide direction, local +z = the rollback direction. u is the distance along
the guide axis from its origin (0.300, -0.367, TOP + 0.028). The guide stands on three pillars (box shapes kept world-vertical)
and is heavy; the rod is the only moving part.

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

INSTRUCTION = ("Pull the grey sample rod uphill through the blue right opening and orange left opening until its "
               "lower tip reaches the green band by the blue opening. Keep the rod from sliding back.")

GREY = (0.55, 0.57, 0.60)
DARK = (0.30, 0.31, 0.34)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)
GREEN = (0.10, 0.70, 0.20)
RED = (0.85, 0.10, 0.08)
WHITE = (0.95, 0.95, 0.95)

# -- guide geometry ------------------------------------------------------------------------------------------------------
ANGLE = math.radians(7.5)
SA, CA = math.sin(ANGLE), math.cos(ANGLE)
GUIDE_XY = (0.300, -0.367)
GUIDE_Z = TOP + 0.028                                     # axis height above the table at u = 0
THETA = math.pi / 2 + ANGLE                               # body frame = world frame turned THETA about x
QUAT_GUIDE = (math.sin(THETA / 2), 0.0, 0.0, math.cos(THETA / 2))          # xyzw
_C, _S = math.cos(THETA), math.sin(THETA)
R_GUIDE = np.array([[1, 0, 0], [0, _C, -_S], [0, _S, _C]])                # local -> world
QUAT_BACK = (math.cos(-THETA / 2), math.sin(-THETA / 2), 0.0, 0.0)         # wxyz: world-aligned box in the body frame

SHROUDS = ((0.0, 0.110), (0.260, 0.490), (0.640, 0.741))   # u ranges (m)
BLUE_OPEN = (0.110, 0.260)
ORANGE_OPEN = (0.490, 0.640)
BORE, OUTSIDE = 0.032, 0.048
ROD_LEN, ROD_W = 0.540, 0.020
ROD_U0 = 0.005                                            # u of the rod's lower tip at rest
TRAVEL = 0.185
BAND_U = (0.177, 0.200)                                   # green band along the guide (leading edge 177 mm)
TARGET_LO, TARGET_HI = 172.0, 185.5                       # rod travel for 1.0 (mm); the band's leading edge is u = 5 + 172


def axis_point(u: float) -> np.ndarray:
    """World point on the guide axis at distance u (nominal pose)."""
    return np.array([GUIDE_XY[0], GUIDE_XY[1] + u * CA, GUIDE_Z + u * SA])


def Lw(v) -> tuple:
    """World offset from the guide origin -> body-frame vector."""
    return tuple(float(x) for x in R_GUIDE.T @ np.asarray(v, float))


def u_point(u: float, dx=0.0) -> np.ndarray:
    return axis_point(u) + [dx, 0.0, 0.0]


def _shroud(u0, u1):
    uc, ln = 0.5 * (u0 + u1), u1 - u0
    wall = (OUTSIDE - BORE) / 2
    off = (OUTSIDE + BORE) / 4
    return [((-off, 0.0, -uc), (wall, OUTSIDE, ln), GREY), ((off, 0.0, -uc), (wall, OUTSIDE, ln), GREY),
            ((0.0, -off, -uc), (BORE, wall, ln), GREY), ((0.0, off, -uc), (BORE, wall, ln), GREY)]


def _flat(world_centre, size, color):
    """A world-aligned box at a world point (relative to the table frame), as a body-frame box spec."""
    rel = np.asarray(world_centre, float) - [GUIDE_XY[0], GUIDE_XY[1], GUIDE_Z]
    return (Lw(rel), tuple(size), color, QUAT_BACK)


def _strip(u0, u1, dx, width, color, lift=0.0, thick=0.002):
    y0, y1 = GUIDE_XY[1] + u0 * CA, GUIDE_XY[1] + u1 * CA
    return _flat([GUIDE_XY[0] + dx, 0.5 * (y0 + y1), TOP + 0.001 + thick / 2 + lift], (width, y1 - y0, thick), color)


GUIDE_BOXES = []
for _a, _b in SHROUDS:
    GUIDE_BOXES += _shroud(_a, _b)
    _uc = 0.5 * (_a + _b)
    _top = GUIDE_Z + _uc * SA - (OUTSIDE / 2) * CA + 0.004          # pillar top, a little inside the bottom wall
    _h = _top - (TOP + 0.001)
    GUIDE_BOXES.append(_flat([GUIDE_XY[0], GUIDE_XY[1] + _uc * CA, TOP + 0.001 + _h / 2], (0.048, 0.040, _h), DARK))
for _dx in (-0.052, 0.052):                                         # coloured strips on the table beside each opening
    GUIDE_BOXES.append(_strip(*BLUE_OPEN, _dx, 0.012, BLUE))
    GUIDE_BOXES.append(_strip(*ORANGE_OPEN, _dx, 0.012, ORANGE))
    GUIDE_BOXES.append(_strip(*BAND_U, _dx, 0.012, GREEN, thick=0.004))

ROD = Slider(
    name="rod", rest=(0.0, 0.0, -(ROD_U0 + ROD_LEN / 2)), travel=TRAVEL, rate=0.0, damping=8.0, mass=0.30,
    friction=0.8,
    boxes=(((0.0, 0.0, -(ROD_U0 + ROD_LEN / 2)), (ROD_W, ROD_W, ROD_LEN), GREY),
           ((0.0, 0.0, -(ROD_U0 + 0.006)), (ROD_W + 0.002, ROD_W + 0.002, 0.012), RED)))   # red lower tip
GUIDE = part("guide", pos=(GUIDE_XY[0], GUIDE_XY[1], GUIDE_Z), mass=8.0, color=GREY, friction=0.8,
             boxes=GUIDE_BOXES, sliders=(ROD,))
OBJECTS = [GUIDE]

# -- scoring thresholds (rod travel s in mm) ----------------------------------------------------------------------------
LEVELS = ((30.0, 0.25), (90.0, 0.5), (160.0, 0.75), (TARGET_LO, 1.0))
HOLD_STEPS = 20              # 1.0 s
STILL_MM = 8.0               # max - min of s over the hold window
ARM_MM = 10.0                # scoring starts once the rod has been seen at <= 10 mm (it spawns at the top and rolls down)
GATE_FROM_STEP = 40
SETTLE_STEPS = 10
GATE_SHIFT_MM = 12.0
GATE_TILT_DEG = 6.0

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(start=None, step=0, scored=None, last=(0.0, False), tripped=False, hist=[], best=0.0, max_s=0.0, armed=False)
    if environment is not None:
        write_pose(environment, "guide", (GUIDE_XY[0], GUIDE_XY[1], float(GUIDE_Z)), QUAT_GUIDE)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _tilt_deg(state) -> float:
    """Angle between the body z (slide) axis and its nominal direction (the guide axis stays at 7.5 degrees)."""
    nominal = R_GUIDE[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[:, 2] @ nominal)))))


def _travel_mm(state) -> float:
    return -1000.0 * float(state.sliders.get("rod", 0.0))


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    guide = objects["guide"]
    if _STATE["step"] < SETTLE_STEPS:
        _STATE["last"] = (0.0, False)
        return _STATE["last"]
    pos = np.asarray(guide.pos, float)
    if _STATE["start"] is None:
        _STATE["start"] = pos.copy()
    shift = 1000.0 * float(np.linalg.norm(pos - _STATE["start"]))
    s = _travel_mm(guide)
    if _STATE["step"] > GATE_FROM_STEP and (shift > GATE_SHIFT_MM or _tilt_deg(guide) > GATE_TILT_DEG):
        _STATE["tripped"] = True
    _STATE["max_s"] = max(_STATE["max_s"], s)
    if s <= ARM_MM:
        _STATE["armed"] = True             # the rod spawns at its upper stop and rolls down in the first 4 s: score only after
    h = _STATE["hist"]
    h.append(s)
    del h[:-HOLD_STEPS]
    if _STATE["armed"] and len(h) == HOLD_STEPS and max(h) - min(h) <= STILL_MM:
        lo = min(h)
        for thr, val in LEVELS:
            if lo >= thr and (val < 1.0 or max(h) <= TARGET_HI):
                _STATE["best"] = max(_STATE["best"], val)
    if _STATE["tripped"]:
        result = (0.0, False)
    else:
        result = (_STATE["best"], _STATE["best"] >= 1.0)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Rod travel 172-185.5 mm and still (within 8 mm) for 1 s; guide never moved > 12 mm or tilted > 6 degrees."""
    return _update(objects)[1]


def score(objects) -> float:
    """Rod travel (mm) held for 1 s with the rod moving <= 8 mm: >= 30 gives 0.25, >= 90 gives 0.5, >= 160 gives 0.75,
    172-185.5 gives 1.0. Zero if after step 40 the guide moved > 12 mm or tilted > 6 degrees."""
    return _update(objects)[0]
