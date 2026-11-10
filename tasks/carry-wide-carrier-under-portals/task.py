"""On the station table: a wide blue carrier (500 x 80 mm plate with a small blue content block in its middle and an
orange handle stem at each long end) rests in a green outline at the near end. Two grey guard portals hang over the
table in front of it, one behind the other; each is a low plate with a slot, and the slots are offset to opposite sides.
The content block fits only through the slot, and the plate itself must stay lower than the underside of the guards.
Both hands lift the carrier by the two orange stems, steer it through slot 1 and slot 2 and set it down level in the
second green outline.

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

INSTRUCTION = ("Carry the blue carrier by its two orange handles through the slots in both grey guards, steering the "
               "blue block through each slot, and set it down level inside the second green outline.")

BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)
GREY = (0.55, 0.55, 0.58)
GREEN = (0.10, 0.70, 0.20)

# -- carrier (own frame: origin at the plate underside centre, +x forward, +y left) ------------------------------------
PLATE = (0.08, 0.50, 0.012)
STEM_Y = 0.23
STEM = (0.05, 0.04, 0.05)
BLOCK = (0.04, 0.05, 0.06)
CARRIER_BOXES = [
    ((0.0, 0.0, PLATE[2] / 2), PLATE, BLUE),
    ((0.0, 0.0, PLATE[2] + BLOCK[2] / 2), BLOCK, BLUE),                          # content, rigidly fixed to the plate
    ((0.0, STEM_Y, PLATE[2] + STEM[2] / 2), STEM, ORANGE),
    ((0.0, -STEM_Y, PLATE[2] + STEM[2] / 2), STEM, ORANGE),
]
START = (0.21, 0.0)
TARGET = (0.45, 0.0)
CARRIER = part("carrier", pos=(START[0], START[1], TOP), mass=0.20, boxes=CARRIER_BOXES, color=BLUE, friction=0.8)

# -- guards: plate over |y| < GUARD_HALF_Y, underside GUARD_Z, slot SLOT_W wide centred SLOT_Y -------------------------
GUARD_Z = TOP + 0.0375                       # underside 37.5 mm above the table: plate top at rest is 12 mm
GUARD_TOP = 1.00
GUARD_HALF_Y = 0.075
GUARD_T = 0.012
SLOT_W = 0.08
PORTALS = [(0.30, +0.02), (0.38, -0.02)]     # (x, slot centre y)


def _guard(name, x, lo, hi):
    return Box(name, pos=(x, 0.5 * (lo + hi), 0.5 * (GUARD_Z + GUARD_TOP)), size=(GUARD_T, hi - lo, GUARD_TOP - GUARD_Z), color=GREY)


GUARDS = []
for _i, (_x, _c) in enumerate(PORTALS, 1):
    GUARDS += [_guard(f"guard{_i}_left", _x, _c + SLOT_W / 2, GUARD_HALF_Y), _guard(f"guard{_i}_right", _x, -GUARD_HALF_Y, _c - SLOT_W / 2)]
GUARDS += [Box("guard_beam", pos=(0.4075, 0.0, GUARD_TOP + 0.015), size=(0.275, 2 * GUARD_HALF_Y, 0.03), color=GREY),
           Box("guard_post", pos=(0.53, 0.0, 0.5 * (TOP + GUARD_TOP + 0.03)), size=(0.03, 2 * GUARD_HALF_Y, GUARD_TOP + 0.03 - TOP), color=GREY)]


def _outline(prefix, cx, cy):
    ix, iy, w, t = 0.11, 0.54, 0.012, 0.003              # inner size (x, y), strip width, strip height
    return [Box(f"{prefix}_n", pos=(cx + ix / 2 + w / 2, cy, TOP + t / 2), size=(w, iy + 2 * w, t), color=GREEN),
            Box(f"{prefix}_s", pos=(cx - ix / 2 - w / 2, cy, TOP + t / 2), size=(w, iy + 2 * w, t), color=GREEN),
            Box(f"{prefix}_e", pos=(cx, cy + iy / 2 + w / 2, TOP + t / 2), size=(ix, w, t), color=GREEN),
            Box(f"{prefix}_w", pos=(cx, cy - iy / 2 - w / 2, TOP + t / 2), size=(ix, w, t), color=GREEN)]


OBJECTS = [*GUARDS, *_outline("start", *START), *_outline("goal", *TARGET), CARRIER]

# -- scoring thresholds -------------------------------------------------------------------------------------------------
REST_Z = TOP                       # carrier origin height when resting on the table (re-measured in the settle run)
X_PAST_1 = 0.335                   # block clear of slot 1 (back face past the guard)
X_PAST_2 = 0.42                    # block clear of slot 2
LIFT_MIN = 0.010                   # centre lifted at least this high when it crosses a guard (carried, not dragged)
GATE_TILT_DEG = 20.0
PLACE_TOL, Z_TOL = 0.012, 0.006
LEVEL_FULL_DEG, LEVEL_PART_DEG = 8.0, 12.0
PART_TOL = 0.030
HOLD_STEPS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), past1=False, past2=False, gate=False, held=0, max_tilt=0.0)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def tilt_deg(state: PartState) -> float:
    """Angle between the carrier's z axis and the vertical (roll and pitch together)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def yaw_deg(state: PartState) -> float:
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def classify(objects, memo: dict):
    """Pure function of the carrier pose and the memo (flags seen so far): (level, all-1.0-conditions-hold, info)."""
    c = objects["carrier"]
    t = tilt_deg(c)
    yaw = yaw_deg(c)
    memo["max_tilt"] = max(memo["max_tilt"], t)
    if t > GATE_TILT_DEG or c.pos[2] < TOP - 0.05:
        memo["gate"] = True
    info = dict(x=float(c.pos[0]), y=float(c.pos[1]), z=float(c.pos[2] - REST_Z), tilt=t, yaw=yaw, gate=memo["gate"])
    if memo["gate"]:
        return 0.0, False, info
    carried = float(c.pos[2]) - REST_Z >= LIFT_MIN
    if c.pos[0] >= X_PAST_1 and carried:
        memo["past1"] = True
    if memo["past1"] and c.pos[0] >= X_PAST_2 and carried:
        memo["past2"] = True
    d = float(np.hypot(c.pos[0] - TARGET[0], c.pos[1] - TARGET[1]))
    info["d"] = d
    dz = abs(float(c.pos[2]) - REST_Z)
    if memo["past2"] and d <= PLACE_TOL and dz <= Z_TOL and t <= LEVEL_FULL_DEG and abs(yaw) <= LEVEL_FULL_DEG:
        return 1.0, True, info
    if memo["past2"] and d <= PART_TOL and t <= LEVEL_PART_DEG and abs(yaw) <= LEVEL_PART_DEG:
        return 0.75, False, info
    if memo["past2"]:
        return 0.5, False, info
    if memo["past1"]:
        return 0.25, False, info
    return 0.0, False, info


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    level, perfect, _ = classify(objects, _STATE)
    _STATE["held"] = _STATE["held"] + 1 if perfect else 0
    result = (level, _STATE["held"] >= HOLD_STEPS)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """The carrier passed both slots (block past x = 0.42 after x = 0.335), and rests within 12 mm of the second outline's
    centre, within 6 mm of table height, tilt and yaw each <= 8 degrees, held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 carrier past slot 1 (x >= 0.335) while lifted >= 10 mm (carried, not dragged); 0.5 likewise past slot 2 (x >= 0.42); 0.75 then within 30 mm of the second outline's
    centre with tilt and yaw <= 12 degrees; 1.0 as in success(). Zero if the carrier tilts over 20 degrees at any time
    or falls off the table."""
    return _update(objects)[0]
