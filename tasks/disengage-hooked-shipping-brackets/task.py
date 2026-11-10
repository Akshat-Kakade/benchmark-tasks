"""On the station table: a blue keyed housing (a flat box with a closed lid and a narrow slot in its front wall, 0.12 kg) and an
orange T-head bracket (a long thin head inside the housing, a stem out through the slot, a handle post at the stem end, 0.12 kg)
are hooked together: the head is longer than the slot, so a straight pull is blocked. Both arms pinch the handle posts, lift
the two free brackets, turn the orange bracket in the mouth of the slot until its head is edge-on to the slot, pull the head
out, and hold the two brackets apart above their own marked outlines.
Nothing is pinned or fastened: both brackets are free rigid bodies built from boxes.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import geom  # noqa: E402
from partkit import PartState, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Hold the blue housing by its back handle and the orange bracket by its stem handle, turn the orange bracket until "
               "its T-head clears the slot and pull it out, then hold both brackets apart above their marked outlines.")

BLUE = (0.10, 0.30, 0.90)
BLUE_LID = (0.35, 0.55, 0.98)
ORANGE = (0.95, 0.50, 0.08)
WHITE = (0.95, 0.95, 0.95)

A_X, A_Y = 0.24, 0.10                    # housing centre (station frame), slot opens toward -y
B_REL_OUT = (0.0, -0.28)                 # target of the T-head centre in the housing frame (outline of the orange bracket)
OUT_T = 0.002
A_Z = TOP + OUT_T + 0.0005               # housing origin (underside centre) at rest: on its outline
B_Z = TOP + 0.0005                       # orange origin at rest: its handle post stands on the table

_AC = [BLUE, BLUE_LID, BLUE, BLUE, BLUE, BLUE, BLUE, BLUE]   # colours of geom.A_BOXES in order (floor, lid, walls, lips, post)
A_BOXES = [(c, s, _AC[i]) for i, (c, s) in enumerate(geom.A_BOXES)]
B_BOXES = [(c, s, ORANGE) for c, s in geom.B_BOXES]

BLUE_P = part("blue", pos=(A_X, A_Y, A_Z), mass=0.12, boxes=A_BOXES, color=BLUE, friction=0.6)
ORANGE_P = part("orange", pos=(A_X, A_Y, B_Z), mass=0.12, boxes=B_BOXES, color=ORANGE, friction=0.6)

OUT_A = Box("outline_blue", pos=(A_X, A_Y, TOP + OUT_T / 2), size=(geom.A_W + 0.02, geom.A_D + 0.02, OUT_T), color=WHITE)
OUT_B = Box("outline_orange", pos=(A_X + B_REL_OUT[0], A_Y + B_REL_OUT[1], TOP + OUT_T / 2), size=(0.17, 0.17, OUT_T), color=WHITE)
OBJECTS = [OUT_A, OUT_B, BLUE_P, ORANGE_P]

# -- scoring thresholds ------------------------------------------------------------------------------------------------
Z_HELD = TOP + 0.020          # both origins at least this high: "lifted and held" (rest heights are TOP + 0.0005 / 0.0025)
GATE_TILT = 20.0
X_MAX, Y_MIN = 0.58, -0.50    # off the table / out of reach
ROT_1, ROT_2 = 15.0, 30.0     # peak relative turn (deg)
SEP_2, SEP_3, SEP_4 = 0.12, 0.17, 0.20
OUT_TOL = 0.030
HOLD_STEPS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), held=0, peak=0.0)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def tilt_deg(state: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def yaw_deg(state: PartState) -> float:
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def rel(objects) -> dict:
    """Orange bracket (T-head centre, turn) in the blue housing's frame, plus tilts, heights and the housing's own placement."""
    a, b = objects["blue"], objects["orange"]
    ya = math.radians(yaw_deg(a))
    c, s = math.cos(ya), math.sin(ya)
    d = np.asarray(b.pos, float) - np.asarray(a.pos, float)
    rx, ry = c * d[0] + s * d[1], -s * d[0] + c * d[1]
    turn = (yaw_deg(b) - yaw_deg(a) + 180.0) % 360.0 - 180.0
    return dict(rx=float(rx), ry=float(ry), turn=float(turn), sep=float(math.hypot(rx, ry)),
                tilt_a=tilt_deg(a), tilt_b=tilt_deg(b), za=float(a.pos[2]), zb=float(b.pos[2]),
                yaw_a=float(yaw_deg(a)), ax=float(a.pos[0]), ay=float(a.pos[1]),
                bx=float(b.pos[0]), by=float(b.pos[1]), out=bool(geom.head_outside((rx, ry, math.radians(turn)))))


def classify(objects, peak: float = 0.0):
    """(level, all-1.0-conditions-hold, info) from the current poses and the peak turn so far (deg)."""
    m = rel(objects)
    m["peak"] = peak
    held = m["za"] >= Z_HELD and m["zb"] >= Z_HELD
    if (not held or max(m["tilt_a"], m["tilt_b"]) > GATE_TILT or m["ax"] > X_MAX or m["bx"] > X_MAX
            or m["ay"] < Y_MIN or m["by"] < Y_MIN):
        return 0.0, False, m
    tilt = max(m["tilt_a"], m["tilt_b"])
    level = 0.0
    if peak >= ROT_1:
        level = 0.25
    if peak >= ROT_2 and m["sep"] >= SEP_2:
        level = 0.5
    if peak >= ROT_1 and m["out"] and m["sep"] >= SEP_3 and tilt <= 12.0:
        level = 0.75
    perfect = (peak >= ROT_1 and m["out"] and m["sep"] >= SEP_4 and tilt <= 8.0 and abs(m["yaw_a"]) <= 8.0
               and math.hypot(m["ax"] - A_X, m["ay"] - A_Y) <= OUT_TOL
               and math.hypot(m["bx"] - (A_X + B_REL_OUT[0]), m["by"] - (A_Y + B_REL_OUT[1])) <= OUT_TOL)
    return (1.0 if perfect else level), perfect, m


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    m = rel(objects)
    if (m["za"] >= Z_HELD and m["zb"] >= Z_HELD and max(m["tilt_a"], m["tilt_b"]) <= GATE_TILT):
        _STATE["peak"] = max(_STATE["peak"], abs(m["turn"]))
    level, perfect, _ = classify(objects, _STATE["peak"])
    _STATE["held"] = _STATE["held"] + 1 if perfect else 0
    result = (level, _STATE["held"] >= HOLD_STEPS)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Both brackets held in the air, the orange T-head out of the slot (all head corners beyond the housing's front face),
    the peak turn of the orange bracket relative to the housing at least 15 deg, head centre 0.20 m or more from the housing
    centre, each bracket within 30 mm of its own outline, housing yaw <= 8 deg, both tilts <= 8 deg; held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if either bracket is not lifted at least 20 mm, tilts more than 20 deg, or leaves the table area. 0.25 peak turn of
    the orange bracket >= 15 deg; 0.5 peak turn >= 30 deg and head 0.12 m from the housing centre; 0.75 head out of the slot,
    0.17 m apart, tilts <= 12 deg; 1.0 as in success()."""
    return _update(objects)[0]
