"""On the station table: a blue rectangular frame (a closed flat rectangle of four joined bars, 0.16 kg) rests on two low grey
pedestals, with an orange upright handle stem on each of its two long ends. A grey post stands upright 0.12 m in front of the
frame, inside a green seating outline. Both arms pinch the handles, lift the frame, carry its opening over the post, and
lower it level so it sits flat on the green outline with the post standing in the middle of the opening and touching no bar.
The frame is one free rigid body (a rigid box assembly): nothing is pinned, hinged or fastened.

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
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Hold the blue rectangular frame by its two orange handles, move its opening over the grey post, and lower it "
               "level onto the green outline without touching the post.")

BLUE = (0.10, 0.30, 0.90)
ORANGE = (0.95, 0.50, 0.08)
GREEN = (0.15, 0.65, 0.25)
GREY = (0.62, 0.62, 0.65)

# frame, in its own frame (origin: centre of the underside; x forward, y left)
FR_X, FR_Y, FR_T = 0.26, 0.34, 0.016        # outer size and bar thickness
OP_X, OP_Y = 0.16, 0.22                     # opening
BAR_X = (FR_X - OP_X) / 2                   # 0.05: front and back bars (full width)
BAR_Y = (FR_Y - OP_Y) / 2                   # 0.06: side bars
HANDLE_Y = 0.185
STEM = (0.04, 0.035, 0.086)                 # handle stem (x, y, z) from the bar underside up: 70 mm above the bar
PINCH_Z = 0.062                             # pinch height above the frame origin

FRAME_BOXES = [
    ((+(OP_X + BAR_X) / 2, 0.0, FR_T / 2), (BAR_X, FR_Y, FR_T), BLUE),
    ((-(OP_X + BAR_X) / 2, 0.0, FR_T / 2), (BAR_X, FR_Y, FR_T), BLUE),
    ((0.0, +(OP_Y + BAR_Y) / 2, FR_T / 2), (OP_X, BAR_Y, FR_T), BLUE),
    ((0.0, -(OP_Y + BAR_Y) / 2, FR_T / 2), (OP_X, BAR_Y, FR_T), BLUE),
    ((0.0, +HANDLE_Y, STEM[2] / 2), STEM, ORANGE),
    ((0.0, -HANDLE_Y, STEM[2] / 2), STEM, ORANGE),
]

X0 = 0.18                      # start x of the frame (its back edge, x 0.05, and the pedestals, x 0.165-0.195, lie behind the final footprint, x 0.21-0.47)
PED_H = 0.085
POST_X, POST_R, POST_H = 0.34, 0.03, 0.07
POST_TOP = TOP + POST_H        # 0.8175
OUT_T = 0.008
SEAT_Z = TOP + OUT_T           # frame origin when seated on the outline: 0.7555
START_Z = TOP + PED_H + 0.0005

FRAME = part("frame", pos=(X0, 0.0, START_Z), mass=0.16, boxes=FRAME_BOXES, color=BLUE, friction=0.6)
POST = Cylinder("post", pos=(POST_X, 0.0, TOP + POST_H / 2), radius=POST_R, height=POST_H, color=GREY)
OUTLINE = Box("seat_outline", pos=(POST_X, 0.0, TOP + OUT_T / 2), size=(0.27, 0.35, OUT_T), color=GREEN)
PED_L = Box("ped_left", pos=(X0, +HANDLE_Y, TOP + PED_H / 2), size=(0.03, 0.03, PED_H), color=GREY)
PED_R = Box("ped_right", pos=(X0, -HANDLE_Y, TOP + PED_H / 2), size=(0.03, 0.03, PED_H), color=GREY)
OBJECTS = [OUTLINE, POST, PED_L, PED_R, FRAME]

# -- scoring thresholds ------------------------------------------------------------------------------------------------
GATE_TILT = 12.0
OFF_BASE = 0.06               # frame centre this far from the post and below the post top: fallen aside
L1_OFF = 0.040
L1_Z = POST_TOP - 0.007        # 0.810
L2_OFF, L2_Z, L2_TILT = 0.025, SEAT_Z + 0.020, 12.0       # 0.75
L3_OFF, L3_Z, L3_TILT, L3_CLEAR = 0.015, SEAT_Z + 0.012, 8.0, 0.008   # 1.0
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


def fit(objects) -> dict:
    """Measured fit of the frame to the post: frame-frame coordinates of the post axis (px, py), horizontal offset of the
    frame centre from the post, frame underside height z, tilt, and the smallest gap between the post and an inner bar
    side (x or y; negative = the post is not in the opening or touches a bar)."""
    f = objects["frame"]
    p = f.to_local((POST_X, 0.0, f.pos[2]))
    px, py = float(p[0]), float(p[1])
    cx = OP_X / 2 - (abs(px) + POST_R)
    cy = OP_Y / 2 - (abs(py) + POST_R)
    return dict(px=px, py=py, off=float(math.hypot(f.pos[0] - POST_X, f.pos[1])), z=float(f.pos[2]),
                tilt=tilt_deg(f), clear=min(cx, cy), inside=bool(abs(px) < OP_X / 2 and abs(py) < OP_Y / 2))


def classify(objects):
    """(level, all-1.0-conditions-hold, info) from the current pose only."""
    m = fit(objects)
    low = m["z"] < POST_TOP                                    # the bars are at or below the post top
    if m["tilt"] > GATE_TILT or (low and m["clear"] < 0.0 and m["off"] < OFF_BASE) or (low and m["off"] >= OFF_BASE):
        return 0.0, False, m                                   # tilted, post hit by a bar, or frame fallen aside
    if not m["inside"]:
        return 0.0, False, m
    if (m["off"] <= L3_OFF and m["z"] <= L3_Z and m["tilt"] <= L3_TILT and m["clear"] >= L3_CLEAR):
        return 1.0, True, m
    if m["off"] <= L2_OFF and m["z"] <= L2_Z and m["tilt"] <= L2_TILT:
        return 0.75, False, m
    if m["off"] <= L1_OFF and m["z"] <= L1_Z:
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
    """The frame sits level on the green outline (underside within 12 mm of it), its centre within 15 mm of the post axis,
    tilt <= 8 deg, the post at least 8 mm from every inner bar side; held for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the frame tilts over 12 deg, a bar reaches the post, or the frame ends fallen aside; 0 if the post is not in the
    opening. 0.25 post inside the opening (frame above it); 0.5 centre within 40 mm and underside below 810 mm (below the
    post top); 0.75 centre within 25 mm, underside <= 20 mm above the outline, tilt <= 12 deg; 1.0 as in success()."""
    return _update(objects)[0]
