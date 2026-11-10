"""On the station table: a grey trim panel rests on four soft spring seats (on a heavy base) with a red push-pin clip
standing in its hole. A blue backing paddle (a fork on a long stem) lies on the table to the left. The right hand
must press the clip home (10 mm) while the left hand slides the paddle in under the panel and backs it up, so the
panel does not sag; then the paddle is withdrawn and the panel must come back to rest.

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
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Press the red clip into the grey panel's hole until it is fully seated, without letting the panel "
               "sag. Support the panel from underneath with the blue paddle while you press.")

GREY = (0.55, 0.55, 0.58)
DARK = (0.25, 0.25, 0.28)
SILVER = (0.80, 0.80, 0.85)
RED = (0.85, 0.10, 0.10)
BLUE = (0.10, 0.25, 0.90)


def _b(x0, x1, y0, y1, z0, z1, color):
    """Box from its extents in mm (part frame) -> (centre, size, colour) in m."""
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000, (z1 - z0) / 1000),
            color)


# -- layout (world frame: x forward from the arms, y left, metres) ------------------------------------------------------
CX = 0.30                       # station centre line in x
SEAT_X, SEAT_Y = 0.085, 0.070   # seat centres (4)
PANEL_HALF = 0.100              # panel is 200 x 200 x 8 mm
PANEL_Z = TOP + 0.064           # panel centre height on the seats at rest (the settled value is PANEL_REST_Z)
HOLE_Y = -0.020                 # hole / clip axis, relative to the panel centre
HOLE = 17                       # half-width of the 34 mm hole (mm)

# base: two rails along y (x = +-85 mm) on the table, four sprung seats on top. It is heavy so it does not walk.
BASE_Z = TOP + 0.020
SEATS = tuple(
    Slider(f"seat{i}", rest=(sx * SEAT_X, sy * SEAT_Y, 0.030),
           boxes=(_b(sx * 85 - 15, sx * 85 + 15, sy * 70 - 15, sy * 70 + 15, 20, 40, SILVER),),
           travel=0.020, rate=125.0, damping=3.0, mass=0.02, friction=0.6)
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]))
BASE = part("base", pos=(CX, 0.0, BASE_Z), mass=4.0, color=DARK, friction=0.8, sliders=SEATS,
            boxes=[_b(-100, -70, -120, 120, -20, 20, DARK), _b(70, 100, -120, 120, -20, 20, DARK)])

# panel with a clip slider: pressed down by up to 10 mm against a 0.6 N/mm detent spring
_h = HOLE_Y * 1000
CLIP_REST_Z = 0.011              # clip frame (centre of its stem) above the panel centre
CLIP = Slider("clip", rest=(0.0, HOLE_Y, CLIP_REST_Z),
              boxes=(_b(-15, 15, _h - 15, _h + 15, -4, 26, RED), _b(-22, 22, _h - 22, _h + 22, 20, 26, RED)),
              travel=0.010, rate=600.0, damping=3.0, mass=0.02, friction=0.5)
PANEL = part("panel", pos=(CX, 0.0, PANEL_Z), mass=0.10, color=GREY, friction=0.6, sliders=(CLIP,),
             boxes=[_b(-100, 100, _h + HOLE, 100, -4, 4, GREY), _b(-100, 100, -100, _h - HOLE, -4, 4, GREY),
                    _b(-100, -HOLE, _h - HOLE, _h + HOLE, -4, 4, GREY), _b(HOLE, 100, _h - HOLE, _h + HOLE, -4, 4, GREY)])

# paddle frame: origin on the clip axis at the top face of the fork's prongs; stem toward +y; post to hold it by
PADDLE_START = (CX, 0.18 + 0.0, TOP + 0.030)
PADDLE = part("paddle", pos=PADDLE_START, mass=0.20, color=BLUE, friction=0.5,
              boxes=[_b(-38, -22, -30, 30, -12, 0, BLUE), _b(22, 38, -30, 30, -12, 0, BLUE),
                     _b(-46, 46, 30, 45, -12, 0, BLUE), _b(-20, 20, 45, 300, -30, -6, BLUE),
                     _b(-20, 20, 260, 300, -30, 30, BLUE)])
OBJECTS = [BASE, PANEL, PADDLE]

# -- scoring thresholds ----------------------------------------------------------------------------------------------
PANEL_REST_Z = TOP + 0.0601      # panel centre at rest on the seats (measured in the settle run: sag 1.9 mm vs 0.062)
GATE_FROM_STEP = 40
PRESS_ENGAGED_MM = 3.0
PRESS_SEATED_MM = 9.0
SAG_SUPPORT_MM = 8.0
SAG_SEATED_MM = 3.0
SAG_BOTTOM_MM = 18.0
REST_TOL_MM = 4.0
SHIFT_GATE_MM = 40.0
TILT_GATE_DEG = 20.0
NEAR_MM = 40.0
CLEAR_Y = 0.15                   # paddle origin this far toward +y of the panel centre: fork 20 mm clear of the panel
SEAT_STEPS = 10
HOLD_STEPS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(start=None, step=0, scored=None, last=(0.0, False), tripped=False, eng=False, supp=False,
                  seat75=False, seat_run=0, seated=False, hist=[])


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _tilt_deg(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def measures(objects):
    """(sag_mm, press_mm, near_mm, gap_mm, tilt_deg): panel sag below rest, clip travel, paddle fork's xy distance from
    the clip axis, paddle top minus panel underside (vertical, mm), panel tilt."""
    panel, paddle = objects["panel"], objects["paddle"]
    sag = 1000.0 * (PANEL_REST_Z - float(panel.pos[2]))
    press = -1000.0 * float(panel.sliders.get("clip", 0.0))
    axis = np.asarray(panel.to_world((0.0, HOLE_Y, 0.0)), float)
    near = 1000.0 * float(np.linalg.norm(np.asarray(paddle.pos[:2]) - axis[:2]))
    gap = 1000.0 * (float(paddle.pos[2]) - (float(panel.pos[2]) - 0.004))
    return sag, press, near, gap, _tilt_deg(panel)


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    panel, base, paddle = objects["panel"], objects["base"], objects["paddle"]
    if _STATE["start"] is None:
        _STATE["start"] = (np.asarray(panel.pos[:2], float).copy(), np.asarray(base.pos[:2], float).copy())
    p0, b0 = _STATE["start"]
    sag, press, near, gap, tilt = measures(objects)
    shift = 1000.0 * max(float(np.linalg.norm(np.asarray(panel.pos[:2]) - p0)),
                         float(np.linalg.norm(np.asarray(base.pos[:2]) - b0)))
    if _STATE["step"] > GATE_FROM_STEP and (shift > SHIFT_GATE_MM or tilt > TILT_GATE_DEG
                                            or (sag > SAG_BOTTOM_MM and press >= PRESS_ENGAGED_MM)
                                            or float(paddle.pos[2]) > TOP + 0.20):
        _STATE["tripped"] = True
    if press >= PRESS_ENGAGED_MM:
        _STATE["eng"] = True
    if near <= NEAR_MM and abs(gap) <= 3.0 and sag <= SAG_SUPPORT_MM:
        _STATE["supp"] = True
    if press >= PRESS_SEATED_MM and sag <= SAG_SEATED_MM:
        _STATE["seat_run"] += 1
        if _STATE["seat_run"] >= 1:
            _STATE["seat75"] = True
        if _STATE["seat_run"] >= SEAT_STEPS:
            _STATE["seated"] = True
    else:
        _STATE["seat_run"] = 0
    h = _STATE["hist"]
    h.append(np.asarray(panel.pos, float).copy())
    del h[:-HOLD_STEPS]
    still = len(h) == HOLD_STEPS and all(np.linalg.norm(p - h[-1]) < 0.0007 for p in h)
    paddle_clear = float(paddle.pos[1]) - float(panel.pos[1]) >= CLEAR_Y
    final = _STATE["seated"] and paddle_clear and abs(sag) <= REST_TOL_MM
    if _STATE["tripped"]:
        result = (0.0, False)
    elif final:
        result = (1.0, bool(still))
    elif _STATE["seat75"]:
        result = (0.75, False)
    elif _STATE["supp"]:
        result = (0.5, False)
    elif _STATE["eng"]:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Clip seated (>= 9 mm for 0.5 s) with the panel sagging <= 3 mm, then the paddle is clear of the panel and the
    panel is back at rest (+-4 mm) and still for 1 s; no zero gate."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 clip pressed >= 3 mm; 0.5 paddle touching the panel underside within 40 mm of the clip axis while sag <= 8 mm;
    0.75 clip >= 9 mm with sag <= 3 mm; 1.0 seated, then paddle clear and panel at rest +-4 mm, still 1 s. Zero if
    (after step 40) the panel bottomed (sag > 18 mm) with the clip pressed, slid or tipped, or the paddle was thrown."""
    return _update(objects)[0]
