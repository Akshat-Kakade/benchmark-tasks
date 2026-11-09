"""On the station table: a thin white slip-sheet (260 x 260 x 4 mm) lies on the table with an orange 80 mm block standing
on its middle. A green tab (40 x 60 x 40 mm) stands up from the sheet's far-left edge. The left hand must pull the sheet
out from under the block by its tab while the right hand keeps the block from moving.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Pull the white sheet out from under the orange block by its green tab with the left hand, while the "
               "right hand keeps the orange block from moving.")

WHITE = (0.95, 0.95, 0.95)
GREEN = (0.10, 0.70, 0.20)
ORANGE = (1.00, 0.50, 0.00)

# -- layout (world frame: x forward from the arms, y left, metres) -----------------------------------------------------
SHEET_T = 0.004
SHEET_SIZE = 0.26
SHEET_XY = (0.30, 0.04)
BLOCK_SIZE = 0.08
BLOCK_XY = (0.30, 0.0)

SHEET = part("sheet", pos=(SHEET_XY[0], SHEET_XY[1], TOP + SHEET_T / 2), mass=0.15, color=WHITE, friction=0.3,
             boxes=[((0, 0, 0), (SHEET_SIZE, SHEET_SIZE, SHEET_T), WHITE),
                    ((0, 0.15, 0.018), (0.04, 0.06, 0.04), GREEN)])       # tab: x 40, y 60, z 40 mm, on the +y edge
LOAD = part("load", pos=(BLOCK_XY[0], BLOCK_XY[1], TOP + SHEET_T + BLOCK_SIZE / 2), mass=0.35, color=ORANGE,
            friction=0.3, boxes=[((0, 0, 0), (BLOCK_SIZE,) * 3, ORANGE)])
OBJECTS = [SHEET, LOAD]

# -- scoring thresholds (mm / degrees) --------------------------------------------------------------------------------
GATE_FROM_STEP = 40
GATE_TILT_DEG = 20.0
GATE_SHIFT_MM = 80.0
GATE_LIFT_MM = 10.0
ENGAGED_MM = 60.0          # 0.25: sheet moved this far in +y
PARTIAL_MM = 100.0         # 0.5: sheet moved this far in +y
CLEAR_GAP_MM = 5.0         # sheet's bounding box this far from the block's footprint
SHIFT_FULL_MM = 12.0
SHIFT_PART_MM = 25.0
TILT_FULL_DEG = 5.0
REST_LIFT_MM = 3.0
HOLD_STEPS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(start=None, step=0, scored=None, last=(0.0, False), tripped=False, block_hist=[], sheet_hist=[])


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _tilt_deg(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def _footprint(state, half_xyz):
    """(low, high) xy of a box centred on the body origin, from its turned corners (partkit's low/high assume the
    bounding box is centred on the origin, which the sheet's is not)."""
    h = np.asarray(half_xyz, float)
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * h
    pts = np.array([state.to_world(c) for c in corners])
    return pts.min(axis=0), pts.max(axis=0)


def _gap_mm(sheet, block) -> float:
    """Largest clearance (mm) between the sheet slab's and the block's footprints along x or y; negative if they overlap."""
    slo, shi = _footprint(sheet, (SHEET_SIZE / 2, SHEET_SIZE / 2, SHEET_T / 2))
    blo, bhi = _footprint(block, (BLOCK_SIZE / 2,) * 3)
    gaps = []
    for ax in (0, 1):
        gaps.append(slo[ax] - bhi[ax])
        gaps.append(blo[ax] - shi[ax])
    return 1000.0 * max(gaps)


def measures(objects, start=None):
    """(sheet_dy_mm, block_shift_mm, tilt_deg, lift_mm, gap_mm) for the given states. ``start`` is (sheet_y, block_xy,
    block_z); without it, the nominal layout."""
    sheet, block = objects["sheet"], objects["load"]
    if start is None:
        start = (SHEET_XY[1], np.array(BLOCK_XY), TOP + SHEET_T + BLOCK_SIZE / 2)
    sy, bxy, bz = start
    dy = 1000.0 * (float(sheet.pos[1]) - sy)
    shift = 1000.0 * float(np.linalg.norm(np.asarray(block.pos[:2]) - bxy))
    lift = 1000.0 * (float(block.pos[2]) - bz)
    return dy, shift, _tilt_deg(block), lift, _gap_mm(sheet, block)


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    sheet, block = objects["sheet"], objects["load"]
    if _STATE["start"] is None:
        _STATE["start"] = (float(sheet.pos[1]), np.asarray(block.pos[:2], float).copy(), float(block.pos[2]))
    dy, shift, tilt, lift, gap = measures(objects, _STATE["start"])
    if _STATE["step"] > GATE_FROM_STEP and (tilt > GATE_TILT_DEG or shift > GATE_SHIFT_MM or lift > GATE_LIFT_MM):
        _STATE["tripped"] = True
    hb, hs = _STATE["block_hist"], _STATE["sheet_hist"]
    hb.append(np.asarray(block.pos, float).copy())
    hs.append(np.asarray(sheet.pos, float).copy())
    del hb[:-HOLD_STEPS], hs[:-HOLD_STEPS]
    still = (len(hb) == HOLD_STEPS and all(np.linalg.norm(p - hb[-1]) < 0.001 for p in hb)
             and all(np.linalg.norm(p - hs[-1]) < 0.002 for p in hs))
    clear = gap >= CLEAR_GAP_MM
    if _STATE["tripped"]:
        result = (0.0, False)
    elif clear and shift <= SHIFT_FULL_MM and tilt <= TILT_FULL_DEG and lift <= REST_LIFT_MM:
        result = (1.0, bool(still))
    elif clear and shift <= SHIFT_PART_MM:
        result = (0.75, False)
    elif clear or dy >= PARTIAL_MM:
        result = (0.5, False)
    elif dy >= ENGAGED_MM:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Sheet fully out (>= 5 mm clear of the block's footprint), block moved <= 12 mm, tilted <= 5 degrees, resting on
    the table (<= 3 mm up), block and sheet still for 1 s; never lifted > 10 mm, tilted > 20 deg or moved > 80 mm."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 sheet pulled 60 mm in +y; 0.5 pulled 100 mm, or clear but the block moved 25-80 mm; 0.75 clear with the block
    moved 12-25 mm or tilted 5-20 degrees; 1.0 clear, block moved <= 12 mm, tilt <= 5, resting, still 1 s. Zero if the
    block was ever lifted > 10 mm, tilted > 20 degrees or moved > 80 mm after step 40."""
    return _update(objects)[0]
