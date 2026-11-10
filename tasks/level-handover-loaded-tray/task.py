"""On the station table: a white sample rack (230 x 90 mm plate with an upright handle at each end, 0.15 kg) carrying three
blue cups stands on a grey pad at the far left. A second grey pad is at the far right, 0.84 m away. Neither arm can reach
both pads, so the rack has to be lifted by one hand, handed over to the other hand in mid-air near the middle of the
table, kept level the whole way (the cups must stay upright), and set down on the right pad.

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

INSTRUCTION = ("Carry the white rack of three blue cups from the grey pad on the left to the grey pad on the right, "
               "handing it from one arm to the other in the air, and keep it level so no cup tips over.")

WHITE = (0.95, 0.95, 0.95)
BLUE = (0.10, 0.30, 0.90)
GREY = (0.62, 0.62, 0.65)

PAD_T = 0.003
PAD_L = (0.30, 0.40)           # left pad centre (the rack starts here)
PAD_R = (0.30, -0.40)          # right pad centre (the rack must end here)
PAD_SIZE = (0.13, 0.34)        # x, y
RACK_Z = TOP + PAD_T           # rack body origin: on its pad

# rack, in its own frame (origin: centre of the plate underside; +y toward the left)
HANDLE_Y = 0.135
RACK_BOXES = [
    ((0.0, 0.0, 0.006), (0.09, 0.23, 0.012), WHITE),                       # plate
    ((0.0, HANDLE_Y, 0.028), (0.06, 0.04, 0.040), WHITE),                  # handle at the +y end
    ((0.0, -HANDLE_Y, 0.028), (0.06, 0.04, 0.040), WHITE),                 # handle at the -y end
]
for _cy in (-0.055, 0.0, 0.055):                                            # shallow 6 mm wells keep each cup in its slot
    RACK_BOXES += [((0.0255, _cy, 0.015), (0.003, 0.054, 0.006), WHITE), ((-0.0255, _cy, 0.015), (0.003, 0.054, 0.006), WHITE),
                   ((0.0, _cy + 0.0255, 0.015), (0.048, 0.003, 0.006), WHITE), ((0.0, _cy - 0.0255, 0.015), (0.048, 0.003, 0.006), WHITE)]
RACK = part("rack", pos=(PAD_L[0], PAD_L[1], RACK_Z), mass=0.15, boxes=RACK_BOXES, color=WHITE, friction=0.6)

CUP_R, CUP_H = 0.0225, 0.060
CUP_Y = (-0.055, 0.0, 0.055)   # slots on the plate, rack frame
CUPS = [part(f"cup_{i}", pos=(PAD_L[0], PAD_L[1] + y, RACK_Z + 0.012 + 0.0005), mass=0.03, friction=0.8, boxes=[((0, 0, 0.001), (0.02, 0.02, 0.002), BLUE)],
             cylinders=[((0, 0, CUP_H / 2), CUP_R, CUP_H, BLUE)], color=BLUE)
        for i, y in enumerate(CUP_Y)]
CUP_NAMES = [c.name for c in CUPS]

PAD_A = Box("pad_left", pos=(PAD_L[0], PAD_L[1], TOP + PAD_T / 2), size=(PAD_SIZE[0], PAD_SIZE[1], PAD_T), color=GREY)
PAD_B = Box("pad_right", pos=(PAD_R[0], PAD_R[1], TOP + PAD_T / 2), size=(PAD_SIZE[0], PAD_SIZE[1], PAD_T), color=GREY)
OBJECTS = [PAD_A, PAD_B, RACK, *CUPS]

# -- scoring thresholds ------------------------------------------------------------------------------------------------
LIFT_MIN = 0.040             # rack lifted at least this far above its pad (0.25)
CROSS_Y = 0.150              # ... and carried past y = +0.15 (0.25)
FAR_Y = -0.300               # carried past y = -0.30: beyond the left arm's reach, so the right hand has it (0.5)
LEVEL_DEG, GATE_DEG = 8.0, 20.0
PLACE_TOL, PLACE_YAW_DEG = 0.015, 8.0
CUP_SLOT_TOL = 0.015
HOLD_STEPS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), lifted=False, crossed=False, far=False, max_tilt=0.0,
                  gate=False, held=0)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def tilt_deg(state: PartState) -> float:
    """Angle between the part's z axis and the world's vertical."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def yaw_deg(state: PartState) -> float:
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def _dist(state: PartState, xy) -> float:
    return float(np.hypot(state.pos[0] - xy[0], state.pos[1] - xy[1]))


def cup_info(objects):
    """(all cups upright on the plate, all within their slots, max cup tilt)."""
    rack = objects["rack"]
    on = slots = True
    worst = 0.0
    for name, y in zip(CUP_NAMES, CUP_Y):
        c = objects[name]
        worst = max(worst, tilt_deg(c))
        lp = rack.to_local(c.pos)
        if abs(lp[0]) > 0.06 or abs(lp[1] - y) > 0.10 or not (0.005 < lp[2] < 0.05) or tilt_deg(c) > GATE_DEG:
            on = False
        if abs(lp[0]) > CUP_SLOT_TOL or abs(lp[1] - y) > CUP_SLOT_TOL:
            slots = False
    return on, on and slots, worst


def classify(objects, memo: dict):
    """Pure function of the current poses and the memo (flags seen so far): (level, all-1.0-conditions-hold, info).
    ``memo`` holds lifted/crossed/far/max_tilt/gate and is updated here."""
    rack = objects["rack"]
    on, slots, cup_tilt = cup_info(objects)
    lift = float(rack.pos[2] - RACK_Z)
    cy = float(rack.pos[1])
    t = tilt_deg(rack)
    if lift > 0.020:                                       # tilt only counts while the rack is carried
        memo["max_tilt"] = max(memo["max_tilt"], t)
    resting = lift < 0.012
    off_pad = resting and _dist(rack, PAD_L) > 0.10 and _dist(rack, PAD_R) > 0.10
    if not on or t > GATE_DEG or memo["max_tilt"] > GATE_DEG or off_pad:
        memo["gate"] = True
    if memo["gate"]:
        return 0.0, False, dict(on=on, lift=lift, y=cy, tilt=t, max_tilt=memo["max_tilt"], gate=True)
    if lift > LIFT_MIN and on and cy < CROSS_Y:
        memo["lifted"] = memo["crossed"] = True
    if memo["crossed"] and cy < FAR_Y and lift > 0.020:
        memo["far"] = True
    info = dict(on=on, slots=slots, lift=lift, y=cy, tilt=t, max_tilt=memo["max_tilt"], gate=False,
                place_d=_dist(rack, PAD_R), yaw=yaw_deg(rack))
    placed = (memo["far"] and resting and _dist(rack, PAD_R) <= PLACE_TOL and abs(yaw_deg(rack)) <= PLACE_YAW_DEG
              and t < 4.0)
    level_ok = memo["max_tilt"] <= LEVEL_DEG
    if placed and level_ok and slots:
        return 1.0, True, info
    if placed and level_ok:
        return 0.75, False, info
    if memo["far"] and level_ok:
        return 0.5, False, info
    if memo["crossed"]:
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
    """The rack was lifted from the left pad, carried over y = -0.30 (only possible after a handover) at <= 8 degrees
    of tilt, and rests on the right pad within 15 mm of its centre and 8 degrees of yaw, all three cups upright on
    the plate within 15 mm of their slots, held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 lifted >= 40 mm with the cups upright and carried past y = +0.15; 0.5 carried past y = -0.30 (handed over)
    level (<= 8 degrees); 0.75 level handover and resting on the right pad within 15 mm / 8 degrees of yaw; 1.0 as in success().
    Zero if a cup falls off or tips over 20 degrees, the rack tilts over 20 degrees while carried, or the rack is
    left on the table away from both pads."""
    return _update(objects)[0]
