"""On the station table: a telescoping extraction hood. An orange mounting foot (a slab with a side tab and a grip post) carries a
blue arm that slides along the x axis; the arm ends in a blue hood mouth (also with a side tab and a grip post) that hangs 4 mm above
a white work mark. A weak spring (modelled in the physics engine) pulls the arm back toward the foot. One hand must hold the mouth
still over the white mark while the other slides the foot about 7 cm toward the robot, so the arm extends and the mouth does not move.

The arm is a prismatic slider of the foot (partkit sliders run along the part's local z axis, downward), so the foot's body frame is
turned -90 degrees about y: local (x, y, z) = world (up, y, -x), and the arm's local travel is world +x. The foot is spawned
upright and turned flat by a write_pose in on_reset (the marks are written again after it).

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

INSTRUCTION = ("Hold the blue hood mouth over the white work mark with your right hand. Move the orange mounting foot toward you "
               "to the green line with your left hand, letting the arm extend so the mouth stays put.")

ORANGE = (1.00, 0.50, 0.00)
BLUE = (0.12, 0.32, 0.90)
WHITE = (0.95, 0.95, 0.95)
GREEN = (0.10, 0.70, 0.20)

# -- layout (station frame, world metres) --------------------------------------------------------------------------------
SHIFT = 0.0                       # whole scene shifted in x (toward the robot when negative)
FOOT_X0 = 0.270 + SHIFT           # foot centre
FOOT_Y0 = 0.0
FOOT_H = 0.035
ORIGIN_Z = TOP + FOOT_H / 2       # foot body origin: foot centre
MOUTH_X = 0.400 + SHIFT           # mouth centre at rest
MOUTH_ZC = TOP + 0.0355           # mouth centre height (bottom 4.5 mm above the table, top level with the arm)
FOOT_REAR0 = FOOT_X0 - 0.060      # foot rear (toward-robot) edge at the start
TARGET_D = 0.0715                 # middle of the 65-78 mm band
GREEN_X = FOOT_REAR0 - TARGET_D - 0.004   # green line centre: just behind the foot's rear edge at the target
QUAT_FLAT = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))   # xyzw: Ry(-90): local z -> world -x, local x -> world up


def L(dx, dy, dz):
    """World offset from the foot origin -> foot-frame vector."""
    return (dz, dy, -dx)


def S(sx, sy, sz):
    """World box size -> foot-frame size."""
    return (sz, sy, sx)


K_SPRING = 40.0                   # N/m, modelled retracting spring
STROKE = 0.080                    # hard stop (m)
FOOT_POST_Y = 0.110               # grip posts (20 mm wide in y, 80 mm tall) on side tabs
MOUTH_POST_Y = -0.110
FOOT_PINCH_Z = TOP + 0.0600       # TCP height pinching the posts
MOUTH_PINCH_Z = TOP + 0.0680

_fz = TOP + 0.0075                # foot-tab centre height (tab 15 mm thick, 0.5 mm above the table)
_mz = TOP + 0.0050 + 0.0075       # mouth-tab centre height
FOOT_BOXES = [
    (L(0, 0, 0), S(0.120, 0.090, FOOT_H), ORANGE),
    (L(0, 0.090, _fz + 0.0005 - ORIGIN_Z), S(0.050, 0.090, 0.015), ORANGE),                         # side tab
    (L(0, FOOT_POST_Y, _fz + 0.0005 + 0.0075 + 0.040 - ORIGIN_Z), S(0.040, 0.020, 0.080), ORANGE),  # grip post (80 mm stem)
]
_arm_c = TOP + FOOT_H + 0.001 + 0.015      # arm bar centre
ARM = Slider(
    name="arm", rest=L(0, 0, 0), travel=STROKE, rate=K_SPRING, damping=5.0, mass=0.12, friction=0.1,
    boxes=((L(0.025, 0, _arm_c - ORIGIN_Z), S(0.150, 0.055, 0.030), BLUE),                                  # arm bar x 0.22..0.37 (rel. foot)
           (L(MOUTH_X - FOOT_X0, 0, MOUTH_ZC - ORIGIN_Z), S(0.060, 0.075, 0.0615), BLUE),                   # hood mouth
           (L(MOUTH_X - FOOT_X0, -0.0825, _mz - ORIGIN_Z), S(0.050, 0.090, 0.015), BLUE),                    # side tab
           (L(MOUTH_X - FOOT_X0, MOUTH_POST_Y, _mz + 0.0075 + 0.040 - ORIGIN_Z), S(0.040, 0.020, 0.080), BLUE)))
FOOT = part("foot", pos=(FOOT_X0, FOOT_Y0, ORIGIN_Z + 0.16), mass=0.18, color=ORANGE, friction=0.1, boxes=FOOT_BOXES,
            sliders=(ARM,))
MARK_W = part("mark_work", pos=(MOUTH_X, 0.0, TOP + 0.001), mass=3.0, color=WHITE, friction=0.8,
              boxes=[((0, 0, 0), (0.010, 0.010, 0.002), WHITE)], cylinders=[((0, 0, 0), 0.0325, 0.002, WHITE)])
MARK_G = part("mark_green", pos=(GREEN_X, 0.0, TOP + 0.0004), mass=3.0, color=GREEN, friction=0.8,
              boxes=[((0, 0, 0), (0.008, 0.120, 0.0008), GREEN)])
OBJECTS = [FOOT, MARK_W, MARK_G]

# -- scoring thresholds ---------------------------------------------------------------------------------------------------
GATE_FROM_STEP = 40
POP_DX = 0.0442           # the teleported foot slides about 0.4x this far toward -x while it settles (measured): write it forward so it rests at FOOT_X0
SETTLE_STEPS = 15         # the freshly placed foot pops and settles in the first steps: no baseline before
GATE_TILT_DEG = 10.0
GATE_LIFT_MM = 20.0
GATE_SHIFT_MM = 120.0     # foot leaves the reachable area
YAW_TILT_MAX = 8.0
LEVELS = (               # (score, D_min mm, D_max mm, d_max mm, yerr_max mm)
    (0.25, 20.0, 1e9, 45.0, 1e9),
    (0.5, 40.0, 1e9, 30.0, 1e9),
    (0.75, 55.0, 1e9, 20.0, 20.0),
)
FULL = dict(D=(65.0, 78.0), d=12.0, yerr=12.0, ext=60.0)
HOLD_STEPS = 20           # 1.0 s

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(start=None, mouth0=None, step=0, scored=None, last=(0.0, False), tripped=False, streak=0)
    if environment is not None:
        write_pose(environment, "foot", (FOOT_X0 + POP_DX, FOOT_Y0, float(ORIGIN_Z)), QUAT_FLAT)
        write_pose(environment, "mark_work", (MOUTH_X, 0.0, TOP + 0.001), (0.0, 0.0, 0.0, 1.0))
        write_pose(environment, "mark_green", (GREEN_X, 0.0, TOP + 0.0004), (0.0, 0.0, 0.0, 1.0))


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _tilt_deg(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 0])))))     # local x is up


def _yaw_deg(state) -> float:
    r = state.rotation()
    return math.degrees(math.atan2(-float(r[1, 2]), -float(r[0, 2])))                       # local z -> world -x


def _ext(state) -> float:
    return -float(state.sliders.get("arm", 0.0))


def mouth_pos(state) -> np.ndarray:
    """World centre of the hood mouth from the foot pose and the arm extension."""
    return state.to_world(L(MOUTH_X - FOOT_X0 + _ext(state), 0.0, MOUTH_ZC - ORIGIN_Z))


def _metrics(state) -> dict:
    st, m0 = _STATE["start"], _STATE["mouth0"]
    pos = np.asarray(state.pos, float)
    m = mouth_pos(state)
    return dict(D=1000.0 * float(st[0] - pos[0]), yerr=1000.0 * abs(float(pos[1] - st[1])),
                d=1000.0 * float(np.linalg.norm(m - m0)), ext=1000.0 * _ext(state),
                tilt=_tilt_deg(state), yaw=abs(_yaw_deg(state)),
                lift=1000.0 * abs(float(pos[2] - st[2])), shift=1000.0 * float(np.linalg.norm(pos[:2] - st[:2])))


def _level(m: dict) -> float:
    best = 0.0
    for sc, dmin, dmax, dd, ye in LEVELS:
        if dmin <= m["D"] <= dmax and m["d"] <= dd and m["yerr"] <= ye:
            best = sc
    return best


def _full_now(m: dict) -> bool:
    lo, hi = FULL["D"]
    return (lo <= m["D"] <= hi and m["d"] <= FULL["d"] and m["yerr"] <= FULL["yerr"] and m["ext"] >= FULL["ext"]
            and m["tilt"] <= YAW_TILT_MAX and m["yaw"] <= YAW_TILT_MAX)


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    foot = objects["foot"]
    if _STATE["step"] < SETTLE_STEPS:
        _STATE["last"] = (0.0, False)
        return _STATE["last"]
    if _STATE["start"] is None:
        _STATE["start"] = np.asarray(foot.pos, float).copy()
        _STATE["mouth0"] = mouth_pos(foot).copy()
    m = _metrics(foot)
    if _STATE["step"] > GATE_FROM_STEP and (m["tilt"] > GATE_TILT_DEG or m["lift"] > GATE_LIFT_MM
                                            or m["shift"] > GATE_SHIFT_MM):
        _STATE["tripped"] = True
    _STATE["streak"] = _STATE["streak"] + 1 if _full_now(m) else 0
    if _STATE["tripped"]:
        result = (0.0, False)
    elif _STATE["streak"] >= HOLD_STEPS:
        result = (1.0, True)
    else:
        result = (min(_level(m), 0.75), False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Foot moved 65-78 mm toward the robot with its y within 12 mm, mouth centre within 12 mm of its start, arm extended >= 60 mm,
    foot and mouth tilt and yaw <= 8 degrees, all held for 1 s; no zero gate tripped."""
    return _update(objects)[1]


def score(objects) -> float:
    """Read at the end. D = foot travel toward -x, d = mouth-centre distance from its start. 0.25: D >= 20 mm, d <= 45 mm;
    0.5: D >= 40 mm, d <= 30 mm; 0.75: D >= 55 mm, d <= 20 mm, foot y error <= 20 mm; 1.0: D 65-78 mm, d <= 12 mm, y error <= 12 mm,
    extension >= 60 mm, tilt and yaw <= 8 degrees, held 1 s. Zero if after step 40 the foot tilted over 10 degrees, lifted over
    20 mm or moved over 120 mm."""
    return _update(objects)[0]
