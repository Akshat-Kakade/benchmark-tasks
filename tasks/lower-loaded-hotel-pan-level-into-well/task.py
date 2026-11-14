"""On the station table: a stainless half-size hotel pan (325 x 265 mm, 40 mm deep, 10 mm flange, filled with 20 mm of cold water: 2.4 kg
simulated as one fused body) with an upright orange handle tab on the flange at each short end. Beside it a black steam-table
counter (40 mm high) has a pit sized for the pan body (6 mm clearance on each side). Both arms must carry the pan by its two
handles and lower it level into the pit so its flange rests on the counter all round, then let go. A pan scores 0 if it tilts
so that water would overflow at its lowest rim corner (a spill), if it is dropped (hits the counter or pit at over 0.5 m/s) or if it is pushed off the table.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Move the full steel pan from the table into the black steam-table well and let go, so it sits in the well "
               "with its flange resting on the counter.")

STEEL = (0.72, 0.72, 0.75)
FOOD = (0.25, 0.50, 0.85)                       # water (drawn as a fused blue box)
ORANGE = (0.95, 0.50, 0.08)
BLACK = (0.08, 0.08, 0.09)

# pan, in its own frame (origin: centre of the underside; x forward, y left; long axis along y)
PAN_X, PAN_Y, PAN_H = 0.265, 0.325, 0.040      # body size (GN 1/2: 325 x 265 mm), 40 mm deep
WALL, FLOOR = 0.002, 0.003
FL_W, FL_T = 0.010, 0.003                       # rim flange: 10 mm wide, 3 mm thick (outside 285 x 345... see below)
FL_X, FL_Y = PAN_X + 2 * FL_W, PAN_Y + 2 * FL_W  # outside of the flange: 285 x 345 mm
FOOD_H = 0.020                                  # cold tap water 20 mm deep (1.68 kg over the 0.261 x 0.321 m inside)
HANDLE_Y = PAN_Y / 2 + FL_W / 2                 # handle tabs stand on the short flange: y = +-0.1675
TAB = (0.024, 0.020, 0.045)                     # upright handle tab on each short flange (x, y, z): 24 mm wide, 20 mm thick, 45 mm tall
BAR_Z = PAN_H + 0.025                           # pinch height above the pan underside (tab mid-height 0.0625; pinch 0.065)
PINCH_Z = BAR_Z

PAN_BOXES = [
    ((0.0, 0.0, FLOOR / 2), (PAN_X, PAN_Y, FLOOR), STEEL),
    ((+(PAN_X - WALL) / 2, 0.0, PAN_H / 2), (WALL, PAN_Y, PAN_H), STEEL),
    ((-(PAN_X - WALL) / 2, 0.0, PAN_H / 2), (WALL, PAN_Y, PAN_H), STEEL),
    ((0.0, +(PAN_Y - WALL) / 2, PAN_H / 2), (PAN_X, WALL, PAN_H), STEEL),
    ((0.0, -(PAN_Y - WALL) / 2, PAN_H / 2), (PAN_X, WALL, PAN_H), STEEL),
    # flange: four strips on the rim top (z 0.037 - 0.040), outside 285 x 345
    ((+(PAN_X + FL_W) / 2, 0.0, PAN_H - FL_T / 2), (FL_W, FL_Y, FL_T), STEEL),
    ((-(PAN_X + FL_W) / 2, 0.0, PAN_H - FL_T / 2), (FL_W, FL_Y, FL_T), STEEL),
    ((0.0, +(PAN_Y + FL_W) / 2, PAN_H - FL_T / 2), (PAN_X, FL_W, FL_T), STEEL),
    ((0.0, -(PAN_Y + FL_W) / 2, PAN_H - FL_T / 2), (PAN_X, FL_W, FL_T), STEEL),
    # water (fused with the pan body: no slosh, no fluid)
    ((0.0, 0.0, FLOOR + FOOD_H / 2), (PAN_X - 2 * WALL, PAN_Y - 2 * WALL, FOOD_H), FOOD),
]
for _s in (1, -1):
    PAN_BOXES.append(((0.0, _s * HANDLE_Y, PAN_H + TAB[2] / 2), TAB, ORANGE))

# well: a 40 mm counter slab with a pit (opening 277 x 337, clearance 6 mm per side for the 265 x 325 body)
WELL_X = 0.40
OPEN_X, OPEN_Y = PAN_X + 0.012, PAN_Y + 0.012
COUNTER_H = 0.040
WALL_X, WALL_Y = 0.030, 0.045                   # counter strip widths around the opening
PAN_X0 = 0.085                                  # pan start centre x (on the table, flange 4 mm clear of the counter)
START_Z = TOP + 0.0005
SEAT_Z = TOP + COUNTER_H - (PAN_H - FL_T)       # pan origin height when the flange underside rests on the counter: TOP + 0.003
LEDGE_Z = TOP + COUNTER_H                       # counter top

PAN = part("pan", pos=(PAN_X0, 0.0, START_Z), mass=2.4, boxes=PAN_BOXES, color=STEEL, friction=0.5)
_ox, _oy = OPEN_X / 2, OPEN_Y / 2
COUNTER = [
    Box("well_front", pos=(WELL_X - _ox - WALL_X / 2, 0.0, TOP + COUNTER_H / 2), size=(WALL_X, OPEN_Y + 2 * WALL_Y, COUNTER_H), color=BLACK),
    Box("well_back", pos=(WELL_X + _ox + WALL_X / 2, 0.0, TOP + COUNTER_H / 2), size=(WALL_X, OPEN_Y + 2 * WALL_Y, COUNTER_H), color=BLACK),
    Box("well_left", pos=(WELL_X, +_oy + WALL_Y / 2, TOP + COUNTER_H / 2), size=(OPEN_X, WALL_Y, COUNTER_H), color=BLACK),
    Box("well_right", pos=(WELL_X, -_oy - WALL_Y / 2, TOP + COUNTER_H / 2), size=(OPEN_X, WALL_Y, COUNTER_H), color=BLACK),
]
OBJECTS = [*COUNTER, PAN]

# -- scoring thresholds ------------------------------------------------------------------------------------------------
# Spill = geometric overflow at the LOWEST RIM CORNER. The free surface stays horizontal; water leaves the pan when a rim corner drops
# below it. With n the pan's up-normal, the corner at (+-0.1325, +-0.1625) m from the centre drops
#   drop = 0.1325 * |n_x / n_z| + 0.1625 * |n_y / n_z|    (m)
# and the water overflows when drop > FREEBOARD = 17 mm (40 mm wall - 3 mm floor - 20 mm water). Alone, tilt across x (about y)
# spills at 7.3 deg and tilt along y (about x) at 6.0 deg; a combination spills earlier (5 deg + 4 deg drops the corner 23 mm).
# Real-world equivalent: carry the pan over a weighed absorbent mat or catch tray; the spill allowed is at most 5 mL.
FREEBOARD = PAN_H - FLOOR - FOOD_H
SPILL_TAN_X = FREEBOARD / (PAN_X / 2)                  # single-axis limits, for tests and the README
SPILL_TAN_Y = FREEBOARD / (PAN_Y / 2)


def corner_drop(state) -> float:
    """How far (m) the lowest rim corner of the pan has dropped below the horizontal water surface (0 when level)."""
    n = state.rotation()[:, 2]
    return float((PAN_X / 2) * abs(n[0] / n[2]) + (PAN_Y / 2) * abs(n[1] / n[2]))
GATE_SPEED = 0.5              # m/s: dropped onto the counter / pit
FALL_Z = TOP - 0.02
LIFT_CLEAR = 0.012            # flange underside this far above the counter top: "lifted over the well" (0.25)
L1_OFF = 0.020
BODY_IN = TOP + COUNTER_H - 0.004   # pan underside below the counter top: body is in the pit (0.5)
SEAT_TOL = 0.002              # every flange corner within 2 mm of the counter top
SEAT_TILT, OFF_MAX = 2.0, 0.008
HOLD_STEPS = 20
OPEN_DIST = 0.040             # a gripper TCP this far from the handle centres and open: released
GRIP_GAP_MIN = 0.065          # finger-tip body distance above this = still pinching a handle (measured: 0.048 open, 0.084 pinching)
STILL_V, STILL_W, STILL_Q = 0.01, 0.10, 0.05

_STATE: dict = {}
_ENV: dict = {"env": None}


def _fresh() -> dict:
    return dict(step=0, scored=None, last=(0.0, False), max_tilt=0.0, max_drop=0.0, max_speed=0.0, gate=False, held=0)


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(_fresh())


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _install_env_hook() -> None:
    """Keep a handle on the sim env so score() can read each gripper (finger gap, TCP) and the arm joint speeds."""
    try:
        from yam_arena import inspect_embodiment as ie
    except Exception:  # pragma: no cover
        return
    cls = ie.YamArenaEmbodiment
    if getattr(cls._task_step, "_pan_hooked", False):
        return
    base = cls._task_step

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        return base(self, *a, **k)

    _task_step._pan_hooked = True
    cls._task_step = _task_step


_install_env_hook()


def grip_state():
    """{"A": (tcp xyz, finger gap, max joint speed), "B": ...} for the left (A) and right (B) arm, or None if unknown."""
    env = _ENV["env"]
    if env is None:
        return None
    base = env.unwrapped
    origin = base.scene.env_origins[0]
    out = {}
    for tag, key in (("A", "left_robot"), ("B", "right_robot")):
        art = base.scene[key]
        names = list(art.body_names)
        pos = (getattr(art.data.body_pos_w, "torch", art.data.body_pos_w)[0] - origin).detach().cpu().numpy().astype(np.float64)
        g = pos[names.index("gripper")]
        tl, tr = pos[names.index("tip_left")], pos[names.index("tip_right")]
        mid = (tl + tr) / 2
        ax = (mid - g) / (np.linalg.norm(mid - g) + 1e-12)
        vel = getattr(art.data.joint_vel, "torch", art.data.joint_vel)[0].detach().cpu().numpy()
        out[tag] = (g + ax * 0.134, float(np.linalg.norm(tl - tr)), float(np.abs(vel[:6]).max()))
    return out


def tilt_deg(state: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def flange_heights(pan: PartState):
    """World heights of the four flange-underside corners."""
    return [float(pan.to_world((sx * FL_X / 2, sy * FL_Y / 2, PAN_H - FL_T))[2]) for sx in (1, -1) for sy in (1, -1)]


def fit(pan: PartState) -> dict:
    h = flange_heights(pan)
    return dict(dx=float(pan.pos[0] - WELL_X), dy=float(pan.pos[1]), off=float(math.hypot(pan.pos[0] - WELL_X, pan.pos[1])),
                z=float(pan.pos[2]), tilt=tilt_deg(pan), roll=math.degrees(math.atan2(pan.rotation()[2, 1], pan.rotation()[2, 2])),
                pitch=math.degrees(math.atan2(-pan.rotation()[2, 0], pan.rotation()[2, 2])), proud_max=max(h) - LEDGE_Z, proud_min=min(h) - LEDGE_Z)


def released(pan: PartState, grip) -> bool:
    if not grip:
        return False
    for tag in ("A", "B"):
        tcp, gap, _ = grip[tag]
        if gap > GRIP_GAP_MIN:                      # finger-tip bodies 48 mm apart when open, 84 mm while pinching a handle
            return False
        if min(float(np.linalg.norm(pan.to_world((0.0, s * HANDLE_Y, BAR_Z)) - tcp)) for s in (1, -1)) < OPEN_DIST:
            return False
    return True


def classify(pan: PartState, memo: dict, grip=None):
    """Pure function of the pose, the grippers and the memo: (level, all-1.0-conditions-hold, info)."""
    m = fit(pan)
    speed = float(np.linalg.norm(pan.lin_vel))
    if memo["step"] > 3:
        memo["max_tilt"] = max(memo["max_tilt"], m["tilt"])
        memo["max_drop"] = max(memo["max_drop"], corner_drop(pan))
        memo["max_speed"] = max(memo["max_speed"], speed)
    if memo["max_drop"] > FREEBOARD or memo["max_speed"] > GATE_SPEED or m["z"] < FALL_Z:
        memo["gate"] = True
    m.update(max_tilt=memo["max_tilt"], max_drop=memo["max_drop"], max_speed=memo["max_speed"], gate=memo["gate"])
    if memo["gate"]:
        return 0.0, False, m
    near = abs(m["dx"]) <= L1_OFF and abs(m["dy"]) <= L1_OFF
    seated = (m["proud_max"] <= SEAT_TOL and m["proud_min"] >= -SEAT_TOL and m["tilt"] <= SEAT_TILT
              and abs(m["dx"]) <= OFF_MAX and abs(m["dy"]) <= OFF_MAX)
    free = released(pan, grip)
    still = (speed < STILL_V and float(np.linalg.norm(pan.ang_vel)) < STILL_W
             and bool(grip) and all(grip[k][2] < STILL_Q for k in grip))
    m.update(seated=seated, free=free, still=still)
    if seated and free and still:
        return 1.0, True, m
    if seated:
        return 0.75, False, m
    if near and pan.pos[2] < BODY_IN:
        return 0.5, False, m
    if near and m["proud_min"] >= LIFT_CLEAR:
        return 0.25, False, m
    return 0.0, False, m


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    grip = grip_state()
    level, perfect, info = classify(objects["pan"], _STATE, grip)
    _STATE["held"] = _STATE["held"] + 1 if perfect else 0
    ok = _STATE["held"] >= HOLD_STEPS
    if perfect and not ok:
        level = 0.75                                       # never 1.0 before the full 1 s hold
    result = (level, ok)
    _STATE["last"] = result
    tr = os.environ.get("TASK_TRACE")
    if tr:
        rec = {k: (round(float(v), 5) if not isinstance(v, (bool, str)) else v) for k, v in info.items()}
        rec.update(step=_STATE["step"], level=level, pos=[round(float(x), 4) for x in objects["pan"].pos])
        if grip:
            rec["grip"] = {k: [[round(float(x), 4) for x in v[0]], round(v[1], 4), round(v[2], 4)] for k, v in grip.items()}
        with open(tr, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return result


def success(objects) -> bool:
    """The pan's four flange corners rest on the counter (each within 2 mm of its top), pan centred within 8 mm of the pit
    centre and tilted at most 2 degrees, it never spilled (lowest rim corner dropped more than 17 mm, the water freeboard) or moved faster than 0.5 m/s, both grippers open and
    40 mm clear of the handles, pan and arms still: true after 1 s (20 steps) of all of this."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the lowest rim corner ever drops more than 17 mm below the water level (single-axis limits 7.3 deg across x, 6.0 deg along y; combined tilts spill earlier), ever moves faster than 0.5 m/s (dropped on the counter) or
    ends off the table. Otherwise from the END pose: 0.25 held over the well (centre within 20 mm, flange underside >= 12 mm
    above the counter); 0.5 pan underside below the counter top (body in the pit) but the flange not seated (a corner more
    than 2 mm proud, or tilt over 2 degrees); 0.75 flange seated on the counter all round, still held or moving; 1.0 as in
    success() (1 s hold)."""
    return _update(objects)[0]
