"""On the station table: a 2 kg red block (80 x 110 x 50 mm) rests on the short end of a grey pry bar. The bar lies along
the table's y axis on a low black fulcrum, its long end (about 6 cm up in the air) pointing at the right arm. An orange
U-shaped chock (two prongs 37 mm tall joined by a bridge) lies on the table on the left side. The robot must press the
long end of the bar down so the block rises, slide the chock's prongs under the raised block with the other hand, and
then let the bar go so the block rests on the chock.

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
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Press down the long end of the grey lever bar to lift the red block, slide the orange chock under the "
               "red block, then let the lever go so the block rests on the chock.")

GREY = (0.55, 0.57, 0.60)
RED = (0.85, 0.10, 0.08)
ORANGE = (1.00, 0.50, 0.00)
BLACK = (0.12, 0.12, 0.13)

X = 0.28                               # the whole fixture lies along this x (both arms reach it)
# -- lever (part frame: origin on the bar's underside at the fulcrum's +y edge; x across, y along the bar, z up) ------
FULCRUM_H = 0.022                      # fulcrum top above the table
SHORT_ARM = 0.13                       # from the pivot (+y) to the bar tip, which rests on the table
THETA = math.atan2(FULCRUM_H, SHORT_ARM)       # rest slope of the bar, 9.6 degrees
BAR_W, BAR_T = 0.030, 0.012
S_TIP, S_LONG = -SHORT_ARM / math.cos(THETA), 0.25   # bar extent along its own axis from the pivot
PIVOT = np.array([X, -0.025, float(TOP) + FULCRUM_H])
PLATFORM_TOP = 0.011                   # level strip the block rests on, above the pivot (local z)
BLOCK_BOTTOM = FULCRUM_H + PLATFORM_TOP        # block bottom above the table at rest (33 mm)
CHOCK_H = 0.037                        # chock prong height: 4 mm taller than the block's resting height
LEVER_ARM_PRESS = 0.20                 # where the right hand presses, along the bar from the pivot


def _bar_box():
    d = np.array([0.0, -math.cos(THETA), math.sin(THETA)])
    n = np.array([0.0, math.sin(THETA), math.cos(THETA)])
    mid = (S_TIP + S_LONG) / 2
    centre = mid * d + n * BAR_T / 2
    psi = -THETA
    return (tuple(centre), (BAR_W, S_LONG - S_TIP, BAR_T), GREY, (math.cos(psi / 2), math.sin(psi / 2), 0.0, 0.0))


PLATFORM = ((0.0, 0.0725, (PLATFORM_TOP - 0.020) / 2), (BAR_W, 0.105, PLATFORM_TOP + 0.020), GREY)
LEVER = part("lever", pos=tuple(float(v) for v in PIVOT), mass=0.40, friction=0.6,
             boxes=[_bar_box(), ((0.0, PLATFORM[0][1], PLATFORM[0][2]), PLATFORM[1], GREY)], color=GREY)

# -- block ---------------------------------------------------------------------------------------------------------
BLOCK_SIZE = (0.080, 0.110, 0.050)
BLOCK_START = (X, 0.045, TOP + BLOCK_BOTTOM + BLOCK_SIZE[2] / 2)
BLOCK = part("block", pos=BLOCK_START, mass=2.0, friction=0.8,
             boxes=[((0, 0, 0), BLOCK_SIZE, RED)], color=RED)

# -- chock (part frame: origin on the bridge's front face at prong mid height) -------------------------------------
PRONG_X, PRONG_W, PRONG_L = 0.028, 0.020, 0.080
CHOCK_START = (X, 0.235, TOP + CHOCK_H / 2)
CHOCK_FINAL_DY = 0.063                 # chock origin y minus block y when the prongs fill the space under the block
BRIDGE_Y = 0.065                       # bridge centre in the chock frame (grip point)
CHOCK = part("chock", pos=CHOCK_START, mass=0.15, friction=0.6, color=ORANGE, boxes=[
    ((-PRONG_X, -PRONG_L / 2, 0), (PRONG_W, PRONG_L, CHOCK_H), ORANGE),
    ((PRONG_X, -PRONG_L / 2, 0), (PRONG_W, PRONG_L, CHOCK_H), ORANGE),
    ((0, 0.010, 0.0185), (0.076, 0.020, 0.030), ORANGE),
    ((0, 0.055, 0.0185), (0.020, 0.070, 0.030), ORANGE),       # handle stem so the fingers stay clear of the block       # bridge: table +22..+52 mm, clear of the bar tip
])

FIXTURE = [
    Box("fulcrum", pos=(X, float(PIVOT[1]) - 0.010, TOP + FULCRUM_H / 2), size=(0.06, 0.020, FULCRUM_H), color=BLACK),
    Box("guide_l", pos=(X - 0.0205, float(PIVOT[1]) - 0.010, TOP + 0.0225), size=(0.008, 0.020, 0.045), color=BLACK),
    Box("guide_r", pos=(X + 0.0205, float(PIVOT[1]) - 0.010, TOP + 0.0225), size=(0.008, 0.020, 0.045), color=BLACK),
]
OBJECTS = [*FIXTURE, LEVER, BLOCK, CHOCK]

# -- scoring thresholds (m, deg); rest values are re-checked in the settle run -----------------------------------------
LIFT_USED = 0.006          # 0.25: block raised 6 mm
LIFT_CLEAR = 0.009         # 0.5: raised 9 mm (chock is 4 mm taller than the rest height; 5 mm clearance)
CHOCK_TRAVEL = 0.08        # 0.5: chock slid 80 mm
UNDER_DY, UNDER_DX = 0.012, 0.010     # chock origin within these of its fully-inserted place
LEVER_RELEASED_DEG = 2.5   # bar back within 2.5 degrees of rest (block resting on the chock keeps it about 1.6 deg up)
FINAL_LIFT = (0.002, 0.007)           # block resting on the chock: 4 mm above its start
FINAL_TILT_DEG = 5.0
GATE_TILT_DEG = 30.0
GATE_MOVE = 0.20           # chock or block carried more than 20 cm from its start
NO_LEVER_LIFT = 0.006      # block raised this much with the bar at rest and no chock under it: lifted by hand

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(z0=None, start=None, lifted=False, cleared=False, inserted=False, gate=False, recent=[], step=0,
                  scored=None, last=(0.0, False))


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def tilt_deg(state) -> float:
    return float(math.degrees(math.acos(float(np.clip(state.rotation()[2, 2], -1.0, 1.0)))))


def lever_deg(lever) -> float:
    """Press angle of the bar about x in degrees, 0 at rest; positive when the long end goes down."""
    R = lever.rotation()
    return float(math.degrees(math.atan2(R[2, 1], R[1, 1])))


def chock_under(block, chock) -> bool:
    dx, dy = chock.pos[0] - block.pos[0], chock.pos[1] - block.pos[1]
    return bool(abs(dx) <= UNDER_DX and abs(dy - CHOCK_FINAL_DY) <= UNDER_DY)


def _metrics(objects) -> dict:
    block, lever, chock = objects["block"], objects["lever"], objects["chock"]
    if _STATE["z0"] is None:
        _STATE["z0"] = float(block.pos[2])
        _STATE["start"] = {"block": np.array(block.pos), "chock": np.array(chock.pos)}
    return {
        "lift": float(block.pos[2]) - _STATE["z0"],
        "block_tilt": tilt_deg(block),
        "lever": lever_deg(lever),
        "chock_travel": float(np.linalg.norm(chock.pos[:2] - _STATE["start"]["chock"][:2])),
        "under": chock_under(block, chock),
        "block_moved": float(np.linalg.norm(block.pos[:2] - _STATE["start"]["block"][:2])),
        "chock_off": float(np.linalg.norm(chock.pos - _STATE["start"]["chock"])),
    }


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    m = _metrics(objects)
    block, chock = objects["block"], objects["chock"]
    lift, lev = m["lift"], m["lever"]
    if (m["block_tilt"] > GATE_TILT_DEG or m["block_moved"] > GATE_MOVE or m["chock_off"] > GATE_MOVE + 0.1
            or (lift > NO_LEVER_LIFT and lev < 1.0 and not m["under"])):
        _STATE["gate"] = True
    _STATE["lifted"] |= bool(lift >= LIFT_USED and lev > 2.0)
    _STATE["cleared"] |= bool(_STATE["lifted"] and lift >= LIFT_CLEAR and m["chock_travel"] >= CHOCK_TRAVEL)
    _STATE["inserted"] |= bool(_STATE["cleared"] and m["under"])
    recent = _STATE["recent"]
    recent.append(np.r_[block.pos, chock.pos])
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - recent[-1]) < 0.0015 for p in recent)
    done = bool(_STATE["inserted"] and m["under"] and abs(lev) <= LEVER_RELEASED_DEG
                and FINAL_LIFT[0] <= lift <= FINAL_LIFT[1] and m["block_tilt"] <= FINAL_TILT_DEG)
    if _STATE["gate"]:
        result = (0.0, False)
    elif done:
        result = (1.0, bool(still))
    elif _STATE["inserted"]:
        result = (0.75, False)
    elif _STATE["cleared"]:
        result = (0.5, False)
    else:
        result = (0.25 if _STATE["lifted"] else 0.0), False
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Block resting on the chock (4 mm above its start, tilt <= 5 deg), chock fully under it, bar back at rest
    within 2.5 degrees, nothing gated, still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 block raised 6 mm with the bar; 0.5 raised 9 mm while the chock has slid 80 mm; 0.75 chock fully under the
    raised block; 1.0 bar released and block resting on the chock. 0 if the block tips past 30 degrees or is carried
    away, the chock is thrown, or the block is lifted without the bar."""
    return _update(objects)[0]
