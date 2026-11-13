"""On the station table a low bench stands over four pieces of a smashed cup. A blue floor-squeegee head lies in front
of the bench; its middle is under the bench overhang, so it can only be held by two orange handle blocks at its two ends
(0.46 m apart), one per arm. Both arms advance the head together along x so that all four pieces are swept out from
under the bench and onto the green pan, then release it. Everything is a free or fixed rigid box; nothing is a joint.

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

INSTRUCTION = ("A cup has smashed under the bench. Get every piece of it out from under the bench and into the green "
               "pan, then let go of the squeegee and leave it and the pieces still.")

BLUE = (0.10, 0.30, 0.90)
ORANGE = (0.95, 0.50, 0.08)
GREEN = (0.10, 0.65, 0.20)
WOOD = (0.55, 0.38, 0.20)
CREAM = (0.97, 0.95, 0.80)

# -- squeegee head ----------------------------------------------------------------------------------------------------
HEAD = (0.030, 0.420, 0.030)           # x, y, z
HEAD_X0 = 0.200
HEAD_Z = TOP + HEAD[2] / 2 + 0.0005
HANDLE_Y = 0.230                       # handle block centres: sockets at the two blade ends (blade ends at 0.210)
BLOCK = (0.040, 0.040, 0.060)          # fingers pinch these across x, 0.040 m in x, 0.060 m tall, standing on the table


def _head_boxes():
    b = [((0.0, 0.0, 0.0), HEAD, BLUE)]
    for s in (-1, 1):
        b.append(((0.0, s * HANDLE_Y, BLOCK[2] / 2 - HEAD[2] / 2), BLOCK, ORANGE))
    return b


HEAD_PART = part("head", pos=(HEAD_X0, 0.0, HEAD_Z), mass=1.0, boxes=_head_boxes(), color=BLUE, friction=0.35)

# -- pieces -----------------------------------------------------------------------------------------------------------
PIECE = (0.022, 0.022, 0.012)
PIECE_MASS = 0.014                     # 22 x 22 x 12 mm of earthenware (2.4 g/cm3)
PIECE_Z = TOP + PIECE[2] / 2 + 0.0005
PIECE_XY = [(0.300, -0.070), (0.285, 0.004), (0.315, -0.004), (0.305, 0.070)]
PIECES = [part(f"shard{i}", pos=(x, y, PIECE_Z), mass=PIECE_MASS, color=CREAM, friction=0.4,
               boxes=[((0, 0, 0), PIECE, CREAM)]) for i, (x, y) in enumerate(PIECE_XY)]
NAMES = ["head"] + [p.name for p in PIECES]
SHARDS = NAMES[1:]

# -- fixed scenery ----------------------------------------------------------------------------------------------------
BENCH_X, BENCH_LEN = 0.265, 0.230                           # slatted bench top: rails x 0.15-0.38, y +-0.12
BENCH_UNDER, RAIL_T, RAIL_W = 0.060, 0.020, 0.030           # table to rail underside; rail thickness, width
BENCH_X_FRONT = BENCH_X + BENCH_LEN / 2
RAIL_Y = (-0.105, -0.035, 0.035, 0.105)                     # 40 mm gaps between rails, the pieces lie in the gaps
RAILS = [Box(f"rail{i}", pos=(BENCH_X, y, TOP + BENCH_UNDER + RAIL_T / 2), size=(BENCH_LEN, RAIL_W, RAIL_T), color=WOOD)
         for i, y in enumerate(RAIL_Y)]
CROSS = Box("cross", pos=(0.160, 0.0, TOP + BENCH_UNDER + RAIL_T / 2), size=(0.020, 0.240, RAIL_T), color=WOOD)
POSTS = [Box(f"post{i}", pos=(0.160, s * 0.110, TOP + BENCH_UNDER / 2), size=(0.020, 0.020, BENCH_UNDER), color=WOOD)
         for i, s in enumerate((-1, 1))]
PAN_SIZE, PAN_X = (0.075, 0.280, 0.003), 0.455
PAN_TOP = 0.0                                               # pan top level with the table (no lip): a 0.5 mm step jammed the pieces
PAN = Box("pan", pos=(PAN_X, 0.0, TOP + PAN_TOP - PAN_SIZE[2] / 2), size=PAN_SIZE, color=GREEN)
WALL_H = 0.012                                              # pan walls on the two sides and the far end; the sweep side is open
WALLS = [Box(f"wall{i}", pos=(PAN_X, s * (PAN_SIZE[1] / 2 + 0.005), TOP + WALL_H / 2), size=(PAN_SIZE[0], 0.010, WALL_H), color=GREEN)
         for i, s in enumerate((-1, 1))]
WALLS.append(Box("wall2", pos=(PAN_X + PAN_SIZE[0] / 2 + 0.005, 0.0, TOP + WALL_H / 2), size=(0.010, PAN_SIZE[1] + 0.020, WALL_H),
                 color=GREEN))
OBJECTS = RAILS + [CROSS] + POSTS + WALLS + [PAN, HEAD_PART] + PIECES

# -- scoring thresholds ------------------------------------------------------------------------------------------------
PAN_X_LO, PAN_X_HI = PAN_X - PAN_SIZE[0] / 2, PAN_X + PAN_SIZE[0] / 2        # 0.4175 .. 0.4925
PAN_Y = PAN_SIZE[1] / 2                                                       # 0.14
INSIDE = 0.008
GATE_TILT, GATE_YAW = 8.0, 8.0
GATE_LIFT, LIFT_STEPS = 0.005, 10
MIN_ADVANCE = 0.160
CLEAR_GAP = 0.004                      # head front face to nearest piece back edge
GRIP_OPEN, HAND_AWAY = 0.058, 0.040
STILL_V, STILL_W = 0.012, 0.25
HOLD_STEPS = 20
SETTLE_STEPS = 10
HANDS = None                           # unit tests put [(tcp, gap, tip_l, tip_r)] x2 here

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), held=0, gate=False, lift=0, hands=None)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1
    if environment is not None:
        try:
            _STATE["hands"] = _read_hands(environment)
        except Exception:  # pragma: no cover
            _STATE["hands"] = None


_reset_state()


def _read_hands(environment):
    base = environment.unwrapped
    origin = base.scene.env_origins[0]
    out = []
    for key in ("left_robot", "right_robot"):
        art = base.scene[key]
        names = list(art.body_names)
        pos = (getattr(art.data.body_pos_w, "torch", art.data.body_pos_w)[0] - origin).detach().cpu().numpy().astype(np.float64)
        g = pos[names.index("gripper")]
        tl, tr = pos[names.index("tip_left")], pos[names.index("tip_right")]
        mid = (tl + tr) / 2
        ax = (mid - g) / (np.linalg.norm(mid - g) + 1e-12)
        out.append((g + ax * 0.134, float(np.linalg.norm(tl - tr)), tl, tr))
    return out


def tilt_deg(state: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def yaw_deg(state: PartState) -> float:
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def head_yaw(state: PartState) -> float:
    return abs((yaw_deg(state) + 90.0) % 180.0 - 90.0)


def head_bottom(state: PartState) -> float:
    R = state.rotation()
    h = np.array(HEAD) / 2
    return float(state.pos[2] - (abs(R[2, 0]) * h[0] + abs(R[2, 1]) * h[1] + abs(R[2, 2]) * h[2]))


def head_ends(state: PartState):
    return [state.to_world((0.0, s * HANDLE_Y, 0.0)) for s in (1, -1)]


def head_front(state: PartState) -> float:
    """Largest x of the squeegee blade face (the 0.03 x 0.42 box)."""
    R = state.rotation()
    h = np.array(HEAD) / 2
    return float(state.pos[0] + abs(R[0, 0]) * h[0] + abs(R[0, 1]) * h[1] + abs(R[0, 2]) * h[2])


def in_pan(state: PartState, margin: float = 0.0) -> bool:
    x, y = float(state.pos[0]), float(state.pos[1])
    return PAN_X_LO + margin <= x <= PAN_X_HI - margin and abs(y) <= PAN_Y - margin


def left_table(state: PartState) -> bool:
    x, y, z = (float(v) for v in state.pos)
    return z < TOP - 0.010 or z > TOP + 0.040 or abs(y) > 0.40 or x > 0.60 or x < 0.0


def hand_to_head(head: PartState, p) -> float:
    """Distance from a hand point to the squeegee (blade box, two stems, two handle blocks)."""
    q = head.to_local(p)
    best = 9.0
    for c, size, _ in _head_boxes():
        d = np.maximum(np.abs(q - np.array(c)) - np.array(size) / 2, 0.0)
        best = min(best, float(np.linalg.norm(d)))
    return best


def measure(objects, hands=None) -> dict:
    h = objects["head"]
    sh = [objects[n] for n in SHARDS]
    e = head_ends(h)
    m = dict(head_tilt=tilt_deg(h), head_yaw=head_yaw(h), head_lift=head_bottom(h) - TOP,
             adv_left=float(e[0][0]) - HEAD_X0, adv_right=float(e[1][0]) - HEAD_X0,
             n_pan=sum(in_pan(s) for s in sh), n_inside=sum(in_pan(s, INSIDE) for s in sh),
             left=[left_table(s) for s in sh],
             gap=min(float(s.pos[0]) - PIECE[0] / 2 - head_front(h) for s in sh),
             under=sum(float(s.pos[0]) < BENCH_X_FRONT + 0.005 for s in sh),
             still=all(float(np.linalg.norm(o.lin_vel)) <= STILL_V and float(np.linalg.norm(o.ang_vel)) <= STILL_W
                       for o in objects.values()))
    if hands is not None:
        m["open"] = all(g[1] <= GRIP_OPEN for g in hands)
        m["away"] = min(hand_to_head(h, p) for g in hands for p in (g[0], g[2], g[3])) >= HAND_AWAY
    else:
        m["open"] = m["away"] = False
    return m


def classify(objects, hands=None):
    """(level, all-1.0-conditions-hold, info). The gate is checked by _advance (it latches)."""
    m = measure(objects, hands)
    n = m["n_pan"]
    level = 0.75 if n >= 3 else 0.5 if n >= 2 else 0.25 if n >= 1 else 0.0
    if min(m["adv_left"], m["adv_right"]) < MIN_ADVANCE:
        level = min(level, 0.5)
    full = (m["n_inside"] == 4 and m["under"] == 0 and m["gap"] >= CLEAR_GAP and m["open"] and m["away"]
            and m["still"] and min(m["adv_left"], m["adv_right"]) >= MIN_ADVANCE)
    if full:
        return 1.0, True, m
    return min(level, 0.75), False, m


def gate_now(objects, m) -> bool:
    return (m["head_tilt"] > GATE_TILT or m["head_yaw"] > GATE_YAW or any(m["left"]))


def _advance(objects, hands) -> tuple[float, bool]:
    level, perfect, m = classify(objects, hands)
    if _STATE["step"] > SETTLE_STEPS:
        if gate_now(objects, m):
            _STATE["gate"] = True
        _STATE["lift"] = _STATE["lift"] + 1 if m["head_lift"] > GATE_LIFT else 0
        if _STATE["lift"] > LIFT_STEPS:
            _STATE["gate"] = True
    if _STATE["gate"]:
        _STATE["held"] = 0
        return 0.0, False
    _STATE["held"] = _STATE["held"] + 1 if perfect else 0
    held = _STATE["held"] >= HOLD_STEPS
    return (1.0 if held else min(level, 0.75)), held


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    hands = HANDS if HANDS is not None else _STATE["hands"]
    _STATE["last"] = _advance(objects, hands)
    return _STATE["last"]


def success(objects) -> bool:
    """All four pieces >= 8 mm inside the pan, none under the bench, squeegee clear of them, both handle blocks
    advanced >= 160 mm, both grippers open and 40 mm from the head, everything still; held 1 s; no gate tripped."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 (latched) if the head tilts over 8 deg or yaws over 8 deg, a piece leaves the table, or the head bottom rises
    5 mm for more than 0.5 s. Else 0.25 / 0.5 / 0.75 for 1 / 2 / 3 piece centres on the pan at the end (at most 0.5 unless
    both handle blocks advanced 160 mm); 1.0 as in success()."""
    return _update(objects)[0]
