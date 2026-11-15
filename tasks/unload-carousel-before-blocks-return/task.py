"""A bench-top indexing carousel stands on the station table: six square aluminium carrier plates (11 x 11 cm) ride
on a circle of 18 cm radius around a hub 34 cm in front of the arm bases. Every 5 s the machine indexes one slot
(60 degrees, clockwise seen from above: 1.2 s of motion, then 3.8 s standing). The loading slot is the one on the
robot's left, beside a green marker; the next slot is on the far side of the hub and the slot opposite the loader, on
the robot's right, is the unload slot. A carrier loaded at the loader is back at the loader after one lap, 30 s.

On the left, on a rubber supply mat, stand three red and two blue 5 cm solid acetal cubes (176 g each). On the right,
beyond the carousel, a polypropylene crate (inner 26 x 36 cm, walls 6 cm above the table) stands on the table.

The robot must load the five blocks onto the carousel at the loading slot, and take every red block off the carousel
into the crate before the carrier it rides on comes back to the loader (a carrier that comes back still holding a
finished part is not free for the next block). The blue blocks are still in process and must stay riding. A red block
counts only if it was seen on a carrier first (a block taken straight off the supply mat never counts), was off every
carrier before its carrier's first return, and now rests on the crate floor, let go and still. A per-step monitor
keeps this for every block.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Load the five blocks onto the carousel at the green marker. Crate each red block before it
           returns to that marker; leave the blue ones riding." ...
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, arm_joints, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"

INSTRUCTION = ("Load the five blocks onto the carousel at the green marker. Crate each red block before it returns "
               "to that marker; leave the blue ones riding.")


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the machine -------------------------------------------------------------------------------------------------
CENTER = (0.34, 0.0)
RADIUS = 0.18
SLOTS = 6
T0, PERIOD, MOVE = 5.0, 5.0, 1.2  # first index starts at T0; one index every PERIOD s, MOVE s of motion each
CARRIER = (0.11, 0.11, 0.012)
CARRIER_Z = TOP + 0.002 + CARRIER[2] / 2  # 2 mm above the table: driven plates ride, they don't rub
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
ALUMINIUM = (0.72, 0.74, 0.76)
DARK = (0.18, 0.18, 0.2)
RED = (0.82, 0.12, 0.1)
BLUE = (0.12, 0.25, 0.85)
GREEN = (0.1, 0.65, 0.2)
OFFWHITE = (0.88, 0.86, 0.8)
RUBBER = (0.35, 0.33, 0.3)
PP = (0.45, 0.5, 0.58)


def index_progress(t: float) -> tuple[float, float]:
    """(indexes done so far as a smooth real number, its rate per second) at episode time ``t``."""
    if t <= T0:
        return 0.0, 0.0
    n = int((t - T0) // PERIOD)
    r = (t - T0) - n * PERIOD
    if r < MOVE:
        return n + (1 - np.cos(np.pi * r / MOVE)) / 2, float(np.pi * np.sin(np.pi * r / MOVE) / (2 * MOVE))
    return float(n + 1), 0.0


def slot_xy(k: float) -> np.ndarray:
    """Centre of carousel slot ``k`` (0 = loader, 3 = unload; a carrier moves from slot k to k + 1 per index)."""
    phi = np.pi / 2 - (np.pi / 3) * k
    return np.array([CENTER[0] + RADIUS * np.cos(phi), CENTER[1] + RADIUS * np.sin(phi)])


LOADER_XY = slot_xy(0)
UNLOAD_XY = slot_xy(3)


def slot_of(c: int, t: float) -> float:
    """Where carrier ``c`` is, as a (real) slot number, at ``t``."""
    return c + index_progress(t)[0]


def carrier_path(c: int):
    def path(t: float):
        u, du = index_progress(t)
        phi = np.pi / 2 - (np.pi / 3) * (c + u)
        pos = np.array([CENTER[0] + RADIUS * np.cos(phi), CENTER[1] + RADIUS * np.sin(phi), CARRIER_Z])
        dphi = -(np.pi / 3) * du
        vel = np.array([-RADIUS * np.sin(phi) * dphi, RADIUS * np.cos(phi) * dphi, 0.0])
        return pos, vel

    return path


def carrier_at_slot(slot: int, index: int) -> int:
    """The carrier standing in ``slot`` after ``index`` indexes."""
    return (slot - index) % SLOTS


def standing_window(index: int) -> tuple[float, float]:
    """Episode times between which the carousel stands still after ``index`` indexes (index >= 1)."""
    return T0 + (index - 1) * PERIOD + MOVE, T0 + index * PERIOD


CARRIERS = [
    DrivenBox(f"carrier_{c}", pos=_floats(carrier_path(c)(0.0)[0]), size=CARRIER, color=ALUMINIUM, path=carrier_path(c))
    for c in range(SLOTS)
]
CARRIER_NAMES = tuple(f"carrier_{c}" for c in range(SLOTS))
HUB = Box("hub", pos=_floats((CENTER[0], CENTER[1], TOP + 0.015)), size=(0.10, 0.10, 0.03), color=DARK)
MARKER = Box("marker", pos=_floats((CENTER[0], 0.262, TOP + 0.001)), size=(0.12, 0.015, 0.002), color=GREEN)

# -- the supply mat and the blocks ------------------------------------------------------------------------------
BLOCK = 0.05
BLOCK_MASS = 0.176  # 50 mm solid acetal (POM, 1.41 g/cm3)
MAT_CENTER = (0.34, 0.44)
MAT_SIZE = (0.34, 0.20, 0.004)
MAT_TOP = TOP + MAT_SIZE[2]
MAT = Box("supply_mat", pos=_floats((*MAT_CENTER, TOP + MAT_SIZE[2] / 2)), size=MAT_SIZE, color=RUBBER)
# name, colour, where it stands on the mat; the demo loads them in this order
BLOCKS = (("red_1", RED, (0.23, 0.37)), ("red_2", RED, (0.32, 0.37)), ("red_3", RED, (0.41, 0.37)),
          ("blue_1", BLUE, (0.275, 0.49)), ("blue_2", BLUE, (0.365, 0.49)))
REDS = tuple(n for n, col, _ in BLOCKS if col == RED)
BLUES = tuple(n for n, col, _ in BLOCKS if col == BLUE)
NAMES = tuple(n for n, _, _ in BLOCKS)
START_XY = {n: xy for n, _, xy in BLOCKS}
BLOCK_OBJECTS = [Box(n, pos=_floats((*xy, MAT_TOP + BLOCK / 2)), size=(BLOCK,) * 3, mass=BLOCK_MASS, color=col)
                 for n, col, xy in BLOCKS]

# -- the crate --------------------------------------------------------------------------------------------------
WALL, FLOOR, WALL_H = 0.008, 0.005, 0.06
IN_X = (0.17, 0.43)
IN_Y = (-0.62, -0.26)
FLOOR_TOP = TOP + FLOOR


def _crate() -> list[Box]:
    (x0, x1), (y0, y1) = IN_X, IN_Y
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    wz = TOP + WALL_H / 2
    return [
        Box("crate_floor", pos=_floats((xc, yc, TOP + FLOOR / 2)), size=_floats((x1 - x0 + 2 * WALL, y1 - y0 + 2 * WALL, FLOOR)), color=PP),
        Box("crate_near", pos=_floats((x0 - WALL / 2, yc, wz)), size=_floats((WALL, y1 - y0 + 2 * WALL, WALL_H)), color=PP),
        Box("crate_far", pos=_floats((x1 + WALL / 2, yc, wz)), size=_floats((WALL, y1 - y0 + 2 * WALL, WALL_H)), color=PP),
        Box("crate_right", pos=_floats((xc, y0 - WALL / 2, wz)), size=_floats((x1 - x0, WALL, WALL_H)), color=PP),
        Box("crate_left", pos=_floats((xc, y1 + WALL / 2, wz)), size=_floats((x1 - x0, WALL, WALL_H)), color=PP),
    ]


OBJECTS = [*CARRIERS, HUB, MARKER, MAT, *_crate(), *BLOCK_OBJECTS]

# -- scoring ---------------------------------------------------------------------------------------------------
SUPPORT_TOL = 0.008  # block bottom within this of the plate / floor carrying it
LIFT_TOL = 0.03  # a loaded blue block counts as taken off its carrier when lifted this far above the plate
RELEASE_CLEAR = 0.04
GRIPPER_LEN = 0.12
LEFT_LOADER_R = 0.12  # a carrier counts as having left the loader when this far from it
RETURN_R = 0.08  # and as back when this close again


def _box_distance(p, low, high) -> float:
    p = np.asarray(p, float)
    return float(np.linalg.norm(np.maximum(np.maximum(low - p, 0.0), p - high)))


def gripper_distance(c, segment) -> float:
    a, b = (np.asarray(v, float) for v in segment[:2])
    return min(_box_distance(a + s * (b - a), c.low, c.high) for s in np.linspace(0.0, 1.0, 25))


_STATE: dict = {}


def reset_state(environment=None) -> None:
    _STATE.update(grippers={}, step=0, blocks={n: dict(loaded=None, carrier=None, away=False, returned=None,
                                                       late=False, lost=False, spoiled=False) for n in NAMES})


reset_state()


def on_carrier(b, carriers) -> int | None:
    """Index of the carrier ``b`` rests on (centre over the plate, bottom within 8 mm of its top), else None."""
    for i, name in enumerate(CARRIER_NAMES):
        c = carriers[name]
        d = b.pos - c.pos
        if abs(d[0]) < CARRIER[0] / 2 and abs(d[1]) < CARRIER[1] / 2 and abs(b.low[2] - (c.pos[2] + CARRIER[2] / 2)) < SUPPORT_TOL:
            return i
    return None


def over_crate(b) -> bool:
    """Centre inside the crate's inner footprint and below the wall tops."""
    return bool(IN_X[0] <= b.pos[0] <= IN_X[1] and IN_Y[0] <= b.pos[1] <= IN_Y[1] and b.low[2] < TOP + WALL_H)


def in_crate(b) -> bool:
    """Resting on the crate floor, whole block inside the inner walls, still, and let go."""
    inside = b.low[0] >= IN_X[0] - 0.002 and b.high[0] <= IN_X[1] + 0.002 and b.low[1] >= IN_Y[0] - 0.002 and b.high[1] <= IN_Y[1] + 0.002
    return bool(inside and abs(b.low[2] - FLOOR_TOP) <= 0.006 and b.resting() and released(b))


def released(b) -> bool:
    return all(gripper_distance(b, g) >= RELEASE_CLEAR + g[2] for g in _STATE["grippers"].values())


def on_table(b) -> bool:
    """Lying on the table, mat or floor (not on a plate, not in the crate): bottom within 1 cm of the table's top."""
    return bool(b.low[2] < TOP + 0.010 and not over_crate(b)) or b.low[2] < TOP - 0.05


def monitor(objects, grippers: dict) -> None:
    """One control step. ``objects`` are the states after the step, ``grippers`` maps each arm to (pad tips, top end,
    opening)."""
    _STATE["grippers"] = {s: (np.asarray(g[0], float), np.asarray(g[1], float), float(g[2]) if len(g) > 2 else 0.0)
                          for s, g in grippers.items()}
    _STATE["step"] += 1
    carriers = {n: objects[n] for n in CARRIER_NAMES}
    for n in NAMES:
        b, st = objects[n], _STATE["blocks"][n]
        on_c = on_carrier(b, carriers)
        if st["loaded"] is None:
            if on_c is not None and released(b):
                st["loaded"], st["carrier"] = _STATE["step"], on_c
            continue
        c = carriers[CARRIER_NAMES[st["carrier"]]]
        d = float(np.linalg.norm(c.pos[:2] - LOADER_XY))
        if d > LEFT_LOADER_R:
            st["away"] = True
        if st["away"] and d < RETURN_R and st["returned"] is None:
            st["returned"] = _STATE["step"]
        if st["returned"] is not None and on_c is not None:
            st["late"] = True  # still riding at / after the carrier's first return
        if on_c is None and on_table(b):
            st["lost"] = True
        if on_c is None and n in BLUES and (b.low[2] > CARRIER_TOP + LIFT_TOL or over_crate(b)):
            st["spoiled"] = True
        if on_c is None and n in BLUES and st["lost"]:
            st["spoiled"] = True


def tally(objects) -> dict:
    """loaded: blocks seen on a carrier; credited: reds loaded, never late nor lost, resting in the crate now;
    spoiled: blues taken off / lost; riding: loaded blues on a carrier now."""
    carriers = {n: objects[n] for n in CARRIER_NAMES}
    blocks = _STATE["blocks"]
    loaded = sum(blocks[n]["loaded"] is not None for n in NAMES)
    credited = sum(blocks[n]["loaded"] is not None and not blocks[n]["late"] and not blocks[n]["lost"] and in_crate(objects[n])
                   for n in REDS)
    spoiled = sum(blocks[n]["spoiled"] or (blocks[n]["loaded"] is not None and on_carrier(objects[n], carriers) is None) for n in BLUES)
    riding = sum(blocks[n]["loaded"] is not None and not blocks[n]["spoiled"] and on_carrier(objects[n], carriers) is not None
                 for n in BLUES)
    return dict(loaded=loaded, credited=credited, spoiled=spoiled, riding=riding)


def score(objects):
    """0.25: at least three blocks loaded onto the carousel; 0.5: one red block credited (seen on a carrier, off every
    carrier before its carrier's first return, resting in the crate, let go, still); 0.75: two reds credited; also 0.75
    for the full condition (the hold decides 1.0: see success()). A blue block taken off its carrier (lifted, lost,
    crated) caps the score at 0.5. A block never seen on a carrier is never credited."""
    t = tally(objects)
    level = 0.0
    if t["loaded"] >= 3:
        level = 0.25
    if t["credited"] >= 1:
        level = 0.5
    if t["credited"] >= 2:
        level = 0.75
    if t["spoiled"] > 0:
        level = min(level, 0.5)
    return level


def success(objects):
    """All five blocks loaded, all three reds credited, both blues still riding (held 1 s by the framework)."""
    t = tally(objects)
    return t["loaded"] == len(NAMES) and t["credited"] == len(REDS) and t["spoiled"] == 0 and t["riding"] == len(BLUES)


@on_reset
def _reset(environment) -> None:
    reset_state()


@on_step
def _monitor(environment) -> None:
    from yam_arena import station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.scene import object_states

    objects = object_states(environment, [*NAMES, *CARRIER_NAMES])
    grippers = {}
    for side in ("left", "right"):
        opening = float(np.max(np.abs(arm_joints(environment, side, FINGER_JOINTS))))
        grippers[side] = gripper_segment(arm_joints(environment, side, ARM_JOINTS), station.BASES[side], opening)
    monitor(objects, grippers)
    path = os.environ.get("TASK_TRACE")
    if path:  # dev trace: one JSON line per control step (block poses, monitor flags, both gripper tips)
        import json

        row = {"step": _STATE["step"], "blocks": {n: [round(float(v), 4) for v in objects[n].pos] for n in NAMES},
               "flags": {n: {k: (v if not isinstance(v, bool) else int(v)) for k, v in _STATE["blocks"][n].items()} for n in NAMES},
               "tips": {s: [round(float(v), 4) for v in g[0]] for s, g in grippers.items()}, "open": {s: round(g[2], 4) for s, g in grippers.items()}}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def gripper_segment(q, base, opening: float = 0.0):
    from yam_arena import ik

    tcp, rot = ik.tcp_pose(q, base)
    tip = tcp + rot[:, 2] * PAD_TIP_PAST_TCP
    return tip, tip - rot[:, 2] * GRIPPER_LEN, float(opening)


def distance(objects):
    """Metres from the first red block not yet in the crate to the crate's centre."""
    goal = np.array([(IN_X[0] + IN_X[1]) / 2, (IN_Y[0] + IN_Y[1]) / 2])
    for n in REDS:
        if not in_crate(objects[n]):
            return float(np.linalg.norm(objects[n].pos[:2] - goal))
    return 0.0


# -- the scripted demo -----------------------------------------------------------------------------------------
# Open loop, timed against the carousel's clock. The left arm loads the five blocks at the loader slot (one every 10 s,
# in the standing windows after indexes 1, 3, 5, 7 and 9); the right arm lifts each red block off its carrier at the
# unload slot (indexes 4, 6, 8: three indexes after its load) and sets it in the crate. It reads nothing from the
# simulator: its timing is the machine's fixed timetable and its places are the layout's fixed places.
HOVER_Z = TOP + 0.16
CENTRE_ABOVE_SURFACE = BLOCK / 2 + PAD_TIP_PAST_TCP  # TCP height over the surface a block stands on, to hold it mid-height
LEFT_YAW, RIGHT_YAW = 0.0, float(np.pi / 2)
REST = {"left": (0.30, 0.30, HOVER_Z + 0.04), "right": (0.30, -0.30, HOVER_Z + 0.04)}
# where each red block is set down in the crate (blue_1, in the blue_removed ablation, goes in the second column)
CRATE_SPOT = {"red_1": (0.26, -0.34), "red_2": (0.26, -0.44), "red_3": (0.26, -0.54),
              "blue_1": (0.34, -0.39), "blue_2": (0.34, -0.49)}
LOAD_INDEX = {n: 2 * j + 1 for j, n in enumerate(NAMES)}  # loaded in the standing window after this index


class _Plan:
    """Waypoints that end at absolute episode times."""

    def __init__(self, yaw: float, rest):
        self.t, self.pts, self.yaw, self.pos, self.finger = 0.0, [], yaw, rest, FINGER_OPEN

    def go(self, pos, finger, until: float, label: str, yaw: float | None = None):
        dur = until - self.t
        if dur < 0.2 - 1e-9:
            raise ValueError(f"{label}: needs to end at {until:.2f} s but the plan is already at {self.t:.2f} s")
        self.pts.append(Waypoint(dur, tuple(float(v) for v in pos), finger, label, self.yaw if yaw is None else yaw))
        self.t, self.pos, self.finger = until, pos, finger


def _pick_place(p: _Plan, name: str, pick_xy, pick_surface: float, place_xy, place_surface: float, t0: float,
                release_t: float, tag: str, pick_at: float | None = None):
    """One cycle ending with the fingers open at ``release_t`` over the place spot and ``name`` set down. ``pick_at``:
    the time the fingers close on the block (the cycle is laid out backwards from the release time)."""
    pz = pick_surface + CENTRE_ABOVE_SURFACE
    qz = place_surface + CENTRE_ABOVE_SURFACE + 0.004
    over = release_t - 7.6 + 2.0  # fingers over the block, open, at hover
    p.go((*pick_xy, HOVER_Z), FINGER_OPEN, over, f"over {name}")
    p.go((*pick_xy, pz), FINGER_OPEN, over + 1.0, f"descend {name}")
    p.go((*pick_xy, pz), FINGER_CLOSED, over + 1.7, f"close {name}")
    p.go((*pick_xy, HOVER_Z), FINGER_CLOSED, over + 2.6, f"lift {name}")
    p.go((*place_xy, HOVER_Z), FINGER_CLOSED, over + 4.2, f"carry {name} {tag}")
    p.go((*place_xy, qz), FINGER_CLOSED, over + 5.1, f"lower {name} {tag}")
    p.go((*place_xy, qz), FINGER_OPEN, over + 5.6, f"release {name} {tag}")
    p.go((*place_xy, HOVER_Z), FINGER_OPEN, over + 6.4, f"rise {name}")


def left_plan(names=NAMES) -> list[Waypoint]:
    """Left arm: each block from the mat onto the carrier standing in the loader slot, set down 1.3 s into the window."""
    p = _Plan(LEFT_YAW, REST["left"])
    for n in names:
        lo, _ = standing_window(LOAD_INDEX[n])
        _pick_place(p, n, START_XY[n], MAT_TOP, tuple(LOADER_XY), CARRIER_TOP, 0.0, lo + 1.3, "to loader")
    p.go(REST["left"], FINGER_OPEN, p.t + 1.5, "left home")
    return p.pts


def right_plan(names=REDS, shift: float = 0.0) -> list[Waypoint]:
    """Right arm: each block lifted off the carrier at the unload slot (indexes 4, 6, 8 for the reds) and crated."""
    p = _Plan(RIGHT_YAW, REST["right"])
    for n in names:
        lo, _ = standing_window(LOAD_INDEX[n] + 3)
        lo += shift
        _pick_place(p, n, tuple(UNLOAD_XY), CARRIER_TOP, CRATE_SPOT[n], FLOOR_TOP, 0.0, lo + 5.3, "to crate")
    p.go(REST["right"], FINGER_OPEN, p.t + 1.5, "right home")
    return p.pts


def _hold(side: str, seconds: float = 12.0) -> list[Waypoint]:
    return [Waypoint(seconds, REST[side], FINGER_OPEN, "hold", LEFT_YAW if side == "left" else RIGHT_YAW)]


def demo():
    """Both arms, per DEMO_VARIANT (default ``full``): settle (hold both, expect 0), left_only, right_only, late (the
    right arm unloads the reds only one lap, 30 s, later: after the first return), blue_removed (the right arm also
    crates blue_1 too)."""
    v = os.environ.get("DEMO_VARIANT", "full")
    if v == "settle":
        return {"left": _hold("left"), "right": _hold("right")}
    if v == "left_only":
        return {"left": left_plan(), "right": _hold("right")}
    if v == "right_only":
        return {"left": _hold("left"), "right": right_plan()}
    if v == "late":
        return {"left": left_plan(REDS), "right": right_plan(shift=SLOTS * PERIOD)}
    if v == "blue_removed":  # four loads (the reds and blue_1); the right arm crates blue_1 as well (a blue block must stay riding)
        return {"left": left_plan((*REDS, "blue_1")), "right": right_plan((*REDS, "blue_1"))}
    if v != "full":
        raise ValueError(f"unknown DEMO_VARIANT {v!r}")
    return {"left": left_plan(), "right": right_plan()}
