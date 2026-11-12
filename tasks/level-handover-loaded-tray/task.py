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

INSTRUCTION = ("Move the white tray of blue cups from the grey pad on the left to the grey pad on the right and"
               " set it down level, with no cup tipped. Neither arm reaches both pads; do not put the tray down in between.")

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
LIFT_MIN = 0.040             # rack lifted at least this far above its pad by the left hand (0.25)
AIR = 0.020                  # "in the air": plate underside at least this far above the pad top
CROSS_Y = 0.150              # ... and carried past y = +0.15 (0.25)
LEVEL_DEG, GATE_DEG = 8.0, 20.0
PLACE_TOL, PLACE_YAW_DEG = 0.015, 8.0
SEAT_TOL = 0.003             # rack plate underside within 3 mm of the pad top = supported by the pad
CUP_SLOT_TOL = 0.015
HOLD_STEPS = 20
HOLD_R = 0.050               # a gripper TCP this close to a handle centre ...
GRIP_GAP_MIN = 0.058         # ... with its finger-tip bodies at least this far apart (measured: 0.048 m open, 0.069 m pinching a handle) holds the rack
STILL_V, STILL_W, STILL_Q = 0.02, 0.15, 0.05   # rack m/s, rack rad/s, arm joint rad/s: "still"
SPILL_DEG = GATE_DEG         # real cups: 30 mL water, 19 mm deep in a 60 mm cup (41 mm freeboard); a cup tilted over 20 degrees
                             # (slosh margin: static spill needs ~60) or leaving the plate counts as spilled

_STATE: dict = {}
_ENV: dict = {"env": None}


def _fresh() -> dict:
    return dict(step=0, scored=None, last=(0.0, False), lifted=False, crossed=False, max_tilt=0.0, gate=False, held=0,
                a_seen=False, b_seen=False, transfer=False)


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
    if getattr(cls._task_step, "_handover_hooked", False):
        return
    base = cls._task_step

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        return base(self, *a, **k)

    _task_step._handover_hooked = True
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
    """Angle between the part's z axis and the world's vertical."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def yaw_deg(state: PartState) -> float:
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def _dist(state: PartState, xy) -> float:
    return float(np.hypot(state.pos[0] - xy[0], state.pos[1] - xy[1]))


def holds(rack: PartState, grip, tag: str) -> bool:
    """Arm ``tag`` holds the rack: its TCP is within HOLD_R of a handle centre and its fingers are closed on it."""
    if not grip or tag not in grip:
        return False
    tcp, gap, _ = grip[tag]
    if gap < GRIP_GAP_MIN:
        return False
    return min(float(np.linalg.norm(rack.to_world((0.0, s * HANDLE_Y, 0.028)) - tcp)) for s in (1, -1)) <= HOLD_R


def cup_info(objects):
    """(all cups upright on the plate, all within their slots, max cup tilt)."""
    rack = objects["rack"]
    on = slots = True
    worst = 0.0
    for name, y in zip(CUP_NAMES, CUP_Y):
        c = objects[name]
        worst = max(worst, tilt_deg(c))
        lp = rack.to_local(c.pos)
        if abs(lp[0]) > 0.06 or abs(lp[1] - y) > 0.10 or not (0.005 < lp[2] < 0.05) or tilt_deg(c) > SPILL_DEG:
            on = False
        if abs(lp[0]) > CUP_SLOT_TOL or abs(lp[1] - y) > CUP_SLOT_TOL:
            slots = False
    return on, on and slots, worst


def classify(objects, memo: dict, grip=None):
    """Pure function of the poses, the gripper readings and the memo: (level, all-1.0-conditions-hold, info).
    A is the left arm, B the right arm. Full credit needs the in-air transfer: A holds the rack lifted off the pad, then B
    holds it (rack still off both pads), then A has let go while B holds it, rack never resting in between."""
    rack = objects["rack"]
    on, slots, cup_tilt = cup_info(objects)
    lift = float(rack.pos[2] - RACK_Z)
    cy = float(rack.pos[1])
    t = tilt_deg(rack)
    if lift > AIR:                                         # tilt only counts while the rack is carried
        memo["max_tilt"] = max(memo["max_tilt"], t)
    resting = lift < 0.012
    off_pad = resting and _dist(rack, PAD_L) > 0.10 and _dist(rack, PAD_R) > 0.10
    if not on or t > GATE_DEG or memo["max_tilt"] > GATE_DEG or off_pad:
        memo["gate"] = True
    if memo["gate"]:
        return 0.0, False, dict(on=on, lift=lift, y=cy, tilt=t, max_tilt=memo["max_tilt"], gate=True,
                                     transfer=memo["transfer"])
    ha, hb = holds(rack, grip, "A"), holds(rack, grip, "B")
    air = lift > AIR
    if resting:                                            # touching a pad between the grips: the chain restarts
        if not memo["transfer"]:
            memo["a_seen"] = memo["b_seen"] = False
    elif air:
        if ha and not memo["a_seen"] and not hb:
            memo["a_seen"] = True
        elif memo["a_seen"] and hb and not memo["b_seen"]:
            memo["b_seen"] = True
        elif memo["b_seen"] and hb and not ha and not memo["transfer"]:
            memo["transfer"] = True                        # A has let go, B holds it, still in the air
    if lift > LIFT_MIN and on and cy < CROSS_Y and memo["a_seen"]:
        memo["lifted"] = memo["crossed"] = True
    seated = abs(lift) <= SEAT_TOL
    still = (float(np.linalg.norm(rack.lin_vel)) < STILL_V and float(np.linalg.norm(rack.ang_vel)) < STILL_W
             and bool(grip) and all(grip[k][2] < STILL_Q for k in grip))
    free = bool(grip) and not ha and not hb and all(grip[k][1] < GRIP_GAP_MIN for k in grip)
    info = dict(on=on, slots=slots, lift=lift, y=cy, tilt=t, max_tilt=memo["max_tilt"], gate=False, a=ha, b=hb,
                transfer=memo["transfer"], place_d=_dist(rack, PAD_R), yaw=yaw_deg(rack), still=still, free=free)
    placed = (memo["transfer"] and seated and _dist(rack, PAD_R) <= PLACE_TOL and abs(yaw_deg(rack)) <= PLACE_YAW_DEG
              and t < 4.0)
    level_ok = memo["max_tilt"] <= LEVEL_DEG
    if placed and level_ok and slots and still and free:
        return 1.0, True, info
    if placed and level_ok:
        return 0.75, False, info
    if memo["transfer"] and level_ok:
        return 0.5, False, info
    if memo["crossed"]:
        return 0.25, False, info
    return 0.0, False, info


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    level, perfect, _ = classify(objects, _STATE, grip_state())
    _STATE["held"] = _STATE["held"] + 1 if perfect else 0
    ok = _STATE["held"] >= HOLD_STEPS
    if perfect and not ok:
        level = 0.75                                       # never 1.0 before the full 1 s hold
    result = (level, ok)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Left arm lifts the rack off the left pad, the right arm takes it while it is still in the air and the left arm lets
    go (checked from both grippers: TCP at a handle, fingers closed on it), the rack is then set down on the right pad
    (plate underside within 3 mm of the pad top, within 15 mm of its centre and 8 degrees of yaw), the carry was
    <= 8 degrees of tilt, all cups upright in their slots, rack and both arms still and both grippers open, for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 left arm lifts the rack >= 40 mm with the cups upright and carries it past y = +0.15; 0.5 verified in-air
    transfer (left hold, then right hold, then left released, rack off both pads) with the carry <= 8 degrees; 0.75 level
    transfer and rack seated on the right pad within 15 mm / 8 degrees of yaw; 1.0 as in success() (1 s hold).
    Zero if a cup spills (tilts over 20 degrees or leaves the plate), the rack tilts over 20 degrees while carried, or
    the rack is left on the table away from both pads."""
    return _update(objects)[0]
