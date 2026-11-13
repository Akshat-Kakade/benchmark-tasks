"""On the station table: a three-slot wheeled rack (6 kg, free to roll, 0.404 m wide) stands in front of the robot and a
loaded quarter sheet pan (0.241 x 0.330 m, 20 mm walls, ten cookies, rear grab tab) rests on a low counter in front of it.
The pan has to go into the middle slot of the rack without letting the rack move.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Put the full baking tray into the second slot of the rolling rack without letting the rack move."

STEEL = (0.78, 0.80, 0.82)
ALU = (0.70, 0.72, 0.76)
CRATE = (0.85, 0.55, 0.25)
COOKIE = (0.55, 0.35, 0.18)
GREY = (0.45, 0.47, 0.50)

# ---- rack (part frame: origin on the base plate's lower face centre, x = from the open front toward the back, y across,
#      z up; world position of the front face is RX, RY) --------------------------------------------------------------
RX, RY = 0.34, -0.03
RACK_DEPTH = 0.250                      # clear depth between the open front and the rear stop panel
RACK_W_IN = 0.354                       # clear width between the side panels
PLATE_T = 0.015
PANEL_T = 0.005
SLOT_TOP = (0.055, 0.110, 0.165)        # rib (ledge) top heights above the table for slots 1, 2, 3
RIB_T, RIB_W = 0.006, 0.025             # rib thickness, inward reach from the side panel (tray overlaps it by 13 mm)
POST = 0.020
POST_H = 0.240                          # front posts: the upright the left or right hand holds
PANEL_H = 0.205
RACK_MASS = 6.0
RACK_FRICTION = 0.45
CASTER_H = 0.010                        # caster feet: four sprung feet whose own friction (0.01, rolling resistance) is the only contact with the table
CASTER_MU = 0.010
YI = RACK_W_IN / 2

_rack_boxes = [
    ((0.128, 0.0, CASTER_H + PLATE_T / 2), (0.256, 2 * (YI + PANEL_T + POST), PLATE_T), ALU),               # base plate
    ((0.125, YI + PANEL_T / 2, CASTER_H + PANEL_H / 2), (0.25, PANEL_T, PANEL_H), ALU),                      # side panels
    ((0.125, -YI - PANEL_T / 2, CASTER_H + PANEL_H / 2), (0.25, PANEL_T, PANEL_H), ALU),
    ((RACK_DEPTH + 0.003, 0.0, CASTER_H + PANEL_H / 2), (0.006, RACK_W_IN + 2 * PANEL_T, PANEL_H), ALU),    # rear stop panel
    ((POST / 2, YI + PANEL_T + POST / 2, CASTER_H + POST_H / 2), (POST, POST, POST_H), ALU),                # front posts
    ((POST / 2, -YI - PANEL_T - POST / 2, CASTER_H + POST_H / 2), (POST, POST, POST_H), ALU),
]
RIB_FRICTION = 0.45                      # steel pan on aluminium rib; the ribs are on light spring sliders so they can carry their own friction
_ribs = []
for _k, _h in enumerate(SLOT_TOP):                                                                # ribs
    for _s in (1, -1):
        _ribs.append((f"rib{_k + 1}{'l' if _s > 0 else 'r'}", (RACK_DEPTH / 2, _s * (YI - RIB_W / 2), _h - RIB_T / 2)))
_Z0 = CASTER_H + PLATE_T / 2            # the body origin sits at the plate's mid-height
_rack_boxes = [((c[0], c[1], c[2] - _Z0), s, col) for c, s, col in _rack_boxes]
_sliders = [Slider(n, rest=(c[0], c[1], c[2] - _Z0), boxes=(((c[0], c[1], c[2] - _Z0), (RACK_DEPTH, RIB_W, RIB_T), ALU),),
                   travel=0.004, rate=8000.0, damping=40.0, mass=0.02, friction=RIB_FRICTION) for n, c in _ribs]
for _i, (_cx, _cy) in enumerate(((0.03, 0.17), (0.03, -0.17), (0.226, 0.17), (0.226, -0.17))):
    _c = (_cx, _cy, CASTER_H / 2 - _Z0)
    _sliders.append(Slider(f"caster{_i}", rest=_c, boxes=((_c, (0.04, 0.04, CASTER_H), GREY),), travel=0.004, rate=20000.0,
                           damping=60.0, mass=0.05, friction=CASTER_MU))
RACK = part("rack", pos=(RX, RY, TOP + _Z0 + 0.001), mass=RACK_MASS, friction=RACK_FRICTION, color=ALU, boxes=_rack_boxes,
            sliders=_sliders)
RACK_BASE_Z = TOP + _Z0                 # world z of the rack frame origin

# ---- tray (origin on the lower face of the bottom, centre) ------------------------------------------------------------
TD, TW, TWALL, THT = 0.241, 0.330, 0.003, 0.020
TAB_W, TAB_H, TAB_T = 0.080, 0.036, 0.008    # folded grab handle on the rear wall
TRAY_MASS = 0.35
TRAY_FRICTION = 0.45
COUNTER_T = 0.015
TRAY_X = 0.2045                         # tray centre x on the counter (front edge at 0.325, 15 mm before the rack)
_tray_boxes = [
    ((0, 0, TWALL / 2), (TD, TW, TWALL), STEEL),
    ((TD / 2 - TWALL / 2, 0, THT / 2 + TWALL / 2), (TWALL, TW, THT), STEEL),
    ((-TD / 2 + TWALL / 2, 0, THT / 2 + TWALL / 2), (TWALL, TW, THT), STEEL),
    ((0, TW / 2 - TWALL / 2, THT / 2 + TWALL / 2), (TD, TWALL, THT), STEEL),
    ((0, -TW / 2 + TWALL / 2, THT / 2 + TWALL / 2), (TD, TWALL, THT), STEEL),
    ((-TD / 2 + TAB_T / 2, 0, TAB_H / 2), (TAB_T, TAB_W, TAB_H), STEEL),                          # rear grab handle
]
for _ix in range(2):
    for _iy in range(5):
        _tray_boxes.append(((-0.015 + 0.06 * _ix, -0.10 + 0.05 * _iy, TWALL + 0.006), (0.04, 0.04, 0.012), COOKIE))
TRAY = part("tray", pos=(TRAY_X, RY, TOP + COUNTER_T + 0.001), mass=TRAY_MASS, friction=TRAY_FRICTION, color=STEEL,
            boxes=_tray_boxes)
TAB_PINCH_Z = 0.026                     # pinch height above the tray's lower face (upper part of the tab)

COUNTER = Box("counter", pos=(TRAY_X + 0.0, RY, TOP + COUNTER_T / 2), size=(0.27, 0.36, COUNTER_T), color=GREY)
OBJECTS = [COUNTER, RACK, TRAY]

# ---- scoring thresholds ---------------------------------------------------------------------------------------------
RACK_SLIDE = 0.040          # zero if the rack moves this far (m) from where it stood
RACK_YAW = 8.0              # zero if the rack turns more than this (deg) or tips
TRAY_SPILL = 15.0           # zero if the tray tilts more than this (deg): the load spills
SLOT_TOL = 0.010            # tray lower face within this of a rib top = at that slot
SEAT_TOL = 0.004            # 1.0: tray lies on the ribs (lower face within this of the rib top)
LEVEL_DEG = 5.0
Y_TOL = 0.012
IN_MOUTH = -0.040           # 0.25: front edge of the tray at most this far before the rack front
HALF = 0.120                # 0.5: front edge this far inside (about half the tray depth)
FULL = RACK_DEPTH - 0.010   # 0.75: front edge within 10 mm of the rear stop
GRIP_OPEN = 0.058           # finger-tip body gap (m) is 0.048 fully open and grows as the jaws close (0.079 on a 20 mm post, 0.099 on nothing/thin): open below this
STILL_V, STILL_W, STILL_Q = 0.010, 0.10, 0.05   # m/s, rad/s, arm joint rad/s
HOLD_STEPS = 20

import os  # noqa: E402

_DEBUG = bool(os.environ.get('RACK_DEBUG'))
_STATE: dict = {}
_ENV: dict = {"env": None}


def _fresh() -> dict:
    return dict(step=0, scored=None, last=(0.0, False), base=None, gate=False, held=0, max_slide=0.0, max_tilt=0.0,
                max_rack_tilt=0.0)


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(_fresh())


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def _install_env_hook() -> None:
    try:
        from yam_arena import inspect_embodiment as ie
    except Exception:  # pragma: no cover
        return
    cls = ie.YamArenaEmbodiment
    if getattr(cls._task_step, "_rack_hooked", False):
        return
    base = cls._task_step

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        return base(self, *a, **k)

    _task_step._rack_hooked = True
    cls._task_step = _task_step


_install_env_hook()


def grip_state():
    """{"L": (tcp xyz, finger gap, max joint speed), "R": ...} or None if unknown."""
    env = _ENV["env"]
    if env is None:
        return None
    base = env.unwrapped
    origin = base.scene.env_origins[0]
    out = {}
    for tag, key in (("L", "left_robot"), ("R", "right_robot")):
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


def yaw_deg(state: PartState) -> float:
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def tray_metrics(tray: PartState) -> dict:
    """Pose of the tray relative to the rack's nominal (start) frame."""
    front = tray.to_world((TD / 2, 0.0, 0.0))
    zb = float(tray.pos[2] - TOP)
    slots = [k for k, h in enumerate(SLOT_TOP) if abs(zb - h) <= SLOT_TOL]
    return dict(depth=float(front[0] - RX), zb=zb, dy=float(tray.pos[1] - RY), tilt=tilt_deg(tray),
                slot=(slots[0] + 1 if slots else 0), seat=min(abs(zb - h) for h in SLOT_TOP),
                seat2=abs(zb - SLOT_TOP[1]))


def _update(objects, grip=None) -> tuple[float, bool]:
    """Score and success of this control step (success() and score() both run each step: evaluate once)."""
    st = _STATE
    if st["scored"] == st["step"]:
        return st["last"]
    st["scored"] = st["step"]
    rack, tray = objects["rack"], objects["tray"]
    if grip is None:
        grip = grip_state()
    if st["step"] >= 15 and st["base"] is None:
        st["base"] = (np.array(rack.pos), yaw_deg(rack))
    if st["base"] is not None:
        slide = float(np.linalg.norm(rack.pos[:2] - st["base"][0][:2]))
        dyaw = abs((yaw_deg(rack) - st["base"][1] + 180.0) % 360.0 - 180.0)
        st["max_slide"] = max(st["max_slide"], slide)
        if slide > RACK_SLIDE or dyaw > RACK_YAW or tilt_deg(rack) > RACK_YAW:
            st["gate"] = True
    m = tray_metrics(tray)
    st["max_tilt"] = max(st["max_tilt"], m["tilt"])
    if m["tilt"] > TRAY_SPILL or tray.pos[2] - TOP < COUNTER_T - 0.010:    # spilled, or fell off the counter / rack
        st["gate"] = True
    level = m["tilt"] <= LEVEL_DEG
    aligned = abs(m["dy"]) <= Y_TOL
    quiet = (np.linalg.norm(rack.lin_vel) <= STILL_V and np.linalg.norm(tray.lin_vel) <= STILL_V
             and np.linalg.norm(tray.ang_vel) <= STILL_W and np.linalg.norm(rack.ang_vel) <= STILL_W)
    still = bool(grip) and quiet and all(g[2] <= STILL_Q for g in grip.values())
    open_ = bool(grip) and all(g[1] <= GRIP_OPEN for g in grip.values())
    full = m["slot"] == 2 and level and aligned and m["depth"] >= FULL
    if st["gate"]:
        score = 0.0
    elif full:
        score = 0.75
    elif m["slot"] == 2 and level and m["depth"] >= HALF:
        score = 0.5
    elif m["slot"] in (1, 3) and level and aligned and m["depth"] >= HALF:
        score = 0.5                      # a full pan in the wrong slot
    elif m["slot"] == 2 and m["depth"] >= IN_MOUTH and m["tilt"] <= 15.0:
        score = 0.25
    else:
        score = 0.0
    hold_ok = (not st["gate"]) and full and m["seat2"] <= SEAT_TOL and still and open_
    st["held"] = st["held"] + 1 if hold_ok else 0
    done = st["held"] >= HOLD_STEPS
    if done:
        score = 1.0
    st["last"] = (score, bool(done))
    if _DEBUG and st["step"] % 10 == 0:
        g = grip or {}
        print(f"[dbg] step {st['step']} score {score} slide {st['max_slide']*1000:.1f}mm depth {m['depth']:.3f} zb {m['zb']:.3f} "
              f"dy {m['dy']*1000:.1f} tilt {m['tilt']:.1f} rack {np.round(rack.pos - np.array([RX, RY, RACK_BASE_Z]), 3).tolist()} "
              f"gate {st['gate']} held {st['held']} " + " ".join(f"{k} gap {v[1]:.3f} qd {v[2]:.2f}" for k, v in g.items()),
              flush=True)
    return st["last"]


def success(objects) -> bool:
    """Pan fully in slot 2 (front edge within 10 mm of the stop, within 10 mm of the slot-2 rib height, level, centred),
    resting on the ribs, rack moved less than 40 mm / 8 degrees at every step, both grippers open, pan, rack and both arms
    still, all for 20 consecutive control steps (1 s)."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the rack ever slid 40 mm or turned 8 degrees, or the pan tilted 15 degrees or fell. Otherwise (end state):
    0.25 pan at the slot-2 mouth; 0.5 at least half in slot 2 (or fully in another slot); 0.75 fully in slot 2 and level;
    1.0 also resting on the ribs with both grippers open and everything still for 1 s."""
    return _update(objects)[0]
