"""On the station table (the "floor"): a small grey steel drum (4 L, 1.2 kg) lies on its side in the aisle in front of the arms, its axis
along y, free to roll. Ahead of it, turned 35 degrees to the left of straight ahead, is a bay: two yellow rails and a
red back bar. The robot must get the drum into the bay and leave it resting against the back bar, hands clear.

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
from partkit import on_reset, on_step, part, write_pose  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Get the steel drum into the angled bay between the two yellow rails, resting against the red back "
               "bar, and take your hands off it.")

STEEL = (0.62, 0.64, 0.66)
DARK = (0.20, 0.22, 0.26)
YELLOW = (1.0, 0.85, 0.0)
RED = (0.85, 0.12, 0.10)
GREEN = (0.10, 0.65, 0.25)

# -- the drum (part frame: origin at the centre, axis = part z; written onto its side in on_reset) ------------------------
R_BODY, R_RING = 0.070, 0.075             # shell and end-chime radius (the drum rolls on the two chimes)
L_BODY, L_RING = 0.270, 0.015             # shell length between the chimes, chime width: 0.300 overall
HALF_L = L_BODY / 2 + L_RING              # 0.150: centre to an end face
MASS = 1.2                                # kg: 0.8 mm steel shell (0.163 m2 x 0.8 mm x 7850) ~ 1.0 kg + two hoop chimes; rings are solid discs in sim, total mass fixed
FRICTION = 0.5
START = (0.231, -0.018)                   # centre xy at rest in the aisle
START_Z = TOP + R_RING
QUAT_SIDE = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))     # part z -> world y
DRUM = part("drum", pos=(START[0], START[1], TOP + HALF_L + 0.001), mass=MASS, friction=FRICTION, color=STEEL,
            boxes=[((0, 0, 0), (0.01, 0.01, 0.01), STEEL)],
            cylinders=[((0, 0, 0), R_BODY, L_BODY, STEEL), ((0, 0, L_BODY / 2 + L_RING / 2), R_RING, L_RING, DARK),
                       ((0, 0, -L_BODY / 2 - L_RING / 2), R_RING, L_RING, DARK)])

# -- the bay (world frame) --------------------------------------------------------------------------------------------
BAY_DEG = 35.0
BAY_C = np.array([0.370, 0.080])          # drum centre when it rests against the back bar (target)
_B = math.radians(BAY_DEG)
LANE = np.array([math.cos(_B), math.sin(_B)])     # along the bay, toward the back bar
SIDE = np.array([-math.sin(_B), math.cos(_B)])    # across the bay (the drum's axis direction when it is in)
INNER_HALF = 0.170                         # rail inner faces 0.340 apart: 20 mm clear of each drum end
RAIL_T, RAIL_H = 0.020, 0.050
RAIL_S0, RAIL_S1 = -0.030, 0.100           # rail extent along the bay from BAY_C
STOP_FACE = 0.076                          # back bar's near face along the bay from BAY_C
STOP_T = 0.020


def _fixed(name, s0, s1, l0, l1, h, color):
    """A fixed box from s0..s1 along the bay and l0..l1 across it (from BAY_C)."""
    s, l = (s0 + s1) / 2, (l0 + l1) / 2
    c = BAY_C + s * LANE + l * SIDE
    return Box(name, pos=(float(c[0]), float(c[1]), TOP + h / 2), size=(s1 - s0, l1 - l0, h), yaw=_B, color=color)


RAILS = [_fixed("rail_left", RAIL_S0, RAIL_S1, INNER_HALF, INNER_HALF + RAIL_T, RAIL_H, YELLOW),
         _fixed("rail_right", RAIL_S0, RAIL_S1, -INNER_HALF - RAIL_T, -INNER_HALF, RAIL_H, YELLOW)]
STOP = _fixed("back_bar", STOP_FACE, STOP_FACE + STOP_T, -INNER_HALF, INNER_HALF, RAIL_H, RED)
HEADING_BAND = _fixed("heading_band", -0.06, 0.06, -0.010, 0.010, 0.001, GREEN)
OBJECTS = [*RAILS, STOP, HEADING_BAND, DRUM]

# -- scoring thresholds (m, deg) ---------------------------------------------------------------------------------------
LIFT_GATE = 0.008          # drum centre above its resting height for more than LIFT_STEPS steps: 0
LIFT_STEPS = 10
PITCH_GATE_DEG = 8.0       # axis tilted out of the horizontal
ADVANCE_1, ADVANCE_2 = 0.06, 0.12
YAW_2 = 15.0
YAW_TOL_075 = 15.0
CENTRE_TOL = 0.020         # 1.0: centre within 20 mm of BAY_C
YAW_TOL_10 = 8.0
ROLL_FRACTION = 0.70       # 1.0: rolled at least this share of travel / radius
GRIP_OPEN_MAX = 0.058      # finger-tip gap <= this = open (0.048 open, ~0.09 closed)
AWAY = 0.120               # TCP this far from both drum end centres = clear
STILL_STEPS = 20
STILL_POS, STILL_YAW = 0.0015, 0.5
BASELINE_STEP = 15         # the teleported part pops in its first steps: baselines taken here

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), base=None, prev_R=None, roll=0.0, lift_n=0, gate="", recent=[],
                  condh=[], touched_stop=False, max_pitch=0.0)
    if environment is not None:
        write_pose(environment, "drum", (START[0], START[1], START_Z + 0.0005), QUAT_SIDE)


RR_COEFF = float(os.environ.get("DRUM_RR", "0.01"))   # rolling resistance coefficient, steel drum on concrete
CTRL_DT = 0.05


def _rolling_resistance(environment) -> None:
    """PhysX rolls a rigid cylinder with no resistance at all (a hair of table tilt walked the drum 0.25 m). A real drum on
    a concrete floor has a rolling resistance coefficient of ~0.01: once per control step the drum's speed is reduced by
    c_rr * g * dt (to zero when it is slower than that), its spin with it. Only the drum; arms and fixtures untouched."""
    if environment is None or RR_COEFF <= 0:
        return
    import torch

    asset = environment.unwrapped.scene["drum"]
    vel = getattr(asset.data.root_vel_w, "torch", asset.data.root_vel_w)
    v = vel[0].detach().clone()
    speed = float(torch.linalg.norm(v[:2]))
    if speed <= 0.0:
        return
    f = max(0.0, speed - RR_COEFF * 9.81 * CTRL_DT) / speed
    v[:2] *= f
    v[3:6] *= f
    asset.write_root_velocity_to_sim_index(root_velocity=v.unsqueeze(0))


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1
    _rolling_resistance(environment)


_reset_state()


def _arms():
    """Per arm: TCP, finger-tip gap; None when the simulator is not attached."""
    try:
        from partkit import env
    except Exception:  # pragma: no cover
        return None
    e = env()
    if e is None:
        return None
    base = e.unwrapped
    origin = base.scene.env_origins[0]
    out = {}
    for side in ("left", "right"):
        art = base.scene[f"{side}_robot"]
        names = list(art.body_names)
        pos = (getattr(art.data.body_pos_w, "torch", art.data.body_pos_w)[0] - origin).detach().cpu().numpy().astype(
            np.float64)
        g, tl, tr = pos[names.index("gripper")], pos[names.index("tip_left")], pos[names.index("tip_right")]
        mid = (tl + tr) / 2
        ax = (mid - g) / (np.linalg.norm(mid - g) + 1e-12)
        out[side] = {"tcp": g + ax * 0.134, "gap": float(np.linalg.norm(tl - tr))}
    return out


# -- measures -----------------------------------------------------------------------------------------------------------
def axis(state) -> np.ndarray:
    return state.rotation()[:, 2]


def yaw_deg(state) -> float:
    """Heading of the drum: 0 with the axis along y, +35 when it lies across the bay."""
    a = axis(state)
    return math.degrees(math.atan2(-a[0], a[1]))


def pitch_deg(state) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, abs(float(axis(state)[2]))))))


def end_centres(state):
    a = axis(state)
    return state.pos + HALF_L * a, state.pos - HALF_L * a


def ring_points(state, n=16) -> np.ndarray:
    """Points on the rims of both end faces (world)."""
    R = state.rotation()
    a, u, v = R[:, 2], R[:, 0], R[:, 1]
    pts = []
    for sgn in (-1, 1):
        for k in range(n):
            t = 2 * math.pi * k / n
            pts.append(state.pos + sgn * HALF_L * a + R_RING * (math.cos(t) * u + math.sin(t) * v))
    return np.array(pts)


def bay_coords(p) -> tuple:
    """(s along the bay, l across it) of a world point, from BAY_C."""
    rel = np.asarray(p, float)[:2] - BAY_C
    return float(rel @ LANE), float(rel @ SIDE)


def touches_rail(state, margin=0.001) -> bool:
    """A point on a drum end face (below the rail top) is within the rail's footprint."""
    for p in ring_points(state):
        if p[2] > TOP + RAIL_H:
            continue
        s, l = bay_coords(p)
        if RAIL_S0 - margin <= s <= RAIL_S1 + margin and INNER_HALF - margin <= abs(l) <= INNER_HALF + RAIL_T + margin:
            return True
    return False


def touches_stop(state, margin=0.003) -> bool:
    for p in ring_points(state, 72):
        if p[2] <= TOP + RAIL_H and STOP_FACE - margin <= bay_coords(p)[0] <= STOP_FACE + STOP_T and abs(bay_coords(p)[1]) <= INNER_HALF:
            return True
    # the shell between the chimes
    R = state.rotation()
    for k in range(24):
        t = 2 * math.pi * k / 24
        q = state.pos + R_BODY * (math.cos(t) * R[:, 0] + math.sin(t) * R[:, 1])
        if q[2] <= TOP + RAIL_H and STOP_FACE - margin <= bay_coords(q)[0] <= STOP_FACE + STOP_T and abs(bay_coords(q)[1]) <= INNER_HALF:
            return True
    return False


def _roll_step(prev_R, R) -> float:
    """Rotation of the drum about its own axis between two steps (rad, + = rolling toward +x for an axis along +y)."""
    d = R @ prev_R.T
    ang = math.acos(max(-1.0, min(1.0, (np.trace(d) - 1) / 2)))
    if ang < 1e-9:
        return 0.0
    w = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) / (2 * math.sin(ang) + 1e-12) * ang
    return float(w @ R[:, 2])


def _metrics(objects) -> dict:
    drum = objects["drum"]
    base = _STATE["base"]
    disp = 0.0 if base is None else float(np.linalg.norm(drum.pos[:2] - base["pos"][:2]))
    s, l = bay_coords(drum.pos)
    return {
        "x": float(drum.pos[0]), "y": float(drum.pos[1]), "z": float(drum.pos[2]),
        "yaw": yaw_deg(drum), "pitch": pitch_deg(drum), "advance": disp,
        "roll": _STATE["roll"], "s": s, "l": l, "centre_err": float(np.linalg.norm(drum.pos[:2] - BAY_C)),
        "gate": _STATE["gate"],
    }


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    drum = objects["drum"]
    R = drum.rotation()
    if _STATE["prev_R"] is not None and _STATE["step"] > BASELINE_STEP:
        _STATE["roll"] += _roll_step(_STATE["prev_R"], R)
    _STATE["prev_R"] = R.copy()
    if _STATE["base"] is None and _STATE["step"] >= BASELINE_STEP:
        _STATE["base"] = {"pos": np.array(drum.pos), "z": float(drum.pos[2]), "yaw": yaw_deg(drum)}
    base = _STATE["base"]
    if base is None:
        _STATE["last"] = (0.0, False)
        return _STATE["last"]
    m = _metrics(objects)
    _STATE["max_pitch"] = max(_STATE["max_pitch"], m["pitch"])
    # gates (history, never undone)
    if m["z"] - base["z"] > LIFT_GATE:
        _STATE["lift_n"] += 1
    if _STATE["lift_n"] > LIFT_STEPS and not _STATE["gate"]:
        _STATE["gate"] = "lifted"
    if m["pitch"] > PITCH_GATE_DEG and not _STATE["gate"]:
        _STATE["gate"] = "pitched"
    if m["z"] < TOP + 0.03 or abs(m["y"]) > 0.45 or m["x"] > 0.60 or m["x"] < 0.0:
        _STATE["gate"] = _STATE["gate"] or "off_floor"
    if touches_stop(drum):
        _STATE["touched_stop"] = True
    if not _STATE["touched_stop"] and touches_rail(drum) and not _STATE["gate"]:
        _STATE["gate"] = "rail_first"
    # derived
    yaw_rel = m["yaw"] - base["yaw"]
    in_bay = (RAIL_S0 <= m["s"] <= STOP_FACE) and abs(m["l"]) <= INNER_HALF - 0.05
    yaw_err = abs(m["yaw"] - BAY_DEG)
    rolled = _STATE["roll"] * R_RING >= ROLL_FRACTION * m["advance"] * (1.0 if m["advance"] > 0 else 0.0)
    arms = _arms()
    free = True
    if arms is not None:
        c1, c2 = end_centres(drum)
        for a in arms.values():
            if a["gap"] > GRIP_OPEN_MAX or min(np.linalg.norm(a["tcp"] - c1), np.linalg.norm(a["tcp"] - c2)) < AWAY:
                free = False
    recent = _STATE["recent"]
    recent.append((np.array(drum.pos), m["yaw"]))
    del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(
        np.linalg.norm(p - recent[-1][0]) < STILL_POS and abs(y - recent[-1][1]) < STILL_YAW for p, y in recent)
    cond = bool(m["centre_err"] <= CENTRE_TOL and yaw_err <= YAW_TOL_10 and rolled and free and _STATE["touched_stop"]
                and pitch_ok(m))
    hist = _STATE["condh"]
    hist.append(cond)
    del hist[:-STILL_STEPS]
    full = bool(still and len(hist) == STILL_STEPS and all(hist))        # every condition held for 20 steps (1 s)
    if _STATE["gate"]:
        result = (0.0, False)
    elif full:
        result = (1.0, True)
    elif in_bay and yaw_err <= YAW_TOL_075:
        result = (0.75, False)
    elif m["advance"] >= ADVANCE_2 and yaw_rel >= YAW_2:
        result = (0.5, False)
    elif m["advance"] >= ADVANCE_1:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    tr = os.environ.get("TASK_TRACE")
    if tr:
        rec = {"step": _STATE["step"], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
               "score": result[0], "stop": _STATE["touched_stop"], "free": free, "still": bool(still), "rolled": bool(rolled)}
        if arms is not None:
            for k, a in arms.items():
                rec[k + "_tcp"] = [round(float(t), 3) for t in a["tcp"]]
                rec[k + "_gap"] = round(a["gap"], 3)
        with open(tr, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return result


def pitch_ok(m) -> bool:
    return m["pitch"] <= PITCH_GATE_DEG


def success(objects) -> bool:
    """Drum centre within 20 mm of the bay centre, yaw 35 +/- 8 deg, resting against the back bar, rolled (not slid)
    there, both grippers open and 120 mm clear, drum still, all for 20 consecutive steps; no gate tripped."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 centre advanced 60 mm; 0.5 advanced 120 mm and turned 15 deg toward the bay; 0.75 centre inside the bay and
    yaw within 15 deg of 35; 1.0 as success(). 0 if lifted, pitched, off the floor or the drum touched a rail before the
    back bar."""
    return _update(objects)[0]
