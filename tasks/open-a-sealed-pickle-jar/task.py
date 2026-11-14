"""On the station table: a glass pickle jar (78 mm across, 122 mm tall, 0.65 kg with its pickles and brine) with a stuck gold
twist-off lug lid (70 mm across, 10 mm above the shoulder ledge), standing free in front of the arms. Beside it on the table lies a steel
lid lifter (a thin blade and an orange grip fin, 0.091 kg). The lid is held by a vacuum seal, which is a latch in this task file: it releases
when the blade is under the lid edge and has raised the edge by 1 mm. After that the lid (0.03 kg) turns only against 0.4 N m, which would
spin the jar on the table unless a hand holds it. Both arms are needed: one holds the jar, the other frees the lid, turns it a quarter turn,
lifts it off and sets it down.

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
from scenekit import UsdBody, before_reset, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = g.TOP

INSTRUCTION = "Open the pickle jar and set the lid down."

_SY = math.sin(g.TOOL_YAW / 2)
_CY = math.cos(g.TOOL_YAW / 2)
JAR = UsdBody("pickle_jar", usd="assets/pickle_jar.usda", pos=(g.JAR_X, g.JAR_Y, TOP),
              size=(2 * g.BODY_R, 2 * g.BODY_R, g.LID_Z0 + g.LID_H), articulation=True,
              joint_pos=(("lid_x", 0.0), ("lid_y", 0.0), ("lid_z", 0.0), ("lid_spin", 0.0)),
              drives=(("lid_x", 0.0, 0.5), ("lid_y", 0.0, 0.5), ("lid_z", 0.0, 0.5), ("lid_spin", 0.0, 100.0)),
              friction=g.JAR_MU)
TOOL = UsdBody("lever_tool", usd="assets/lever_tool.usda", pos=(g.TOOL_X, g.TOOL_Y, TOP), rot_xyzw=(0.0, 0.0, _SY, _CY),
               size=(0.125, 0.024, 0.042), mass=g.TOOL_MASS, friction=(0.6, 0.5))
OBJECTS = [JAR, TOOL]

SEALED_TORQUE = 30.0                      # N m of drive friction while the seal holds (the real seal: > 3 N m)
_STATE: dict = {}
_LOCK = {"lid_x": (-0.0005, 0.0005), "lid_y": (-0.0005, 0.0005), "lid_z": (-0.0005, g.POP_WINDOW), "lid_spin": (-0.02, 0.02)}   # a zero-width limit is not enforced


def _fresh() -> dict:
    return dict(step=0, popped=False, unlatched=False, clear=False, pop_run=0, level=0.0, ok=False, held=0, gate=False, max_tilt=0.0,
                spin=0.0, max_spin=0.0, lid=(0.0, 0.0, 0.0), lid_prev=None, lid_speed=0.0, info={}, pop_best=None)


def _write(environment, joint: str, limits=None, effort=None) -> None:
    import torch

    asset = environment.unwrapped.scene["pickle_jar"]
    ids = asset.find_joints([joint])[0]
    if limits is not None:
        asset.write_joint_position_limit_to_sim_index(
            limits=torch.tensor([[list(limits)]], dtype=torch.float32, device=asset.device), joint_ids=ids,
            warn_limit_violation=False)
    if effort is not None:
        asset.write_joint_effort_limit_to_sim_index(
            limits=torch.tensor([[effort]], dtype=torch.float32, device=asset.device), joint_ids=ids)


def _lock(environment) -> None:
    for name, lim in _LOCK.items():
        _write(environment, name, limits=lim)
    _write(environment, "lid_spin", effort=SEALED_TORQUE)     # the seal: far beyond what a wrist can give


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(_fresh())
    if environment is not None:
        _lock(environment)


@before_reset
def _restart_lid(environment) -> None:
    from scenekit import write_joint_positions

    _lock(environment)
    write_joint_positions(environment, "pickle_jar", {"lid_x": 0.0, "lid_y": 0.0, "lid_z": 0.0, "lid_spin": 0.0})


_reset_state()


def grip_state():
    """{"L": (tcp xyz, finger-tip gap), "R": ...} for the left and right gripper, or None if unknown."""
    from scenekit import env

    e = env()
    if e is None:
        return None
    base = e.unwrapped
    origin = base.scene.env_origins[0]
    out = {}
    for tag, key in (("L", "left_robot"), ("R", "right_robot")):
        art = base.scene[key]
        names = list(art.body_names)
        pos = (getattr(art.data.body_pos_w, "torch", art.data.body_pos_w)[0] - origin).detach().cpu().numpy().astype(np.float64)
        gr = pos[names.index("gripper")]
        tl, tr = pos[names.index("tip_left")], pos[names.index("tip_right")]
        mid = (tl + tr) / 2
        ax = (mid - gr) / (np.linalg.norm(mid - gr) + 1e-12)
        out[tag] = (gr + ax * 0.134, float(np.linalg.norm(tl - tr)))
    return out


def tool_points(tool) -> tuple[np.ndarray, np.ndarray, float]:
    """World tip and tail of the lifter blade, and the angle between its pointing direction (flat) and the direction to the jar axis."""
    R = tool.rotation()
    tip = np.asarray(tool.pos) + R @ np.array(g.TIP_LOCAL)
    tail = np.asarray(tool.pos) + R @ np.array(g.TAIL_LOCAL)
    d = tip - tail
    to_jar = np.array([g.JAR_X, g.JAR_Y]) - tip[:2]
    return tip, tail, d[:2], to_jar


def prise(jar, tool, lid_up=0.0) -> dict:
    """Is the lifter's blade tip under the lid edge and is the lid edge really raised (lid_up, m) by it? Hovering near the rim does not count."""
    tip, tail, d2, to_jar = tool_points(tool)
    axis = np.asarray(jar.pos)[:2]
    r = float(np.linalg.norm(tip[:2] - axis))
    z = float(tip[2] - TOP)
    raised = float(tail[2] - tip[2])
    aim = math.acos(max(-1.0, min(1.0, float(np.dot(d2, axis - tip[:2]) / (np.linalg.norm(d2) * np.linalg.norm(axis - tip[:2]) + 1e-12)))))
    ok = g.RIM_R[0] <= r <= g.RIM_R[1] and g.RIM_Z[0] <= z <= g.RIM_Z[1] + lid_up and lid_up >= g.POP_LIFT and aim <= g.TOOL_AIM
    return dict(ok=ok, r=r, z=z, raised=raised, up=lid_up, aim=math.degrees(aim))


def loosened(spin: float) -> bool:
    """Unlatch test: the lid has turned the loosening way (counter-clockwise seen from above, positive spin) by the lug quarter turn.
    A tightening turn, of any size, never unlatches."""
    return g.LOOSEN_SIGN * spin >= g.UNLATCH


def tilt_rad(jar) -> float:
    return math.acos(max(-1.0, min(1.0, float(jar.rotation()[2, 2]))))


def seg_dist(p, a, b) -> float:
    ab = b - a
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / float(np.dot(ab, ab)))))
    return float(np.linalg.norm(p - (a + t * ab)))


def classify(jar, lid, lid_speed, memo, grip) -> tuple[float, bool, dict]:
    """(level, all-1.0-conditions-hold, info) from the poses, the grippers and the memo of latched events."""
    tilt = tilt_rad(jar)
    dist = float(math.hypot(lid[0] - jar.pos[0], lid[1] - jar.pos[1]))
    lid_h = float(lid[2] - TOP)
    if memo["step"] > 3:
        memo["max_tilt"] = max(memo["max_tilt"], tilt)
        if memo["unlatched"] and tilt > g.SPILL_TILT:
            memo["gate"] = True
        if jar.pos[2] < TOP - 0.02:
            memo["gate"] = True
    info = dict(tilt=math.degrees(tilt), lid_dist=dist, lid_h=lid_h, lid_speed=lid_speed, popped=memo["popped"], unlatched=memo["unlatched"],
                spin=math.degrees(memo["spin"]), max_tilt=math.degrees(memo["max_tilt"]), gate=memo["gate"])
    if memo["gate"]:
        return 0.0, False, info
    jar_speed = float(np.linalg.norm(jar.lin_vel))
    axis_lo, axis_hi = np.array([jar.pos[0], jar.pos[1], TOP]), np.array([jar.pos[0], jar.pos[1], TOP + g.LID_Z0 + g.LID_H])
    grip_ok = False
    if grip:
        grip_ok = all(grip[k][1] <= g.OPEN_GAP and seg_dist(grip[k][0] + [0, 0, 0], axis_lo, axis_hi) >= g.GRIP_CLEAR
                      and float(np.linalg.norm(grip[k][0] - np.asarray(lid))) >= g.GRIP_CLEAR for k in grip)
    stay = float(math.hypot(jar.pos[0] - g.JAR_X, jar.pos[1] - g.JAR_Y))
    rest = (memo["unlatched"] and dist >= g.LID_CLEAR and g.LID_REST_Z[0] <= lid_h <= g.LID_REST_Z[1] and lid_speed < g.STILL_V)
    perfect = (rest and tilt <= g.JAR_TILT_OK and stay <= g.JAR_STAY and grip_ok and jar_speed < g.STILL_V
               and float(np.linalg.norm(jar.ang_vel)) < 0.10)
    info.update(rest=rest, grip_ok=grip_ok, stay=stay)
    if perfect:
        return 1.0, True, info
    if memo["unlatched"] and dist >= g.LID_OFF:
        return 0.75, False, info
    if memo["unlatched"]:
        return 0.5, False, info
    if memo["popped"]:
        return 0.25, False, info
    return 0.0, False, info


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    S = _STATE
    S["step"] += 1
    st = object_states(environment, ["pickle_jar", "lever_tool"])
    jar, tool = st["pickle_jar"], st["lever_tool"]
    j = jar.joints
    R = jar.rotation()
    lid = np.asarray(jar.pos) + R @ np.array([j["lid_x"], j["lid_y"], g.LID_CZ + j["lid_z"]])
    prev = S["lid_prev"]
    S["lid_speed"] = 0.0 if prev is None else float(np.linalg.norm(lid - prev)) * 20.0
    S["lid_prev"] = lid
    S["lid"] = tuple(float(v) for v in lid)
    S["spin"] = float(j["lid_spin"])
    S["max_spin"] = max(S["max_spin"], abs(S["spin"]))
    if not S["popped"]:
        p = prise(jar, tool, float(j["lid_z"]))
        S["pop_run"] = S["pop_run"] + 1 if p["ok"] else 0
        S["prise"] = p
        if S["pop_run"] >= g.POP_STEPS:
            S["popped"] = True                                   # the seal is broken: the lid turns against its friction
            _write(environment, "lid_spin", limits=(-g.TIGHT_STOP, g.SPIN_LIMIT), effort=g.SEAL_TORQUE)
    elif not S["unlatched"] and loosened(S["spin"]):
        S["unlatched"] = True                                    # the lugs clear the thread: a loose lid
        _write(environment, "lid_spin", limits=(-60.0, 60.0), effort=0.002)
        for n in ("lid_x", "lid_y"):
            _write(environment, n, limits=(-0.45, 0.45))
        _write(environment, "lid_z", limits=(0.0, 0.25))               # it stays over the neck until it is clear of the jar
    if S["unlatched"] and not S["clear"] and math.hypot(j["lid_x"], j["lid_y"]) >= g.BODY_R + g.LID_R:
        S["clear"] = True                                        # beside the jar: it can come down to the table
        _write(environment, "lid_z", limits=(-g.LID_CZ - 0.002, 0.25))
    level, perfect, info = classify(jar, lid, S["lid_speed"], S, grip_state())
    S["held"] = S["held"] + 1 if perfect else 0
    S["ok"] = S["held"] >= g.HOLD_STEPS
    if perfect and not S["ok"]:
        level = 0.75                                             # never 1.0 before the full 1 s hold
    S["level"], S["info"] = level, info
    tr = os.environ.get("TASK_TRACE")
    if tr:
        rec = {k: (round(float(v), 5) if not isinstance(v, (bool, str)) else v) for k, v in info.items()}
        rec.update(step=S["step"], level=level, jar=[round(float(x), 4) for x in jar.pos], lid=[round(float(x), 4) for x in lid],
                   tool=[round(float(x), 4) for x in tool.pos], prise=S.get("prise"), pop_run=S["pop_run"],
                   jyaw=round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2))
        gs = grip_state()
        if gs:
            rec["grip"] = {k: [[round(float(x), 4) for x in v[0]], round(v[1], 4)] for k, v in gs.items()}
        with open(tr, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def success(objects) -> bool:
    """The seal was broken with the lifter and the lid unlatched (turned 80 degrees relative to the jar in the loosening direction, counter-clockwise seen from above; tightening never unlatches); the lid now lies still on the table
    at least 30 mm from the jar's footprint; the jar stands upright (tilt <= 10 degrees), within 60 mm of its start and still; both
    grippers are open and at least 70 mm from the jar axis and from the lid centre. True after 1 s (20 steps) of all of this. The
    jar never tilted past 30 degrees after the lid came free (brine at the rim) and never left the table."""
    return bool(_STATE["ok"])


def score(objects) -> float:
    """0 if the jar ever tilts past 30 degrees once the lid is unlatched, or leaves the table. Otherwise from the end state:
    0.25 seal broken (the lifter's blade tip under the lid edge, raising it 1 mm, for 5 steps; latched); 0.5 lid unlatched
    (turned 80 degrees relative to the jar counter-clockwise seen from above, the loosening way) and still on the jar (latched); 0.75 lid off the jar (centre 50 mm from the axis) but held,
    in the air or not at rest; 1.0 as in success() (1 s hold)."""
    return float(_STATE["level"])
