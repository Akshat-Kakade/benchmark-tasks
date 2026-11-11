"""On the station table, ahead of the arms: a grey inspection rocker. A beam (34 cm, along the table's width) tips
about a pivot at its middle. At the beam's left end sit a dark ballast block and a blue handle; near its right end a
round turntable carries an orange counterweight (an arm with a block and an orange handle). The beam rests with its
right end down on the red stop, the counterweight turned out toward the right end. A dark post stands ahead of the
turntable, in the counterweight's path: the weight cannot swing past it unless the beam is tipped so that the
turntable rises. A green stop sits under the beam's left end, and a yellow mark on the beam, between the turntable
and the pivot, shows where the weight must end up.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 2.0

INSTRUCTION = ("Tip the rocker beam over so its left end rests on the green stop; while it is tipped, turn the "
               "orange counterweight round to the yellow mark. Let go of both handles and leave the rocker resting "
               "on the green stop.")

ROCKER = UsdBody("rocker", usd="assets/rocker.usda", pos=(g.XC, 0.0, g.TOP), size=(0.10, 0.40, 0.10),
                 articulation=True, joint_pos=(("beam_pivot", g.STOP), ("weight_turn", 0.0)),
                 joint_drive=(("beam_pivot", 0.08, 0.0), ("weight_turn", 0.01, 0.3)), friction=(0.8, 0.7))
OBJECTS = [ROCKER]

# -- scoring thresholds ---------------------------------------------------------------------------------------------
BEAM_TOL = math.radians(5.0)      # beam within this of the green-stop angle (-STOP)
WEIGHT_TOL = math.radians(8.0)    # weight turned in to the mark: within this of pi
CROSSED = math.radians(100.0)     # the weight has passed the post once its angle exceeds this
CLEAR_MM = 8.0                    # the weight clears the post: its bottom is this far above the post top
PEN_MM = 6.0                      # post penetration (gate): the weight's bottom this far below the post top in front
POST_WINDOW = (math.radians(55.0), math.radians(125.0))
HOLD_R = 0.045                    # a gripper holds a handle when its tool point is this near the handle centre
HOLD_OPEN = 0.065                 # ... and its fingers are less than this far apart
REST_SPEED = math.radians(8.0)    # rad/s: the beam counts as at rest below this
TOUCH_R = 0.06                    # a gripper this near the beam or the weight block is still touching the rocker

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(cleared=False, crossed=False, why=[], theta=g.STOP, psi=0.0, held={"left": False, "right": False},
                  opening={}, tcp={}, pose_ok=False, released=True, rest=False, peak=0.0, last_theta=None)


_reset_state()


def _flag(why: str) -> None:
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def _arms(environment):
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    from scenekit import arm_joints

    out = {}
    for side in ("left", "right"):
        q = arm_joints(environment, side, ARM_JOINTS)
        tcp, _ = ik.tcp_pose(q, station.BASES[side])
        out[side] = (np.asarray(tcp, float), float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum()))
    return out


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["rocker"])["rocker"]
    theta = float(st.joints.get("beam_pivot", g.STOP))
    psi = float(st.joints.get("weight_turn", 0.0))
    arms = _arms(environment)

    gap = g.block_bottom(theta) - g.POST_TOP
    if gap * 1000 >= CLEAR_MM:
        _STATE["cleared"] = True
    if psi >= CROSSED and _STATE["cleared"]:
        _STATE["crossed"] = True
    if POST_WINDOW[0] <= psi <= POST_WINDOW[1] and gap * 1000 < -PEN_MM:
        _flag("weight pushed through the post")

    handles = {"left": g.left_stem(theta), "right": g.weight_stem(theta, psi)}
    touch_pts = [g.beam_to_world(theta, (0, y, 0.0)) for y in (-0.17, -0.1, 0.0, 0.1, 0.17)]
    touch_pts.append(g.block_centre(theta, psi))
    held = {s: False for s in ("left", "right")}
    released = True
    for side, (tcp, opening) in arms.items():
        for hs, hp in handles.items():
            if np.linalg.norm(tcp - hp) < HOLD_R and opening < HOLD_OPEN:
                held[side] = True
        near = min([float(np.linalg.norm(tcp - hp)) for hp in handles.values()]
                   + [float(np.linalg.norm(tcp - p)) for p in touch_pts])
        if near < TOUCH_R:
            released = False
    last = _STATE["last_theta"]
    speed = 0.0 if last is None else abs(theta - last) * 20.0
    _STATE.update(theta=theta, psi=psi, held=held, released=released, last_theta=theta,
                  opening={s: a[1] for s, a in arms.items()}, tcp={s: a[0] for s, a in arms.items()},
                  rest=speed < REST_SPEED,
                  pose_ok=bool(abs(psi - g.PSI_TARGET) <= WEIGHT_TOL and abs(theta + g.STOP) <= BEAM_TOL),
                  gap_mm=gap * 1000)


def violation() -> bool:
    return bool(_STATE["why"])


def success(objects) -> bool:
    """The counterweight turned in to within 8 degrees of the yellow mark, the beam within 5 degrees of the
    green-stop angle and at rest on it, no gripper holding or touching the rocker, no violation; held 2 s."""
    return bool(not violation() and _STATE["pose_ok"] and _STATE["rest"] and _STATE["released"])


def score(objects) -> float:
    """0 after any violation (the weight forced through the post). Otherwise
    0.25 once the beam has been tipped far enough for the weight to clear the post (8 mm above its top), 0.5 once the
    weight has then been swung past the post (more than 100 degrees), 0.75 while the beam rests on the green stop
    with the weight turned in to the mark but a gripper still holds or touches the rocker (or it is still moving);
    1.0 comes from success()."""
    if violation():
        return 0.0
    if _STATE["pose_ok"]:
        return 0.75
    return 0.5 if _STATE["crossed"] else 0.25 if _STATE["cleared"] else 0.0
