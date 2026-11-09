"""Scripted full-score demo for the drive-drawer task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

The right arm pulls the failed drive (red lamp) and puts it in the returns bin; only then does the left arm, waiting in
front of the drawer with a spare, seat it in that slot. When the slot's lamp turns green (rebuild finished), the right
arm pulls the predictive drive (amber) to the bin and the left arm seats the second spare. Only one arm is over the
drawer at a time. The demo knows the hidden state (it reads the lamp state from the task), which makes it a feasibility
proof, not a policy.
DEMO_VARIANT=parallel | simultaneous | amber_first | no_wait | failed_only | settle runs the ablations (see README).
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20


def _rot_axis(axis, angle) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)

    def aim(self, pos, rot):
        """Track a pose this step (for closed-loop moves)."""
        self._solve(pos, rot)


def reachable_yaw(arm, pos, yaw):
    """The grip yaw, or the same parallel grip turned 180 degrees, whichever the arm reaches more exactly. A wrist at
    its joint limit otherwise leaves the fingers several degrees off and they twist light parts as they close."""
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    best = None
    for y in (yaw, yaw + math.pi, yaw - math.pi):
        _, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(y)), arm.q, base_pos=arm.base, iters=200)
        if best is None or pe + 0.05 * re < best[0]:
            best = (pe + 0.05 * re, y)
    return best[1]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def both(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


def until(cond, timeout_s: float = 60.0):
    """Wait (one yield per step) until ``cond()`` is true."""
    for _ in range(round(timeout_s * CONTROL_HZ)):
        if cond():
            return
        yield


# Heights of the tool point (TCP, 13 mm above the pad tips), from the task's geometry.
GRIP_BELOW_TOP = 0.015   # the pads hold the drive's top 28 mm, their tips 22 mm above the slot walls
VARIANT = os.environ.get("DEMO_VARIANT", "full")   # full | parallel | simultaneous | amber_first | no_wait | failed_only
LANE_X = 0.13            # the arms move sideways here, in front of the drawer and its lamp panel
HIGH_Z = 0.93            # TCP height for crossing the bin and tray walls (reachable near each arm's own side only)


def settle():
    """An explicit hold: both arms stay at their start pose, fingers open, for 8 s."""
    yield from wait(8.0)


def script(task, ctx):
    """The demo as a generator: one yield per control step. The right arm takes drives out to the bin, the left arm
    brings spares in; only one arm is ever over the drawer."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    tcp_slot = T.FLOOR_Z + T.DRIVE_H - GRIP_BELOW_TOP + float(os.environ.get("SEAT_DZ", "0.0"))
    tcp_tray = T.TOP + T.DRIVE_H - GRIP_BELOW_TOP
    tcp_bin = T.BIN_FLOOR_Z + 0.004 + T.DRIVE_H - GRIP_BELOW_TOP
    carry = T.WALL_TOP + 0.012 + T.DRIVE_H - GRIP_BELOW_TOP   # carried drive's bottom 12 mm over the slot walls
    flags = {"r_out1": False, "l_out1": False, "r_out2": False, "r_lifted": False}
    rot = {}

    def rot_for(arm, p):
        key = (arm.name, round(p[0], 2), round(p[1], 2))
        if key not in rot:
            rot[key] = grip_rotation(reachable_yaw(arm, np.asarray(p), 0.0))
        return rot[key]

    def go(arm, p, dur, grip=None):
        p = np.asarray(p, float)
        yield from arm.move(p, rot_for(arm, p), dur, grip)

    def pos_of(name):
        return np.asarray(ctx.states()[name].pos, float)

    def pick(arm, name, tcp_z):
        d = pos_of(name)
        yield from go(arm, (d[0], d[1], carry), 2.5, grip=1.0)
        yield from go(arm, (d[0], d[1], tcp_z), 1.5)
        yield from arm.set_grip(0.0, 0.8)
        yield from wait(0.3)
        yield from go(arm, (d[0], d[1], carry), 2.0)            # straight up, slowly

    def to_bin(arm, name, y):
        yield from go(arm, (LANE_X, arm.pos[1], carry), 1.5)
        flags["r_lifted"] = True       # out of the drawer, in front of it
        yield from go(arm, (LANE_X, y, HIGH_Z), 2.0)
        yield from go(arm, (T.BIN_XY[0], y, HIGH_Z), 1.0)
        yield from go(arm, (T.BIN_XY[0], y, tcp_bin), 1.5)
        yield from wait(0.3)
        yield from arm.set_grip(1.0, 1.0)
        yield from wait(0.5)
        yield from go(arm, (T.BIN_XY[0], y, HIGH_Z), 1.0)

    def place(arm, name, k):
        sy = T.SLOT_Y[k]
        yield from go(arm, (LANE_X, arm.pos[1], carry), 1.5)
        yield from go(arm, (LANE_X, sy, carry), 2.0)
        yield from go(arm, (T.DRAWER_X, sy, carry), 1.5)
        yield from wait(0.5)
        # closed loop: put the held drive's measured centre over the slot centre
        for _ in range(2):
            off = pos_of(name)[:2] - arm.pos[:2]
            yield from go(arm, (T.DRAWER_X - off[0], sy - off[1], carry), 0.8)
            yield from wait(0.3)
        x, y = arm.pos[0], arm.pos[1]
        yield from go(arm, (x, y, tcp_slot + 0.03), 1.2)
        yield from go(arm, (x, y, tcp_slot), 1.5)
        yield from wait(0.3)
        yield from arm.set_grip(1.0, 1.0)
        yield from wait(0.5)
        yield from go(arm, (x, y, carry + 0.01), 1.2)
        if VARIANT == "no_wait" and k == 1:   # the arm already at the drawer pulls the amber drive next door at once
            y3 = T.SLOT_Y[2]
            yield from go(arm, (T.DRAWER_X, y3, carry), 1.0)
            yield from go(arm, (T.DRAWER_X, y3, tcp_slot), 1.0)
            yield from arm.set_grip(0.0, 0.8)
            yield from go(arm, (T.DRAWER_X, y3, carry), 1.5)
            yield from wait(3.0)
            return
        yield from go(arm, (LANE_X, y, carry + 0.01), 1.2)

    def park(arm, xy):
        yield from go(arm, (*xy, HIGH_Z), 2.0, grip=1.0)

    rebuilt = lambda k: T.array().status[k] == "rebuilt"  # noqa: E731  (what the slot's lamp shows: green)

    def right():
        first, second = ("drive3", "drive2") if VARIANT == "amber_first" else ("drive2", "drive3")
        yield from pick(R, first, tcp_slot)
        yield from to_bin(R, first, -0.36)
        flags["r_out1"] = True
        if VARIANT == "failed_only":
            yield from park(R, (T.BIN_XY[0], -0.42))
            return
        if VARIANT == "no_wait":   # the left arm does the early pull (see place())
            yield from park(R, (T.BIN_XY[0], -0.42))
            return
        yield from until(lambda: flags["l_out1"])
        yield from until(lambda: rebuilt(1))
        yield from pick(R, second, tcp_slot)
        yield from to_bin(R, second, -0.48)
        flags["r_out2"] = True
        yield from park(R, (T.BIN_XY[0], -0.42))

    def left():
        if VARIANT in ("parallel", "simultaneous"):   # the greedy shortcut: the left arm pulls the amber drive too
            if VARIANT == "parallel":   # as soon as the right arm has the failed drive out of the drawer
                yield from until(lambda: flags["r_lifted"])
            yield from pick(L, "drive3", tcp_slot)
            yield from wait(3.0)
            return
        yield from pick(L, "spare1", tcp_tray)
        yield from go(L, (LANE_X, L.pos[1], carry), 1.5)
        yield from until(lambda: flags["r_out1"] or R.pos[1] < -0.30)
        yield from place(L, "spare1", 1)
        if VARIANT == "failed_only":
            flags["l_out1"] = True
            yield from park(L, (T.TRAY_XY[0][0], 0.41))
            return
        yield from pick(L, "spare2", tcp_tray)
        yield from go(L, (LANE_X, L.pos[1], carry), 1.5)
        flags["l_out1"] = True
        yield from until(lambda: flags["r_out2"])
        yield from place(L, "spare2", 2)
        yield from park(L, (T.TRAY_XY[0][0], 0.41))

    yield from both(right(), left())
    # wait for the last rebuild (the lamp turns green) with both arms parked
    yield from until(lambda: rebuilt(2) or T.array().data_loss, 15.0)
    yield from wait(1.0)



def _yaw_of(p):
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = settle() if os.environ.get("DEMO_VARIANT") == "settle" else script(task, ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 100 == 0:
            arr = task.array()
            msg = (f"step {steps} score {info.get('score')} status {arr.status} loss {arr.data_loss} "
                   f"L {np.round(ctx.left.pos, 3)} R {np.round(ctx.right.pos, 3)}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
