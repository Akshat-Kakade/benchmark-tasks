"""Scripted full-score demo for the telescoping extraction hood, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1500] [--no-record]

The right hand pinches the hood mouth's grip post and, closed loop on the measured mouth position, holds it over the white mark.
The left hand pinches the foot's grip post and slides the foot toward the robot at 20 mm/s (closed loop on the measured foot
position and y), so the arm extends. Both hands keep holding at the end (releasing lets the spring pull the mouth back).

DEMO_VARIANT: settle (hold only), no_mouth (left only, mouth unheld), no_foot (mouth held, foot never moves), whole (both hands move
together: the whole assembly slides, no extension), stop50 (foot stops after 50 mm), probe (open-loop arm lag probe along the path).
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
TOP = 0.7475
SPEED = float(os.environ.get("SPEED", 0.020))           # m/s, foot travel rate
HOVER_Z = TOP + 0.16          # left hand hover height
HOVER_R = TOP + 0.12          # right hand: the IK ceiling at x 0.40, y -0.11 (0.15 leaves a 15 mm error)
JAW_L = float(os.environ.get("JAW_L", 0.0157))          # y offset of the TCP from the post centre, left hand
JAW_R = float(os.environ.get("JAW_R", 0.0))             # right hand


def _rz(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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

    def tcp(self, obs, sl):
        """Measured TCP position (world) from the observed joint angles."""
        from yam_arena import ik

        q = np.asarray(obs.state["joint_pos"], float)[sl][:6]
        return ik.tcp_pose(q, self.base)[0]

    def aim(self, pos, rot):
        self._solve(pos, rot)


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def _grips(base, probe_points):
    """The flip (0 or pi) of the straight-down grip (fingers open along world y) that IK reaches best over the whole path."""
    from yam_arena import ik
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation

    worst = {}
    for flip in (0.0, math.pi):
        rot = grip_rotation(flip)
        errs = [ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(rot), np.array(READY, float), base_pos=base,
                         iters=200)[1] for p in probe_points]
        worst[flip] = max(errs)
    return min(worst, key=worst.get), worst


def waypoints(task):
    """Key tool points (world) of both arms along the whole motion, for the offline IK reach check on the VM."""
    zl, zr = task.FOOT_PINCH_Z, task.MOUTH_PINCH_Z
    left, right = [], []
    for k in range(0, 9):                                    # foot post along the travel 0 .. 80 mm
        x = task.FOOT_X0 - 0.010 * k
        left.append((x, task.FOOT_POST_Y + JAW_L, zl))
    left.append((task.FOOT_X0, task.FOOT_POST_Y + JAW_L, HOVER_Z))
    left.append((task.FOOT_X0 - task.TARGET_D, task.FOOT_POST_Y + JAW_L, HOVER_Z))
    right = [(task.MOUTH_X, task.MOUTH_POST_Y + JAW_R, z) for z in (zr, HOVER_R)]
    right += [(task.MOUTH_X - task.TARGET_D, task.MOUTH_POST_Y + JAW_R, zr)]     # whole-assembly ablation end point
    return right, left


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    variant = os.environ.get("DEMO_VARIANT", "")
    R, L = ctx.right, ctx.left
    right_pts, left_pts = waypoints(task)
    flip_r, _ = _grips(R.base, right_pts)
    flip_l, _ = _grips(L.base, left_pts)
    rot_r = grip_rotation(flip_r)
    rot_l = grip_rotation(flip_l)
    if variant == "settle":
        yield from wait(8.0)
        return
    yield from wait(1.0)                                         # the foot is turned flat by on_reset and settles
    st0 = ctx.states()["foot"]
    p0 = st0.pos.copy()
    m0 = task.mouth_pos(st0)
    zl, zr = task.FOOT_PINCH_Z, task.MOUTH_PINCH_Z
    hold_r = np.array([m0[0], m0[1] + task.MOUTH_POST_Y + JAW_R, zr])
    hold_l = np.array([p0[0], p0[1] + task.FOOT_POST_Y + JAW_L, zl])
    offr = np.zeros(2)                                           # integrated correction of the mouth hold
    offl = np.zeros(2)                                           # integrated correction of the foot position
    rcmd = hold_r.copy()
    lcmd = hold_l.copy()

    def mouth_err():
        s = ctx.states()["foot"]
        return task.mouth_pos(s)[:2] - m0[:2]

    def brace():
        """One step of the closed-loop mouth hold: integrate the measured mouth drift into the right command."""
        if variant == "no_mouth":
            return
        offr[:] = np.clip(offr - 0.10 * mouth_err(), -0.03, 0.03)
        R.aim(np.array([rcmd[0] + offr[0], rcmd[1] + offr[1], rcmd[2]]), rot_r)

    if variant == "probe":
        for k in range(0, 9):
            tgt = np.array([task.FOOT_X0 - 0.010 * k, 0.11 + JAW_L, zl])
            yield from L.move(tgt + [0, 0, 0.0], rot_l, 1.5, grip=1.0)
            yield from wait(1.0)
            tl = L.tcp(ctx.obs, slice(0, 6))
            print("probe left", k, "cmd", np.round(tgt, 4), "meas", np.round(tl, 4), "err_mm", np.round(1000 * (tl - tgt), 1), flush=True)
        return

    # 1. Right arm: pinch the mouth post.
    if variant != "no_mouth":
        yield from R.move(hold_r + [0, 0, HOVER_R - zr], rot_r, 2.5, grip=1.0)
        yield from R.move(hold_r, rot_r, 1.5)
        for _ in range(40):                                      # null the steady error of the free arm
            rcmd += 0.15 * (hold_r - R.tcp(ctx.obs, slice(7, 13)))
            R.aim(rcmd, rot_r)
            yield
        yield from R.set_grip(0.0, 0.8)
        yield from wait(0.3)

    # 2. Left arm: pinch the foot post.
    if True:
        for p, d in ((hold_l + [0, 0, HOVER_Z - zl], 2.5), (hold_l, 1.5)):
            gen = L.move(p, rot_l, d, grip=1.0)
            for _ in gen:
                brace()
                yield
        for _ in range(40):
            lcmd += 0.15 * (hold_l - L.tcp(ctx.obs, slice(0, 6)))
            L.aim(lcmd, rot_l)
            brace()
            yield
        for _ in L.set_grip(0.0, 0.8):
            brace()
            yield
        for _ in range(10):
            brace()
            yield

    # 3. Slide the foot toward the robot (closed loop on the measured foot position), then hold.
    lcmd0 = lcmd.copy()
    goal = 0.050 if variant == "stop50" else task.TARGET_D
    if variant == "no_foot":
        goal = 0.0
    ramp = 0.0
    held = 0
    for k in range(int(40.0 * CONTROL_HZ)):
        ramp = min(goal, ramp + SPEED / CONTROL_HZ)
        s = ctx.states()["foot"]
        e = np.array([(p0[0] - ramp) - s.pos[0], p0[1] - s.pos[1]])
        offl[:] = np.clip(offl + 0.10 * e, -0.05, 0.05)
        L.aim(np.array([lcmd0[0] - ramp + offl[0], lcmd0[1] + offl[1], lcmd0[2]]), rot_l)
        if variant == "whole":
            rc = rcmd.copy()
            rc[0] -= ramp
            R.aim(np.array([rc[0], rc[1], rc[2]]), rot_r)
        else:
            brace()
        if ramp >= goal:
            held += 1
        if held >= int(4.0 * CONTROL_HZ):
            break
        yield
    yield from wait(1.0)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    obs = None

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1500, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
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
    gen = script(task, ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    ctx.obs = obs
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
        ctx.obs = res.observation
        snap(res.observation)
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % int(os.environ.get("DEBUG_EVERY", 10)) == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                s = ctx.states()
                pl = s["foot"]
                msg += f" foot {pl.pos[0]:.4f},{pl.pos[1]:.4f},{pl.pos[2]:.4f} slider {pl.sliders['arm']:.4f} q {np.round(pl.quat_xyzw, 3)}"
                if task._STATE["start"] is not None:
                    m = task._metrics(pl)
                    mp = task.mouth_pos(pl)
                    msg += (f" D {m['D']:.1f} yerr {m['yerr']:.1f} d {m['d']:.1f} ext {m['ext']:.1f} tilt {m['tilt']:.1f} yaw {m['yaw']:.1f} "
                            f"mouth {mp[0]:.3f},{mp[1]:.3f},{mp[2]:.3f} tripped {task._STATE['tripped']}")
                tl, tr = ctx.left.tcp(ctx.obs, slice(0, 6)), ctx.right.tcp(ctx.obs, slice(7, 13))
                msg += (f" L {tl[0]:.3f},{tl[1]:.3f},{tl[2]:.3f} cmdL {ctx.left.pos[0]:.3f},{ctx.left.pos[1]:.3f},{ctx.left.pos[2]:.3f}"
                        f" R {tr[0]:.3f},{tr[1]:.3f},{tr[2]:.3f} cmdR {ctx.right.pos[0]:.3f},{ctx.right.pos[1]:.3f},{ctx.right.pos[2]:.3f}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][min(25, len(frames[top]) - 1)])   # lamps/parts settled
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
