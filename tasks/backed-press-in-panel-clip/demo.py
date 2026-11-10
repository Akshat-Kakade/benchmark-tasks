"""Scripted full-score demo for the backed-press task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1800] [--no-record]

The left arm pinches the paddle's post, lifts it and slides it in under the panel until its fork surrounds the clip axis
(closed loop on the measured paddle pose), then raises it until it touches the panel's underside. The right arm brings a
closed fist down on the clip's cap and pushes until the measured clip travel reaches 10 mm, while the left hand
raises the paddle as needed to hold the panel's measured sag near zero. After a short hold the right hand lifts away and
the left withdraws the paddle (lowered a little, then out along +y) and lets go.
DEMO_VARIANT: right_only (no paddle), left_only (paddle only), both_no_press (both arms, right does not press),
release_first (the right hand lifts away before the paddle is withdrawn: a second valid order is not needed).
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
POST = (0.0, 0.28, 0.0)      # paddle post centre in the paddle frame
POST_Z = TOP + 0.030         # TCP height pinching the post at rest
HOVER_Z = TOP + 0.16
BACK_Z = TOP + 0.0561 # paddle origin height that just touches the panel underside
FIST_DZ = 0.017              # closed fingers reach this far below the TCP


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

    def aim(self, pos, rot):
        self._solve(pos, rot)


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def _grips(base, probe_points):
    """The flip (0 or pi) of the straight-down grip, with fingers opening along world x, that IK reaches best."""
    from yam_arena import ik
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation

    worst = {}
    for flip in (0.0, math.pi):
        rot = _rz(math.pi / 2) @ grip_rotation(flip)
        errs = [ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(rot), np.array(READY, float), base_pos=base,
                         iters=200)[1] for p in probe_points]
        worst[flip] = max(errs)
    return min(worst, key=worst.get), worst


def waypoints(task):
    """Key tool points (world) of both arms, for the offline IK reach check on the VM."""
    ax, ay = task.CX, task.HOLE_Y
    cap = task.PANEL_REST_Z + 0.026 + FIST_DZ
    right = [(ax, ay, HOVER_Z), (ax, ay, cap + 0.01), (ax, ay, cap - 0.010)]
    left = [(ax, 0.18 + 0.28, POST_Z), (ax, 0.18 + 0.28, BACK_Z + 0.02), (ax, ay + 0.28, BACK_Z), (ax, ay + 0.28 + 0.15, BACK_Z)]
    return right, left


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    variant = os.environ.get("DEMO_VARIANT", "")
    R, L = ctx.right, ctx.left
    if variant == "hold":
        yield from wait(8.0)
        return
    right_pts, left_pts = waypoints(task)
    flip_r, _ = _grips(R.base, right_pts)
    flip_l, _ = _grips(L.base, left_pts)
    rot_r = _rz(math.pi / 2) @ grip_rotation(flip_r)
    rot_l = _rz(math.pi / 2) @ grip_rotation(flip_l)
    st = ctx.states()
    pad0, panel0 = st["paddle"], st["panel"]
    axis = np.asarray(panel0.to_world((0.0, task.HOLE_Y, 0.0)), float)
    use_left = variant != "right_only"
    use_right = variant != "left_only"
    press = variant not in ("both_no_press", "left_only")

    # 1. Left arm: grasp the post, lift, slide the fork in under the panel, closed loop on the paddle's measured pose.
    off = np.zeros(3)                       # TCP minus paddle origin (hand-held offset), measured after the grasp
    if use_left:
        post = np.asarray(pad0.to_world(POST), float)
        yield from L.move(np.array([post[0], post[1], HOVER_Z]), rot_l, 2.5, grip=1.0)
        yield from L.move(np.array([post[0], post[1], POST_Z]), rot_l, 1.5)
        yield from L.set_grip(0.0, 0.8)
        yield from wait(0.3)
        p = ctx.states()["paddle"]
        off = L.pos - np.asarray(p.pos, float)
        cmd = L.pos.copy()
        # lift to a height where the fork clears the table, 5 cm below the panel underside, then slide in y
        for tz, ty, tx, n in ((TOP + 0.05, None, None, 40), (None, axis[1], axis[0], 400)):
            for _ in range(n):
                p = np.asarray(ctx.states()["paddle"].pos, float)
                if ty is None:
                    cmd[2] += 0.5 * (tz - p[2]) * 0.2
                else:
                    step = np.clip(axis[:2] - p[:2], -0.0015, 0.0015) * 1.0
                    cmd[:2] += step
                    cmd[2] += 0.2 * (TOP + 0.05 - p[2])
                    if np.linalg.norm(axis[:2] - p[:2]) < 0.002:
                        break
                L.aim(cmd, rot_l)
                yield
        # raise until the fork touches the panel's underside, then 1.5 mm more
        for _ in range(200):
            s = ctx.states()
            p = np.asarray(s["paddle"].pos, float)
            sag, pressed, near, gap, tilt = task.measures(s)
            cmd[:2] += 0.2 * (axis[:2] - p[:2])
            if gap >= -0.3 or sag <= -1.0:
                break
            cmd[2] += 0.0004
            L.aim(cmd, rot_l)
            yield
        preload = cmd[2] + 0.0015
        cmd[2] = preload
        L.aim(cmd, rot_l)
        yield from wait(0.5)

    # 2. Right arm: fist on the clip cap, press down until the measured clip travel reaches 10 mm.
    seat_steps = 0
    if use_right:
        s = ctx.states()
        panel = s["panel"]
        ax = np.asarray(panel.to_world((0.0, task.HOLE_Y, 0.0)), float)
        yield from R.move(np.array([ax[0], ax[1], HOVER_Z]), rot_r, 2.5, grip=0.0)
        top = float(panel.pos[2]) + 0.026 + FIST_DZ
        yield from R.move(np.array([ax[0], ax[1], top + 0.012]), rot_r, 1.5)
        rc = R.pos.copy()
        for _ in range(int(14.0 * CONTROL_HZ)):
            s = ctx.states()
            sag, pressed, near, gap, tilt = task.measures(s)
            if press:
                if pressed >= 9.8:
                    seat_steps += 1 if sag <= 2.5 else 0
                else:
                    rc[2] -= 0.006 / CONTROL_HZ
                if seat_steps >= 30:
                    break
            else:
                yield from wait(2.0)
                break
            if use_left:                      # keep the panel near its rest height with the backing paddle
                cmd[2] += float(np.clip(0.3 * (sag + 1.0), -1.5, 1.5)) / 1000
                L.aim(cmd, rot_l)
            R.aim(rc, rot_r)
            yield
        yield from R.move(R.pos + [0.0, 0.0, 0.08], rot_r, 1.0)
    yield from wait(0.5)

    # 3. Left arm: lower a little, withdraw the paddle out along +y, let go, retreat.
    if use_left:
        cmd[2] -= 0.008
        for _ in range(int(0.8 * CONTROL_HZ)):
            L.aim(cmd, rot_l)
            yield
        for _ in range(int(8.0 * CONTROL_HZ)):
            p = np.asarray(ctx.states()["paddle"].pos, float)
            if p[1] - axis[1] >= 0.22:
                break
            cmd[1] += 0.050 / CONTROL_HZ
            L.aim(cmd, rot_l)
            yield
        yield from L.set_grip(1.0, 0.8)
        yield from L.move(L.pos + [0.0, 0.0, 0.08], rot_l, 1.0)
    yield from wait(2.0)


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


def run(task_path: str, out_dir: str, max_steps: int = 1800, record: bool = True, cam_width: int = 320,
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
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                s = ctx.states()
                sag, pr, near, gap, tilt = task.measures(s)
                msg += (f" sag {sag:.1f} press {pr:.1f} near {near:.0f} gap {gap:.1f} "
                        f"tripped {task._STATE['tripped']}")
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
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
