"""Scripted full-score demo for the lever-and-chock task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2000] [--no-record]

The right arm presses the long end of the bar down (closed loop on the measured bar angle) until the block has risen
about 11 mm and holds it there. The left arm grips the chock's bridge across its width, slides the prongs under the
raised block (closed loop on the measured chock pose), lets go and backs off. The right arm then slowly lets the bar
come up until the block rests on the chock, and both arms withdraw.

DEMO_VARIANT: "" full demo; "no_chock" press and release only (expect <= 0.25); "no_lever" slide the chock without
pressing (expect 0); "palm" press with open fingers instead of closed ones (must also reach 1.0).
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
PRESS_DEG = float(os.environ.get("PRESS_DEG", 9.5))
PRESS_BAND = (PRESS_DEG - 1.0, PRESS_DEG + 2.0)
SPEED = 0.0006             # m per control step of the closed-loop press (12 mm/s)
INSERT_SPEED = 0.05        # m/s chock slide
PAD_TO_TCP = 0.013


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


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


def _pick_yaw(arm, targets, candidates, base_rot_fn) -> float:
    """The candidate grip yaw with the smallest worst IK error over every planned target."""
    from yam_arena import ik
    from yam_arena.constants import READY

    worst = {}
    for yaw in candidates:
        errs = []
        for p in targets:
            _, pe, _ = ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(base_rot_fn(yaw)),
                                np.array(READY, float), base_pos=arm.base, iters=200)
            errs.append(pe)
        worst[yaw] = max(errs)
    return min(worst, key=worst.get)


def press_point(T, lever):
    """Top surface of the bar at LEVER_ARM_PRESS from the pivot, world frame, from the measured bar pose."""
    d = np.array([0.0, -math.cos(T.THETA), math.sin(T.THETA)])
    n = np.array([0.0, math.sin(T.THETA), math.cos(T.THETA)])
    return lever.to_world(T.LEVER_ARM_PRESS * d + n * T.BAR_T)


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    R, L = ctx.right, ctx.left
    s0 = ctx.states()
    down_r = grip_rotation(0.0)

    # chock grip: fingers open along x (pinch the bridge's ends); pick the yaw that IK reaches best
    def chock_rot(yaw):
        return grip_rotation(yaw)

    c0 = s0["chock"]
    grip_pt = c0.to_world((0, T.BRIDGE_Y, 0.0185 + 0.0))
    yaw = _pick_yaw(L, [grip_pt + [0, 0, 0.10], grip_pt, grip_pt + [0, -0.13, 0]],
                    (math.pi / 2, -math.pi / 2), chock_rot)
    rot_l = chock_rot(yaw)

    press_closed = 0.0 if variant != "palm" else 1.0

    # 1. Right arm over the long end of the bar, then press in small steps until the bar reaches PRESS_DEG.
    def above(lever, h):
        p = press_point(T, lever)
        return np.array([p[0], p[1], p[2] + PAD_TO_TCP + h])

    if variant != "no_lever":
        yield from R.move(above(s0["lever"], 0.10), down_r, 2.5, grip=press_closed)
        yield from R.move(above(ctx.states()["lever"], 0.008), down_r, 1.5)
        z_cmd = R.pos[2]
        for _ in range(int(8.0 * CONTROL_HZ)):
            lev = ctx.states()["lever"]
            if T.lever_deg(lev) >= PRESS_DEG:
                break
            z_cmd -= SPEED
            p = press_point(T, lev)
            R.aim(np.array([p[0], p[1], z_cmd]), down_r)
            yield
    hold = R.pos.copy()

    def hold_press():
        """Keep the bar inside the band for one step (closed loop on the measured angle)."""
        nonlocal hold
        a = T.lever_deg(ctx.states()["lever"])
        if a < PRESS_BAND[0]:
            hold = hold - [0, 0, 0.5 * SPEED]
        elif a > PRESS_BAND[1]:
            hold = hold + [0, 0, 0.5 * SPEED]
        R.aim(hold, down_r)

    if variant == "no_chock":
        for _ in range(int(2.0 * CONTROL_HZ)):
            hold_press()
            yield
        yield from R.move(R.pos + [0, 0, 0.06], down_r, 1.5)
        return

    # 2. Left arm grips the chock's bridge across its width.
    def grip_at(c):
        return c.to_world((0, T.BRIDGE_Y, 0.0185))

    yield from L.move(grip_at(ctx.states()["chock"]) + [0, 0, 0.10], rot_l, 3.0, grip=1.0)
    for _ in range(int(0.5 * CONTROL_HZ)):
        hold_press()
        yield
    yield from L.move(grip_at(ctx.states()["chock"]), rot_l, 1.2)
    yield from L.set_grip(0.0, 1.0)

    # 3. Slide the prongs under the raised block (closed loop on the measured chock pose).
    y_goal = None
    for _ in range(int(8.0 * CONTROL_HZ)):
        st = ctx.states()
        blk, ck = st["block"], st["chock"]
        y_goal = blk.pos[1] + T.CHOCK_FINAL_DY
        if ck.pos[1] <= y_goal + 0.006:
            break
        tgt = L.pos + [-INSERT_SPEED / CONTROL_HZ, 0, 0]
        tgt[1] = L.pos[1] - INSERT_SPEED / CONTROL_HZ
        tgt[0] = T.X
        L.aim(tgt, rot_l)
        hold_press()
        yield
    for _ in range(int(0.6 * CONTROL_HZ)):
        hold_press()
        yield
    yield from L.set_grip(1.0, 0.8)
    yield from L.move(L.pos + [0, 0.06, 0.0], rot_l, 1.0)
    yield from L.move(L.pos + [0, 0.0, 0.08], rot_l, 1.0)

    # 4. Let the bar come up slowly until the block rests on the chock, then withdraw.
    for _ in range(int(8.0 * CONTROL_HZ)):
        hold = hold + [0, 0, SPEED]
        R.aim(hold, down_r)
        yield
        if T.lever_deg(ctx.states()["lever"]) < 0.8:
            break
    yield from R.move(R.pos + [0, 0, 0.05], down_r, 1.5)
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


def run(task_path: str, out_dir: str, max_steps: int = 2000, record: bool = True, cam_width: int = 320,
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
                st = ctx.states()
                m = task._metrics(st)
                msg += (f" lift_mm {1000 * m['lift']:.1f} lever_deg {m['lever']:.1f} block_tilt {m['block_tilt']:.1f} "
                        f"chock_travel_mm {1000 * m['chock_travel']:.0f} under {m['under']} "
                        f"chock_y {st['chock'].pos[1]:.3f} blk_y {st['block'].pos[1]:.3f} L {np.round(ctx.left.pos,3).tolist()} g {ctx.left.grip:.2f} stages {task._STATE['lifted']} {task._STATE['cleared']} "
                        f"{task._STATE['inserted']} gate {task._STATE['gate']}")
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
            imageio.imwrite(out / "setup.png", frames[top][min(20, len(frames[top]) - 1)])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
