"""Scripted full-score demo for the lower-frame-over-post task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1100] [--no-record]

Each arm pinches a handle stem (fingers open along x, so the jaws resist pitch), both lift the frame 30 mm, carry its opening over the post in closed
loop on the MEASURED frame pose (position, yaw, roll), then lower it at 16 mm/s while still correcting, and hold on the
outline. DEMO_VARIANT: settle, one_arm, no_center, stop_above, off_alt / off_alt2 / yaw5 (other valid solutions).
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
        self.read = None          # callable -> actual joint positions (6) or None
        self.fb = False           # integrate measured TCP error into the command
        self.corr = np.zeros(3)
        self.gain = 0.12

    def feedback(self, on: bool):
        self.fb = on
        if not on:
            self.corr = np.zeros(3)

    def _track(self):
        if self.fb and self.read is not None:
            from yam_arena import ik

            qa = self.read()
            if qa is not None:
                pa, _ = ik.tcp_pose(np.asarray(qa, float), self.base)
                self.corr = np.clip(self.corr + self.gain * (self.pos - pa), -0.04, 0.04)

    def _solve(self, pos, rot):
        from yam_arena import ik

        self.pos, self.rot = np.asarray(pos, float), rot
        self._track()
        q, pe, _ = ik.solve(self.pos + self.corr, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q = q
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




def reachable_yaw(arm, pos, cands=(0.0, math.pi)):
    """0 or pi: the straight-down grip turn (fingers open along the world y axis) that IK reaches best at ``pos``."""
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    best = None
    for y in cands:
        _, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(y)), arm.q, base_pos=arm.base, iters=200)
        if best is None or pe + 0.05 * re < best[0]:
            best = (pe + 0.05 * re, y)
    return best[1]




def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])



LIFT_Z = 0.030             # carry height of the frame underside above its rest on the pedestals
DESCENT = 0.016            # m/s (<= 20 mm/s) during the last part of the lowering


def script(task, ctx):
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":
        yield from wait(8.0)
        return
    for _ in range(12):
        yield
    f0 = ctx.states()["frame"]
    z0 = float(f0.pos[2])
    hy, pz = task.HANDLE_Y, task.PINCH_Z
    off_xy = {"off_alt": (0.008, -0.008), "off_alt2": (-0.008, 0.008)}.get(variant, (0.0, 0.0))
    yaw_t = math.radians(5.0) if variant == "yaw5" else 0.0
    px_t, py_t = task.POST_X + off_xy[0], off_xy[1]
    if variant == "no_center":                     # carries the frame 70 mm too far forward and never corrects it
        px_t += 0.07
    seat_z = task.SEAT_Z + 0.004
    yaw_l = reachable_yaw(L, np.array([px_t, hy, z0 + LIFT_Z + pz]), (math.pi / 2, -math.pi / 2))
    yaw_r = reachable_yaw(Rr, np.array([px_t, -hy, z0 + LIFT_Z + pz]), (math.pi / 2, -math.pi / 2))
    gl, gr = grip_rotation(yaw_l), grip_rotation(yaw_r)
    print("YAW", yaw_l, yaw_r, flush=True)

    def fr():
        return ctx.states()["frame"]

    hl = f0.to_world((0.0, hy, pz))
    hr = f0.to_world((0.0, -hy, pz))

    def grab(arm, h, g, name):
        yield from arm.move(h + [0, 0, 0.09], g, 2.5, grip=1.0)
        yield from arm.move(h, g, 1.5)
        pre = fr().pos.copy()
        yield from arm.set_grip(0.0, 1.0)
        yield from wait(0.4)
        print("CLOSE", name, "shift", np.round(fr().pos - pre, 4).tolist(), flush=True)

    if variant == "one_arm":
        yield from par(grab(L, hl, gl, "left"), wait(1.0))
        L.feedback(True)
        yield from L.move(L.pos + [0, 0, LIFT_Z], gl, 2.0)
        yield from L.move(L.pos + [0.12, 0, 0], gl, 3.0)
        yield from wait(4.0)
        return
    yield from par(grab(L, hl, gl, "left"), grab(Rr, hr, gr, "right"))
    L.feedback(True)
    Rr.feedback(True)
    yield from par(L.move(L.pos + [0, 0, LIFT_Z], gl, 2.0), Rr.move(Rr.pos + [0, 0, LIFT_Z], gr, 2.0))
    yield from wait(0.8)
    f = fr()
    locL, locR = f.to_local(L.pos), f.to_local(Rr.pos)        # hand positions in the frame's own frame, as grasped
    p0 = f.pos.copy()
    closed = variant != "no_center"
    ang = 0.0                                                  # yaw correction (rad)
    tr = np.zeros(3)                                           # common-mode position correction
    roll_c = 0.0                                               # differential height correction
    n_move = int(5.0 * CONTROL_HZ)
    hold_above = int(4.0 * CONTROL_HZ)

    def step_ctl(target):
        nonlocal ang, tr, roll_c
        f = fr()
        if closed:
            tr = np.clip(tr + 0.06 * (target - f.pos), -0.04, 0.04)
            yaw_m = math.atan2(f.rotation()[1, 0], f.rotation()[0, 0])
            ang = float(np.clip(ang - 0.15 * (yaw_m - yaw_t) + 0.0, -0.4, 0.4))
            roll = math.atan2(f.rotation()[2, 1], f.rotation()[2, 2])
            roll_c = float(np.clip(roll_c - 0.10 * roll, -0.05, 0.05))
        Rz = _rz(ang + yaw_t * 0.0)
        base = np.asarray(target) + tr
        L.aim(base + Rz @ locL + [0, 0, roll_c], _rz(ang) @ gl)
        Rr.aim(base + Rz @ locR - [0, 0, roll_c], _rz(ang) @ gr)

    k = 0
    # 1) carry the opening over the post at constant height
    for k in range(1, n_move + 1):
        s = ik.smoothstep(k / n_move)
        step_ctl(np.array([p0[0] + s * (px_t - p0[0]), p0[1] + s * (py_t - p0[1]), p0[2]]))
        yield
    for _ in range(int(1.5 * CONTROL_HZ)):                     # settle over the post, correcting
        step_ctl(np.array([px_t, py_t, p0[2]]))
        yield
    if variant == "stop_above":
        for _ in range(hold_above):
            step_ctl(np.array([px_t, py_t, p0[2]]))
            yield
        print("FINAL", {kk: round(v, 4) for kk, v in task.fit(ctx.states()).items()}, flush=True)
        return
    # 2) lower in closed loop (<= 16 mm/s), then hold at the seat
    z = p0[2]
    n_low = 0
    while z > seat_z + 1e-4 or n_low < 1:
        z = max(seat_z, z - DESCENT / CONTROL_HZ)
        step_ctl(np.array([px_t, py_t, z]))
        n_low += 1
        if os.environ.get("DEMO_DEBUG") and n_low % 20 == 0:
            print("FIT", {kk: round(v, 4) for kk, v in task.fit(ctx.states()).items()}, flush=True)
        yield
    for _ in range(int(3.0 * CONTROL_HZ)):
        step_ctl(np.array([px_t, py_t, seat_z]))
        yield
    print("FINAL", {kk: round(v, 4) for kk, v in task.fit(ctx.states()).items()}, flush=True)
    yield from wait(1.0)


def par(*gens):
    """Run generators in lockstep (one yield per control step) until all are done."""
    live = list(gens)
    while live:
        nxt = []
        for g in live:
            try:
                next(g)
                nxt.append(g)
            except StopIteration:
                pass
        live = nxt
        if live:
            yield


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.last_state = np.zeros(14)
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.left.read = lambda: np.asarray(self.last_state)[0:6] if np.any(self.last_state) else None
        self.right.read = lambda: np.asarray(self.last_state)[7:13] if np.any(self.last_state) else None

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


def run(task_path: str, out_dir: str, max_steps: int = 1100, record: bool = True, cam_width: int = 320,
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
        ctx.last_state = next(iter(res.observation.state.values()))
        snap(res.observation)
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % int(os.environ.get("DEBUG_EVERY", "20")) == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                level, perfect, inf = task.classify(st); inf = {k: float(v) for k, v in inf.items()}
                msg += f" | level {level} " + " ".join(f"{k} {v:.3f}" for k, v in inf.items())
                from yam_arena import ik as _ik
                for nm, arm_, sl in (("L", ctx.left, slice(0, 6)), ("R", ctx.right, slice(7, 13))):
                    pa, _r = _ik.tcp_pose(np.asarray(ctx.last_state)[sl], arm_.base)
                    msg += f"\n   {nm} cmd {np.round(arm_.pos + arm_.corr, 3).tolist()} actual {np.round(pa, 3).tolist()} g {arm_.grip:.2f}"
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
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1100)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
