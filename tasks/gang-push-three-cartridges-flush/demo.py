"""Scripted full-score demo for the gang-push cartridges task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 800] [--no-record]

Each arm pinches one handle of the pusher (fingers open along y), then both push the bar forward at 25 mm/s, closed
loop on the MEASURED pusher pose (shared carrier integrator) and the measured cartridge x (stop at the target depth,
trim the bar yaw so the three lanes finish level). DEMO_VARIANT: settle, probe (reach check), one_arm, no_yaw.
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



HANDLE_Z = 0.0775          # pinch height above the pusher origin (handle centre)
HY = 0.170
SPEED = 0.025              # push speed (m/s)
X_TARGET = 0.322           # target mean cartridge centre


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
    st = ctx.states()
    p0 = st["pusher"]
    hl = p0.to_world((0.0, HY, HANDLE_Z))
    hr = p0.to_world((0.0, -HY, HANDLE_Z))
    end_x = p0.pos[0] + 0.115
    yaw_l = reachable_yaw(L, np.array([end_x, hl[1], hl[2]]))
    yaw_r = reachable_yaw(Rr, np.array([end_x, hr[1], hr[2]]))
    if os.environ.get("YAW_L"):
        yaw_l = float(os.environ["YAW_L"]) * math.pi / 2
    if os.environ.get("YAW_R"):
        yaw_r = float(os.environ["YAW_R"]) * math.pi / 2
    gl, gr = grip_rotation(yaw_l), grip_rotation(yaw_r)
    print("YAW", yaw_l, yaw_r, flush=True)

    def pu():
        return ctx.states()["pusher"]

    def grab(arm, h, g, name):
        yield from arm.move(h + [0, 0, 0.08], g, 2.5, grip=1.0)
        yield from arm.move(h, g, 1.5)
        yield from arm.set_grip(0.0, 1.0)
        yield from wait(0.4)

    if variant == "probe":          # IK / reach check along the stroke without touching anything
        for dx in (0.0, 0.05, 0.10, 0.115):
            yield from par(L.move(hl + [dx, 0, 0.0], gl, 1.5, grip=1.0), Rr.move(hr + [dx, 0, 0.0], gr, 1.5, grip=1.0))
            print("PROBE", dx, "worst_mm", round(L.worst_mm, 1), round(Rr.worst_mm, 1), flush=True)
        return
    if variant == "one_arm":
        yield from par(grab(L, hl, gl, "pusher"), wait(1.0))
        L.feedback(True)
    else:
        yield from par(grab(L, hl, gl, "pusher"), grab(Rr, hr, gr, "pusher"))
        L.feedback(True)
        Rr.feedback(True)
    p = pu()
    R0 = p.rotation()[:2, :2]
    offL = R0.T @ (L.pos[:2] - p.pos[:2])          # hand-to-pusher offsets in the pusher frame (as grasped)
    offR = R0.T @ (Rr.pos[:2] - p.pos[:2])
    zL, zR = L.pos[2] - p.pos[2], Rr.pos[2] - p.pos[2]
    yaw0 = math.radians(task.yaw_deg(p))
    ref = np.array([p.pos[0], p.pos[1]])
    y_hold = p.pos[1]
    psi = 0.0                                      # commanded yaw bias (rad) to even the three cartridges
    corr = np.zeros(3)                             # shared carrier integrator: x, y, yaw
    stroke_done = False
    k_end = 0
    tmax = 22.0 if variant == "one_arm" else 16.0
    ablate_yaw = variant == "no_yaw"
    for k in range(int(tmax * CONTROL_HZ)):
        st = ctx.states()
        p = st["pusher"]
        xs = [float(st[n].pos[0]) for n in task.NAMES]
        mean_x = sum(xs) / 3.0
        yawm = math.radians(task.yaw_deg(p))
        if not stroke_done:
            ref[0] += SPEED / CONTROL_HZ
            if mean_x >= X_TARGET - 0.012:
                stroke_done = True
        else:
            ref[0] += float(np.clip(0.6 * (X_TARGET - mean_x), -0.4 * SPEED, 0.5 * SPEED)) / CONTROL_HZ
        if stroke_done and not ablate_yaw:
            psi = float(np.clip(psi + 0.02 * (xs[2] - xs[0]) / 0.21 * 1.0, -0.06, 0.06))   # +y lane (cart2) ahead -> rotate back
        ref[1] = y_hold
        if variant != "one_arm":
            corr[0] = float(np.clip(corr[0] + 0.05 * (ref[0] - p.pos[0]), -0.04, 0.04))
            corr[1] = float(np.clip(corr[1] + 0.05 * (ref[1] - p.pos[1]), -0.03, 0.03))
            corr[2] = float(np.clip(corr[2] + 0.05 * (yaw0 + psi - yawm), -0.3, 0.3))
        a = yaw0 + psi + corr[2]
        Rz = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        tgt = ref + np.array([corr[0], corr[1]])
        pl = tgt + Rz @ offL
        L.aim(np.array([pl[0], pl[1], p.pos[2] + zL]), _rz(a - yaw0) @ gl)
        if variant != "one_arm":
            pr = tgt + Rz @ offR
            Rr.aim(np.array([pr[0], pr[1], p.pos[2] + zR]), _rz(a - yaw0) @ gr)
        if k % 20 == 0 and os.environ.get("DEMO_DEBUG"):
            m = task.measure(ctx.states())
            print("M", {kk: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for kk, v in m.items()}, flush=True)
        if variant != "one_arm" and stroke_done:
            k_end += 1
            if k_end > 6 * CONTROL_HZ:
                break
        yield
    m = task.measure(ctx.states())
    print("FINAL", {kk: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for kk, v in m.items()}, flush=True)
    yield from wait(2.0)


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


def run(task_path: str, out_dir: str, max_steps: int = 800, record: bool = True, cam_width: int = 320,
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
                level, perfect, inf = task.classify(st)
                msg += f" | level {level} x {np.round(inf['x'], 4).tolist()} y {np.round(inf['yerr'], 4).tolist()} p_yaw {inf['p_yaw']:.1f}"
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
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
