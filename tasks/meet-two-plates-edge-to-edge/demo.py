"""Scripted full-score demo for the two-plates edge-to-edge task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 800] [--no-record]

Each arm pinches the handle block of its plate (fingers open along y), lifts the plate 40 mm, then both bring the inner
edges together in closed loop on the MEASURED plate poses (position, height and yaw of each plate), settling on a 15 mm
seam centred on y = 0, and hold. DEMO_VARIANT: settle, one_arm, no_level (left plate held 25 mm high, no correction), overlap (target
gap -20 mm), yaw15 (right plate turned 15 deg), gap10 / gap22 / mid_alt (other valid seams).
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



HZ = 0.032                 # pinch height above the plate origin
LIFT = 0.040               # lift above the rest height
TARGET_GAP = 0.015


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
    st0 = ctx.states()
    zb0, zo0 = float(st0["blue"].pos[2]), float(st0["orange"].pos[2])
    x_c = float(st0["blue"].pos[0])
    hc = task.HC
    gap = {"overlap": -0.020, "gap10": 0.010, "gap22": 0.022}.get(variant, TARGET_GAP)
    mid = 0.010 if variant == "mid_alt" else 0.0
    yb_t = mid + gap / 2 + task.PL_Y / 2          # target plate centres (y)
    yo_t = mid - gap / 2 - task.PL_Y / 2
    z_t = max(zb0, zo0) + LIFT
    yaw_l = reachable_yaw(L, np.array([x_c, yb_t + hc, z_t + HZ]))
    yaw_r = reachable_yaw(Rr, np.array([x_c, yo_t - hc, z_t + HZ]))
    if os.environ.get("YAW_L"):
        yaw_l = float(os.environ["YAW_L"]) * math.pi / 2
    if os.environ.get("YAW_R"):
        yaw_r = float(os.environ["YAW_R"]) * math.pi / 2
    gl, gr = grip_rotation(yaw_l), grip_rotation(yaw_r)
    print("YAW", yaw_l, yaw_r, flush=True)

    def pl(n):
        return ctx.states()[n]

    hl = pl("blue").to_world((0.0, hc, HZ))
    hr = pl("orange").to_world((0.0, -hc, HZ))

    def grab(arm, h, g, name):
        yield from arm.move(h + [0, 0, 0.09], g, 2.5, grip=1.0)
        yield from arm.move(h, g, 1.5)
        pre = pl(name).pos.copy()
        yield from arm.set_grip(0.0, 1.0)
        yield from wait(0.4)
        print("CLOSE", name, "shift", np.round(pl(name).pos - pre, 4).tolist(), flush=True)

    if variant == "one_arm":
        yield from par(grab(L, hl, gl, "blue"), wait(1.0))
        L.feedback(True)
        yield from L.move(L.pos + [0, 0, LIFT], gl, 2.0)
        yield from wait(5.0)
        return
    yield from par(grab(L, hl, gl, "blue"), grab(Rr, hr, gr, "orange"))
    L.feedback(True)
    Rr.feedback(True)
    high = 0.025 if variant == "no_level" else 0.0       # no_level: left plate raised 25 mm more, never corrected
    yield from par(L.move(L.pos + [0, 0, LIFT + high], gl, 2.0), Rr.move(Rr.pos + [0, 0, LIFT], gr, 2.0))
    yield from wait(0.8)
    b, o = pl("blue"), pl("orange")
    offL = L.pos - b.pos                      # hand-to-plate offsets as grasped (plates carried)
    offR = Rr.pos - o.pos
    b0, o0 = b.pos.copy(), o.pos.copy()
    closed = variant != "no_level"
    yaw15 = math.radians(15) if variant == "yaw15" else 0.0
    a_l = a_r = 0.0                            # hand yaw corrections (rad)
    trL = np.zeros(3)
    trR = np.zeros(3)
    n = int(7.0 * CONTROL_HZ)
    for k in range(1, n + 1 + int(5.0 * CONTROL_HZ)):
        s = ik.smoothstep(min(k, n) / n)
        zb_t, zo_t = ((b0[2] + o0[2]) / 2,) * 2 if closed else (b0[2], o0[2])
        tb = np.array([x_c, b0[1] + s * (yb_t - b0[1]), zb_t])
        to = np.array([x_c, o0[1] + s * (yo_t - o0[1]), zo_t])
        if closed or k <= n:
            b, o = pl("blue"), pl("orange")
            if closed:
                trL = np.clip(trL + 0.06 * (tb - b.pos), -0.03, 0.03)
                trR = np.clip(trR + 0.06 * (to - o.pos), -0.03, 0.03)
                a_l = float(np.clip(a_l - 0.15 * math.radians(task.yaw_deg(b)), -0.4, 0.4))
                a_r = float(np.clip(a_r - 0.15 * (math.radians(task.yaw_deg(o)) - yaw15), -0.4, 0.4))
        L.aim(tb + offL + trL, _rz(a_l) @ gl)
        Rr.aim(to + offR + trR, _rz(a_r + yaw15 * s) @ gr)
        if k % 20 == 0 and os.environ.get("DEMO_DEBUG"):
            m = task.seam(ctx.states())
            print("SEAM", {kk: round(v, 4) for kk, v in m.items()}, flush=True)
        yield
    print("FINAL", {kk: round(v, 4) for kk, v in task.seam(ctx.states()).items()}, flush=True)
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
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
