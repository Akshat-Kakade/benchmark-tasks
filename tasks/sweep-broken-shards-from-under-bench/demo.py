"""Open-loop scripted full-score demo for the squeegee sweep task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 800] [--no-record]

Each arm pinches one handle block of the squeegee head (fingers open along x), both advance 0.215 m together at
30 mm/s from fixed waypoints (task constants + each arm's own joint_pos, which the feedback integrates into the command),
pull back 25 mm, open and lift away. The demo never reads an object pose.
DEMO_VARIANT: settle, probe, left_only, right_only, half_stroke, three_pieces_only, no_release, poke.
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




def reachable_yaw(arm, pos, cands=(math.pi / 2, -math.pi / 2)):
    """+-pi/2: the straight-down grip turn (fingers open along the world x axis) that IK reaches best at ``pos``."""
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



TOP = 0.7475
HEAD_X0 = 0.200            # squeegee head centre x at start (task.py HEAD_X0)
HY = 0.230                 # handle block y offsets (task.HANDLE_Y)
PINCH_Z = TOP + 0.0005 + 0.040   # pinch height on the 60 mm tall handle blocks
SPEED = 0.030              # m/s
STROKE = 0.215             # advance of the head (task: both handle blocks must advance >= 0.160)
BACK = 0.025               # pull back before opening so the blade is clear of the pieces


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":
        yield from wait(8.0)
        return
    for _ in range(12):
        yield
    stroke = {"half_stroke": 0.10}.get(variant, STROKE)
    dy = 0.14 if variant == "three_pieces_only" else 0.0
    hl = np.array([HEAD_X0, HY, PINCH_Z])
    hr = np.array([HEAD_X0, -HY, PINCH_Z])
    end_x = HEAD_X0 + stroke
    yaw_l = reachable_yaw(L, np.array([end_x, hl[1], hl[2]]))
    yaw_r = reachable_yaw(Rr, np.array([end_x, hr[1], hr[2]]))
    if os.environ.get("YAW_L"):
        yaw_l = float(os.environ["YAW_L"]) * math.pi / 2
    if os.environ.get("YAW_R"):
        yaw_r = float(os.environ["YAW_R"]) * math.pi / 2
    gl, gr = grip_rotation(yaw_l), grip_rotation(yaw_r)
    print("YAW", yaw_l, yaw_r, flush=True)

    def grab(arm, h, g):
        yield from arm.move(h + [0, 0, 0.08], g, 2.5, grip=1.0)
        yield from arm.move(h, g, 1.5)
        yield from arm.set_grip(0.0, 1.0)
        yield from wait(0.4)

    if variant == "probe":          # IK / reach check along the stroke without touching anything
        for dx in (0.0, 0.05, 0.10, 0.15, 0.215):
            yield from par(L.move(hl + [dx, 0, 0.0], gl, 1.5, grip=1.0), Rr.move(hr + [dx, 0, 0.0], gr, 1.5, grip=1.0))
            print("PROBE", dx, "worst_mm", round(L.worst_mm, 1), round(Rr.worst_mm, 1), flush=True)
        return
    if variant == "poke":           # a closed hand goes in at piece height instead of the squeegee
        h = np.array([0.24, 0.035, TOP + 0.015])
        yield from L.move(h + [-0.05, 0, 0.10], gl, 2.5, grip=0.0)
        yield from L.move(h, gl, 2.0)
        yield from L.move(h + [0.20, 0, 0], gl, 7.0)
        return
    arms = [(L, hl, gl), (Rr, hr, gr)]
    if variant == "left_only":
        arms = arms[:1]
    elif variant == "right_only":
        arms = arms[1:]
    yield from par(*[grab(a, h, g) for a, h, g in arms])
    for a, _, _ in arms:
        a.feedback(True)
    if dy:      # three_pieces_only: slide the head sideways first so the blade misses the -y piece
        for k in range(1, 61):
            for a, h, g in arms:
                a.aim(h + [0.0, dy * k / 60, 0.0], g)
            yield
        arms = [(a, h + [0.0, dy, 0.0], g) for a, h, g in arms]
    n = int(stroke / SPEED * CONTROL_HZ)
    for k in range(1, n + 1):
        s = k / n
        for a, h, g in arms:
            a.aim(h + [stroke * s, 0.0, 0.0], g)
        yield
    for a, h, g in arms:          # let the blade come to rest, then pull back
        a.aim(h + [stroke, 0.0, 0.0], g)
    yield from wait(0.5)
    nb = int(BACK / 0.02 * CONTROL_HZ)
    for k in range(1, nb + 1):
        for a, h, g in arms:
            a.aim(h + [stroke - BACK * k / nb, 0.0, 0.0], g)
        yield
    yield from wait(0.5)
    if variant == "no_release":
        yield from wait(4.0)
        return
    for a, _, _ in arms:
        a.feedback(False)
    yield from par(*[a.set_grip(1.0, 1.0) for a, _, _ in arms])
    yield from par(*[a.move(a.pos + [0, 0, 0.12], g, 1.5) for a, _, g in arms])
    yield from wait(2.5)


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
