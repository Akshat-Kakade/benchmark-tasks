"""Scripted full-score demo for the slip-sheet task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2000] [--no-record]

The right arm brings a closed fist down on the block's +y face (the side the sheet is pulled toward) and presses
until the measured block pose says it is in contact, then a few mm more. The left arm pinches the green tab across
its 40 mm width and pulls the sheet in +y at 40 mm/s until the measured sheet is clear of the block, re-pushing the
brace if the block creeps. Then the left hand lets go, both hands retreat upward and hold.
Writes one MP4 per camera, setup.png and result.json to the output folder.

DEMO_VARIANT: no_brace (right arm idle), no_pull (brace only), pinch (right hand pinches the block's x sides instead
of the fist brace: the second valid solution), fast (no brace, pull at 150 mm/s).
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
PULL_SPEED = 0.040           # m/s
OVERSHOOT = 0.006            # brace pushed this far past first contact
CLEAR_MARGIN = 0.03          # keep pulling this far past "clear"
BRACE_Z = TOP + 0.004 + 0.045   # TCP height for the fist on the block's +y face (pads 32 mm above the table)
TAB_Z = TOP + 0.030          # TCP height pinching the tab (pad bottoms 17 mm above the table)
HOVER_Z = TOP + 0.16


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
    sx, sy = task.SHEET_XY
    bx, by = task.BLOCK_XY
    half = task.BLOCK_SIZE / 2
    tab_y0 = sy + 0.15
    right = [(bx, by + half + 0.045, z) for z in (HOVER_Z, BRACE_Z)] + [(bx, by, BRACE_Z), (bx, by, HOVER_Z)]
    left = [(sx, y, z) for y in (tab_y0, tab_y0 + 0.20) for z in (HOVER_Z, TAB_Z)]
    return right, left


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    variant = os.environ.get("DEMO_VARIANT", "")
    R, L = ctx.right, ctx.left
    right_pts, left_pts = waypoints(task)
    flip_r, _ = _grips(R.base, right_pts)
    flip_l, _ = _grips(L.base, left_pts)
    rot_r = _rz(math.pi / 2) @ grip_rotation(flip_r)
    rot_l = _rz(math.pi / 2) @ grip_rotation(flip_l)
    st = ctx.states()
    block0, sheet0 = st["load"], st["sheet"]
    half = task.BLOCK_SIZE / 2
    by0 = float(block0.pos[1])

    # 1. Right arm: fist on the block's +y face.
    brace_y0 = None
    if variant == "pinch":
        yield from R.move(np.array([block0.pos[0], by0, HOVER_Z]), rot_r, 2.5, grip=1.0)
        yield from R.move(np.array([block0.pos[0], by0, BRACE_Z]), rot_r, 1.5)
        yield from R.set_grip(0.0, 1.0)
        yield from wait(0.3)
    elif variant not in ("no_brace", "fast"):
        yield from R.move(np.array([block0.pos[0], by0 + half + 0.045, HOVER_Z]), rot_r, 2.5, grip=0.0)
        yield from R.move(np.array([block0.pos[0], by0 + half + 0.045, BRACE_Z]), rot_r, 1.5)
        y = by0 + half + 0.045
        for _ in range(int(6.0 * CONTROL_HZ)):            # advance 20 mm/s until the block is pushed 1.5 mm
            b = ctx.states()["load"]
            if by0 - float(b.pos[1]) > 0.0015:
                break
            y -= 0.020 / CONTROL_HZ
            R.aim(np.array([block0.pos[0], y, BRACE_Z]), rot_r)
            yield
        brace_y0 = y - OVERSHOOT
        yield from R.move(np.array([block0.pos[0], brace_y0, BRACE_Z]), rot_r, 0.6)

    if variant == "no_pull":
        yield from wait(4.0)
        return

    # 2. Left arm: pinch the tab across its 40 mm width.
    tab = sheet0.to_world((0.0, 0.15, 0.018))
    yield from L.move(np.array([tab[0], tab[1], HOVER_Z]), rot_l, 2.5, grip=1.0)
    yield from L.move(np.array([tab[0], tab[1], TAB_Z]), rot_l, 1.5)
    yield from L.set_grip(0.0, 0.8)
    yield from wait(0.3)

    # 3. Pull +y until the measured sheet is clear of the block, keeping the brace against creep.
    speed = 0.150 if variant == "fast" else PULL_SPEED
    ty = float(tab[1])
    target_extra = None
    for _ in range(int(14.0 * CONTROL_HZ)):
        s = ctx.states()
        gap = task._gap_mm(s["sheet"], s["load"])
        if gap >= task.CLEAR_GAP_MM and target_extra is None:
            target_extra = ty + CLEAR_MARGIN
        if target_extra is not None and ty >= target_extra:
            break
        ty += speed / CONTROL_HZ
        L.aim(np.array([tab[0], ty, TAB_Z]), rot_l)
        if brace_y0 is not None:
            creep = max(0.0, float(s["load"].pos[1]) - by0)
            R.aim(np.array([block0.pos[0], brace_y0 - creep, BRACE_Z]), rot_r)
        yield

    # 4. Let go, retreat up, hold.
    yield from L.set_grip(1.0, 0.8)
    yield from L.move(L.pos + [0.0, 0.0, 0.08], rot_l, 1.0)
    if variant != "no_brace" and variant != "fast":
        if variant == "pinch":
            yield from R.set_grip(1.0, 0.8)
        yield from R.move(R.pos + [0.0, 0.0, 0.08], rot_r, 1.0)
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
                dy, shift, tilt, lift, gap = task.measures(s, task._STATE["start"])
                msg += (f" sheet_dy {dy:.1f} block_shift {shift:.1f} tilt {tilt:.1f} lift {lift:.1f} gap {gap:.1f} "
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
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
