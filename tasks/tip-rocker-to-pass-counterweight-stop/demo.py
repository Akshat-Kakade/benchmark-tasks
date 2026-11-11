"""Scripted full-score demo for the tip-rocker task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1500] [--no-record]      (DEMO_VARIANT env or --variant)

1. The left arm pinches the blue handle at the beam's left end and presses it down, closed loop on the measured beam
   angle, until the beam rests on the green stop (the turntable rises about 4 cm), and keeps pressing.
2. The right arm pinches the orange handle and swings the counterweight round the turntable, over the post, to the
   yellow mark, closed loop on the measured weight angle, then lets go.
3. The left arm lets go and lifts away. The beam stays on the green stop.

Variants (ablations): settle (both arms still), tip_release (tip, let go before turning the weight), no_tip (turn the
weight without tipping), one_arm (the right arm alone lifts the orange handle and turns the weight), left_only (the
left arm tips, holds, and nobody turns the weight).
Writes one MP4 per camera, setup.png and result.json to the output folder.
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
DT = 1.0 / CONTROL_HZ
VARIANT = os.environ.get("DEMO_VARIANT", "")
TIME_SCALE = float(os.environ.get("DEMO_TIME_SCALE", "1.0"))

# tunables (metres, radians, seconds)
TIP_RATE = math.radians(float(os.environ.get("TIP_RATE", "10")))          # beam tip rate of the command, rad/s
MAX_LEAD = math.radians(float(os.environ.get("MAX_LEAD", "16")))         # command at most this far past the measured angle
PRESS_DEG = float(os.environ.get("PRESS_DEG", "8.0"))                   # press this far past the stop angle
SWING_RATE = math.radians(float(os.environ.get("SWING_RATE", "22")))    # weight swing rate, rad/s
SWING_LEAD = math.radians(15.0)                                         # wait if the weight lags the command by more
SWING_END = math.radians(float(os.environ.get("SWING_END", "183")))     # swing command end (limit is 190)
GRIP_Z = float(os.environ.get("GRIP_Z", "0.0"))                          # tool point above the handle's mid-height
LEFT_YAW = float(os.environ.get("LEFT_YAW", "0.0"))


class Arm:
    """One arm commanded in Cartesian space (IK on the URDF), each move starting where the last command left it."""

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.worst_mm = 0.0

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=80)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, err * 1000)

    def move(self, pos, rot=None, seconds: float = 1.0, grip: float | None = None):
        from yam_arena import ik

        rot = self.rot if rot is None else rot
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(seconds * TIME_SCALE * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self.aim(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, seconds: float = 0.8):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def together(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        if gens:
            yield


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")

    @property
    def S(self):
        return self.task._STATE

    def theta(self) -> float:
        return float(self.S["theta"])

    def psi(self) -> float:
        return float(self.S["psi"])

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def left_rot():
    from yam_arena.demo import grip_rotation

    return grip_rotation(LEFT_YAW)


def right_rot(psi: float):
    from yam_arena.demo import grip_rotation

    return grip_rotation(psi - math.pi / 2)        # fingers open along the weight's direction of travel


def grab_left(ctx, g):
    L = ctx.left
    p = g.left_stem(ctx.theta()) + [0, 0, GRIP_Z]
    yield from L.move(p + [0, 0, 0.07], left_rot(), 2.5, grip=1.0)
    yield from L.move(p, left_rot(), 1.5)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.4)


def tip(ctx, g):
    """Left arm: press the left end down until the beam rests on the green stop, keep pressing. The command runs
    ahead of the measured beam angle at TIP_RATE and never more than MAX_LEAD ahead (the arm is soft)."""
    L = ctx.left
    stop = -g.STOP
    floor = stop - math.radians(PRESS_DEG)
    cmd = ctx.theta()
    for _ in range(round(12 * CONTROL_HZ)):
        cmd = max(cmd - TIP_RATE * DT, floor, ctx.theta() - MAX_LEAD)
        L.aim(g.left_stem(cmd) + [0, 0, GRIP_Z], left_rot())
        yield
        if ctx.theta() <= stop + math.radians(0.7):
            break
    for _ in range(round(0.8 * CONTROL_HZ)):
        L.aim(g.left_stem(floor) + [0, 0, GRIP_Z], left_rot())
        yield


def hold_left(ctx, g):
    L = ctx.left
    while True:
        L.aim(g.left_stem(-g.STOP - math.radians(PRESS_DEG)) + [0, 0, GRIP_Z], left_rot())
        yield


def grab_right(ctx, g, theta=None):
    R = ctx.right
    th = ctx.theta() if theta is None else theta
    p = g.weight_stem(th, 0.0) + [0, 0, GRIP_Z]
    yield from R.move(p + [0, 0, 0.07], right_rot(0.0), 2.5, grip=1.0)
    yield from R.move(p, right_rot(0.0), 1.5)
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.4)


def swing(ctx, g):
    """Right arm: carry the weight round the turntable, closed loop on the measured weight angle."""
    R = ctx.right
    cmd = max(ctx.psi(), 0.0)
    for _ in range(600):
        if ctx.psi() > cmd - SWING_LEAD:
            cmd = min(cmd + SWING_RATE * DT, SWING_END)
        R.aim(g.weight_stem(ctx.theta(), cmd) + [0, 0, GRIP_Z], right_rot(cmd))
        yield
        if ctx.psi() >= math.radians(179.0):
            break
    for _ in range(round(0.5 * CONTROL_HZ)):
        R.aim(g.weight_stem(ctx.theta(), cmd) + [0, 0, GRIP_Z], right_rot(cmd))
        yield


def let_go_right(ctx, g):
    R = ctx.right
    yield from R.set_grip(1.0, 1.0)
    yield from R.move(R.pos + [0, 0, 0.07], None, 1.5)
    yield from R.move(R.pos + [0, -0.05, 0.0], None, 1.5)


def let_go_left(ctx, g):
    L = ctx.left
    yield from L.set_grip(1.0, 1.0)
    yield from L.move(L.pos + [0, 0, 0.07], None, 1.5)
    yield from L.move(L.pos + [0, 0.05, 0.0], None, 1.5)


def script(ctx, g, variant: str):
    if variant == "settle":
        yield from wait(8.0)
        return
    if variant == "one_arm":
        yield from grab_right(ctx, g, theta=g.STOP)
        R = ctx.right
        # lift the right end with the handle until the beam is tipped, then swing
        while ctx.theta() > -g.STOP + math.radians(0.7):
            cmd = ctx.theta() - MAX_LEAD
            R.aim(g.weight_stem(max(cmd, -g.STOP - math.radians(PRESS_DEG)), ctx.psi()), right_rot(ctx.psi()))
            yield
        yield from swing(ctx, g)
        yield from let_go_right(ctx, g)
        yield from wait(3.0)
        return
    if variant == "no_tip":
        yield from grab_right(ctx, g)
        yield from swing(ctx, g)
        yield from let_go_right(ctx, g)
        yield from wait(3.0)
        return
    yield from grab_left(ctx, g)
    yield from tip(ctx, g)
    if variant == "tip_release":
        yield from let_go_left(ctx, g)
        yield from wait(6.0)
        return
    if variant == "left_only":
        yield from hold_left_for(ctx, g, 6.0)
        yield from let_go_left(ctx, g)
        yield from wait(3.0)
        return
    right = chain(grab_right(ctx, g), swing(ctx, g), let_go_right(ctx, g))
    yield from left_driver(ctx, g, right)
    yield from let_go_left(ctx, g)
    yield from wait(3.0)


def chain(*gens):
    for gen in gens:
        yield from gen


def hold_left_for(ctx, g, seconds):
    L = ctx.left
    for _ in range(round(seconds * CONTROL_HZ)):
        L.aim(g.left_stem(-g.STOP - math.radians(PRESS_DEG)) + [0, 0, GRIP_Z], left_rot())
        yield


def left_driver(ctx, g, right):
    """Run the right arm's chain while the left arm keeps pressing."""
    L = ctx.left
    done = {"v": False}

    def right_wrapped():
        yield from right
        done["v"] = True

    def left_press():
        while not done["v"]:
            L.aim(g.left_stem(-g.STOP - math.radians(PRESS_DEG)) + [0, 0, GRIP_Z], left_rot())
            yield

    return together(right_wrapped(), left_press())


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1500, record: bool = True, variant: str = "",
        cam_width: int = 320, cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    sys.modules.setdefault("task", task)
    import geometry as g

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    ctx = Ctx(emb, task)
    gen = script(ctx, g, variant)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    trace = []
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        S = task._STATE
        if steps % 10 == 0:
            tl, tr = S["tcp"].get("left", np.zeros(3)), S["tcp"].get("right", np.zeros(3))
            trace.append([steps, round(math.degrees(S["theta"]), 2), round(math.degrees(S["psi"]), 1),
                          round(S.get("gap_mm", 0.0), 1), [round(float(v), 4) for v in tl],
                          [round(float(v), 4) for v in ctx.left.pos], [round(float(v), 4) for v in tr],
                          [round(float(v), 4) for v in ctx.right.pos], round(S["opening"].get("left", 0), 4)])
        if steps % 50 == 0:
            print(f"step {steps} score {info.get('score')} theta {math.degrees(S['theta']):.2f} psi "
                  f"{math.degrees(S['psi']):.1f} gap {S.get('gap_mm', 0):.1f} held {S['held']} released "
                  f"{S['released']} pose_ok {S['pose_ok']} rest {S['rest']} why {S['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "theta_deg": round(math.degrees(S["theta"]), 2), "psi_deg": round(math.degrees(S["psi"]), 1),
              "cleared": bool(S["cleared"]), "crossed": bool(S["crossed"]), "variant": variant,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top and len(frames[top]) > 25:
            imageio.imwrite(out / "setup.png", frames[top][25])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "trace.json").write_text(json.dumps(trace))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, variant=a.variant, cam_width=a.cam_width,
        cam_height=a.cam_height)
