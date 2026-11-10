"""Scripted full-score demo for the anchor-plate proof test, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1500] [--no-record]

The left hand pinches the orange fin (fingers open along y) and, closed loop on the measured plate pose, keeps the
plate from sliding. The right hand pinches the yellow handle and pulls it along +y with a closed loop on the measured
carriage travel (spring gauge) to 62 mm, holds it, eases it back to rest and lets go; both hands then retreat.
Writes one MP4 per camera, setup.png and result.json to the output folder.

DEMO_VARIANT: no_left (plate unbraced), no_right (brace only), no_hold (no hold in the band), overload (pull to 86 mm),
fist (left hand braces with a closed fist on the post's +x face: the second valid solution).
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
TARGET = 0.062               # m, middle of the band
PULL_SPEED = 0.030           # m/s, cap on the command rate
POST_Z = TOP + 0.040         # TCP height pinching the post (spans 15-55 mm above the table)
FIST_Z = TOP + 0.049
HANDLE_Z = TOP + 0.050       # handle spans 28-68 mm above the table
HOVER_Z = TOP + 0.16
HOLD_STEPS = 45
FIST_OFF = float(os.environ.get("FIST_OFF", 0.034))   # fist centre beyond the fin centre
JAW_OFFSET = float(os.environ.get("JAW_OFFSET", 0.0157))   # the fixed jaw sits this far -y of the TCP


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
    """The flip (0 or pi) of the straight-down grip, with fingers opening along world x, that IK reaches best."""
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


def _pts(task, travel=0.0):
    """Nominal world points: (post, handle at the given travel)."""
    o = np.array([task.PLATE_XY[0], task.PLATE_XY[1], task.ORIGIN_Z])
    return o + [0.0, task.POST_Y, 0.0], o + [0.0, task.CAR_Y0 + travel, 0.0]


def waypoints(task):
    """Key tool points (world) of both arms, for the offline IK reach check on the VM."""
    post, h0 = _pts(task)
    _, h1 = _pts(task, 0.09)
    right = [(h[0], h[1], z) for h in (h0, h1) for z in (HOVER_Z, HANDLE_Z)]
    left = [(post[0], post[1], z) for z in (HOVER_Z, POST_Z)] + [(post[0], post[1] + FIST_OFF, FIST_Z)]
    return right, left


def _handle(plate, task):
    """Measured handle centre (world) from the plate pose and carriage travel."""
    t = -float(plate.sliders["carriage"])
    return plate.to_world(task.L(0.0, task.CAR_Y0 + t, 0.0)), t


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
    yield from wait(0.7)                                         # the plate is turned flat by on_reset and settles
    plate0 = ctx.states()["plate"]
    p0 = plate0.pos.copy()
    post = plate0.to_world(task.L(0.0, task.POST_Y, 0.0))

    # 1. Left arm: pinch the post (or put a closed fist on its +x face).
    off = np.zeros(2)                                           # integrated correction of the hold position
    cmd3 = np.zeros(3)
    hold = None
    if variant != "no_left":
        if variant == "fist":
            hold = np.array([post[0], post[1] + FIST_OFF, FIST_Z])
            cmd3[:] = hold
            yield from L.move(hold + [0, 0, HOVER_Z - FIST_Z], rot_l, 2.5, grip=0.0)
            yield from L.move(hold, rot_l, 1.5)
        else:
            hold = np.array([post[0], post[1] + JAW_OFFSET, POST_Z])
            cmd3[:] = hold
            yield from L.move(hold + [0, 0, HOVER_Z - POST_Z], rot_l, 2.5, grip=1.0)
            yield from L.move(hold, rot_l, 1.5)
            for _ in range(40):                                  # null the steady error of the free arm
                cmd3[:] += 0.15 * (hold - L.tcp(ctx.obs, slice(0, 6)))
                L.aim(cmd3, rot_l)
                yield
            yield from L.set_grip(0.0, 0.8)
        yield from wait(0.3)

    def brace():
        """One step of the closed-loop hold: integrate the measured plate drift into the left command."""
        if variant == "no_left":
            return
        d = ctx.states()["plate"].pos[:2] - p0[:2]
        off[:] = np.clip(off - 0.12 * d, -0.03, 0.03)
        L.aim(np.array([cmd3[0] + off[0], cmd3[1] + off[1], cmd3[2]]), rot_l)

    if variant == "no_right":
        for _ in range(int(4.0 * CONTROL_HZ)):
            brace()
            yield
        return

    # 2. Right arm: pinch the handle.
    h, _ = _handle(ctx.states()["plate"], task)
    grasp = np.array([h[0], h[1], HANDLE_Z])
    for p, d in ((grasp + [0, 0, HOVER_Z - HANDLE_Z], 2.5), (grasp, 1.5)):
        gen = R.move(p, rot_r, d, grip=1.0)
        for _ in gen:
            brace()
            yield
    rcmd = grasp.copy()
    for _ in range(40):                                          # null the steady error of the free arm
        rcmd += 0.15 * (grasp - R.tcp(ctx.obs, slice(7, 13)))
        R.aim(rcmd, rot_r)
        brace()
        yield
    for _ in R.set_grip(0.0, 0.8):
        brace()
        yield
    for _ in range(6):
        brace()
        yield

    # 3. Pull to the target travel, hold, ease back (closed loop on the measured travel).
    target = 0.086 if variant == "overload" else TARGET
    x = float(R.pos[1])                                         # command along the pull axis (+y); R.pos[0], z carry the null
    held = 0
    for k in range(int(30.0 * CONTROL_HZ)):
        pl = ctx.states()["plate"]
        t = -float(pl.sliders["carriage"])
        if variant == "no_hold":
            tgt = 0.0 if t >= task.BAND[0] + 0.004 else 0.09
        elif abs(t - target) < 0.004:
            held += 1
            tgt = target
        else:
            tgt = target
        if held >= HOLD_STEPS or (variant == "no_hold" and tgt == 0.0):
            break
        step = float(np.clip(1.5 * (tgt - t), -PULL_SPEED, PULL_SPEED)) / CONTROL_HZ
        x = min(x + step, grasp[1] + 0.16)
        R.aim(np.array([R.pos[0], x, R.pos[2]]), rot_r)
        brace()
        yield
    for _ in range(int(8.0 * CONTROL_HZ)):                       # ease back to rest
        t = -float(ctx.states()["plate"].sliders["carriage"])
        if t <= 0.006:
            break
        x -= 0.040 / CONTROL_HZ
        R.aim(np.array([R.pos[0], x, R.pos[2]]), rot_r)
        brace()
        yield

    # 4. Let go, retreat up, hold.
    for _ in R.set_grip(1.0, 0.8):
        brace()
        yield
    for _ in R.move(R.pos + [0.0, 0.0, 0.08], rot_r, 1.0):
        brace()
        yield
    if variant != "no_left":
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
                pl = s["plate"]
                sh = 1000 * float(np.linalg.norm(pl.pos[:2] - task._STATE["start"][:2]))
                yaw = math.degrees(math.atan2(pl.rotation()[0, 2], -pl.rotation()[1, 2]))
                tl, tr = ctx.left.tcp(ctx.obs, slice(0, 6)), ctx.right.tcp(ctx.obs, slice(7, 13))
                msg += f" L {tl[0]:.3f},{tl[1]:.3f},{tl[2]:.3f} g{ctx.left.grip:.2f} cmdL {ctx.left.pos[0]:.3f},{ctx.left.pos[1]:.3f},{ctx.left.pos[2]:.3f} R {tr[0]:.3f},{tr[1]:.3f},{tr[2]:.3f}"
                msg += (f" d {1000 * (pl.pos[0] - task._STATE['start'][0]):.1f},{1000 * (pl.pos[1] - task._STATE['start'][1]):.1f} yaw {yaw:.1f} travel {-1000 * pl.sliders['carriage']:.1f} shift {sh:.1f} tilt {task._tilt_deg(pl):.1f} "
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
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
