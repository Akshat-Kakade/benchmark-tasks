"""Scripted full-score demo for the blind-plate isolation task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1000] [--no-record]

The left hand pinches the yellow lug and pulls the lower flange down until the measured gap is 32 mm. The right hand
pinches the orange knob, lifts the plate off the tray, and slides it into the gap closed loop on the measured plate
pose, centres it on the green mark and lets go. The left hand then eases the flange up until the joint closes on the
plate (measured gap <= 13 mm), lets go, and both hands retreat.

DEMO_VARIANT: settle (hold still), no_left (plate pushed at a closed joint), no_right (flange held open, no plate),
no_release (stop with the flange still held open), offcentre (plate centred 10 mm off).
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
GAP_OPEN = 0.032             # m, flange pulled down to this measured gap
LUG_Y = 0.125                # pinch point on the lug
LUG_Z = TOP + 0.110          # lug centre height, flange closed
KNOB_Z = 0.020               # knob centre above the plate centre
KNOB_DY = -0.14
SPEED = 0.04                 # m/s, plate feed speed
JAW_X = float(os.environ.get("JAW_X", 0.0))      # TCP offset along the finger axis (x)
HOVER_L = TOP + 0.19
HOVER_R = TOP + 0.20


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


def waypoints(task):
    """Key tool points (world) of both arms, for the offline IK reach check on the VM."""
    p0 = task.PLATE_START
    kn = lambda y, z: (task.JX, y + KNOB_DY, z + KNOB_Z)
    right = [(task.JX, p0[1] + KNOB_DY, HOVER_R), kn(p0[1], p0[2]), kn(p0[1], task.UPPER_UNDER - 0.016),
             kn(-0.11, task.UPPER_UNDER - 0.016), kn(0.0, task.UPPER_UNDER - 0.016), kn(0.0, TOP + 0.20 - KNOB_Z)]
    left = [(task.JX, LUG_Y, HOVER_L), (task.JX, LUG_Y, LUG_Z), (task.JX, LUG_Y, LUG_Z - GAP_OPEN),
            (task.JX, LUG_Y, LUG_Z + 0.08)]
    return right, left


def _gap(ctx):
    return -float(ctx.states()["joint"].sliders["flange"])


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    variant = os.environ.get("DEMO_VARIANT", "")
    R, L = ctx.right, ctx.left
    right_pts, left_pts = waypoints(task)
    flip_r, _ = _grips(R.base, right_pts)
    flip_l, _ = _grips(L.base, left_pts)
    rot_r, rot_l = grip_rotation(flip_r), grip_rotation(flip_l)
    if variant == "settle":
        yield from wait(8.0)
        return
    yield from wait(0.7)
    st0 = ctx.states()
    pl0 = st0["plate"].pos.copy()
    jx, jy = task.JX, task.JY
    off_l = np.zeros(3)

    # 1. Left hand: pinch the lug, pull the flange down to the target gap (closed loop on the measured gap).
    lug = np.array([jx + JAW_X, LUG_Y, LUG_Z])
    lcmd = lug.copy()
    if variant != "no_left":
        yield from L.move(lug + [0, 0, HOVER_L - LUG_Z], rot_l, 2.5, grip=1.0)
        yield from L.move(lug, rot_l, 1.5)
        for _ in range(30):
            lcmd += 0.15 * (lug - L.tcp(ctx.obs, slice(0, 6)))
            L.aim(lcmd, rot_l)
            yield
        yield from L.set_grip(0.0, 0.8)
        zc = lcmd[2]
        for _ in range(int(8 * CONTROL_HZ)):
            g = _gap(ctx)
            if g >= GAP_OPEN - 0.001:
                break
            zc -= min(0.02, 1.2 * (GAP_OPEN + 0.004 - g)) / CONTROL_HZ
            L.aim(np.array([lcmd[0], lcmd[1], zc]), rot_l)
            yield
        lhold = np.array([lcmd[0], lcmd[1], zc])
        for _ in range(15):
            yield
    else:
        lhold = None

    def left_hold():
        """Hold the flange at the target gap on the measured gap."""
        if lhold is None:
            return
        g = _gap(ctx)
        lhold[2] -= 0.05 * (GAP_OPEN - g)
        L.aim(lhold, rot_l)

    if variant == "no_right":
        for _ in range(int(4.0 * CONTROL_HZ)):
            left_hold()
            yield
        return

    # 2. Right hand: pinch the knob, lift the plate to the gap height.
    p = ctx.states()["plate"].pos
    knob = np.array([p[0] + JAW_X, p[1] + KNOB_DY, p[2] + KNOB_Z])
    for q, d in ((knob + [0, 0, HOVER_R - knob[2]], 2.5), (knob, 1.5)):
        for _ in R.move(q, rot_r, d, grip=1.0):
            left_hold()
            yield
    rcmd = knob.copy()
    for _ in range(30):
        rcmd += 0.15 * (knob - R.tcp(ctx.obs, slice(7, 13)))
        R.aim(rcmd, rot_r)
        left_hold()
        yield
    for _ in R.set_grip(0.0, 0.8):
        left_hold()
        yield
    for _ in range(6):
        left_hold()
        yield

    # 3. Closed loop on the measured plate pose: lift, then feed along +y to the green mark.
    ty = jy + (0.010 if variant == "offcentre" else 0.0)
    z_mid = task.UPPER_UNDER - GAP_OPEN / 2 + 0.002
    des = np.array([p[0], p[1], p[2]])
    corr = np.zeros(3)
    stage, still, t_end = "lift", 0, None
    for k in range(int(40 * CONTROL_HZ)):
        pm = ctx.states()["plate"].pos
        if stage == "lift":
            des[2] = min(des[2] + 0.04 / CONTROL_HZ, z_mid)
            if abs(des[2] - z_mid) < 1e-6 and abs(pm[2] - z_mid) < 0.003:
                stage = "feed"
        elif stage == "feed":
            des[1] = min(des[1] + SPEED / CONTROL_HZ, ty)
            des[0] += 0.05 * (jx - des[0])
            if abs(des[1] - ty) < 1e-6 and abs(pm[1] - ty) < 0.003:
                still += 1
            if still >= 10:
                break
        corr = np.clip(corr + 0.10 * (des - pm), -0.04, 0.04) if k % 1 == 0 else corr
        R.aim(des + corr + [JAW_X, KNOB_DY, KNOB_Z], rot_r)
        left_hold()
        yield
    if variant == "no_left":
        return
    # 4. Let go of the plate, lift the hand clear and back out.
    for _ in R.set_grip(1.0, 0.8):
        left_hold()
        yield
    for _ in R.move(R.pos + [0, 0, 0.06], rot_r, 1.0):
        left_hold()
        yield
    for _ in R.move(R.pos + [0, -0.12, 0.0], rot_r, 1.5):
        left_hold()
        yield
    if variant == "no_release":
        for _ in range(int(4.0 * CONTROL_HZ)):
            left_hold()
            yield
        return
    # 5. Ease the flange up until it closes on the plate, then let go.
    zc = lhold[2]
    for _ in range(int(14 * CONTROL_HZ)):
        g = _gap(ctx)
        if g <= 0.0135:
            break
        zc += 0.012 / CONTROL_HZ
        L.aim(np.array([lhold[0], lhold[1], zc]), rot_l)
        yield
    for _ in range(10):
        yield
    for _ in L.set_grip(1.0, 0.8):
        yield
    for _ in L.move(L.pos + [0, 0, 0.08], rot_l, 1.0):
        yield
    for _ in R.move(R.pos + [0, -0.05, 0.05], rot_r, 1.0):
        yield
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


def run(task_path: str, out_dir: str, max_steps: int = 1000, record: bool = True, cam_width: int = 320,
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
                pl, jt = s["plate"], s["joint"]
                tl, tr = ctx.left.tcp(ctx.obs, slice(0, 6)), ctx.right.tcp(ctx.obs, slice(7, 13))
                msg += (f" L {tl[0]:.3f},{tl[1]:.3f},{tl[2]:.3f} R {tr[0]:.3f},{tr[1]:.3f},{tr[2]:.3f}"
                        f" plate {pl.pos[0]:.3f},{pl.pos[1]:.3f},{pl.pos[2]:.4f} tilt {task._tilt_deg(pl):.1f}"
                        f" gap {task._gap_mm(jt):.1f} lvl {task._STATE['level']} tripped {task._STATE['tripped']}")
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
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
