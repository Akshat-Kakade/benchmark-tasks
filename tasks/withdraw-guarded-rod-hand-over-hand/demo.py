"""Scripted full-score demo for the guarded sample rod: the hands take turns pulling the rod uphill and exchange grips.

    python demo.py task.py demo_out [--max-steps 1500] [--no-record]

Right hand pinches the rod in the blue opening and pulls it 58 mm, the left hand takes the rod in the orange opening while
the right still holds it, the right lets go and the left pulls to 118 mm, the right takes over again and pulls to 178 mm.
Every pull is closed loop on the measured rod travel; every exchange has a stationary overlap. Writes MP4s, setup.png and
result.json to the output folder.

DEMO_VARIANT: right_only, left_only (one hand, full reach), stop1, stop2 (stop after 1 / 2 strokes, still holding),
release (pull one stroke and open the hand), settle (hold plan, nothing touched), probe (hover and pinch the rod, no pull).
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
SA, CA = math.sin(math.radians(7.5)), math.cos(math.radians(7.5))
AX = np.array([0.0, CA, SA])                 # uphill direction
PULL_SPEED = 0.025                           # m/s, cap on the command rate
HOVER_Z = TOP + 0.15
S_STEPS = (0.058, 0.118, 0.178)              # rod travel at the end of each stroke (m)
V_R1, V_L, V_R3 = 0.145, 0.470, 0.030        # grip points measured from the rod's lower tip (m)
V_L_ONLY = 0.520
HOLD_S = 1.2
ONE_ARM_S = 0.085                            # one hand: farthest the hand can follow the rod before the shroud stops it
JAW_OFFSET = float(os.environ.get("JAW_OFFSET", 0.0157))
YAW_R = float(os.environ.get("YAW_R", math.pi / 2))
YAW_L = float(os.environ.get("YAW_L", math.pi / 2))
ZOFF = float(os.environ.get("ZOFF", 0.0))


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
    """The yaw (+-pi/2: fingers opening along world x) of the straight-down grip that IK reaches best."""
    from yam_arena import ik
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation

    worst = {}
    for yaw in (math.pi / 2, -math.pi / 2):
        rot = grip_rotation(yaw)
        errs = [ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(rot), np.array(READY, float), base_pos=base,
                         iters=200)[1] for p in probe_points]
        worst[yaw] = max(errs)
    return min(worst, key=worst.get), worst


def grip_point(task, v, s):
    """Nominal TCP point (world) at rod coordinate v (from the tip) with the rod pulled s metres."""
    return task.axis_point(task.ROD_U0 + s + v) + [0.0, 0.0, ZOFF]


def waypoints(task):
    """Key tool points (world) for both arms, for the offline IK reach check on the VM."""
    right = []
    for v, s0, s1 in ((V_R1, 0.0, S_STEPS[0]), (V_R3, S_STEPS[1], S_STEPS[2]), (V_R1, 0.0, 0.10)):
        for s in (s0, s1):
            p = grip_point(task, v, s)
            right += [p, p + [0, 0, HOVER_Z - p[2]]]
    left = []
    for v, s0, s1 in ((V_L, S_STEPS[0], S_STEPS[1]), (V_L_ONLY, 0.0, 0.10)):
        for s in (s0, s1):
            p = grip_point(task, v, s)
            left += [p, p + [0, 0, HOVER_Z - p[2]]]
    return right, left


def together(*gens):
    """Step several generators in lock-step until all are done."""
    gens = list(gens)
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        yield


class Hold:
    """A hand pinching the rod: its command is base + d * uphill, with d driven by the measured rod travel."""

    def __init__(self, arm, rot, base, ctx):
        self.arm, self.rot, self.base, self.ctx, self.d = arm, rot, np.asarray(base, float).copy(), ctx, 0.0
        self.anchor = None

    def aim(self):
        self.arm.aim(self.base + self.d * AX, self.rot)

    def servo(self, s_ref):
        if self.anchor is None:
            self.anchor = self.d
        self.d = float(np.clip(self.d + 0.12 * (s_ref - self.ctx.s()), self.anchor - 0.02, self.anchor + 0.02))   # no wind-up
        self.aim()

    def pull(self, target, speed=PULL_SPEED, tol=0.002, dwell=HOLD_S):
        held = 0
        for _ in range(int(40 * CONTROL_HZ)):
            err = target - self.ctx.s()
            if abs(err) < tol:
                held += 1
            if held >= dwell * CONTROL_HZ:
                return
            self.d = min(self.d + float(np.clip(1.5 * err, -speed, speed)) / CONTROL_HZ, target + 0.02)   # no wind-up if blocked
            self.anchor = None
            self.aim()
            yield


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    variant = os.environ.get("DEMO_VARIANT", "")
    R, L = ctx.right, ctx.left
    if variant == "settle":
        yield from wait(8.0)
        return
    right_pts, left_pts = waypoints(task)
    yaw_r, _ = _grips(R.base, right_pts)
    yaw_l, _ = _grips(L.base, left_pts)
    yaw_r, yaw_l = (yaw_r if "YAW_R" not in os.environ else YAW_R), (yaw_l if "YAW_L" not in os.environ else YAW_L)
    rot = {"R": grip_rotation(yaw_r), "L": grip_rotation(yaw_l)}
    yaw = {"R": yaw_r, "L": yaw_l}
    arms = {"R": R, "L": L}
    sl = {"R": slice(7, 13), "L": slice(0, 6)}
    print(f"yaw R {math.degrees(yaw_r):.0f} L {math.degrees(yaw_l):.0f}", flush=True)
    for _ in range(300):                                         # the rod spawns at its upper stop and rolls down
        if ctx.s() < 0.002:
            break
        yield
    yield from wait(0.7)

    def point(v):
        """Measured TCP target at rod coordinate v: rod tip + v along the guide, from the measured guide pose."""
        g = ctx.states()["guide"]
        s = ctx.s()
        c = g.to_world((0.0, 0.0, -(task.ROD_U0 + s + v)))
        return c + [0.0, 0.0, ZOFF]

    def jaw(hand):
        c, sn = math.cos(yaw[hand]), math.sin(yaw[hand])
        return np.array([c * 0.0 - sn * (-JAW_OFFSET), sn * 0.0 + c * (-JAW_OFFSET), 0.0])    # Rz(yaw) @ (0, -J, 0)

    def hover(hand, v, s_pred, duration=2.5):
        g = ctx.states()["guide"]
        p = g.to_world((0.0, 0.0, -(task.ROD_U0 + s_pred + v))) - jaw(hand)
        return arms[hand].move(np.array([p[0], p[1], HOVER_Z]), rot[hand], duration, grip=1.0)

    def grab(hand, v, bg):
        """Descend on the rod with open fingers, null the free-arm error, close. bg() runs every step."""
        arm = arms[hand]
        tgt = point(v) - jaw(hand)
        cmd = tgt.copy()
        for _ in arm.move(np.array([tgt[0], tgt[1], HOVER_Z]), rot[hand], 0.6, grip=1.0):
            bg()
            yield
        for _ in arm.move(tgt, rot[hand], 1.6, grip=1.0):
            bg()
            yield
        for _ in range(40):
            cmd += 0.15 * (tgt - arm.tcp(ctx.obs, sl[hand]))
            arm.aim(cmd, rot[hand])
            bg()
            yield
        ctx.grab_cmd[hand] = cmd.copy()
        for _ in arm.set_grip(0.0, 0.8):
            bg()
            yield
        for _ in range(10):
            bg()
            yield

    def release(hand, bg):
        arm = arms[hand]
        for _ in arm.set_grip(1.0, 0.8):
            bg()
            yield
        for _ in arm.move(arm.pos + [0.0, 0.0, 0.10], rot[hand], 1.0):
            bg()
            yield

    def noop():
        pass

    only = {"right_only": "R", "left_only": "L"}.get(variant)
    if only:
        v = V_R1 if only == "R" else V_L_ONLY
        yield from grab(only, v, noop)
        if os.environ.get("DEMO_DEBUG_GRIP"):
            return
        h = Hold(arms[only], rot[only], ctx.grab_cmd[only], ctx)
        yield from h.pull(ONE_ARM_S, dwell=3.0)
        for _ in range(int(3 * CONTROL_HZ)):
            h.servo(ONE_ARM_S)
            yield
        return

    # stroke 1: right pulls
    yield from grab("R", V_R1, noop)
    hr = Hold(R, rot["R"], ctx.grab_cmd["R"], ctx)
    if variant == "probe":
        yield from wait(2.0)
        return
    yield from together(hr.pull(S_STEPS[0]), hover("L", V_L, S_STEPS[0]))
    if variant == "release":
        yield from release("R", noop)
        yield from wait(4.0)
        return
    if variant == "stop1":
        for _ in range(int(3 * CONTROL_HZ)):
            hr.servo(S_STEPS[0])
            yield
        return
    # exchange 1: left takes the rod while the right holds, then the right lets go
    yield from grab("L", V_L, lambda: hr.servo(S_STEPS[0]))
    hl = Hold(L, rot["L"], ctx.grab_cmd["L"], ctx)
    for _ in range(int(0.5 * CONTROL_HZ)):
        hr.servo(S_STEPS[0])
        hl.servo(S_STEPS[0])
        yield
    yield from release("R", lambda: hl.servo(S_STEPS[0]))
    # stroke 2: left pulls, the right moves above the next grip point
    yield from together(hl.pull(S_STEPS[1]), hover("R", V_R3, S_STEPS[1], 3.5))
    if variant == "stop2":
        for _ in range(int(3 * CONTROL_HZ)):
            hl.servo(S_STEPS[1])
            yield
        return
    # exchange 2
    yield from grab("R", V_R3, lambda: hl.servo(S_STEPS[1]))
    hr = Hold(R, rot["R"], ctx.grab_cmd["R"], ctx)
    for _ in range(int(0.5 * CONTROL_HZ)):
        hr.servo(S_STEPS[1])
        hl.servo(S_STEPS[1])
        yield
    yield from release("L", lambda: hr.servo(S_STEPS[1]))
    # stroke 3: right pulls to the target and holds
    yield from hr.pull(S_STEPS[2], dwell=2.5)
    for _ in range(int(2 * CONTROL_HZ)):
        hr.servo(S_STEPS[2])
        yield


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
    grab_cmd: dict = {}

    def s(self):
        return -float(self.states()["guide"].sliders["rod"])

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
                g = ctx.states()["guide"]
                sh = 1000 * float(np.linalg.norm(g.pos - task._STATE["start"])) if task._STATE["start"] is not None else 0.0
                tl, tr = ctx.left.tcp(ctx.obs, slice(0, 6)), ctx.right.tcp(ctx.obs, slice(7, 13))
                msg += (f" s {1000 * ctx.s():.1f} L {tl[0]:.3f},{tl[1]:.3f},{tl[2]:.3f} g{ctx.left.grip:.2f} "
                        f"R {tr[0]:.3f},{tr[1]:.3f},{tr[2]:.3f} g{ctx.right.grip:.2f} shift {sh:.1f} "
                        f"gd {1000 * (g.pos[0] - task.GUIDE_XY[0]):.1f},{1000 * (g.pos[1] - task.GUIDE_XY[1]):.1f},{1000 * (g.pos[2] - task.GUIDE_Z):.1f} "
                        f"tilt {task._tilt_deg(g):.1f} tripped {task._STATE['tripped']}")
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
