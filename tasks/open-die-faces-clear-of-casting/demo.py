"""Scripted full-score demo for the die-faces task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1600] [--no-record]

Both hands pinch their outer tab (fingers open along y) and pull outward in step, closed loop on the measured slider
travel of each face (the command of each arm integrates its own travel error, so the two faces stay level), then hold
the end state until success. Writes one MP4 per camera, setup.png and result.json to the output folder.

DEMO_VARIANT: settle (hold 8 s), one_arm (left only), short (stop both at 45 mm), uneven (left 70 mm, right 50 mm).
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
TARGET = 0.072               # m, inside the 65-75 mm band
PULL_SPEED = 0.025           # m/s
TAB_Z = 0.85                 # TCP height of the pinch on the post
HOVER_Z = TAB_Z + 0.10
JAW_OFFSET = float(os.environ.get("JAW_OFFSET", 0.0157))   # the fixed jaw sits this far -y of the TCP (world)


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
    out = {}
    for name, sign in (("left", 1), ("right", -1)):
        pts = []
        for t in (0.0, 0.075):
            y = sign * (task.TAB_OUT0 + t)
            pts += [(task.CX, y, z) for z in (HOVER_Z, TAB_Z)]
        out[name] = pts
    return out["right"], out["left"]


def _tab(dies, task, sign, travel=0.0):
    """Measured tab centre (world) from the base pose and the slider reading."""
    st = dies["die_l" if sign > 0 else "die_r"]
    _, Lf, _ = task.side_frame(sign)
    t = -float(st.sliders["face"])
    return st.to_world(Lf(0.0, task.TAB_OUT0 - task.BASE_OUT, task.TAB_Z - task.ORIGIN_Z) + np.array([0.0, 0.0, -t])), t


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    variant = os.environ.get("DEMO_VARIANT", "")
    arms = {1: ctx.left, -1: ctx.right}
    right_pts, left_pts = waypoints(task)
    flips = {1: _grips(ctx.left.base, left_pts)[0], -1: _grips(ctx.right.base, right_pts)[0]}
    rots = {s: grip_rotation(flips[s]) for s in (1, -1)}
    if variant == "settle":
        yield from wait(1.0)
        for k, v in ctx.states().items():
            print("STATE", k, np.round(v.pos, 4), np.round(v.quat_xyzw, 4), getattr(v, "sliders", None), flush=True)
        yield from wait(8.0)
        return
    yield from wait(0.7)
    active = (1,) if variant == "one_arm" else (1, -1)
    target = {1: 0.045 if variant == "short" else TARGET, -1: 0.045 if variant == "short" else TARGET}
    if variant == "uneven":
        target[-1] = 0.050
    grasp, cmd = {}, {}
    dies = ctx.states()
    for s in active:
        tab, _ = _tab(dies, task, s)
        grasp[s] = np.array([tab[0], tab[1] + JAW_OFFSET, TAB_Z])
        cmd[s] = grasp[s].copy()

    # 1. Both hands over the tabs, then down, then null the steady error and close.
    for phase in ("hover", "down"):
        gens = [arms[s].move(grasp[s] + [0, 0, HOVER_Z - TAB_Z if phase == "hover" else 0.0], rots[s],
                             2.5 if phase == "hover" else 1.5, grip=1.0) for s in active]
        for _ in range(int((2.5 if phase == "hover" else 1.5) * CONTROL_HZ)):
            for g in gens:
                next(g, None)
            yield
    for _ in range(40):
        for s in active:
            cmd[s] += 0.15 * (grasp[s] - arms[s].tcp(ctx.obs, slice(0, 6) if s > 0 else slice(7, 13)))
            arms[s].aim(cmd[s], rots[s])
        yield
    gens = [arms[s].set_grip(0.0, 0.8) for s in active]
    for _ in range(int(0.8 * CONTROL_HZ) + 6):
        for g in gens:
            next(g, None)
        yield

    # 2. Pull outward in step: a shared ramp, each arm integrating its own travel error.
    ramp = 0.0
    out = {s: 0.0 for s in active}          # commanded outward displacement of each arm
    for k in range(int(40.0 * CONTROL_HZ)):
        st = ctx.states()
        tr = {s: _tab(st, task, s)[1] for s in active}
        ramp = min(ramp + PULL_SPEED / CONTROL_HZ, max(target.values()))
        for s in active:
            goal = min(ramp, target[s])
            out[s] += (PULL_SPEED / CONTROL_HZ if ramp < target[s] else 0.0) + 0.06 * (goal - tr[s])
            out[s] = float(np.clip(out[s], -0.01, 0.085))
            arms[s].aim(np.array([grasp[s][0] + 0.0, grasp[s][1] + s * out[s], grasp[s][2]]), rots[s])
        yield
        if k > 60 and all(abs(tr[s] - target[s]) < 0.003 for s in active) and ramp >= max(target.values()):
            break
    for _ in range(int(6.0 * CONTROL_HZ)):                       # hold; the run ends on success
        st = ctx.states()
        tr = {s: _tab(st, task, s)[1] for s in active}
        for s in active:
            out[s] += 0.06 * (target[s] - tr[s])
            arms[s].aim(np.array([grasp[s][0], grasp[s][1] + s * out[s], grasp[s][2]]), rots[s])
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
                tl, tr = ctx.left.tcp(ctx.obs, slice(0, 6)), ctx.right.tcp(ctx.obs, slice(7, 13))
                msg += (f" L {tl[0]:.3f},{tl[1]:.3f},{tl[2]:.3f} R {tr[0]:.3f},{tr[1]:.3f},{tr[2]:.3f} travel "
                        f"{-1000 * s['die_l'].sliders['face']:.1f},{-1000 * s['die_r'].sliders['face']:.1f} cast "
                        f"{1000 * (s['casting'].pos[0] - task._STATE['start'][0][0]):.1f},"
                        f"{1000 * (s['casting'].pos[1] - task._STATE['start'][0][1]):.1f} tilt {task._tilt_deg(s['casting']):.1f} "
                        f"clr {task._clearances_mm(s)[0]:.1f},{task._clearances_mm(s)[1]:.1f} base "
                        f"{1000 * (s['die_l'].pos[1] - task._STATE['start'][1][1]):.1f},{1000 * (s['die_r'].pos[1] - task._STATE['start'][2][1]):.1f} tripped {task._STATE['tripped']}")
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
