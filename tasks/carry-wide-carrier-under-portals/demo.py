"""Scripted full-score demo for the carry-under-portals task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1800] [--no-record]

Both hands pinch the orange stems (fingers open along y), lift the carrier 15 mm, steer it through slot 1 (y = +0.03),
shift sideways between the guards, pass slot 2 (y = -0.03), then lower it into the second green outline, release and
retreat straight up. Closed loop on the measured carrier pose and on each hand's measured position.

DEMO_VARIANT: settle (hold only), one_hand (only the left hand carries), no_shift (stays at y = +0.03: jams in guard 2),
roll12 (carrier rolled 12 degrees while carried: plate hits the guard), alt (lift 20 mm, shift earlier, final yaw 4 deg:
second valid solution), final_off (set down 25 mm to the side).
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


SZ = 0.037                 # pinch height above the carrier origin (stem mid-height)
LIFT = 0.015               # carry height of the carrier above its rest
SPEED = 0.025              # m/s along the path


def path_for(variant: str, x0: float, y0: float):
    s0, s1 = (0.322, 0.358) if variant != "alt" else (0.318, 0.36)
    if variant == "no_shift":
        return [(x0, y0), (x0, 0.02), (0.43, 0.02)]
    return [(x0, y0), (x0, 0.02), (s0, 0.02), (s1, -0.02), (0.425, -0.02), (0.45, 0.0)]


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":                      # explicit hold plan (an empty plan would fall back to the built-in demo)
        yield from wait(8.0)
        return
    if variant == "probe":                       # reach probe: left hand to a row of points, print actual vs commanded
        from yam_arena import ik as _ik
        gl0 = grip_rotation(0.0)
        for zz in (0.784, 0.82):
            for xx in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
                yield from L.move(np.array([xx, 0.175, zz]), gl0, 1.5, grip=1.0)
                yield from wait(1.0)
                pa, _r = _ik.tcp_pose(np.asarray(ctx.last_state)[0:6], L.base)
                print("PROBE", xx, zz, "actual", np.round(pa, 3).tolist(), flush=True)
        return
    for _ in range(12):                          # let the teleported parts settle before reading baselines
        yield
    c0 = ctx.states()["carrier"]
    z_rest = float(c0.pos[2])
    x0, y0 = float(c0.pos[0]), float(c0.pos[1])
    SY = task.STEM_Y
    gl = grip_rotation(float(os.environ.get("YAW_L", "0")) * math.pi / 2)
    gr = grip_rotation(float(os.environ.get("YAW_R", "0")) * math.pi / 2)
    oL = float(os.environ.get("OFF_L", "0"))
    oR = float(os.environ.get("OFF_R", "0"))
    lift = 0.020 if variant == "alt" else float(os.environ.get("LIFT", LIFT))
    one = variant == "one_hand"

    def car():
        return ctx.states()["carrier"]

    hl = np.array([x0, y0 + SY + oL, z_rest + SZ])
    hr = np.array([x0, y0 - SY + oR, z_rest + SZ])
    # ---- approach and grip ----------------------------------------------------------------------------------------------
    def go_l():
        yield from L.move(hl + [0, 0, 0.09], gl, 2.5, grip=1.0)
        yield from L.move(hl, gl, 1.5)
        yield from L.set_grip(0.0, 1.0)

    def go_r():
        yield from Rr.move(hr + [0, 0, 0.09], gr, 2.5, grip=1.0)
        yield from Rr.move(hr, gr, 1.5)
        yield from Rr.set_grip(0.0, 1.0)

    # both arms move together: run the two generators in lockstep
    gens = [go_l()] if one else [go_l(), go_r()]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        yield
    pre = car().pos.copy()
    yield from wait(0.5)
    print("CLOSE carrier shift", np.round(car().pos - pre, 4).tolist(), flush=True)
    c = car()
    offL = np.array(L.pos) - c.pos              # hand-to-carrier offsets as grasped
    offR = np.array(Rr.pos) - c.pos
    if os.environ.get("ARM_FB", "0") == "1":
        L.feedback(True)
        if not one:
            Rr.feedback(True)
    corr = np.zeros(3)
    yc = rc = 0.0
    gain = float(os.environ.get("CGAIN", "0.05"))
    roll_target = math.radians(12) if variant == "roll12" else 0.0

    def aim(tx, ty, tz, yaw=0.0):
        """Command both hands so the carrier origin goes to (tx, ty, tz); closed loop on its measured pose (position,
        yaw, roll integrators shared by both hands, so the two grips never pull against each other)."""
        nonlocal corr, yc, rc
        cm = car()
        corr = np.clip(corr + gain * (np.array([tx, ty, tz]) - cm.pos), -0.04, 0.04)
        R = cm.rotation()
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw_m = math.atan2(R[1, 0], R[0, 0])
        yc = float(np.clip(yc + 0.03 * (yaw - yaw_m), -0.25, 0.25))
        rc = float(np.clip(rc + 0.03 * (roll_target - roll), -0.2, 0.2))
        t = np.array([tx, ty, tz]) + corr
        a = yaw + yc
        Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        dz = SY * math.sin(roll_target + rc)
        L.aim(t + Rz @ offL + [0, 0, dz], _rz(a) @ gl)
        if not one:
            Rr.aim(t + Rz @ offR + [0, 0, -dz], _rz(a) @ gr)

    # ---- lift, steer through both slots -------------------------------------------------------------------------------
    n = int(2.0 * CONTROL_HZ)
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        aim(x0, y0, z_rest + lift * s)
        yield
    pts = path_for(variant, x0, y0)
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(seg)
    n = int(total / SPEED * CONTROL_HZ)
    for k in range(1, n + 1):
        d = ik.smoothstep(k / n) * total
        j = 0
        while j < len(seg) - 1 and d > seg[j]:
            d -= seg[j]
            j += 1
        f = min(1.0, d / seg[j])
        px = pts[j][0] + f * (pts[j + 1][0] - pts[j][0])
        py = pts[j][1] + f * (pts[j + 1][1] - pts[j][1])
        aim(px, py, z_rest + lift)
        yield
    yield from wait(0.5)
    if variant in ("no_shift", "one_hand"):
        yield from wait(3.0)
        return
    # ---- lower into the outline, trim on the measured pose, release ------------------------------------------------------
    ex = 0.025 if variant == "final_off" else 0.0
    fyaw = math.radians(4) if variant == "alt" else 0.0
    gx, gy = task.TARGET[0], task.TARGET[1] + ex
    n = int(2.0 * CONTROL_HZ)
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        aim(gx, gy, z_rest + lift * (1 - s) + 0.001 * s, fyaw * s)
        yield
    for _ in range(int(1.5 * CONTROL_HZ)):
        aim(gx, gy, z_rest, fyaw)
        yield
    print("PLACED carrier", np.round(car().pos, 4).tolist(), flush=True)
    yield from wait(0.5)
    L.feedback(False)
    Rr.feedback(False)
    gens = [L.set_grip(1.0, 0.8), Rr.set_grip(1.0, 0.8)]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        yield
    yield from wait(0.5)
    print("RELEASED carrier", np.round(car().pos, 4).tolist(), flush=True)
    gens = [L.move(L.pos + [0, 0, 0.12], L.rot, 2.0), Rr.move(Rr.pos + [0, 0, 0.12], Rr.rot, 2.0)]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        yield
    yield from wait(2.0)


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


def run(task_path: str, out_dir: str, max_steps: int = 1800, record: bool = True, cam_width: int = 320,
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
                level, perfect, inf = task.classify(st, dict(task._STATE))
                rk = st["carrier"]
                msg += f" | carrier {np.round(rk.pos, 3).tolist()} tilt {task.tilt_deg(rk):.1f} yaw {task.yaw_deg(rk):.1f} "
                msg += " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in inf.items())
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
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
