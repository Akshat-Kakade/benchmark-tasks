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



GRIP_YAWS = (math.pi / 2, -math.pi / 2)    # fingers open along world x (pinch the handle stem across x)
PEN = 0.003                                # each paddle presses this far into the blank's face (at the reference pose)
CLOSE_GAP = 0.020
WITHDRAW = 0.030


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def face_target(task, theta, sign, gap, shift=(0.0, 0.0)):
    """Paddle-centre target (x, y) for the tool pushing the blank's -x face (blue, sign +1) or +x face (orange, sign -1)
    when the blank is at yaw ``theta`` about the registered centre; ``gap`` > 0 stands off the face."""
    c, s = math.cos(theta), math.sin(theta)
    lx, ly = -sign * (task.H + task.PAD_FACE + gap), sign * task.PAD_A
    return np.array([task.CX + c * lx - s * ly + shift[0], task.CY + s * lx + c * ly + shift[1]])


def hand_local(task):
    return np.array([task.STEM_X, task.STEM_Y, task.GRIP_Z])


def path_points(task, theta_end):
    """(arm name, hand position, hand yaw offset phi) along the whole planned motion, for the IK yaw choice and scan."""
    pts = []
    z = task.TOOL_Z + 0.006
    for th in np.radians(np.linspace(0.0, theta_end, 7)):
        for name, sign, phi0 in (("left", 1, 0.0), ("right", -1, math.pi)):
            for gap in (-PEN, WITHDRAW) if abs(th) > 0.99 * abs(math.radians(theta_end)) else (-PEN,):
                t = face_target(task, th, sign, gap)
                R = _rz(th + phi0)
                pts.append((name, t_to_hand(t, z, R, task), th))
    return pts


def t_to_hand(t_xy, z, R, task):
    return np.array([t_xy[0], t_xy[1], z]) + R @ hand_local(task)


def plan_yaw(arm, name, pts):
    """Grip yaw (+-pi/2) with the smallest worst IK error over the planned path (LESSON 12)."""
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    best = None
    for c in GRIP_YAWS:
        worst, q = 0.0, arm.q
        for n, p, th in pts:
            if n != name:
                continue
            q, pe, re = ik.solve(p, ik.matrix_to_quat_xyzw(_rz(th) @ grip_rotation(c)), q, base_pos=arm.base, iters=200)
            worst = max(worst, pe + 0.05 * re)
        if best is None or worst < best[0]:
            best = (worst, c)
    return best[1], best[0]


def actual_tcp(arm):
    from yam_arena import ik

    q = arm.read() if arm.read is not None else None
    if q is None:
        return arm.pos.copy()
    return ik.tcp_pose(np.asarray(q, float), arm.base)[0]


class ToolCtl:
    """Holds one push tool by its handle stem; commands the hand so the MEASURED tool pose reaches the target."""

    def __init__(self, arm, name, g0, phi0, tcp_gain=0.08):
        self.arm, self.name, self.g0, self.phi0, self.gain = arm, name, g0, phi0, tcp_gain
        self.tr = np.zeros(3)
        self.a = 0.0
        self.hl = None

    def measure_offset(self, ctx, task):
        t = ctx.states()[self.name]
        self.hl = t.rotation().T @ (actual_tcp(self.arm) - t.pos)

    def aim(self, ctx, task, tpos, tphi, fb=True):
        t = ctx.states()[self.name]
        if fb:
            self.tr = np.clip(self.tr + self.gain * (tpos - t.pos), -0.06, 0.06)
            e = wrap(math.radians(task.yaw_deg(t)) - tphi)
            self.a = float(np.clip(self.a - 0.15 * e, -0.5, 0.5))
        hand = tpos + _rz(tphi) @ self.hl
        self.arm.aim(hand + self.tr, _rz(tphi - self.phi0 + self.a) @ self.g0)


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":
        yield from wait(8.0)
        return
    for _ in range(12):
        yield
    theta_end = {"yaw40": -40.0, "yaw50": -50.0, "yaw30": -30.0, "yaw60": -60.0}.get(variant, -45.0)
    pts = path_points(task, theta_end)
    yl, el = plan_yaw(L, "left", pts)
    yr, er = plan_yaw(Rr, "right", pts)
    if os.environ.get("YAW_L"):
        yl = float(os.environ["YAW_L"]) * math.pi / 2
    if os.environ.get("YAW_R"):
        yr = float(os.environ["YAW_R"]) * math.pi / 2
    gl, gr = grip_rotation(yl), grip_rotation(yr)
    print("YAW", yl, yr, "ik_err", round(el, 4), round(er, 4), flush=True)
    st0 = ctx.states()
    one_tool = variant == "one_tool"

    def grab(arm, name, g):
        t = ctx.states()[name]
        h = t.to_world(hand_local(task))
        yield from arm.move(h + [0, 0, 0.09], g, 2.5, grip=1.0)
        yield from arm.move(h, g, 1.5)
        pre = ctx.states()[name].pos.copy()
        yield from arm.set_grip(0.0, 1.0)
        yield from wait(0.4)
        print("CLOSE", name, "shift", np.round(ctx.states()[name].pos - pre, 4).tolist(), flush=True)

    jobs = [grab(L, "blue_tool", gl)]
    if not one_tool:
        jobs.append(grab(Rr, "orange_tool", gr))
    yield from par(*jobs)
    cb = ToolCtl(L, "blue_tool", gl, 0.0)
    co = ToolCtl(Rr, "orange_tool", gr, math.pi)
    no_fb = variant == "open_loop"
    z_t = task.TOOL_Z + 0.006
    # lift 6 mm (tool bottoms just off the table), then measure the hand-to-tool offsets
    for k in range(1, 41):
        s = k / 40
        for ctl, nm, sign, phi in ((cb, "blue_tool", 1, 0.0), (co, "orange_tool", -1, math.pi)):
            if ctl is co and one_tool:
                continue
            t = ctx.states()[nm]
            ctl.arm.aim(ctl.arm.pos + np.array([0, 0, 0.006 / 40]), ctl.arm.rot)
        yield
    yield from wait(0.8)
    cb.measure_offset(ctx, task)
    if not one_tool:
        co.measure_offset(ctx, task)
    shift = np.zeros(2)
    theta_ref = 0.0
    ofs = 0.0                                  # face-pose offset trimmed on the MEASURED blank yaw
    gap = CLOSE_GAP
    rate = math.radians(4.0)                   # reference turning rate, deg/s
    lag_max = math.radians(4.0)
    dt = 1.0 / CONTROL_HZ
    stage = "close"
    hold = 0
    nsteps = int(60 * CONTROL_HZ)
    for k in range(nsteps):
        b = ctx.states()["blank"]
        th_m = math.radians(task.yaw_deg(b))
        if stage == "close":
            gap = max(-PEN, gap - 0.006 * dt * 4)       # 24 mm/s
            if gap <= -PEN:
                stage = "turn"
        elif stage == "turn":
            if (th_m - theta_ref) <= lag_max:            # blank not lagging more than 4 deg behind the reference
                theta_ref = max(math.radians(theta_end), theta_ref - rate * dt)
            if theta_ref <= math.radians(theta_end) + 1e-9 and abs(th_m - theta_ref) < math.radians(2.0):
                hold += 1
                if hold > int(1.0 * CONTROL_HZ):
                    stage = "withdraw"
            if variant != "open_loop":
                ofs = float(np.clip(ofs + 0.02 * (theta_ref - th_m), -math.radians(20), math.radians(20)))
                shift = np.clip(shift + 0.01 * (np.array([task.CX, task.CY]) - b.pos[:2]), -0.012, 0.012)
        elif stage == "withdraw":
            if variant == "no_withdraw":
                stage = "end"
            else:
                gap = min(WITHDRAW, gap + 0.015 * dt)
                if gap >= WITHDRAW:
                    stage = "end"
        elif stage == "end":
            hold += 1
            if hold > int(4.5 * CONTROL_HZ):
                break
        for ctl, sign, phi, nm in ((cb, 1, 0.0, "blue_tool"), (co, -1, math.pi, "orange_tool")):
            if ctl is co and one_tool:
                continue
            xy = face_target(task, theta_ref + ofs, sign, gap, shift)
            ctl.aim(ctx, task, np.array([xy[0], xy[1], z_t]), theta_ref + ofs + phi, fb=not no_fb)
        if k % 20 == 0 and os.environ.get("DEMO_DEBUG"):
            print("STAGE", stage, "ref", round(math.degrees(theta_ref), 1), "ofs", round(math.degrees(ofs), 1), "gap", round(gap, 3),
                  {kk: round(v, 4) for kk, v in task._metrics(ctx.states()).items()}, flush=True)
        yield
    print("FINAL", {kk: round(v, 4) for kk, v in task._metrics(ctx.states()).items()}, flush=True)


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
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
