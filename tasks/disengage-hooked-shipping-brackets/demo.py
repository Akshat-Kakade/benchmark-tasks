"""Scripted full-score demo for the hooked-brackets task (closed loop on the measured bracket poses).

    python demo.py task.py demo_out [--max-steps 1500] [--no-record]

Each arm pinches the handle post of its bracket (fingers open along y) and lifts it 35 mm. The left hand then holds the blue
housing still (integrated on its measured pose); the right hand moves the orange bracket along a planned collision-free path
in the housing's frame (geom.plan: slide to the slot mouth, turn the T-head about 80 deg while it passes the slot, slide
out) at a pace gated by the measured tracking error, then turns it back and holds 0.28 m from the housing.
DEMO_VARIANT: settle (hold only), one_arm (left only), no_turn (pull straight), stop_early (turn, stop before leaving),
keep_turn (stay turned at the end, also valid).
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




HZ = 0.070                 # pinch height above the bracket origin (handle posts are 90 mm tall)
LIFT = 0.035
PACE = 0.030               # m/s along the planned path (rotation counted as 0.10 m per rad)
ROT_W = 0.10


def _smooth(p, geom, margin):
    """Resample by path length and smooth with a moving average; keep it only if still collision-free."""
    d = np.r_[0.0, np.cumsum(np.hypot(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])), ROT_W * np.diff(p[:, 2])))]
    s = np.linspace(0, d[-1], max(8, int(d[-1] / 0.004)))
    q = np.stack([np.interp(s, d, p[:, k]) for k in range(3)], 1)
    k = 9
    pad = np.r_[np.repeat(q[:1], k, 0), q, np.repeat(q[-1:], k, 0)]
    sm = np.stack([np.convolve(pad[:, c], np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k] for c in range(3)], 1)
    sm[0], sm[-1] = q[0], q[-1]
    ok = not geom.collides((sm[:, 0], sm[:, 1], sm[:, 2]), margin).any()
    out = sm if ok else q
    dd = np.r_[0.0, np.cumsum(np.hypot(np.hypot(np.diff(out[:, 0]), np.diff(out[:, 1])), ROT_W * np.diff(out[:, 2])))]
    return out, dd, ok


def script(task, ctx):
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    import geom

    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":
        yield from wait(8.0)
        for n, p in ctx.states().items():
            print("REST", n, np.round(p.pos, 4).tolist(), "yaw", round(task.yaw_deg(p), 2), "tilt", round(task.tilt_deg(p), 2), flush=True)
        print("REL", task.rel(ctx.states()), flush=True)
        yield from wait(1.0)
        return
    for _ in range(20):
        yield
    path0, why = geom.plan(margin=0.010, sign=+1)
    path, cum, smooth_ok = _smooth(path0, geom, 0.008)
    print("PATH nodes", len(path0), "smoothed", smooth_ok, "max turn deg", round(math.degrees(abs(path[:, 2]).max())), flush=True)
    yaw_l = float(os.environ.get("YAW_L", "0")) * math.pi / 2
    yaw_r = float(os.environ.get("YAW_R", "0")) * math.pi / 2
    gl, gr = grip_rotation(yaw_l), grip_rotation(yaw_r)

    def st(n):
        return ctx.states()[n]

    hl = st("blue").to_world((0.0, geom.A_POST_C[1], HZ))
    hr = st("orange").to_world((0.0, geom.B_POST_Y, HZ))

    def grab(arm, h, g, name):
        yield from arm.move(h + [0, 0, 0.09], g, 2.5, grip=1.0)
        yield from arm.move(h, g, 1.5)
        pre = st(name).pos.copy()
        yield from arm.set_grip(0.0, 1.0)
        yield from wait(0.4)
        print("CLOSE", name, "shift", np.round(st(name).pos - pre, 4).tolist(), flush=True)

    if variant == "one_arm":
        yield from par(grab(L, hl, gl, "blue"), wait(1.0))
        L.feedback(True)
        yield from L.move(L.pos + [0, 0, LIFT], gl, 2.0)
        yield from wait(5.0)
        return
    yield from par(grab(L, hl, gl, "blue"), grab(Rr, hr, gr, "orange"))
    L.feedback(True)
    Rr.feedback(True)
    yield from par(L.move(L.pos + [0, 0, LIFT], gl, 2.0), Rr.move(Rr.pos + [0, 0, LIFT], gr, 2.0))
    yield from wait(1.0)
    a, b = st("blue"), st("orange")
    offL = L.pos - a.pos                      # hand-to-bracket offsets as grasped
    offR = b.rotation().T @ (Rr.pos - b.pos)
    a0 = a.pos.copy()
    dz_b = float(b.pos[2] - a.pos[2])
    yaw_a0 = math.radians(task.yaw_deg(a))
    print("CARRY a", np.round(a.pos, 4).tolist(), "b", np.round(b.pos, 4).tolist(), "tilt", round(task.tilt_deg(a), 2), round(task.tilt_deg(b), 2), flush=True)
    trL, trR = np.zeros(3), np.zeros(3)
    a_l = a_r = 0.0
    s = 0.0
    total = float(cum[-1])
    stop_early = variant == "stop_early"
    no_turn = variant == "no_turn"
    if no_turn:                                   # straight pull: ignore the planned turn, drive y only
        path = np.stack([np.zeros(len(path)) + path[:, 0] * 0.0, path[:, 1], np.zeros(len(path))], 1)
    t_end_path = None
    turn_back = None                              # (start step, from-phi)
    k = 0
    final_rel = np.array([task.B_REL_OUT[0], task.B_REL_OUT[1], 0.0 if variant != "keep_turn" else float(path[-1, 2])])
    dt = 1.0 / CONTROL_HZ
    hold_steps = 0
    max_steps = int(70 * CONTROL_HZ)
    for k in range(max_steps):
        a, b = st("blue"), st("orange")
        ya = math.radians(task.yaw_deg(a))
        ca, sa = math.cos(ya), math.sin(ya)
        Ra = np.array([[ca, -sa], [sa, ca]])
        # desired T-head pose in the housing frame
        if s < total:
            idx = int(np.searchsorted(cum, s))
            idx = min(idx, len(path) - 1)
            rel = path[idx].copy()
        else:
            if turn_back is None:
                turn_back = (k, path[-1].copy())
            u = min(1.0, (k - turn_back[0]) / (4.0 * CONTROL_HZ))
            u = ik.smoothstep(u)
            rel = turn_back[1] + u * (final_rel - turn_back[1])
        pb_xy = np.array([a.pos[0], a.pos[1]]) + Ra @ rel[:2]
        pb_des = np.array([pb_xy[0], pb_xy[1], a.pos[2] + dz_b])
        yaw_des = ya + rel[2]
        e_pos = float(np.linalg.norm(b.pos - pb_des))
        e_yaw = (math.radians(task.yaw_deg(b)) - yaw_des + math.pi) % (2 * math.pi) - math.pi
        ok = e_pos < 0.008 and abs(e_yaw) < math.radians(4.0)
        if s < total and ok and not (stop_early and task_peak(ctx) >= 40.0):
            s = min(total, s + PACE * dt)
        trL = np.clip(trL + 0.06 * (a0 - a.pos), -0.03, 0.03)
        trR = np.clip(trR + 0.05 * (pb_des - b.pos), -0.03, 0.03)
        a_l = float(np.clip(a_l - 0.15 * (ya - yaw_a0), -0.4, 0.4))
        a_r = float(np.clip(a_r - 0.15 * e_yaw, -0.6, 0.6))
        R_hand = _rz(yaw_des + a_r) @ gr
        hand = pb_des + _rz(yaw_des) @ offR + trR
        L.aim(a0 + offL + trL, _rz(a_l) @ gl)
        Rr.aim(hand, R_hand)
        if k % 20 == 0 and os.environ.get("DEMO_DEBUG"):
            m = task.rel(ctx.states())
            print("REL", {kk: (round(v, 4) if not isinstance(v, bool) else v) for kk, v in m.items()}, "s", round(s, 3), "/", round(total, 3),
                  "e_pos", round(e_pos, 4), "e_yaw", round(math.degrees(e_yaw), 1), flush=True)
        yield
        if s >= total and turn_back is not None and k - turn_back[0] > 4.0 * CONTROL_HZ + 80:
            break
        if stop_early and task_peak(ctx) >= 40.0 and k > 0 and hold_steps > 100:
            break
        if stop_early and task_peak(ctx) >= 40.0:
            hold_steps += 1
    print("FINAL", {kk: (round(v, 4) if not isinstance(v, bool) else v) for kk, v in task.rel(ctx.states()).items()}, flush=True)
    yield from wait(2.0)


def task_peak(ctx):
    return float(ctx.emb.task.module._STATE.get("peak", 0.0))


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


def run(task_path: str, out_dir: str, max_steps: int = 1500, record: bool = True, cam_width: int = 320,
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
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
