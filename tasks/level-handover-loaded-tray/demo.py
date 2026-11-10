"""Scripted full-score demo for the level-handover task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1800] [--no-record]

The left arm pinches the rack's outer handle (+y end), lifts the rack 70 mm and carries it level to the middle of the
table. The right arm pinches the other handle (-y end); the left hand lets go and moves away; the right arm carries the
rack to the right pad, lowers it, lets go and withdraws. Closed loop on the measured rack pose and on each hand's
measured position (obs joint positions).

DEMO_VARIANT: settle (hold only), no_handover (left stops at y=+0.15 holding the rack), early_release (left lets go at the
middle before the right hand has it), tilt20 (left tips the rack 20 degrees while carrying), place_yaw15 (rack set down
15 degrees off), mid_alt (handover at y=+0.06: second valid solution).
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


HY = 0.135                 # handle offset from the rack centre (y), rack frame
HZ = 0.030                 # pinch height above the rack origin
LIFT = 0.070               # carry height of the rack above its pad
PAD_THK = 0.003


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":                      # explicit hold plan (an empty plan would fall back to the built-in demo)
        yield from wait(8.0)
        return
    for _ in range(12):                          # let the teleported parts settle before reading baselines
        yield
    r0 = ctx.states()["rack"]
    z_rest = float(r0.pos[2])                    # rack origin height at rest (measured)
    x_c = float(r0.pos[0])
    yL = float(r0.pos[1])
    yR = task.PAD_R[1]
    mid = 0.06 if variant == "mid_alt" else 0.0
    yaw_l = reachable_yaw(L, np.array([x_c, mid + HY, z_rest + LIFT + HZ]))
    yaw_r = reachable_yaw(Rr, np.array([x_c, mid - HY, z_rest + LIFT + HZ]))
    if os.environ.get("YAW_L"):
        yaw_l = float(os.environ["YAW_L"]) * math.pi / 2
    if os.environ.get("YAW_R"):
        yaw_r = float(os.environ["YAW_R"]) * math.pi / 2
    gl, gr = grip_rotation(yaw_l), grip_rotation(yaw_r)
    print('YAW', yaw_l, yaw_r, flush=True)
    oL = float(os.environ.get('OFF_L', '0'))
    oR = float(os.environ.get('OFF_R', '0'))

    def rack_now():
        return ctx.states()["rack"]

    # ---- left hand: grab the +y handle on the left pad, lift, carry to the middle -------------------------------------
    hl = np.array([x_c, yL + HY + oL, z_rest + HZ])
    yield from L.move(hl + [0, 0, 0.09], gl, 2.5, grip=1.0)
    yield from L.move(hl, gl, 1.5)
    pre = rack_now().pos.copy()
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.4)
    print('LCLOSE rack shift', np.round(rack_now().pos - pre, 4).tolist(), flush=True)
    L.feedback(True)
    tilt = math.radians(20) if variant == "tilt20" else 0.0
    stop_y = 0.15 if variant == "no_handover" else mid

    def left_at(cy, dz, ang=0.0):
        p = np.array([x_c, cy + HY + oL, z_rest + dz + HZ])
        return p, _rx(ang) @ gl

    yield from L.move(*left_at(yL, LIFT), 2.0)
    n = int(5.0 * CONTROL_HZ)
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        ang = tilt * math.sin(math.pi * s) if tilt else 0.0
        L.aim(*left_at(yL + s * (stop_y - yL), LIFT, ang))
        yield
    yield from wait(1.0)
    if variant == "no_handover":
        yield from wait(3.0)
        return
    if variant == "early_release":
        yield from L.set_grip(1.0, 0.8)
        yield from wait(3.0)
        return

    # ---- right hand: grab the -y handle (measured rack pose), then the left lets go ------------------------------------
    r = rack_now()
    hr = r.to_world((0.0, -HY + oR, HZ))
    yield from Rr.move(hr + [0, 0, 0.09], gr, 3.0, grip=1.0)
    yield from Rr.move(hr, gr, 1.5)
    pre = rack_now().pos.copy()
    yield from Rr.set_grip(0.0, 1.0)
    yield from wait(0.5)
    print('RCLOSE rack shift', np.round(rack_now().pos - pre, 4).tolist(), flush=True)
    off = np.array(Rr.pos) - rack_now().pos      # hand-to-rack offset as grasped, rack still carried by the left hand
    Rr.gain = float(os.environ.get('GAIN_R', '0.04'))
    Rr.feedback(True)
    yield from L.set_grip(1.0, 0.8)
    yield from wait(0.3)
    yield from L.move(L.pos + [0, 0.0, 0.12], L.rot, 2.0)   # straight up: moving in y would drag the inner finger into the handle
    L.feedback(False)

    # ---- right hand carries the rack to the right pad, lowers it, lets go -----------------------------------------------
    r = rack_now()
    y0 = float(r.pos[1])
    place_yaw = math.radians(15) if variant == "place_yaw15" else 0.0
    n = int(6.0 * CONTROL_HZ)
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        c = np.array([x_c, y0 + s * (yR - y0), z_rest + LIFT])
        Rr.aim(c + off, _rz(place_yaw * s) @ gr)
        yield
    yield from wait(0.5)
    n = int(3.5 * CONTROL_HZ)
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        c = np.array([x_c, yR, z_rest + LIFT * (1 - s) + 0.001 * s])
        Rr.aim(c + off, _rz(place_yaw) @ gr)
        yield
    # trim the position on the measured rack pose until it rests on the pad, then let go
    for _ in range(int(2.0 * CONTROL_HZ)):
        r = rack_now()
        e = np.array([x_c, yR, z_rest]) - r.pos
        e[2] = max(e[2], -0.01) if r.pos[2] > z_rest + 0.0015 else 0.0
        Rr.aim(Rr.pos + 0.25 * e, _rz(place_yaw) @ gr)
        yield
    print('PLACED rack', np.round(rack_now().pos, 4).tolist(), flush=True)
    yield from wait(0.5)
    Rr.feedback(False)
    yield from Rr.set_grip(1.0, 0.8)
    yield from wait(0.8)
    print('RELEASED rack', np.round(rack_now().pos, 4).tolist(), flush=True)
    yield from Rr.move(Rr.pos + [0, 0, 0.12], Rr.rot, 3.0)
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
                rk = st["rack"]
                msg += f" | rack {np.round(rk.pos, 3).tolist()} tilt {task.tilt_deg(rk):.1f} yaw {task.yaw_deg(rk):.1f} "
                msg += " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in inf.items())
                msg += ' cups ' + ' '.join(str(np.round(rk.to_local(st[c].pos)[:2] * 1000, 0).tolist()) for c in task.CUP_NAMES)
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
