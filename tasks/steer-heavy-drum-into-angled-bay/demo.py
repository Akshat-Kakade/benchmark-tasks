"""Scripted full-score demo for the angled drum bay, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1000] [--no-record]

OPEN LOOP. The script uses only the fixed layout (constants of task.py) and each arm's own joint angles (to trim its
hand onto the planned point). It never reads the drum's pose. Both hands, closed, go to the drum's two chimes at axle
height (right hand behind its chime, left hand in front of its chime) and turn the drum 35 degrees in place with a couple
(the right chime pushed forward, the left pushed back). The left hand then lifts over the drum and comes down behind its
chime, and both hands push the chimes forward at the same speed along the new heading: the drum rolls into the bay and
stops against the back bar. The hands open, back off and lift.

DEMO_VARIANT: "" full demo; "hold" settle (no motion, expect 0.0); "left_only" / "right_only" one hand does its part of
the plan alone; "alt" the right hand alone walks between the two chimes; "equal" no couple, both hands push straight
(no turn); "half_turn" the couple turns only half the angle; "no_release" the hands stay on the chimes.
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
PAD_HALF = 0.022           # closed hand: TCP to its face along the push direction (measured in the cabinet task)
GAP = 0.004                # pad starts this far behind the chime rim
PAD_Z = 0.000              # TCP height relative to the drum axle
PUSH_S = float(os.environ.get("DEMO_PUSH_S", "8.0"))         # seconds for the straight push
TURN_S = float(os.environ.get("DEMO_TURN_S", "8.0"))         # seconds for the turn in place
OVER = float(os.environ.get("DEMO_OVER", "0.0"))             # extra pad travel beyond the planned end (m)
CORR_GAIN = 0.12
CORR_MAX = 0.04
HAND_YAW = float(os.environ.get("DEMO_HAND_YAW", "0.0"))     # hand yaw follows this share of the heading change
CHIME_Y = 0.1425           # chime centre from the drum centre along its axis


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


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
        self.pe_mm = 0.0
        self.corr = np.zeros(3)
        self.measured = None        # callable returning this arm's measured joint angles (6,)

    def _solve(self, pos, rot):
        from yam_arena import ik

        if self.measured is not None and CORR_GAIN > 0:      # integrate the measured TCP error into the command
            act, _ = ik.tcp_pose(self.measured(), self.base)
            self.corr = np.clip(self.corr + CORR_GAIN * (self.pos - act), -CORR_MAX, CORR_MAX)
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.pe_mm = pe * 1000
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


def smooth(u: float) -> float:
    return 3 * u * u - 2 * u ** 3


def rim_point(task, sgn: int, c, psi: float, front: bool = False, behind: float = 0.0) -> np.ndarray:
    """Hand TCP (closed hand) touching the chime of end `sgn` (+1 left, -1 right) of a drum with centre c, heading psi:
    behind the chime's rim (pushing forward) or, with front=True, in front of it (pushing back), at axle height."""
    h = np.array([math.cos(psi), math.sin(psi)])
    a = np.array([-math.sin(psi), math.cos(psi)])
    off = (task.R_RING + PAD_HALF + GAP + behind) * (1.0 if front else -1.0)
    xy = np.asarray(c, float) + sgn * CHIME_Y * a + off * h
    return np.array([xy[0], xy[1], task.TOP + task.R_RING + PAD_Z])


def hand_rot(psi: float):
    from yam_arena.demo import grip_rotation

    return grip_rotation(HAND_YAW * psi)


def script(task, ctx):
    variant = os.environ.get("DEMO_VARIANT", "")
    R, L = ctx.right, ctx.left
    if variant == "hold":
        yield from wait(8.0)
        return
    S = np.array(task.START, float)
    full = math.radians(task.BAY_DEG)
    turn = {"half_turn": 0.5 * full, "equal": 0.0}.get(variant, full)
    d = np.array([math.cos(turn), math.sin(turn)])
    length = float(np.linalg.norm(np.array(task.BAY_C) - S))
    both = variant not in ("left_only", "right_only", "alt")
    use_l = variant != "right_only" and variant != "alt"
    use_r = variant != "left_only"
    hover = 0.07

    # ---- 1. hands closed. Right behind its chime, left in FRONT of its chime (it holds that end back) --------------------
    def right_at(c, psi, **kw):
        return rim_point(task, -1, c, psi, **kw)

    def left_front(c, psi):
        return rim_point(task, 1, c, psi, front=True)

    if variant == "alt":
        yield from alt_walk(task, ctx, S)
        return
    if use_r:
        yield from R.move(right_at(S, 0.0, behind=0.02) + [0, 0, hover], hand_rot(0.0), 3.0, grip=0.0)
    if use_l and turn > 0:
        yield from L.move(left_front(S, 0.0) + [0.02, 0, hover], hand_rot(0.0), 3.0, grip=0.0)
    elif use_l:
        yield from L.move(rim_point(task, 1, S, 0.0, behind=0.02) + [0, 0, hover], hand_rot(0.0), 3.0, grip=0.0)
    if use_r:
        yield from R.move(right_at(S, 0.0), hand_rot(0.0), 2.0)
    if use_l and turn > 0:
        yield from L.move(left_front(S, 0.0), hand_rot(0.0), 2.0)
    elif use_l:
        yield from L.move(rim_point(task, 1, S, 0.0), hand_rot(0.0), 2.0)
    yield from wait(0.5)

    # ---- 2. turn in place: the right hand pushes its end forward, the left hand pushes its end back (a couple) ----------
    if turn > 0:
        steps = int(TURN_S * CONTROL_HZ)
        for k in range(1, steps + 1):
            psi = turn * smooth(k / steps)
            if use_r:
                R.aim(right_at(S, psi), hand_rot(psi))
            if use_l:
                L.aim(left_front(S, psi), hand_rot(psi))
            yield
        yield from wait(0.5)
        # ---- 3. the left hand lets go, lifts over the drum and comes down behind its chime ---------------------------------
        if use_l:
            yield from L.set_grip(1.0, 0.5)
            yield from L.move(L.pos + [0, 0, 0.17], hand_rot(turn), 1.5)
            yield from L.move(rim_point(task, 1, S, turn, behind=0.02) + [0, 0, 0.17], hand_rot(turn), 2.5)
            yield from L.move(rim_point(task, 1, S, turn, behind=0.0), hand_rot(turn), 1.5, grip=0.0)
            yield from wait(0.5)

    # ---- 4. push both chimes along the heading at the same speed: the drum rolls straight into the bay ---------------
    steps = int(PUSH_S * CONTROL_HZ)
    for k in range(1, steps + 1):
        c = S + smooth(k / steps) * length * d
        if use_r:
            R.aim(right_at(c, turn, behind=-OVER * k / steps), hand_rot(turn))
        if use_l:
            L.aim(rim_point(task, 1, c, turn, behind=-OVER * k / steps), hand_rot(turn))
        yield
    yield from wait(1.0)
    last = S + length * d

    # ---- 5. release (open), back off along the heading, lift ---------------------------------------------------------
    if variant != "no_release":
        hands = [(sgn, arm) for sgn, arm in ((-1, R), (1, L)) if (sgn == -1 and use_r) or (sgn == 1 and use_l)]
        for _, arm in hands:
            yield from arm.set_grip(1.0, 0.8)
        for sgn, arm in hands:
            yield from arm.move(rim_point(task, sgn, last, turn, behind=0.06), hand_rot(turn), 1.5)
        for _, arm in hands:
            yield from arm.move(arm.pos + [0, 0, 0.10], hand_rot(turn), 1.5)
    yield from wait(3.0)


def alt_walk(task, ctx, S):
    """One hand (the right) alone: pushes the right chime forward, hops over to push the left chime forward, and so on."""
    R = ctx.right
    c = np.array(S, float)
    yield from R.move(rim_point(task, -1, c, 0.0, behind=0.02) + [0, 0, 0.07], hand_rot(0.0), 3.0, grip=0.0)
    for cycle in range(3):
        for sgn in (-1, 1):
            tip = rim_point(task, sgn, c, 0.0)
            if cycle or sgn == 1:
                yield from R.move(R.pos + [0, 0, 0.17], hand_rot(0.0), 1.5)
                yield from R.move(rim_point(task, sgn, c, 0.0, behind=0.02) + [0, 0, 0.17], hand_rot(0.0), 2.5)
            yield from R.move(rim_point(task, sgn, c, 0.0), hand_rot(0.0), 1.5)
            steps = int(2.0 * CONTROL_HZ)
            for k in range(1, steps + 1):
                R.aim(rim_point(task, sgn, c + [0.03 * smooth(k / steps), 0.0], 0.0), hand_rot(0.0))
                yield
            c = c + [0.03, 0.0]
    yield from R.move(R.pos + [0, 0, 0.10], hand_rot(0.0), 1.5)
    yield from wait(3.0)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.q14 = np.zeros(14)
        self.left.measured = lambda: self.q14[0:6]
        self.right.measured = lambda: self.q14[7:13]

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1000, record: bool = True, cam_width: int = 320,
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

    ctx.q14 = np.asarray(next(iter(obs.state.values())), float)
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
        ctx.q14 = np.asarray(next(iter(res.observation.state.values())), float)
        snap(res.observation)
        if steps % 100 == 0:
            print(f"step {steps} score {info.get('score')} ik_mm L {ctx.left.pe_mm:.1f} R {ctx.right.pe_mm:.1f}", flush=True)
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
            imageio.imwrite(out / "setup.png", frames[top][min(20, len(frames[top]) - 1)])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=320)
    ap.add_argument("--cam-height", type=int, default=176)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
