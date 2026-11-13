"""Open-loop scripted demo for the braced sheet-pan insertion, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1000] [--no-record]

Fixed layout, no object pose reads: waypoints come from the task's constants (rack and counter positions) and the only
feedback is each arm's own measured joint angles (the command is nudged so the measured gripper pose matches the plan).
The left hand pinches the rack's front-left post and holds it (closed, not moving) from before the push until after
the right hand has let go; the right hand pinches the pan's rear grab tab, lifts the pan to rib height, slides it in
until it meets the rear stop, sets it on the ribs, opens and withdraws; the left hand lets go last.

DEMO_VARIANT: "" full; "one_arm" right hand only (rack rolls away: 0.0); "brace_only" left hand only (0.0);
"slot1" / "slot3" pan carried to the wrong slot (0.5); "half_in" push only half way (0.5); "no_release" left hand never
lets go of the post (0.75); "hold" do nothing (0.0).
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
CORR_GAIN = 0.12           # nudge the command so the measured TCP (from the arm's own joint angles) follows the plan
CORR_MAX = 0.04
YAW = math.pi / 2          # fingers open along x: pinch the tab / post across x
OPEN = 1.0
CARRY_CLEAR = 0.012        # pan lower face above the rib top while sliding in
LIFT_S, PUSH_S = 2.5, 9.0
PITCH_DEG = float(os.environ.get('PITCH_DEG', 6.0))   # the pan sags nose-down in the pinch by about this much: the hand is pre-tilted
PRESS = 0.0015             # the hand holds the pan this far into the ribs (their springs give) so it rides on them
LEFT_DX = 0.008            # the left jaws close about 8 mm behind the TCP: aim at the post accordingly
PUSH_OVER = 0.004          # command this far past the stop position


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


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    slot = {"slot1": 0, "slot3": 2}.get(variant, 1)
    R, L = ctx.right, ctx.left
    rot = grip_rotation(YAW)
    c, s_ = math.cos(math.radians(PITCH_DEG)), math.sin(math.radians(PITCH_DEG))
    rot_r = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]]) @ rot     # right hand, pitched nose-up
    if variant == "hold":
        yield from wait(8.0)
        return
    ztop = T.TOP
    # pan rear tab on the counter (tab centre, pinch height) and where it ends when the pan meets the rear stop
    tab_x0 = T.TRAY_X - T.TD / 2 + T.TAB_T / 2
    tab_xend = T.RX + T.RACK_DEPTH - T.TD + T.TAB_T / 2
    zpan0 = ztop + T.COUNTER_T + 0.001
    pinch0 = np.array([tab_x0, T.RY, zpan0 + T.TAB_PINCH_Z])
    rib = ztop + T.SLOT_TOP[slot]
    carry_dz = (rib + CARRY_CLEAR) - zpan0
    # rack's front-left post (the upright the left hand holds), pinched across x
    post = np.array([T.RX + T.POST / 2 + LEFT_DX, T.RY + T.YI + T.PANEL_T + T.POST / 2, ztop + 0.175])

    do_left = variant not in ("one_arm",)
    do_right = variant not in ("brace_only",)

    # ---- 1. both hands to their grips -------------------------------------------------------------------------------
    if do_left:
        yield from L.move(post + [0, 0, 0.10], rot, 3.0, grip=OPEN)
        yield from L.move(post, rot, 1.5)
        yield from L.set_grip(0.0, 1.0)
    if not do_right:
        yield from wait(10.0)
        return
    yield from R.move(pinch0 + [0, 0, 0.08], rot, 3.0, grip=OPEN)
    yield from R.move(pinch0, rot, 1.5)
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.5)
    # ---- 2. lift the pan to rib height, tilt-free ------------------------------------------------------------------
    yield from R.move(pinch0 + [0, 0, carry_dz], rot_r, LIFT_S)
    yield from wait(0.5)
    # ---- 3. slide the pan in ----------------------------------------------------------------------------------------
    travel = tab_xend - tab_x0 + PUSH_OVER
    if variant == "half_in":
        travel = 0.5 * (tab_xend - tab_x0)
    carry = pinch0 + [0, 0, carry_dz]
    first = min(0.100, travel)            # in open air over the ribs until the pan's front edge is inside the mouth
    yield from R.move(carry + [first, 0, 0], rot_r, 0.3 * PUSH_S)
    if travel > first:
        low = carry[2] - CARRY_CLEAR - PRESS                # then settle onto the ribs while sliding on, and slide on them
        yield from R.move(np.array([carry[0] + first + 0.060, carry[1], low]), rot_r, 0.3 * PUSH_S)
        yield from R.move(np.array([carry[0] + travel, carry[1], low]), rot_r, 0.4 * PUSH_S)
    yield from wait(0.5)
    # ---- 4. set it on the ribs, let go, withdraw straight up then back -----------------------------------------------
    yield from wait(0.5)
    yield from R.set_grip(1.0, 1.0)
    yield from wait(0.3)
    yield from R.move(R.pos + [0, 0, 0.06], rot_r, 1.5)
    yield from R.move(R.pos + [-0.12, 0, 0.0], rot_r, 2.0)
    yield from wait(1.0)
    if do_left and variant != "no_release":
        yield from L.set_grip(1.0, 2.0)
        yield from wait(0.5)
        yield from L.move(L.pos + [0, 0, 0.09], rot, 2.0)
    yield from wait(2.5)


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
            print(f"step {steps} score {info.get('score')}", flush=True)
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
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
