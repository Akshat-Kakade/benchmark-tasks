"""Scripted full-score demo for the pickle-jar task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1800] [--no-record]

OPEN LOOP: every waypoint comes from the fixed layout in geometry.py (jar, lever tool and lid positions). The only feedback is each
arm's own joint positions (the measured TCP error is integrated into the command in free air); the demo never reads the jar, lid or tool
pose. The left arm grips the jar body from the side and holds it. The right arm picks up the lever tool by its grip fin, slides its blade
under the lid edge on the near side and levers it up (the seal breaks), puts the tool down, pinches the lid from above and
turns the wrist about 100 degrees (the lid unlatches against the jar held by the left arm), lifts the lid off, sets it on the table
and lets go; the left arm lets go last.
DEMO_VARIANT: settle, hover, tighten, right_only, left_only, no_tool, pop_only, lid_not_released.
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

    def feedback(self, on: bool, keep: bool = False):
        """Integrate the measured TCP error (free air). Off with keep=True freezes the correction (in contact, no windup)."""
        self.fb = on
        if not on and not keep:
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



sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as g  # noqa: E402

T = g.TOP
HOLD_TILT = math.radians(30)                 # the left hand comes in from the +y side, 30 deg below horizontal
HOLD_YAW = -math.pi / 2                      # finger axis along x (IK scan: this yaw reaches, +pi/2 does not)
HOLD_Z = 0.060                               # pinch height on the jar body
HOLD_DY = 0.0
PRISE_YAW = math.pi / 2                      # grip yaw for the fin: fingers open along x (across the 10 mm fin)
TIP_R = 0.0335       # blade tip 33.5 mm from the jar axis: under the 35 mm lid skirt, outside the 32.5 mm neck
TIP_Z = 0.110                                 # blade underside 5 mm above the shoulder ledge (ledge 105 mm, lid bottom 115 mm)
TIP_DX = -0.006   # measured: the left grip pulls the jar 5-10 mm toward the base in x
TIP_LIFT = 0.010                              # the tip is then lifted 10 mm: the lid edge goes up
LID_PINCH_Z = 0.130
TWIST = 0.9          # wrist yaw from -TWIST to +TWIST (rad): 103 deg for 0.9
SET_X, SET_Y = g.JAR_X, -0.17                # where the lid is set down (centre)


def hold_rot():
    return _rx(-math.pi / 2 + HOLD_TILT) @ _grip(HOLD_YAW)


def _grip(yaw):
    from yam_arena.demo import grip_rotation

    return grip_rotation(yaw)


def tool_rot(th):
    return _rx(-th) @ _grip(PRISE_YAW)


def tcp_for_tip(tip, th):
    """TCP at the grip fin for a lever whose tip is at ``tip`` and whose handle end is raised by pitch ``th`` (tool yawed 90 deg: tip toward +y)."""
    xt = np.array([0.0, math.cos(th), -math.sin(th)])
    zt = np.array([0.0, math.sin(th), math.cos(th)])
    return np.asarray(tip, float) - g.TIP_LOCAL[0] * xt + g.FIN_Z * zt


def script(task, ctx):
    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":
        yield from wait(8.0)
        return
    for _ in range(40):                          # let the jar and the tool settle on the table
        yield
    hold = hold_rot()
    appr = hold[:, 2]
    at_h = np.array([g.JAR_X, g.JAR_Y + HOLD_DY, T + HOLD_Z])
    r_fin = _grip(PRISE_YAW)
    tool_at = np.array([g.TOOL_X, g.TOOL_Y, T + g.FIN_Z])
    pinch = np.array([g.JAR_X, g.JAR_Y, T + LID_PINCH_Z])

    def grab_jar():
        L.feedback(True)
        yield from L.move(at_h - appr * 0.10, hold, 2.5, grip=1.0)
        yield from L.move(at_h, hold, 1.5)
        L.feedback(False, keep=True)
        yield from L.set_grip(0.0, 1.0)
        yield from wait(0.5)

    def pick_tool():
        Rr.feedback(True)
        yield from Rr.move(tool_at + [0, 0, 0.08], r_fin, 2.5, grip=1.0)
        yield from Rr.move(tool_at, r_fin, 1.2)
        Rr.feedback(False, keep=True)
        yield from Rr.set_grip(0.0, 0.8)
        yield from wait(0.4)
        yield from Rr.move(tool_at + [0, 0, 0.10], r_fin, 1.0)

    def prise():
        tip_out = np.array([g.JAR_X + TIP_DX, g.JAR_Y - 0.075, T + TIP_Z + 0.004])
        tip = np.array([g.JAR_X + TIP_DX, g.JAR_Y - TIP_R, T + TIP_Z])
        up = tip + [0, 0, 0.0 if variant == "hover" else TIP_LIFT]
        yield from Rr.move(tcp_for_tip(tip_out, 0.0), tool_rot(0.0), 3.0)
        yield from Rr.move(tcp_for_tip(tip + [0, -0.02, 0], 0.0), tool_rot(0.0), 1.5)
        yield from Rr.move(tcp_for_tip(tip, 0.0), tool_rot(0.0), 1.5)       # blade slid under the lid edge, on the ledge
        yield from wait(0.3)
        yield from Rr.move(tcp_for_tip(up, 0.0), tool_rot(0.0), 1.5)        # lever up: the blade lifts the lid edge
        yield from wait(1.0)
        yield from Rr.move(tcp_for_tip(up + [0, -0.04, 0], 0.0), tool_rot(0.0), 1.0)    # back out
        yield from Rr.move(tool_at + [0, 0, 0.08], r_fin, 2.5)
        yield from Rr.move(tool_at + [0, 0, 0.003], r_fin, 1.2)
        yield from Rr.set_grip(1.0, 0.8)
        yield from Rr.move(tool_at + [0, 0, 0.10], r_fin, 1.0)

    def twist():
        sgn = -1.0 if variant == "tighten" else 1.0      # the wrist turns anticlockwise (loosening); "tighten" turns it the wrong way
        r0 = _grip(-sgn * TWIST)
        yield from Rr.move(pinch + [0, 0, 0.07], r0, 2.5, grip=1.0)
        yield from Rr.move(pinch, r0, 1.2)
        Rr.feedback(False, keep=True)
        yield from Rr.set_grip(0.0, 0.8)
        yield from wait(0.4)
        n = 12
        for k in range(1, n + 1):
            yield from Rr.move(pinch, _grip(sgn * (-TWIST + 2 * TWIST * k / n)), 0.9 if variant == "right_only_slow" else 0.25)
        yield from wait(0.5)

    def lift_and_set_down(release=True):
        r1 = _grip(TWIST)
        yield from Rr.move(pinch + [0, 0, 0.03], r1, 1.0)
        yield from Rr.move(np.array([SET_X, SET_Y, T + 0.06]), r1, 2.5)
        yield from Rr.move(np.array([SET_X, SET_Y, T + 0.024]), r1, 1.5)
        yield from wait(0.5)
        if not release:
            yield from wait(3.0)
            return
        yield from Rr.set_grip(1.0, 0.8)
        yield from wait(0.3)
        yield from Rr.move(np.array([SET_X, SET_Y, T + 0.10]), r1, 1.5)
        yield from wait(0.5)

    def release_jar():
        yield from L.set_grip(1.0, 1.0)
        yield from wait(0.5)
        yield from L.move(at_h - appr * 0.12, hold, 2.0)
        yield from wait(2.5)

    if variant == "left_only":
        yield from grab_jar()
        yield from wait(8.0)
        return
    if not variant.startswith("right_only"):
        yield from grab_jar()
    if variant != "no_tool":
        yield from pick_tool()
        yield from prise()
        if variant in ("pop_only", "hover"):
            yield from wait(4.0)
            return
    yield from twist()
    if variant == "tighten":
        yield from wait(3.0)
        return
    yield from lift_and_set_down(release=variant != "lid_not_released")
    if variant == "lid_not_released":
        return
    if not variant.startswith("right_only"):
        yield from release_jar()
    else:
        yield from wait(3.0)


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
