"""Scripted full-score demo for the table-saw crosscut task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1500] [--no-record]

Open loop in the fixed layout (task constants); the only feedback is each arm's own joint positions (proprioception: the measured TCP is
integrated into the command), never an object pose. Each arm takes its push stick by the grip fin and pushes its half of the plank's rear edge
through the blade, both at the same speed, one on each side of the blade. DEMO_VARIANT: settle, one_right, one_left, shuffle, lift_over.
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
        self.cap = 0.04

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
                self.corr = np.clip(self.corr + self.gain * (self.pos - pa), -self.cap, self.cap)

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



os.environ["SAW_NO_HOOKS"] = "1"        # constants only: this copy of task.py must not register simulator hooks
os.environ["SAW_NO_HOOKS"] = "1"        # constants only: this copy of task.py must not register simulator hooks
import task as T  # noqa: E402
os.environ.pop("SAW_NO_HOOKS")

SPEED_ENTRY, SPEED = 0.020, 0.035      # push speeds (m/s): the first 60 mm (the plank meets the blade) is slower
HAND_BEHIND = -T.FIN_X                 # fin is this far behind the heel's front face
CARRY_DZ = 0.007                       # lift of the stick (heel bottom 7 mm above the saw table)
FIN_Z = T.SAW_TOP + 0.0005 + T.BAR_Z1 + T.FIN[2] / 2 - T.HEEL_Z0   # fin centre height
READY_FACE_X = T.SLED_FACE0 - T.FENCE_X - 0.030   # heel front face just behind the gauge fence's rear face
END_FACE_X = T.CUT_DONE + 0.012 - T.FENCE_X       # the gauge fence front (the plank's rear edge) is 12 mm past the cut-complete line


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


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


def reachable_yaw(arm, pos, cands):
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    best = None
    for y in cands:
        _, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(y)), arm.q, base_pos=arm.base, iters=200)
        if best is None or pe + 0.05 * re < best[0]:
            best = (pe + 0.05 * re, y)
    return best[1]


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":
        yield from wait(8.0)
        return
    for _ in range(12):
        yield
    if variant == "lift_over":     # bypass test: an upward force on the plank (2x its weight) stands for lifting it over the blade; no arm can do this
        os.environ["SAW_TEST_LIFT"] = "9.0"
    arms = {"R": (ctx.right, -T.STICK_Y), "L": (ctx.left, T.STICK_Y)}
    use = {"one_right": ("R",), "one_left": ("L",)}.get(variant, ("R", "L"))
    rots = {}
    for k in use:
        arm, y = arms[k]
        fin_rest = np.array([T.STICK_FACE_REST - HAND_BEHIND, y, FIN_Z])
        rots[k] = grip_rotation(0.0)      # fingers open along y: pinch the fin; reach-checked offline over the whole push (<= 0.1 mm)

    def grab(k):
        arm, y = arms[k]
        gr = rots[k]
        fin_rest = np.array([T.STICK_FACE_REST - HAND_BEHIND, y, FIN_Z])
        yield from arm.move(fin_rest + [0, 0, 0.09], gr, 3.0, grip=1.0)
        yield from arm.move(fin_rest, gr, 1.5)
        yield from arm.set_grip(0.0, 1.0)
        yield from wait(0.4)
        arm.gain, arm.cap = 0.06, 0.02
        arm.feedback(True)
        yield from arm.move(fin_rest + [0, 0, CARRY_DZ], gr, 1.2)
        ready = np.array([READY_FACE_X - HAND_BEHIND, y, FIN_Z + CARRY_DZ])
        yield from arm.move(ready - [0.03, 0, 0], gr, 2.0)
        yield from wait(0.5)
        yield from arm.move(ready, gr, 1.5)
        yield from wait(0.5)

    yield from par(*[grab(k) for k in use])

    def aim_all(face):
        for k in use:
            arm, y = arms[k]
            arm.aim(np.array([face - HAND_BEHIND, y, FIN_Z + CARRY_DZ]), rots[k])

    face = READY_FACE_X
    if variant == "shuffle":      # back-and-forth over the same 20 mm region of the cut: the cut front must not add up
        lo = T.BLADE_X0 - T.PLANK_B - 0.005 + 0.012 - T.FENCE_X      # heel face where the plank front is 12 mm past the blade's leading edge
        hi = lo + 0.020
        while face < lo:
            face += SPEED_ENTRY / CONTROL_HZ
            aim_all(face)
            yield
        for _ in range(8):
            for tgt in (hi, lo):
                while abs(face - tgt) > 1e-4:
                    face += math.copysign(min(SPEED / CONTROL_HZ, abs(tgt - face)), tgt - face)
                    aim_all(face)
                    yield
        for _ in range(int(2 * CONTROL_HZ)):
            yield
    else:
        while face < END_FACE_X:
            face += (SPEED_ENTRY if face - READY_FACE_X < 0.060 else SPEED) / CONTROL_HZ
            aim_all(face)
            yield

    def away(k):
        arm, y = arms[k]
        gr = rots[k]
        yield from wait(1.5)
        yield from arm.set_grip(1.0, 1.0)
        yield from arm.move(arm.pos + [0, 0, 0.04], gr, 1.0)
        yield from arm.move(np.array([arm.pos[0] - 0.12, y, arm.pos[2] + 0.04]), gr, 2.0)

    yield from par(*[away(k) for k in use])
    yield from wait(4.0)
    st = ctx.states()
    print("FINAL", {k: np.round(np.asarray(st[k].pos), 4).tolist() for k in ("board", "half_l", "half_r")}, flush=True)


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
                inf = task._STATE.get('info', {})
                msg += f" | held {task._STATE['held']} gate {task._STATE['gate']} " + ', '.join(f'{k} {round(v,3) if isinstance(v,float) else v}' for k, v in inf.items() if k in ('rear','front','zone','hand_piece','tilt','speed','lifted','why','kerf','wander'))
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
