"""Scripted full-score demo for the table-saw rip task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1500] [--no-record]

Open loop in the fixed layout (task constants); the only feedback is each arm's own joint positions (proprioception: the
measured TCP is integrated into the command), never an object pose. The right arm takes the push stick by its grip fin
and pushes the board from behind with the forked heel; the left arm presses the board's outer edge toward the fence for
the first 60 mm (the entry), then lifts away. DEMO_VARIANT: settle, probe, one_right, one_left, bare_hand, early_lift, lift_over, no_release.
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
import task as T  # noqa: E402
os.environ.pop("SAW_NO_HOOKS")

SPEED_ENTRY, SPEED = 0.020, 0.035      # push speeds (m/s): the first 60 mm (the board enters the blade) is slower
HAND_BEHIND = -T.FIN_Y                 # fin is this far behind the heel's front face
CARRY_DZ = 0.007                       # lift of the stick (heel bottom 7 mm above the saw table)
FIN_Z = T.SAW_TOP + 0.0005 + T.OZ + T.BAR_Z1 + T.FIN[2] / 2 - T.HEEL_Z0 - T.OZ   # fin centre height on the station table
REST_FACE_Y = T.STICK_START[1]
READY_FACE_Y = T.BOARD_REAR0 - 0.030   # heel front face just behind the board's rear end
END_FACE_Y = T.CUT_DONE + 0.012
LEFT_X_PRESS = float(os.environ.get("LEFT_X", "0.204"))       # hold-down pad over the board, straddling the kerf, next to the fence
XS = T.BLADE_X + float(os.environ.get("STICK_DX", "0.0"))   # the stick line, shifted toward the fence
LEFT_X_RIDE = float(os.environ.get("LEFT_X_RIDE", "0.204"))   # the pad slides the board to the fence, then rides at this x
LEFT_Y0 = T.BOARD_REAR0 + 0.015
LEFT_Y1 = LEFT_Y0 + (T.LOAD_END_REAR - T.BOARD_REAR0)         # the hand rides with the rear end until the loads end
LEFT_Z = T.TOP + 0.030 + 0.019 + 0.013 - float(os.environ.get("PRESS_MM", "10")) / 1000   # pad bottom (13 mm below TCP) this far below the board top


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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


def home(arm, duration: float):
    """Return an arm to its ready pose (joint-space blend), which keeps the hand 180 mm above the table, clear of the blade zone."""
    from yam_arena import ik
    from yam_arena.constants import READY

    q0, q1, g0 = arm.q.copy(), np.array(READY, float), arm.grip
    n = max(1, round(duration * CONTROL_HZ))
    for k in range(1, n + 1):
        s_ = ik.smoothstep(k / n)
        arm.q = q0 + s_ * (q1 - q0)
        arm.grip = g0 + s_ * (1.0 - g0)
        arm.cmd = np.r_[arm.q, arm.grip]
        arm.pos, arm.rot = ik.tcp_pose(arm.q, arm.base)
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

    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":
        yield from wait(8.0)
        return
    for _ in range(12):
        yield
    fin_rest = np.array([T.STICK_START[0], REST_FACE_Y + T.FIN_Y, FIN_Z])
    yaw_r = float(os.environ["YAW_R"]) * math.pi / 2 if os.environ.get("YAW_R") else reachable_yaw(Rr, fin_rest + [0, 0, CARRY_DZ], (0.0, math.pi))
    yaw_l = float(os.environ["YAW_L"]) * math.pi / 2 if os.environ.get("YAW_L") else reachable_yaw(L, np.array([LEFT_X_PRESS, LEFT_Y0, LEFT_Z]), (math.pi / 2, -math.pi / 2))
    gr, gl = grip_rotation(yaw_r), grip_rotation(yaw_l)
    print("YAW", yaw_r, yaw_l, flush=True)
    if variant == "probe":          # IK / reach check along the whole path, no contact
        pts_r = [fin_rest, fin_rest + [0, 0, CARRY_DZ], np.array([XS, READY_FACE_Y + T.FIN_Y, FIN_Z + CARRY_DZ]),
                 np.array([XS, END_FACE_Y + T.FIN_Y, FIN_Z + CARRY_DZ])]
        for p in pts_r:
            yield from Rr.move(p, gr, 1.5, grip=1.0)
            print("PROBE R", np.round(p, 3).tolist(), "worst_mm", round(Rr.worst_mm, 1), flush=True)
        for p in [(0.33, LEFT_Y0, LEFT_Z + 0.05), (0.33, LEFT_Y0, LEFT_Z), (LEFT_X_PRESS, LEFT_Y0, LEFT_Z), (LEFT_X_PRESS, LEFT_Y1, LEFT_Z),
                  (0.36, LEFT_Y1, LEFT_Z + 0.08)]:
            L.worst_mm = 0.0
            yield from L.move(np.array(p), gl, 1.5, grip=0.0)
            print("PROBE L", np.round(p, 3).tolist(), "worst_mm", round(L.worst_mm, 1), flush=True)
        return

    def grab():
        yield from Rr.move(fin_rest + [0, 0, 0.09], gr, 3.0, grip=1.0)
        yield from Rr.move(fin_rest, gr, 1.5)
        yield from Rr.set_grip(0.0, 1.0)
        yield from wait(0.4)
        Rr.gain, Rr.cap = 0.06, 0.02
        Rr.feedback(True)
        yield from Rr.move(fin_rest + [0, 0, CARRY_DZ], gr, 1.2)
        ready = np.array([XS, READY_FACE_Y + T.FIN_Y, FIN_Z + CARRY_DZ])
        stage = ready - [0, 0.06, 0]
        yield from Rr.move(stage, gr, 3.0)
        yield from wait(1.0)
        yield from Rr.move(ready, gr, 3.0)
        yield from wait(1.0)

    def left_prepare():
        yield from L.move(np.array([0.25, -0.03, T.SAW_TOP + 0.20]), gl, 3.5, grip=0.0)       # high over the blade zone (> 150 mm)
        yield from L.move(np.array([LEFT_X_PRESS, LEFT_Y0, T.SAW_TOP + 0.20]), gl, 2.0)
        yield from L.move(np.array([LEFT_X_PRESS, LEFT_Y0, LEFT_Z + 0.008]), gl, 2.0)
        yield from wait(0.5)

    if variant == "one_left":       # left arm alone: a bare closed hand pushes the board from its rear end (no stick)
        yield from L.move(np.array([T.OFF_CX, T.BOARD_REAR0 - 0.05, LEFT_Z + 0.09]), gl, 3.0, grip=0.0)
        yield from L.move(np.array([T.OFF_CX, T.BOARD_REAR0 - 0.05, LEFT_Z]), gl, 1.5)
        L.feedback(True)
        y = T.BOARD_REAR0 - 0.05
        for k in range(int(30 * CONTROL_HZ)):
            y += 0.03 / CONTROL_HZ
            L.aim(np.array([T.OFF_CX, y, LEFT_Z]), gl)
            yield
        return
    if variant == "bare_hand":      # both arms: left holds, right pushes with a bare closed hand behind the board
        yield from par(left_prepare(), Rr.move(np.array([T.STRIP_CX + 0.02, T.BOARD_REAR0 - 0.06, FIN_Z + 0.05 + CARRY_DZ]), gr, 4.0, grip=0.0))
        yield from Rr.move(np.array([T.STRIP_CX + 0.02, T.BOARD_REAR0 - 0.06, T.TOP + 0.045]), gr, 1.5)
        Rr.feedback(True)
        yield from L.move(np.array([LEFT_X_PRESS, LEFT_Y0, LEFT_Z]), gl, 1.5)
        y = T.BOARD_REAR0 - 0.06
        for k in range(int(25 * CONTROL_HZ)):
            y += SPEED / CONTROL_HZ
            Rr.aim(np.array([T.STRIP_CX + 0.02, y, T.TOP + 0.045]), gr)
            if variant != "bare_hand" or y < LEFT_Y1 - 0.03:
                L.aim(np.array([LEFT_X_PRESS, LEFT_Y0 + (y - (T.BOARD_REAR0 - 0.06)), LEFT_Z]), gl)
            yield
        return
    if variant == "one_right":
        yield from grab()
    else:
        yield from par(grab(), left_prepare())
    # -- push, the left arm presses the board edge toward the fence over the first 60 mm ---------------------------------
    if variant != "one_right":
        L.feedback(False)
        yield from L.move(np.array([LEFT_X_PRESS, LEFT_Y0, LEFT_Z]), gl, 2.0)
        yield from wait(0.5)
        if LEFT_X_RIDE != LEFT_X_PRESS:            # hold the board to the fence: the pad drags it toward the fence
            yield from L.move(np.array([LEFT_X_RIDE, LEFT_Y0, LEFT_Z]), gl, 1.5)
            yield from wait(0.5)
    face = READY_FACE_Y
    lift_at = LEFT_Y1 if variant != "early_lift" else LEFT_Y0 + 0.01
    left_gen = None
    left_done = variant == "one_right"

    def left_away():
        L.feedback(False)
        yield from L.move(np.array([LEFT_X_RIDE, L.pos[1], T.SAW_TOP + 0.22]), gl, 2.0, grip=1.0)      # straight up above 150 mm before going home
        yield from home(L, 3.0)

    def right_away():
        yield from wait(1.5)
        yield from Rr.set_grip(1.0, 1.0)
        yield from Rr.move(Rr.pos + [0, 0, 0.04], gr, 1.0)
        yield from Rr.move(np.array([XS, Rr.pos[1] - 0.12, Rr.pos[2] + 0.04]), gr, 2.0)

    k = 0
    while face < END_FACE_Y:
        face += (SPEED_ENTRY if face - READY_FACE_Y < 0.060 else SPEED) / CONTROL_HZ
        Rr.aim(np.array([XS, face + T.FIN_Y, FIN_Z + CARRY_DZ]), gr)
        if left_gen is not None:
            try:
                next(left_gen)
            except StopIteration:
                left_gen = None
        elif not left_done:
            ly = LEFT_Y0 + max(0.0, face - READY_FACE_Y - 0.030)
            if ly < lift_at:
                L.aim(np.array([LEFT_X_RIDE, ly, LEFT_Z]), gl)
            else:
                left_done = True
                left_gen = left_away()
        if variant == "lift_over":       # bypass test: the board is teleported 25 mm up while it passes over the blade (no arm can do this)
            from partkit import write_pose

            bs = ctx.states()["board"]
            if bs.high[1] > T.BLADE_Y0 + 0.008 and bs.low[1] < T.BLADE_Y1 + 0.010:
                write_pose(ctx.emb._env, "board", (bs.pos[0], bs.pos[1], bs.pos[2] + (0.025 if bs.pos[2] < T.SAW_TOP + 0.02 else 0.0)), bs.quat_xyzw,
                           bs.lin_vel, (0, 0, 0))
        k += 1
        if k % 40 == 0 and os.environ.get("DEMO_DEBUG"):
            print("PUSH face", round(face, 3), flush=True)
        yield
    if left_gen is not None:
        for _ in left_gen:
            yield

    if variant == "one_right":
        yield from right_away()
    else:
        yield from right_away()
    yield from wait(4.0)
    st = ctx.states()
    print("FINAL", {k: np.round(np.asarray(st[k].pos), 4).tolist() for k in ("strip", "offcut", "stick")}, "blade_off", task._STATE.get("blade_boxes_off"), "w", task._STATE.get("wmin"), task._STATE.get("wmax"), flush=True)


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
                msg += f" | held {task._STATE['held']} gate {task._STATE['gate']} " + ', '.join(f'{k} {round(v,3) if isinstance(v,float) else v}' for k, v in inf.items() if k in ('rear','front','zone','hand_piece','tilt','speed','lifted','why'))
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
