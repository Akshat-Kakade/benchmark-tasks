"""Scripted full-score demo for the hotel-pan task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 900] [--no-record]

OPEN LOOP: the waypoints come from the fixed layout in task.py (pan start, handle height, well position). The only feedback is
each arm's own joint positions (the measured TCP error is integrated into the command), never the pan or well pose.
Each arm pinches a handle tab (fingers open along x so the jaws resist pitch), both lift the pan 60 mm, carry it 0.315 m to the well, lower it, release and withdraw.
DEMO_VARIANT: settle, spill, spill2, one_arm, tilt, jam, drop, no_release, stop_above, off_alt, off_alt2, slow.
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



LIFT = 0.060               # pan carry height above its start (underside 60 mm above the table, 20 mm over the counter)
SLOW = 0.010               # m/s during the last 25 mm of the descent


def script(task, ctx):
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    L, Rr = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant == "settle":
        yield from wait(8.0)
        return
    for _ in range(40):                          # let the pan settle on the table
        yield
    hy = task.HANDLE_Y
    z_h = task.TOP + 0.0005 + task.PINCH_Z       # TCP height at the handle bar centre, pan on the table
    x0, xw = task.PAN_X0, task.WELL_X
    x_bias = 0.004                               # measured: the carried pan sits 4.6 mm behind the commanded x (pitch of the 65 mm tall pinch)
    off = {"off_alt": (0.003, 0.004), "off_alt2": (-0.003, -0.004), "jam": (-0.015, 0.0)}.get(variant, (0.0, 0.0))
    z_seat = task.SEAT_Z + task.PINCH_Z - 0.002  # command 2 mm below the seated handle height (arm compliance)
    z_carry = z_h + LIFT
    ypos = lambda s, x, z: np.array([x, s * hy, z])
    yaws = []
    for arm, s in ((L, 1), (Rr, -1)):
        best = None
        for yw in (math.pi / 2, -math.pi / 2):
            err = 0.0
            for (x, z) in ((x0, z_h), (x0, z_carry), (xw, z_carry), (xw, z_seat)):
                _, pe, re = ik.solve(ypos(s, x, z), ik.matrix_to_quat_xyzw(grip_rotation(yw)), arm.q, base_pos=arm.base, iters=200)
                err = max(err, pe + 0.05 * re)
            if best is None or err < best[0]:
                best = (err, yw)
        yaws.append(best[1])
    gl, gr = grip_rotation(yaws[0]), grip_rotation(yaws[1])
    print("YAW", yaws, flush=True)

    def grab(arm, s, g):
        h = ypos(s, x0, z_h)
        yield from arm.move(h + [0, 0, 0.07], g, 2.5, grip=1.0)
        yield from arm.move(h, g, 1.5)
        yield from arm.set_grip(0.0, 1.0)
        yield from wait(0.5)

    if variant == "one_arm":
        yield from par(grab(L, 1, gl), wait(1.0))
        L.feedback(True)
        yield from L.move(L.pos + [0, 0, LIFT], gl, 2.0)
        yield from L.move(ypos(1, xw, z_carry), gl, 5.0)
        yield from wait(3.0)
        return
    yield from par(grab(L, 1, gl), grab(Rr, -1, gr))
    L.feedback(True)
    Rr.feedback(True)
    yield from par(L.move(ypos(1, x0, z_carry), gl, 2.5), Rr.move(ypos(-1, x0, z_carry), gr, 2.5))
    yield from wait(0.5)
    if variant == "spill":                       # one side raised 45 mm in the air (7.6 deg about x), then straight on as normal
        yield from L.move(L.pos + [0, 0, 0.045], gl, 1.5)
        yield from L.move(L.pos - [0, 0, 0.045], gl, 1.5)
    # carry over the well (one smooth move per arm, same timing)
    dzc = 0.02 if variant == "spill2" else 0.0   # spill2: left side 20 mm high during the carry (3.5 deg roll, alone harmless; with the carry's 4-5 deg pitch the low corner overflows)
    yield from par(L.move(ypos(1, xw + x_bias + off[0], z_carry + dzc) + [0, off[1], 0], gl, 6.0),
                   Rr.move(ypos(-1, xw + x_bias + off[0], z_carry) + [0, off[1], 0], gr, 6.0))
    yield from wait(2.0)
    if variant == "stop_above":
        yield from wait(5.0)
        return
    if variant == "drop":                        # lets go over the well without lowering
        yield from par(L.set_grip(1.0, 0.5), Rr.set_grip(1.0, 0.5))
        yield from wait(4.0)
        return
    # lower: fast to 25 mm above the seat, then slowly
    z_mid = z_seat + 0.025
    d1 = (z_carry - z_mid) / 0.020
    d2 = (z_mid - z_seat) / SLOW
    dz_l = 0.05 if variant == "tilt" else 0.0    # tilt variant: the left arm lowers 50 mm less
    yield from par(L.move(ypos(1, xw + x_bias + off[0], z_mid + dz_l) + [0, off[1], 0], gl, d1),
                   Rr.move(ypos(-1, xw + x_bias + off[0], z_mid) + [0, off[1], 0], gr, d1))
    yield from par(L.move(ypos(1, xw + x_bias + off[0], z_seat + dz_l) + [0, off[1], 0], gl, d2),
                   Rr.move(ypos(-1, xw + x_bias + off[0], z_seat) + [0, off[1], 0], gr, d2))
    yield from wait(1.5)
    if variant == "no_release":
        yield from wait(6.0)
        return
    yield from par(L.set_grip(1.0, 1.0), Rr.set_grip(1.0, 1.0))
    yield from wait(0.5)
    yield from par(L.move(L.pos + [0, 0, 0.08], gl, 2.0), Rr.move(Rr.pos + [0, 0, 0.08], gr, 2.0))
    yield from wait(2.0)


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


def run(task_path: str, out_dir: str, max_steps: int = 1100, record: bool = True, cam_width: int = 320,
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
    ap.add_argument("--max-steps", type=int, default=1100)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
