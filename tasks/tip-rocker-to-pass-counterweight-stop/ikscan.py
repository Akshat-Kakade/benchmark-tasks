"""IK reach check of every waypoint the demo uses (no simulator needed, but needs yam_arena on the path):
    python ikscan.py
Prints the worst position/rotation error per arm and yaw choice; < 1 mm and < 0.02 rad is a pass."""

from __future__ import annotations

import math
import sys

import numpy as np

import geometry as g


def solve(side, pos, yaw, q_init=None):
    from yam_arena import ik, station
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation

    q0 = np.array(READY, float) if q_init is None else q_init
    q, err, rot = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(yaw)), q_init=q0, base_pos=station.BASES[side],
                           iters=200)
    return q, err * 1000, rot


def scan_left(yaw):
    worst, q = 0.0, None
    for th in np.radians(np.linspace(g.STOP_DEG, -g.STOP_DEG - 2, 9)):
        for dz in (0.0, 0.07):
            p = g.left_stem(th) + [0, 0, dz]
            q, e, r = solve("left", p, yaw, q)
            worst = max(worst, e, r * 100)
    return worst


def scan_right(mode):
    worst, q, rows = 0.0, None, []
    th = -math.radians(g.STOP_DEG)
    for k, psi in enumerate(np.radians(np.linspace(0, 190, 20))):
        p = g.weight_stem(th, psi)
        yaw = {"follow": psi - math.pi / 2, "follow_pi": psi + math.pi / 2, "fixed_0": 0.0,
               "fixed_pi2": math.pi / 2, "fixed_m": -math.pi / 2}[mode]
        for dz in ((0.0, 0.07) if k == 0 else (0.0,)):
            q, e, r = solve("right", p + [0, 0, dz], yaw, q)
            worst = max(worst, e, r * 100)
            rows.append((round(math.degrees(psi)), round(e, 2), round(r, 3)))
    return worst, rows


if __name__ == "__main__":
    for yaw in (0.0, math.pi, math.pi / 2, -math.pi / 2):
        print(f"left yaw {math.degrees(yaw):5.0f}: worst (mm or 0.01rad) {scan_left(yaw):.2f}")
    for mode in ("follow", "follow_pi", "fixed_0", "fixed_pi2", "fixed_m"):
        w, rows = scan_right(mode)
        print(f"right {mode}: worst {w:.2f}", rows[:3] if w < 1 else [r for r in rows if r[1] > 1 or r[2] > 0.02][:4])
    p = g.weight_stem(-math.radians(g.STOP_DEG), 0.0)
    print("weight stem at ψ=0 tipped:", p.round(3), "left stem start:", g.left_stem(g.STOP).round(3),
          "tipped:", g.left_stem(-g.STOP).round(3))
    sys.exit(0)
