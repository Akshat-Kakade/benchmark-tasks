"""Offline reach check of the whole planned path for both arms (needs yam_arena kinematics; run on the VM).
    python ikscan.py [A_X A_Y]"""
import math
import sys

import numpy as np

import geom
import task as T
from yam_arena import ik, station
from yam_arena.constants import READY
from yam_arena.demo import grip_rotation

LIFT, ZG = 0.035, 0.07


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def scan(arm, targets, yaw_opts):
    base = np.array(station.BASES[arm])
    out = {}
    for yw in yaw_opts:
        q = np.array(READY, float)
        worst_p = worst_r = 0.0
        for pos, phi in targets:
            R = rz(phi) @ grip_rotation(yw)
            q, pe, re = ik.solve(np.asarray(pos), ik.matrix_to_quat_xyzw(R), q, base_pos=base, iters=200)
            worst_p, worst_r = max(worst_p, pe * 1000), max(worst_r, re)
        out[yw] = (round(worst_p, 1), round(float(np.degrees(worst_r)) if worst_r < 7 else worst_r, 1))
    return out


if __name__ == "__main__":
    ax, ay = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (T.A_X, T.A_Y)
    z = T.TOP + LIFT + ZG
    left = [(np.array([ax, ay + geom.A_D / 2 + 0.02, z]), 0.0)]
    print("left (hold A):", scan("left", left, (0.0, math.pi, math.pi / 2, -math.pi / 2)))
    for sign in (+1, -1):
        p, why = geom.plan(margin=0.010, sign=sign)
        tg = []
        for x, y, f in p[::2]:
            h = np.array([ax + x, ay + y, 0.0]) + rz(f) @ np.array([0.0, geom.B_POST_Y, 0.0])
            tg.append((np.array([h[0], h[1], z]), f))
        hs = np.array([t[0] for t in tg])
        print("sign", sign, "handle x range", hs[:, 0].min().round(3), hs[:, 0].max().round(3), "y", hs[:, 1].min().round(3), hs[:, 1].max().round(3))
        print("  right (hold B):", scan("right", tg, (0.0, math.pi, math.pi / 2, -math.pi / 2)))
