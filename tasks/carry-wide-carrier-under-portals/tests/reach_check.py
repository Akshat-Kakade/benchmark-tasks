"""IK reach scan of both hands along the whole planned path (run on a machine with yam_arena)."""
import math, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import demo, task as T
from yam_arena import ik, station
from yam_arena.constants import READY
from yam_arena.demo import grip_rotation

worst = {}
for side, sy in (("left", +1), ("right", -1)):
    base = np.asarray(station.BASES[side], float)
    q = np.array(READY, float)
    for yaw in (0.0, math.pi):
        q = np.array(READY, float); w = 0.0; wr = 0.0
        pts = []
        for (x, y) in demo.path_for("", 0.21, 0.0):
            pass
        P = demo.path_for("", 0.21, 0.0)
        for a, b in zip(P, P[1:]):
            for s in np.linspace(0, 1, 25):
                pts.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
        for (x, y) in pts:
            for z in (T.TOP + demo.SZ, T.TOP + demo.SZ + 0.02, T.TOP + demo.SZ + 0.09):
                tgt = np.array([x, y + sy * T.STEM_Y, z])
                q, pe, re_ = ik.solve(tgt, ik.matrix_to_quat_xyzw(grip_rotation(yaw)), q, base_pos=base, iters=200)
                w = max(w, pe * 1000); wr = max(wr, re_)
        print(side, "yaw", round(yaw, 2), "worst pos err mm", round(w, 2), "rot err", round(float(wr), 3))
print("hand gap m", 2 * T.STEM_Y)
