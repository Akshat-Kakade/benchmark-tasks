"""IK reach check along the whole planned motion (no simulator): python ikscan.py  (needs yam_arena on the path)."""
import math
import sys

import numpy as np

import demo
import task


def main():
    from yam_arena import ik, station
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation

    pts = demo.path_points(task, -45.0)
    worst = 0.0
    for name in ("left", "right"):
        base = np.asarray(station.BASES[name], float)
        for c in demo.GRIP_YAWS:
            q = np.array(READY, float)
            errs = []
            for n, p, th in pts:
                if n != name:
                    continue
                q, pe, re = ik.solve(p, ik.matrix_to_quat_xyzw(demo._rz(th) @ grip_rotation(c)), q, base_pos=base, iters=200)
                errs.append((round(pe * 1000, 1), round(math.degrees(re), 1)))
            print(name, "grip yaw", round(c, 2), "pos mm / rot deg:", errs)
            worst = max(worst, max(e[0] for e in errs)) if c == demo.plan_yaw.__defaults__ else worst
    hl = [p for n, p, th in pts if n == "left"]
    hr = [p for n, p, th in pts if n == "right"]
    print("hand separation min (m):", round(min(float(np.linalg.norm(a - b)) for a in hl for b in hr), 3))


def sep_only():
    pts = demo.path_points(task, -45.0)
    hl = [p for n, p, th in pts if n == "left"]
    hr = [p for n, p, th in pts if n == "right"]
    print("hand separation min (m):", round(min(float(np.linalg.norm(a - b)) for a in hl for b in hr), 3))
    print("left hand x,y range:", np.round(np.min(hl, 0), 3), np.round(np.max(hl, 0), 3))


if __name__ == "__main__":
    sep_only() if "--sep" in sys.argv else main()
