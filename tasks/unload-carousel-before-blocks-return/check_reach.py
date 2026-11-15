"""Offline reach check (run where yam_arena is installed): every waypoint of the demo variants through the IK, with the
worst error per waypoint label, and the closest the two TCPs come to each other.   python check_reach.py [variant ...]"""
import os
import sys

import numpy as np
import task
from yam_arena import ik, station
from yam_arena.demo import ArmPlan


def check(variant: str, dt: float = 0.1, verbose: bool = False):
    os.environ["DEMO_VARIANT"] = variant
    plans = {s: ArmPlan(list(w), station.BASES[s]) for s, w in task.demo().items()}
    dur = max(p.duration for p in plans.values())
    per = {s: {} for s in plans}
    tcps = {s: [] for s in plans}
    for k in range(int(dur / dt) + 1):
        for s, p in plans.items():
            _, label = p.action7(k * dt)
            mm, deg = p.worst_ik_mm, p.worst_ik_deg
            pos, _ = ik.tcp_pose(p.q, p.base_pos)
            tcps[s].append(pos)
            # per-sample error: recompute from the target
            tp, tr, _, _ = p.target(k * dt)
            err = float(np.linalg.norm(pos - tp)) * 1000
            per[s][label] = max(per[s].get(label, 0.0), err)
    n = min(len(v) for v in tcps.values())
    gap = min(float(np.linalg.norm(tcps["left"][i] - tcps["right"][i])) for i in range(n))
    worst = {s: round(max(per[s].values()), 2) for s in per}
    if verbose:
        for s in per:
            bad = {k: round(v, 1) for k, v in per[s].items() if v > 3}
            print(variant, s, "labels over 3 mm:", bad)
    return worst, round(gap, 3), round(dur, 1)


if __name__ == "__main__":
    for v in (sys.argv[1:] or ("full", "late", "blue_removed", "left_only", "right_only", "settle")):
        print(v, check(v, verbose=True))
