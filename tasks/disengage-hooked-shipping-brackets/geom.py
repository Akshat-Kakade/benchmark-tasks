"""Pure geometry of the two brackets (no simulator imports): box lists, a 2-D footprint collision test and a path planner.

Frames: bracket A (blue, housing) origin at the centre of its underside, slot opening toward local -y; bracket B (orange,
T-head bracket) origin at the centre of its head, stem toward local -y. Boxes: (centre xyz, size xyz).
"""
from __future__ import annotations

import math

import numpy as np

# -- bracket A: a keyed housing (floor, lid, side walls, back wall, two front lips with a slot between them) -------------
A_W, A_D = 0.30, 0.18
WALL = 0.02
LIP_T = 0.010                         # front lip thickness (y)
FLOOR_T, LID_T = 0.008, 0.008
CAV_Z0, CAV_Z1 = FLOOR_T, 0.036      # cavity height band (0.008 .. 0.036)
A_H = CAV_Z1 + LID_T                 # 0.044
SLOT = 0.11                         # gap between the two lips (x)
LIP_X0 = SLOT / 2
A_BOXES = [
    ((0.0, 0.0, FLOOR_T / 2), (A_W, A_D, FLOOR_T)),
    ((0.0, 0.0, CAV_Z1 + LID_T / 2), (A_W, A_D, LID_T)),
    ((A_W / 2 - WALL / 2, 0.0, (CAV_Z0 + CAV_Z1) / 2), (WALL, A_D, CAV_Z1 - CAV_Z0)),
    ((-(A_W / 2 - WALL / 2), 0.0, (CAV_Z0 + CAV_Z1) / 2), (WALL, A_D, CAV_Z1 - CAV_Z0)),
    ((0.0, A_D / 2 - WALL / 2, (CAV_Z0 + CAV_Z1) / 2), (A_W - 2 * WALL, WALL, CAV_Z1 - CAV_Z0)),
]
_lipw = (A_W / 2 - WALL) - LIP_X0
for s in (+1, -1):
    A_BOXES.append(((s * (LIP_X0 + _lipw / 2), -(A_D / 2 - LIP_T / 2), (CAV_Z0 + CAV_Z1) / 2), (_lipw, LIP_T, CAV_Z1 - CAV_Z0)))
A_POST = (0.04, 0.04, 0.09)           # handle post on the back wall (left hand)
A_POST_C = (0.0, A_D / 2 + A_POST[1] / 2, A_POST[2] / 2)
A_BOXES.append((A_POST_C, A_POST))

# -- bracket B: T-head, stem, handle post -------------------------------------------------------------------------------
HEAD = (0.13, 0.018, 0.012)           # long axis x
HEAD_Z = 0.022
STEM_W = 0.012
STEM_L = A_D / 2 + 0.06                 # stem reaches 60 mm past the lips before the handle post
B_POST = (0.04, 0.04, 0.09)
B_POST_Y = -(STEM_L + B_POST[1] / 2 - 0.02)     # post centre y (post overlaps the stem end by 20 mm)
B_BOXES = [
    ((0.0, 0.0, HEAD_Z), HEAD),
    ((0.0, -STEM_L / 2, HEAD_Z), (STEM_W, STEM_L, HEAD[2])),
    ((0.0, B_POST_Y, B_POST[2] / 2), B_POST),
]
HEAD_CORNERS = [(sx * HEAD[0] / 2, sy * HEAD[1] / 2) for sx in (1, -1) for sy in (1, -1)]
FRONT_Y = -A_D / 2                    # outer face of the lips (A frame)


def _rect(c, s, ang, infl):
    """Corners (4,2) of a box footprint, rotated by ``ang`` about the origin after offsetting by c."""
    hx, hy = s[0] / 2 + infl, s[1] / 2 + infl
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) + np.array(c[:2])


def collides(pose, margin=0.0, a_boxes=A_BOXES, b_boxes=B_BOXES) -> np.ndarray:
    """pose = arrays (x, y, phi) of B in the A frame (broadcast). True where any B box (inflated by ``margin`` in xy)
    overlaps any A box whose z band overlaps. Vectorised separating-axis test."""
    x, y, ph = [np.asarray(v, float) for v in pose]
    c, s = np.cos(ph), np.sin(ph)
    out = np.zeros(np.broadcast(x, y, ph).shape, bool)
    for bc, bs in b_boxes:
        bz0, bz1 = bc[2] - bs[2] / 2, bc[2] + bs[2] / 2
        # B box corners in the A frame, shape (..., 4, 2)
        loc = _rect(bc, bs, 0.0, margin)
        cx = x[..., None] + c[..., None] * loc[:, 0] - s[..., None] * loc[:, 1]
        cy = y[..., None] + s[..., None] * loc[:, 0] + c[..., None] * loc[:, 1]
        for ac, as_ in a_boxes:
            az0, az1 = ac[2] - as_[2] / 2, ac[2] + as_[2] / 2
            if bz1 <= az0 or bz0 >= az1:
                continue
            ax0, ax1 = ac[0] - as_[0] / 2, ac[0] + as_[0] / 2
            ay0, ay1 = ac[1] - as_[1] / 2, ac[1] + as_[1] / 2
            sep = (cx.max(-1) < ax0) | (cx.min(-1) > ax1) | (cy.max(-1) < ay0) | (cy.min(-1) > ay1)   # A axes
            # B axes: project both onto B's two axes
            for ax_ in ((c, s), (-s, c)):
                u, v = ax_
                pb = (u[..., None] * (cx - x[..., None]) + v[..., None] * (cy - y[..., None]))
                ac_x = np.array([ax0, ax1, ax1, ax0])
                ac_y = np.array([ay0, ay0, ay1, ay1])
                pa = u[..., None] * (ac_x - x[..., None]) + v[..., None] * (ac_y - y[..., None])
                sep = sep | (pb.max(-1) < pa.min(-1)) | (pb.min(-1) > pa.max(-1))
            out |= ~sep
    return out


def head_outside(pose, margin=0.0) -> np.ndarray:
    """True where every head corner lies beyond the outer face of the lips (A frame y <= FRONT_Y - margin)."""
    x, y, ph = [np.asarray(v, float) for v in pose]
    c, s = np.cos(ph), np.sin(ph)
    ys = [y + s * hx + c * hy for hx, hy in HEAD_CORNERS]
    return np.max(ys, axis=0) <= FRONT_Y - margin


def plan(margin=0.010, start=(0.0, 0.0, 0.0), res=0.004, dphi=math.radians(4.0), goal_y=-0.30, ymax=0.06,
         phi_max=math.radians(100), sign=+1):
    """Dijkstra over (x, y, phi) of B in the A frame. Returns (path (n,3), grid info) from ``start`` to the cheapest pose
    with the head outside and y <= goal_y, or (None, ...) when no path exists."""
    import heapq
    xs = np.arange(-0.12, 0.12 + 1e-9, res)
    ys = np.arange(goal_y - 0.04, ymax + 1e-9, res)
    nph = int(round(phi_max / dphi))
    phs = np.arange(-nph, nph + 1) * dphi
    X, Y, P = np.meshgrid(xs, ys, phs, indexing="ij")
    col = collides((X, Y, P), margin)
    if sign:
        col |= (P * sign < -1e-9)           # only turn one way
    ix0, iy0, ip0 = (int(round((start[0] - xs[0]) / res)), int(round((start[1] - ys[0]) / res)), nph)
    if col[ix0, iy0, ip0]:
        return None, "start in collision"
    dist = np.full(col.shape, np.inf)
    prev = {}
    dist[ix0, iy0, ip0] = 0.0
    pq = [(0.0, (ix0, iy0, ip0))]
    goal = None
    w_rot = res / dphi * 0.25            # rotating is cheaper than translating a little, so rotate early
    nbrs = [(dx, dy, dp) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dp in (-1, 0, 1) if (dx, dy, dp) != (0, 0, 0)]
    while pq:
        d, (i, j, k) = heapq.heappop(pq)
        if d > dist[i, j, k]:
            continue
        if ys[j] <= goal_y and not col[i, j, k]:
            goal = (i, j, k)
            break
        for dx, dy, dp in nbrs:
            a, b, cc = i + dx, j + dy, k + dp
            if not (0 <= a < len(xs) and 0 <= b < len(ys) and 0 <= cc < len(phs)) or col[a, b, cc]:
                continue
            nd = d + math.hypot(dx, dy) * res + abs(dp) * res * 0.25 * 4 / 4 * 1.0
            if nd < dist[a, b, cc]:
                dist[a, b, cc] = nd
                prev[(a, b, cc)] = (i, j, k)
                heapq.heappush(pq, (nd, (a, b, cc)))
    if goal is None:
        return None, "no path"
    path = [goal]
    while path[-1] in prev:
        path.append(prev[path[-1]])
    path = path[::-1]
    return np.array([(xs[i], ys[j], phs[k]) for i, j, k in path]), "ok"
