"""Geometry of the tip-rocker scene (metres, station frame; the rocker's own frame has its origin on the table top
under the beam's pivot). Pure NumPy, so the task file, the asset builder, the demo and the tests share it.

The rocker: a beam on a horizontal pivot (axis along x, height PZ). The beam runs along y. Near its right end
(y = YT) a turntable (vertical axis) carries an eccentric counterweight on an arm; a fixed post stands in the
weight's path at the front (x > 0). A ballast block at the left end balances the weight when the weight is at the
front. With the weight turned out to the right the beam is right-heavy and rests on the right stop; with the
weight turned in to the yellow mark it is left-heavy and rests on the green stop at the left end. The beam's tip
to the left raises the turntable by about 4 cm, which lifts the weight over the post.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475            # table top
XC = 0.30               # rocker x on the station
PZ = 0.042              # pivot height above the table
STOP_DEG = 8.0          # beam angle at the stops (theta > 0: left end up, right end down)
LIMIT_DEG = STOP_DEG    # joint limit = the stops (articulated links do not collide with their parent)

# beam (rocker frame, relative to the pivot)
BEAM_HALF = (0.020, 0.170, 0.008)
YT = -0.150             # turntable axis y
TURN_TOP = 0.014        # turntable top above the pivot
DISK_R = 0.035
RHO = 0.050             # weight block centre from the axis
W_HALF = (0.015, 0.015, 0.0125)    # weight block (bottom on the turntable top)
ARM_HALF = (0.010, 0.030, 0.004)
STEM_R, STEM_H = 0.016, 0.040
YL = 0.150              # ballast and left handle y
BAL_HALF = (0.020, 0.020, 0.010)   # ballast block on the beam top
MARK_Y = (-0.085, -0.055)          # yellow mark on the beam top (where the weight turns to)

M_WEIGHT = 0.20         # weight body (block + arm + stem)
M_BAR, M_DISK, M_STEM_L = 0.06, 0.02, 0.02

PSI_START, PSI_TARGET = 0.0, math.pi   # weight angle: 0 turned out to the right, pi turned in to the yellow mark
PSI_LIMITS_DEG = (-10.0, 190.0)

POST_X, POST_Y, POST_R = RHO, -0.148, 0.010
POST_CLEAR_TIPPED = 0.020          # weight bottom above the post top with the beam on the green stop


def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


STOP = math.radians(STOP_DEG)
PIVOT_W = np.array([XC, 0.0, TOP + PZ])


def turntable_z(theta: float) -> float:
    """Height (relative to the pivot) of the turntable top at the weight's front position, beam angle theta."""
    return YT * math.sin(theta) + TURN_TOP * math.cos(theta)


# post top (rocker frame, above the table): the weight bottom with the beam on the green stop minus the clearance
POST_TOP = PZ + turntable_z(-STOP) - POST_CLEAR_TIPPED


def block_bottom(theta: float) -> float:
    """Weight bottom above the table when the weight is at the front, for beam angle theta."""
    return PZ + turntable_z(theta)


def beam_to_world(theta: float, p) -> np.ndarray:
    """A point given in the beam frame (relative to the pivot) -> station frame."""
    return PIVOT_W + rx(theta) @ np.asarray(p, float)


def axis_point() -> np.ndarray:
    return np.array([0.0, YT, TURN_TOP])


def weight_to_world(theta: float, psi: float, p) -> np.ndarray:
    """A point in the weight frame (origin on the turntable axis, psi = 0 pointing to -y) -> station frame."""
    return beam_to_world(theta, axis_point() + rz(psi) @ np.asarray(p, float))


def weight_stem(theta: float, psi: float) -> np.ndarray:
    """Weight handle: centre of the stem's mid-height."""
    return weight_to_world(theta, psi, (0.0, -RHO, W_HALF[2] * 2 + STEM_H / 2))


def left_stem(theta: float) -> np.ndarray:
    return beam_to_world(theta, (0.0, YL, BEAM_HALF[2] + 2 * BAL_HALF[2] + STEM_H / 2))


def block_centre(theta: float, psi: float) -> np.ndarray:
    return weight_to_world(theta, psi, (0.0, -RHO, W_HALF[2]))


def stop_top() -> float:
    """Stop block top height (above the table): the beam's lowest outer corner touches it at the stop angle."""
    y, z = BEAM_HALF[1], -BEAM_HALF[2]
    return PZ + (-y * math.sin(STOP)) + z * math.cos(STOP) + 0.0004
