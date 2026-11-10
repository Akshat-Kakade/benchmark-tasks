"""On the station table: a flat yellow blank (120 x 120 x 24 mm, white edge strip on its front edge) sits in a low grey
guard (four posts and a rail frame, 250 x 250 mm clear, rails 92 mm above the table). A blue and an orange push tool
lie on the table either side of the blank: each is a long thin bar with a 40 mm wide flat paddle at one end and a tall
handle stem outside the guard. Using only the two tools (hands stay outside the guard), turn the blank clockwise until
its white edge points to the green mark, keeping its centre between the four white ticks. The blank and tools are free
rigid boxes; the blank has low friction with the table.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Keep both grippers outside the grey guard. Use the blue and orange push tools to turn the yellow blank "
               "clockwise until its white edge points to the green mark, keeping the blank centred between the four "
               "white ticks.")

BLUE = (0.10, 0.30, 0.90)
ORANGE = (0.95, 0.50, 0.08)
YELLOW = (0.95, 0.85, 0.10)
WHITE = (0.95, 0.95, 0.95)
GREEN = (0.10, 0.70, 0.20)
GREY = (0.45, 0.45, 0.48)

CX, CY = 0.30, 0.0              # registered blank centre
BL, BT = 0.12, 0.024            # blank side and thickness
H = BL / 2
GUARD_HALF = 0.125              # guard rail centre lines at +-0.125; clear area 0.25 x 0.25 (inner edge 0.115)
CLEAR_HALF = 0.115
ROOF_Z = 0.84                   # rail underside (tools pass below; a gripper cannot)
REST_B = TOP + 0.0005 + BT / 2

# push tool, local frame: origin at the paddle centre, +x = push-face normal, +y along the bar toward the handle
PAD = (0.018, 0.040, 0.020)     # paddle (x, y, z): flat 40 mm wide push face at x = +0.009
PAD_FACE = PAD[0] / 2
PAD_A = 0.035                   # paddle centre offset along the blank's edge from the edge midpoint
STEM_Y = 0.170                  # handle stem axis, local y (world |y| 0.205 at rest)
STEM_X = -0.004
GRIP_Z = 0.075                  # pinch height above the tool origin
TOOL_Z = TOP + 0.0005 + PAD[2] / 2
TOOL_MASS = 0.08


def _tool_boxes(color):
    return [((0.0, 0.0, 0.0), PAD, color),
            ((STEM_X, 0.1025, -0.004), (0.010, 0.165, 0.012), color),            # bar, flush with the paddle's back
            ((STEM_X, STEM_Y, 0.0575), (0.030, 0.030, 0.135), color),            # handle stem (pinched across x)
            ((STEM_X, 0.205, 0.010), (0.020, 0.040, 0.040), color)]              # tail: balances the bar at the stem


GAP0 = 0.020                    # paddle-to-face clearance at the start
BLANK = part("blank", pos=(CX, CY, REST_B), mass=0.15, friction=0.2, color=YELLOW,
             boxes=[((0, 0, 0), (BL, BL, BT), YELLOW), ((H - 0.007, 0, BT / 2 + 0.0005), (0.012, BL, 0.001), WHITE)])
BLUE_T = part("blue_tool", pos=(CX - H - PAD_FACE - GAP0, CY + PAD_A, TOOL_Z), yaw=0.0, mass=TOOL_MASS, friction=0.5,
              color=BLUE, boxes=_tool_boxes(BLUE))
ORANGE_T = part("orange_tool", pos=(CX + H + PAD_FACE + GAP0, CY - PAD_A, TOOL_Z), yaw=math.pi, mass=TOOL_MASS,
                friction=0.5, color=ORANGE, boxes=_tool_boxes(ORANGE))


def _guard():
    out = []
    g, w, pz = GUARD_HALF, 0.02, ROOF_Z - TOP
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        out.append(Box(f"post{i}", pos=(CX + sx * g, sy * g, TOP + pz / 2), size=(w, w, pz), color=GREY))
    for i, s in enumerate((-1, 1)):
        out.append(Box(f"rail_x{i}", pos=(CX + s * g, 0.0, ROOF_Z + 0.006), size=(w, 2 * g + w, 0.012), color=GREY))
        out.append(Box(f"rail_y{i}", pos=(CX, s * g, ROOF_Z + 0.006), size=(2 * g - w, w, 0.012), color=GREY))
    return out


def _marks():
    out = []
    for i, (dx, dy, sx, sy) in enumerate([(-0.095, 0, 0.016, 0.006), (0.095, 0, 0.016, 0.006),
                                          (0, -0.095, 0.006, 0.016), (0, 0.095, 0.006, 0.016)]):
        out.append(Box(f"tick{i}", pos=(CX + dx, CY + dy, TOP + 0.001), size=(sx, sy, 0.002), color=WHITE))
    r = 0.108                   # green angle mark: a short line on the table along the target direction (-45 deg)
    a = math.radians(-45.0)
    out.append(Box("angle_mark", pos=(CX + r * math.cos(a), CY + r * math.sin(a), TOP + 0.001), size=(0.030, 0.008, 0.002),
                   color=GREEN, yaw=a))
    return out


OBJECTS = _guard() + _marks() + [BLANK, BLUE_T, ORANGE_T]

# -- scoring thresholds -------------------------------------------------------------------------------------------------
GATE_TILT = 8.0
HANDLE_KEEP_OUT = 0.14          # a tool's handle stem inside |x - CX|, |y| < 0.14 means a hand is inside the guard
TARGET_YAW = -45.0
YAW_FULL = (-53.0, -37.0)
YAW_075 = (-57.0, -33.0)
YAW_05 = -25.0
DRIFT_FULL, DRIFT_075, DRIFT_05 = 0.015, 0.030, 0.045
TIP_CLEAR = 0.078               # paddle centre farther than this from the blank centre along the blank's axes
HOLD_STEPS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), held=0)


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def tilt_deg(state: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(state.rotation()[2, 2])))))


def yaw_deg(state: PartState) -> float:
    R = state.rotation()
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def stem_point(tool: PartState) -> np.ndarray:
    return np.asarray(tool.to_world((STEM_X, STEM_Y, GRIP_Z)), float)


def _metrics(objects) -> dict:
    """Blank yaw (deg, negative clockwise), centre drift from (CX, CY), tilt, farthest corner from the centre in the guard
    frame, handle stems' distances into the guard, and the tool paddles' blank-frame distances."""
    b = objects["blank"]
    yaw = yaw_deg(b)
    c = np.asarray(b.pos, float)
    R = b.rotation()
    corner = max(max(abs(float((c + R @ np.array([sx * H, sy * H, 0.0]))[0] - CX)),
                     abs(float((c + R @ np.array([sx * H, sy * H, 0.0]))[1] - CY)))
                 for sx in (-1, 1) for sy in (-1, 1))
    stems = [stem_point(objects[n]) for n in ("blue_tool", "orange_tool")]
    stem_in = min(max(abs(float(s[0]) - CX), abs(float(s[1]) - CY)) for s in stems)
    tips = []
    for n in ("blue_tool", "orange_tool"):
        l = b.to_local(objects[n].pos)
        tips.append(float(max(abs(l[0]), abs(l[1]))))
    return dict(yaw=yaw, drift=float(math.hypot(c[0] - CX, c[1] - CY)), tilt=tilt_deg(b), corner=corner, stem_in=stem_in,
                tip_min=min(tips))


def _classify_m(m):
    if m["tilt"] > GATE_TILT or m["corner"] > GUARD_HALF or m["stem_in"] < HANDLE_KEEP_OUT or m["drift"] > 0.08:
        return 0.0, False
    yaw, d = m["yaw"], m["drift"]
    if YAW_FULL[0] <= yaw <= YAW_FULL[1] and d <= DRIFT_FULL and m["tip_min"] >= TIP_CLEAR:
        return 1.0, True
    if YAW_075[0] <= yaw <= YAW_075[1] and d <= DRIFT_075:
        return 0.75, False
    if -100.0 <= yaw <= YAW_05 and d <= DRIFT_05:
        return 0.5, False
    if -180.0 < yaw <= -12.0:
        return 0.25, False
    return 0.0, False


def classify(objects):
    m = _metrics(objects)
    level, perfect = _classify_m(m)
    return level, perfect, m


def _update(objects) -> tuple[float, bool]:
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    level, perfect, _ = classify(objects)
    _STATE["held"] = _STATE["held"] + 1 if perfect else 0
    result = (level, _STATE["held"] >= HOLD_STEPS)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Blank yaw -37 to -53 deg (clockwise), centre within 15 mm of the registered centre, tilt <= 8 deg, both tool
    paddles clear of the blank (centres > 78 mm from its centre along its axes), no handle stem inside the guard,
    blank inside the guard's clear area; held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the blank tilts over 8 deg, a corner reaches the guard posts or rails (125 mm from the centre), its centre is over 80 mm off, or a tool's handle stem
    is inside the guard footprint. 0.25 turned >= 12 deg clockwise; 0.5 yaw <= -25 deg, drift <= 45 mm; 0.75 yaw -33 to
    -57 deg, drift <= 30 mm; 1.0 as in success()."""
    return _update(objects)[0]
