"""On the station table: three yellow cartridges stand side by side in three marked lanes, and a long blue pusher bar lies
square against their back faces. The bar can only be held by two orange handles on stems at its two ends (0.34 m apart).
Both arms hold the bar and push all three cartridges forward together until their front edges line up at the green stops.
The cartridges and the pusher are free rigid boxes; nothing is pinned or fastened. The task is pushing (not insertion).

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

INSTRUCTION = ("Grip the blue pusher by both orange handles and push the three yellow cartridges along their lanes "
               "until their front edges form one line at the green stops.")

BLUE = (0.10, 0.30, 0.90)
ORANGE = (0.95, 0.50, 0.08)
YELLOW = (0.95, 0.85, 0.10)
GREEN = (0.10, 0.65, 0.20)
WHITE = (0.95, 0.95, 0.95)

LANE_Y = (-0.105, 0.0, 0.105)
LANE_W, LANE_T = 0.085, 0.002          # painted lane strips (flat, 2 mm)
C_X0 = 0.220                           # cartridge start centre x
C_SIZE = (0.055, 0.055, 0.035)         # x, y, z
BAR_X0 = 0.178                         # pusher bar face centre at start
BAR = (0.025, 0.35, 0.025)
HANDLE_Y = 0.170
HANDLE = (0.040, 0.035, 0.025)
HANDLE_Z = 0.0775                      # handle centre above the bar centre (TOP + 0.10 at rest)
STEM = (0.020, 0.020, 0.0525)
STOP_X = 0.360                         # stop faces
BAR_Z = TOP + LANE_T + BAR[2] / 2 + 0.0005
C_Z = TOP + LANE_T + C_SIZE[2] / 2 + 0.0005


def _bar_boxes():
    b = [((0.0, 0.0, 0.0), BAR, BLUE)]
    for s in (-1, 1):
        b.append(((0.0, s * HANDLE_Y, BAR[2] / 2 + STEM[2] / 2), STEM, BLUE))
        b.append(((0.0, s * HANDLE_Y, HANDLE_Z), HANDLE, ORANGE))
    return b


PUSHER = part("pusher", pos=(BAR_X0, 0.0, BAR_Z), mass=0.22, boxes=_bar_boxes(), color=BLUE, friction=0.2)
_FR = (0.45, 0.60, 0.45)
CARTS = [part(f"cart{i}", pos=(C_X0, LANE_Y[i], C_Z), mass=0.09, color=YELLOW, friction=_FR[i],
              boxes=[((0, 0, 0), C_SIZE, YELLOW)]) for i in range(3)]
NAMES = [c.name for c in CARTS]

LANES = [Box(f"lane{i}", pos=(0.27, LANE_Y[i], TOP + LANE_T / 2), size=(0.20, LANE_W, LANE_T), color=WHITE) for i in range(3)]
STOPS = [Box(f"stop{i}", pos=(STOP_X + 0.010, LANE_Y[i], TOP + 0.010), size=(0.020, LANE_W, 0.020), color=GREEN) for i in range(3)]
OBJECTS = LANES + STOPS + [PUSHER] + CARTS

# -- scoring thresholds ------------------------------------------------------------------------------------------------
GATE_TILT, GATE_Y = 8.0, 0.015          # lane half-clearance (0.085 lane vs 0.055 cartridge = 15 mm each side)
GATE_FRONT = STOP_X + 0.001             # cartridge front edge beyond the stop face
X_LO, X_HI, SPREAD1 = 0.312, 0.332, 0.012
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


def front_x(state: PartState) -> float:
    """Largest x of the cartridge footprint (front edge, including yaw)."""
    y = math.radians(yaw_deg(state))
    return float(state.pos[0]) + 0.0275 * (abs(math.cos(y)) + abs(math.sin(y)))


def measure(objects) -> dict:
    cs = [objects[n] for n in NAMES]
    x = [float(c.pos[0]) for c in cs]
    p = objects["pusher"]
    return dict(x=x, adv=[xi - C_X0 for xi in x], yerr=[float(c.pos[1]) - LANE_Y[i] for i, c in enumerate(cs)],
                yaw=[abs((yaw_deg(c) + 90.0) % 180.0 - 90.0) for c in cs], tilt=[tilt_deg(c) for c in cs],
                front=[front_x(c) for c in cs], spread=max(x) - min(x),
                p_x=float(p.pos[0]), p_yaw=abs((yaw_deg(p) + 90.0) % 180.0 - 90.0), p_tilt=tilt_deg(p))


def classify(objects):
    """(level, all-1.0-conditions-hold, info) from the current poses only."""
    m = measure(objects)
    if (max(m["tilt"]) > GATE_TILT or max(abs(e) for e in m["yerr"]) > GATE_Y or max(m["front"]) > GATE_FRONT):
        return 0.0, False, m
    face = m["p_x"] + BAR[0] / 2
    square = (m["p_yaw"] <= 6.0 and m["p_tilt"] <= 8.0
              and min(m["x"]) - 0.0275 - 0.012 <= face <= max(m["x"]) - 0.0275 + 0.012)
    if (all(X_LO <= xi <= X_HI for xi in m["x"]) and m["spread"] <= SPREAD1 and max(abs(e) for e in m["yerr"]) <= 0.012
            and max(m["yaw"]) <= 8.0 and square):
        return 1.0, True, m
    if all(xi >= 0.300 for xi in m["x"]) and m["spread"] <= 0.025:
        return 0.75, False, m
    if all(a >= 0.060 for a in m["adv"]):
        return 0.5, False, m
    if max(m["adv"]) >= 0.040:
        return 0.25, False, m
    return 0.0, False, m


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
    """All three cartridge centres at x 0.312-0.332, centre-x spread <= 12 mm, each within 12 mm of its lane centre and
    yawed <= 8 deg, no gate tripped, the pusher still square behind them (yaw <= 6 deg); held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if any cartridge tilts over 8 deg, leaves its lane (|y error| > 15 mm) or crosses a stop face. Otherwise
    0.25 one cartridge advanced >= 40 mm; 0.5 all three >= 60 mm; 0.75 all centres x >= 0.300 and spread <= 25 mm;
    1.0 as in success()."""
    return _update(objects)[0]
