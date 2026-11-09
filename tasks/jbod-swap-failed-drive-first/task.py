"""On the station table, straight ahead of the arms: a grey top-loading drive drawer fixed to the table, with four
slots side by side (left to right as seen from the arms: slot 4, 3, 2, 1, at 10 cm pitch). Each slot holds one dark grey
drive (4 x 9 x 9 cm, silver top edge) standing upright, 5 cm of it above the slot walls. In front of each slot a lamp
sits on the drawer's front panel: green, amber, red, white (blinking while that slot rebuilds) or dark. On the left of
the table is a white tray with two pockets holding blue spare drives; on the right is an open black returns bin.
The drawer, tray and bin are fixed; the drives are loose (150 g each).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import on_reset, on_step, part, write_pose  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("In the grey top-loading drive drawer, the drive in the red-lit slot has failed and the drive in the "
               "amber-lit slot is about to fail: replace both with the blue spares from the white tray on the left, and "
               "put the removed drives in the black returns bin on the right. The four drives form one RAID 5 array, "
               "so never have two of its slots without a working drive: wait for a replaced slot's white rebuilding "
               "lamp to turn green before pulling another drive.")

STEEL = (0.55, 0.57, 0.60)
DRIVE_GREY, SILVER = (0.25, 0.25, 0.27), (0.80, 0.81, 0.83)
SPARE_BLUE, TRAY_WHITE, BIN_BLACK = (0.10, 0.35, 0.90), (0.92, 0.92, 0.92), (0.08, 0.08, 0.08)
LAMP_COLORS = {"green": (0.05, 0.85, 0.10), "amber": (1.00, 0.55, 0.00), "red": (0.95, 0.05, 0.05),
               "white": (1.00, 1.00, 1.00)}

# -- geometry (world frame: x forward from the arms, y left, z up, m) -------------------------------------------------
DRAWER_X = 0.26
SLOT_Y = (-0.15, -0.05, 0.05, 0.15)          # slots 1..4 (index 0..3), right to left
FLOOR_H = 0.005
FLOOR_Z = TOP + FLOOR_H                       # the slot floor
SLOT_IN_X, SLOT_IN_Y, WALL, WALL_H = 0.100, 0.052, 0.004, 0.040
WALL_TOP = FLOOR_Z + WALL_H
DRIVE_X, DRIVE_Y, DRIVE_H, DRIVE_KG, TAB_H = 0.090, 0.040, 0.090, 0.150, 0.010
DRAWER_LO = np.array([DRAWER_X - 0.060, -0.210, TOP])
DRAWER_HI = np.array([DRAWER_X + 0.060, 0.210, FLOOR_Z + DRIVE_H])   # incl. seated drives
PANEL_X, PANEL_DX, PANEL_H = 0.185, 0.030, 0.040                      # lamp panel in front of the slots
LAMP_SIZE = (0.030, 0.050, 0.006)
LAMP_Z = TOP + PANEL_H + LAMP_SIZE[2] / 2 + 0.0005
TRAY_XY = ((0.22, 0.36), (0.22, 0.46))       # pockets for spare S1, S2
TRAY_WALL_H = 0.030
BIN_XY, BIN_IN, BIN_WALL, BIN_H, BIN_FLOOR = (0.22, -0.42), (0.160, 0.200), 0.005, 0.060, 0.005
BIN_FLOOR_Z = TOP + BIN_FLOOR
BIN_RIM_Z = TOP + BIN_H

ORIGINALS = ("drive1", "drive2", "drive3", "drive4")   # standing in slots 1-4 at the start
SPARES = ("spare1", "spare2")
DRIVES = ORIGINALS + SPARES
OK, PREDICTIVE, FAILED = "ok", "predictive", "failed"
HEALTH0 = (OK, FAILED, PREDICTIVE, OK)        # hidden initial state: slot 2 failed, slot 3 predictive

OUT_ABOVE = 0.015      # a drive is out of its slot once its bottom is 15 mm above the slot floor
SEAT_GAP, SEAT_TILT = 0.005, 10.0
REBUILD_S = 8.0
CONTROL_HZ = 20
CLEAR_DIST = 0.060     # both grippers this far from the drawer at the end


def _cell(prefix, cx, cy, in_x, in_y, wall, h, z0, color):
    ox, oy = in_x / 2 + wall / 2, in_y / 2 + wall / 2
    z = z0 + h / 2
    return [Box(f"{prefix}_n", pos=(cx, cy + oy, z), size=(in_x + 2 * wall, wall, h), color=color),
            Box(f"{prefix}_s", pos=(cx, cy - oy, z), size=(in_x + 2 * wall, wall, h), color=color),
            Box(f"{prefix}_e", pos=(cx + ox, cy, z), size=(wall, in_y, h), color=color),
            Box(f"{prefix}_w", pos=(cx - ox, cy, z), size=(wall, in_y, h), color=color)]


def _drive(name, xy, z_bottom, color):
    body = ((0.0, 0.0, -TAB_H / 2), (DRIVE_X, DRIVE_Y, DRIVE_H - TAB_H), color)
    tab = ((0.0, 0.0, DRIVE_H / 2 - TAB_H / 2), (DRIVE_X, DRIVE_Y, TAB_H), SILVER)
    return part(name, pos=(xy[0], xy[1], z_bottom + DRIVE_H / 2 + 0.001), boxes=[body, tab], mass=DRIVE_KG,
                color=color, friction=0.8)


def _lamp_name(k, c):
    return f"light_s{k + 1}_{c}"


def _hidden(k, ci):
    """Where an unlit lamp waits: under the table, out of every camera's view."""
    return (DRAWER_X, SLOT_Y[k], TOP - 0.30 - 0.02 * ci)


LAMPS = [Box(_lamp_name(k, c), pos=_hidden(k, ci), size=LAMP_SIZE, mass=0.01, color=rgb)
         for k in range(4) for ci, (c, rgb) in enumerate(LAMP_COLORS.items())]

OBJECTS = [
    Box("drawer_floor", pos=(DRAWER_X, 0.0, TOP + FLOOR_H / 2), size=(0.120, 0.420, FLOOR_H), color=STEEL),
    *[w for k, y in enumerate(SLOT_Y) for w in _cell(f"slot{k + 1}", DRAWER_X, y, SLOT_IN_X, SLOT_IN_Y, WALL, WALL_H,
                                                     FLOOR_Z, STEEL)],
    Box("lamp_panel", pos=(PANEL_X, 0.0, TOP + PANEL_H / 2), size=(PANEL_DX, 0.420, PANEL_H), color=(0.30, 0.31, 0.33)),
    *[w for i, (x, y) in enumerate(TRAY_XY) for w in _cell(f"tray{i + 1}", x, y, SLOT_IN_X, SLOT_IN_Y, WALL,
                                                           TRAY_WALL_H, TOP, TRAY_WHITE)],
    Box("bin_floor", pos=(*BIN_XY, TOP + BIN_FLOOR / 2), size=(BIN_IN[0] + 2 * BIN_WALL, BIN_IN[1] + 2 * BIN_WALL,
                                                               BIN_FLOOR), color=BIN_BLACK),
    *_cell("bin", *BIN_XY, *BIN_IN, BIN_WALL, BIN_H - BIN_FLOOR, BIN_FLOOR_Z, BIN_BLACK),
    *[_drive(n, (DRAWER_X, SLOT_Y[k]), FLOOR_Z, DRIVE_GREY) for k, n in enumerate(ORIGINALS)],
    *[_drive(n, TRAY_XY[i], TOP, SPARE_BLUE) for i, n in enumerate(SPARES)],
    *LAMPS,
]


# -- measuring a drive -------------------------------------------------------------------------------------------------
def _rot(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt_deg(s) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(_rot(s.quat_xyzw)[2, 2])))))


def bottom_z(s) -> float:
    """World z of the drive's lowest corner (from its body centre and orientation)."""
    r = _rot(s.quat_xyzw)
    half = np.array([DRIVE_X, DRIVE_Y, DRIVE_H]) / 2
    return float(s.pos[2] - np.abs(r[2, :]) @ half)


def slot_of(s):
    """Index of the slot whose inner footprint holds the drive's centre (and it is below the wall top), else None."""
    x, y, z = (float(v) for v in s.pos)
    if z > WALL_TOP + DRIVE_H / 2 or abs(x - DRAWER_X) > SLOT_IN_X / 2:
        return None
    for k, sy in enumerate(SLOT_Y):
        if abs(y - sy) <= SLOT_IN_Y / 2:
            return k
    return None


def seated(s, k) -> bool:
    return slot_of(s) == k and bottom_z(s) - FLOOR_Z <= SEAT_GAP and tilt_deg(s) <= SEAT_TILT


def in_bin(s) -> bool:
    """Centre inside the bin's inner footprint and the drive's lowest corner below the rim (inside, not on top)."""
    x, y, _ = (float(v) for v in s.pos)
    return abs(x - BIN_XY[0]) <= BIN_IN[0] / 2 and abs(y - BIN_XY[1]) <= BIN_IN[1] / 2 and bottom_z(s) < BIN_RIM_Z


def in_tray(s) -> bool:
    x, y, z = (float(v) for v in s.pos)
    return any(abs(x - tx) <= SLOT_IN_X / 2 and abs(y - ty) <= SLOT_IN_Y / 2 and z < TOP + DRIVE_H / 2 + 0.01
               for tx, ty in TRAY_XY)


def stowed(s) -> bool:
    """In a slot (below the wall top), a tray pocket or the returns bin: not lying loose or held."""
    return slot_of(s) is not None or in_tray(s) or in_bin(s)


# -- the hidden array state machine --------------------------------------------------------------------------------
class Array:
    def __init__(self):
        self.step = 0
        self.out = {n: False for n in ORIGINALS}          # original drive ever lifted out of its slot
        self.seat_since = {}                              # (spare, slot) -> step first seen seated
        self.data_loss = False
        self.ok_pulled = False
        self.d3_out_after_rebuild = None                  # set when drive3 first leaves
        self.status = list(HEALTH0)
        self.hands_clear = True

    def slot_status(self, objects):
        st = []
        for k in range(4):
            orig = objects[ORIGINALS[k]]
            if not self.out[ORIGINALS[k]]:
                st.append(HEALTH0[k])
                continue
            spare = next((n for n in SPARES if seated(objects[n], k)), None)
            if spare is None:
                if seated(orig, k):   # the original put back: it keeps its own health
                    st.append(HEALTH0[k])
                else:
                    st.append("empty")
                continue
            since = self.seat_since.get((spare, k), self.step)
            st.append("rebuilt" if (self.step - since) / CONTROL_HZ >= REBUILD_S else "rebuilding")
        return st

    def update(self, objects):
        self.step += 1
        for k, n in enumerate(ORIGINALS):
            s = objects[n]
            if not self.out[n] and (slot_of(s) != k or bottom_z(s) - FLOOR_Z > OUT_ABOVE):
                self.out[n] = True
                if HEALTH0[k] == OK:
                    self.ok_pulled = True
                if n == "drive3":
                    self.d3_out_after_rebuild = self.status[1] == "rebuilt"
        for n in SPARES:
            for k in range(4):
                key = (n, k)
                if seated(objects[n], k):
                    self.seat_since.setdefault(key, self.step)
                else:
                    self.seat_since.pop(key, None)
        self.status = self.slot_status(objects)
        working = sum(s in (OK, PREDICTIVE, "rebuilt") for s in self.status)
        if working < 3:
            self.data_loss = True

    def lamps(self):
        """Colour shown per slot (None = dark)."""
        if self.data_loss:
            return ["red"] * 4
        blink_on = (self.step // (CONTROL_HZ // 2)) % 2 == 0
        m = {OK: "green", PREDICTIVE: "amber", FAILED: "red", "rebuilt": "green", "empty": None,
             "rebuilding": "white" if blink_on else None}
        return [m[s] for s in self.status]

    def level(self, objects) -> float:
        if self.data_loss or self.ok_pulled or not all(stowed(objects[n]) for n in DRIVES):
            return 0.0
        st = self.status
        if not (self.out["drive2"] and in_bin(objects["drive2"])):
            return 0.0
        if st[1] != "rebuilt":
            return 0.25
        if not (self.out["drive3"] and self.d3_out_after_rebuild and in_bin(objects["drive3"])):
            return 0.5
        if st[2] != "rebuilt" or self.lamps() != ["green"] * 4 or not self.hands_clear:
            return 0.75
        return 1.0


_A: dict = {"array": Array(), "last": None}
HANDS_OVERRIDE = None   # unit tests set a list of gripper points here


def array() -> Array:
    return _A["array"]


@on_reset
def _reset(environment=None) -> None:
    _A["array"] = Array()
    _A["last"] = None
    if environment is not None:
        _show_lamps(environment, _A["array"])


def hand_points(environment):
    """Gripper body and both finger tips of both arms (world, m)."""
    if HANDS_OVERRIDE is not None:
        return [np.asarray(p, float) for p in HANDS_OVERRIDE]
    if environment is None:
        return []
    base = environment.unwrapped
    origin = base.scene.env_origins[0]
    pts = []
    for key in ("left_robot", "right_robot"):
        art = base.scene[key]
        names = list(art.body_names)
        pos = (getattr(art.data.body_pos_w, "torch", art.data.body_pos_w)[0] - origin).detach().cpu().numpy()
        pts += [pos[names.index(b)].astype(np.float64) for b in ("gripper", "tip_left", "tip_right")]
    return pts


def drawer_distance(p) -> float:
    d = np.maximum(np.maximum(DRAWER_LO - p, 0.0), p - DRAWER_HI)
    return float(np.linalg.norm(d))


def _show_lamps(environment, a: Array) -> None:
    shown = a.lamps()
    for k in range(4):
        for ci, c in enumerate(LAMP_COLORS):
            pos = (PANEL_X, SLOT_Y[k], LAMP_Z) if shown[k] == c else _hidden(k, ci)
            write_pose(environment, _lamp_name(k, c), pos, (0.0, 0.0, 0.0, 1.0))


def _trace(environment, objects, a: Array) -> None:
    path = os.environ.get("TASK_TRACE")
    if not path:
        return
    row = {"step": a.step, "status": a.status, "loss": a.data_loss, "ok_pulled": a.ok_pulled,
           "level": a.level(objects), "hands_clear": a.hands_clear,
           "drives": {n: {"pos": [round(float(v), 4) for v in objects[n].pos], "tilt": round(tilt_deg(objects[n]), 1),
                          "bottom_mm": round(1000 * (bottom_z(objects[n]) - FLOOR_Z), 1)} for n in DRIVES},
           "hands": [[round(float(v), 4) for v in p] for p in hand_points(environment)]}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


@on_step
def _step(environment) -> None:
    from yam_arena.scene import object_states

    names = DRIVES
    objects = object_states(environment, names)
    a = array()
    a.update(objects)
    pts = hand_points(environment)
    a.hands_clear = all(drawer_distance(p) > CLEAR_DIST for p in pts)
    _show_lamps(environment, a)
    _trace(environment, objects, a)


def _offline_update(objects):
    """Without the simulator (unit tests), score()/success() drive the state machine themselves, once per call."""
    from partkit import env

    if env() is None and objects is not _A["last"]:
        _A["last"] = objects
        a = array()
        a.update(objects)
        if HANDS_OVERRIDE is not None:
            a.hands_clear = all(drawer_distance(np.asarray(p, float)) > CLEAR_DIST for p in HANDS_OVERRIDE)


def score(objects) -> float:
    """0 on data loss (fewer than 3 working slots at any time), a healthy drive pulled, or any drive left outside a slot,
    tray pocket or the returns bin. 0.25 failed drive in the bin; 0.5 plus a spare rebuilt in its slot; 0.75 plus the
    predictive drive pulled after that rebuild and in the bin; 1.0 plus a spare rebuilt in its slot, all lamps green,
    both grippers 6 cm clear of the drawer."""
    _offline_update(objects)
    return array().level(objects)


def success(objects) -> bool:
    """Full score; must hold for 1 s."""
    return score(objects) == 1.0
