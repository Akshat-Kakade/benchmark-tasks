"""Scoring boundaries, every partial level, zero gate, two valid solutions, blind words. Run: python -m pytest tests -q (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def qx(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def st(pos, quat):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=quat, lin_vel=z, ang_vel=z, low=z, high=z)


def world(xs=(0.322, 0.322, 0.322), dy=(0, 0, 0), yaw=(0, 0, 0), tilt=(0, 0, 0), pyaw=0.0, px=None):
    w = {}
    for i, n in enumerate(T.NAMES):
        w[n] = st((xs[i], T.LANE_Y[i] + dy[i], T.C_Z), qx(tilt[i]) if tilt[i] else qz(yaw[i]))
    px = (min(xs) - 0.0275 - 0.0125) if px is None else px
    w["pusher"] = st((px, 0.0, T.BAR_Z), qz(pyaw))
    return w


def lv(w):
    return T.classify(w)[0]


def test_start_zero():
    w = world(xs=(T.C_X0,) * 3, px=T.BAR_X0)
    assert lv(w) == 0.0


def test_full_and_two_valid():
    l, ok, m = T.classify(world())
    assert l == 1.0 and ok, m
    assert lv(world(xs=(0.317, 0.322, 0.327))) == 1.0       # second valid end state: 10 mm spread, different depths
    assert lv(world(xs=(0.318, 0.318, 0.318))) == 1.0


def test_gates():
    assert lv(world(tilt=(0, 9, 0))) == 0.0
    assert lv(world(dy=(0, 0.016, 0))) == 0.0                 # out of its lane
    assert lv(world(xs=(0.322, 0.322, 0.335))) == 0.0        # front edge crosses the stop (0.335+0.0275 > 0.361)
    assert lv(world(xs=(0.322, 0.322, 0.332))) == 1.0 or lv(world(xs=(0.322, 0.322, 0.332))) == 0.75


def test_075():
    assert lv(world(xs=(0.305, 0.315, 0.328))) == 0.75        # spread 23 mm
    assert lv(world(dy=(0.013, 0, 0))) == 0.75
    assert lv(world(yaw=(10, 0, 0))) == 0.75
    assert lv(world(pyaw=8)) == 0.75
    assert lv(world(xs=(0.302, 0.302, 0.302))) == 0.75        # short of the 1.0 band
    assert lv(world(xs=(0.322, 0.322, 0.322), px=0.20)) == 0.75   # pusher not behind them (finger poke)


def test_05_025():
    assert lv(world(xs=(0.280, 0.322, 0.322))) == 0.5         # all >= 60 mm, one short of 0.300
    assert lv(world(xs=(0.285, 0.285, 0.320))) == 0.5         # spread too big for 0.75
    assert lv(world(xs=(0.255, 0.322, 0.322))) == 0.25        # one cartridge only 35 mm
    assert lv(world(xs=(0.225, 0.262, 0.225))) == 0.25
    assert lv(world(xs=(0.225, 0.255, 0.225))) == 0.0         # nobody 40 mm


def test_boundaries():
    assert lv(world(xs=(0.311, 0.311, 0.311))) == 0.75
    assert lv(world(xs=(0.312, 0.312, 0.312))) == 1.0
    assert lv(world(xs=(0.2999, 0.2999, 0.2999))) == 0.5
    assert lv(world(xs=(0.2799, 0.2799, 0.2799))) == 0.25 or lv(world(xs=(0.2799, 0.2799, 0.2799))) == 0.5


def test_hold_counter():
    T._reset_state()
    for i in range(T.HOLD_STEPS + 2):
        T._STATE["step"] += 1
        s = T.success(world())
    assert s


def test_blind_words():
    p = pathlib.Path(T.__file__).parent
    txt = T.__doc__ + "".join(f.read_text() for f in [p / "task.py"] + ([p / "README.md"] if (p / "README.md").exists() else []))
    assert not re.search(r"(?i)robocurve|/home/|team|\.brev", txt.replace("Dream-1", "").replace("--policy dream", "").replace("dream_run", ""))
