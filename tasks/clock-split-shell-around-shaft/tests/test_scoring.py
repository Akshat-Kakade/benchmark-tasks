"""Scoring boundaries, every partial level, zero gate, two valid solutions, geometry, blind words.
Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402

LZ = T.REST + 0.030


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def qx(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def st(pos, quat):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=quat, lin_vel=z, ang_vel=z, low=z, high=z)


def world(turn=90.0, dxs=0.0, dys=0.0, syaw=0.0, cx=0.0, cy=0.0, zs=LZ, zl=LZ, tilt=0.0):
    """Shaft at its start (+ shift), lifted; shell cavity centre offset (cx, cy) from the shaft axis, turned `turn` deg cw
    relative to a shaft that is yawed `syaw` deg."""
    sx, sy = T.SH_X + dxs, T.SH_Y + dys
    sh = st((sx, sy, zs), qx(tilt) if tilt else qz(syaw))
    sl = st((sx + cx, sy + cy, zl), qz(syaw - turn))
    return {"shaft": sh, "shell": sl}


def lv(w):
    return T.classify(w)[0]


def test_start_zero():
    w = {"shaft": st((T.SH_X, 0, T.REST), qz(0)), "shell": st((T.SH_X, 0, T.REST), qz(0))}
    assert lv(w) == 0.0


def test_full():
    l, ok, m = T.classify(world())
    assert l == 1.0 and ok, m


def test_two_valid_solutions():
    for turn, cx in ((84.0, 0.0), (96.0, 0.0), (90.0, 0.008), (90.0, -0.006)):
        assert lv(world(turn=turn, cx=cx)) == 1.0, (turn, cx)


def test_turn_levels():
    assert lv(world(turn=10)) == 0.0
    assert lv(world(turn=20)) == 0.25
    assert lv(world(turn=54)) == 0.25
    assert lv(world(turn=55)) == 0.5
    assert lv(world(turn=69)) == 0.5
    assert lv(world(turn=70)) == 0.75
    assert lv(world(turn=77)) == 0.75
    assert lv(world(turn=78)) == 1.0
    assert lv(world(turn=102)) == 1.0
    assert lv(world(turn=103)) == 0.75
    assert lv(world(turn=110)) == 0.75
    assert lv(world(turn=111)) == 0.5
    assert lv(world(turn=-60)) == 0.0              # wrong direction


def test_together_is_zero():
    assert lv(world(turn=0, syaw=-90.0)) == 0.0    # shell and shaft turned together: no relative turn


def test_shaft_drift_levels():
    assert lv(world(dxs=0.014)) == 1.0
    assert lv(world(dxs=0.016)) == 0.75
    assert lv(world(dxs=0.026)) == 0.5             # over 25 mm: not 0.75
    assert lv(world(dxs=0.031)) == 0.0             # gate
    assert lv(world(syaw=7.0, turn=90.0)) == 1.0
    assert lv(world(syaw=10.0)) == 0.75
    assert lv(world(syaw=15.0)) == 0.5
    assert lv(world(syaw=25.0)) == 0.25


def test_concentricity_and_contact():
    assert lv(world(cx=0.011)) == 1.0
    assert lv(world(cx=0.014)) == 0.75            # gap 6 mm < 8 mm
    assert lv(world(cx=0.025)) == 0.0             # wall touches the shaft: gate
    assert lv(world(cx=0.0, cy=0.016)) == 0.75    # offset along the open direction: conc > 15 mm
    m = T.measure(world())
    assert abs(m["clear"] - T.CLEAR0) < 1e-6 and T.CLEAR0 >= 0.018


def test_not_held_zero():
    assert lv(world(zs=T.REST + 0.010)) == 0.0     # shaft left standing
    assert lv(world(zl=T.REST + 0.010)) == 0.0     # shell lowered
    assert lv(world(zs=T.REST + 0.016)) == 1.0


def test_tilt_gate():
    assert lv(world(tilt=13)) == 0.0
    assert lv(world(tilt=7)) in (0.75, 1.0)


def test_baseline_uses_measured_start():
    w = world(dxs=0.012)
    assert T.classify(w, base=(T.SH_X + 0.012, 0.0, 0.0))[0] == 1.0


def test_hold_counter():
    T._reset_state()
    w = world()
    for k in range(T.BASE_STEP + 30):
        T._STATE["step"] = k + 1
        s = T.success(w)
    assert s and T.score(w) == 1.0


def test_geometry():
    # every shell box must stay >= CLEAR0 from the shaft cylinder when concentric; handle stem clears the shaft stem circle
    for (c, s, _), name in zip(T.SHELL_BOXES[:3], ("back", "side-", "side+")):
        x0, x1 = c[0] - s[0] / 2, c[0] + s[0] / 2
        y0, y1 = c[1] - s[1] / 2, c[1] + s[1] / 2
        dx = max(x0, 0, -x1) if (x0 > 0 or x1 < 0) else 0.0
        dy = max(y0, 0, -y1) if (y0 > 0 or y1 < 0) else 0.0
        assert math.hypot(dx, dy) - T.SH_R >= 0.018 - 1e-9, name
    assert T.STEM_H[2] > T.SHELL_H + 0.04 and T.SH_H > T.SHELL_H + 0.03   # stems rise above the shell top
    assert 0.07 < T.R_H - T.STEM_H[1] / 2 + 1e-9                                     # handle clears the shell outer wall
    # shaft stem (above the shell top) is not swept by any shell box
    assert T.SH_H > T.SHELL_H


def test_blind_words():
    words = ["/home/", "/users/", "github.com", "token", "api_key"]
    here = pathlib.Path(__file__).resolve().parents[1]
    for p in list(here.glob("*.py")) + [here / "README.md", here / "task.json"]:
        if not p.exists() or p.name == "test_scoring.py":
            continue
        txt = p.read_text().lower()
        for w in words:
            assert not re.search(r"\b" + re.escape(w) + r"\b", txt), (p.name, w)
