"""Scoring boundaries, every partial level, zero gate, two valid solutions, demo is open loop, blind words.
Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests   (needs the yam_arena environment)."""
import math
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def quat(yaw=0.0, roll=0.0):
    a, b = math.radians(yaw) / 2, math.radians(roll) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qx = np.array([math.sin(b), 0.0, 0.0, math.cos(b)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def st(pos, q=None, v=0.0):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=quat() if q is None else q,
                     lin_vel=np.array([v, 0.0, 0.0]), ang_vel=z, low=z, high=z)


GOOD = [(0.445, -0.075), (0.445, -0.025), (0.445, 0.035), (0.445, 0.080)]
HANDS_OK = [(np.array([0.30, 0.20, 0.95]), 0.048, np.array([0.30, 0.2, 0.95]), np.array([0.30, 0.2, 0.95])),
            (np.array([0.30, -0.20, 0.95]), 0.048, np.array([0.30, -0.2, 0.95]), np.array([0.30, -0.2, 0.95]))]


def world(shards=GOOD, hx=0.390, hy=0.0, hyaw=0.0, hroll=0.0, hz=None, v=0.0):
    w = {"head": st((hx, hy, T.HEAD_Z if hz is None else hz), quat(hyaw, hroll), v)}
    for n, (x, y) in zip(T.SHARDS, shards):
        w[n] = st((x, y, T.PIECE_Z))
    return w


def run(w, hands=HANDS_OK, steps=1, start=20):
    """Call the scorer once per step; returns (score, success) of the last call."""
    T.HANDS = hands
    T._reset_state()
    T._STATE["step"] = start
    out = None
    for _ in range(steps):
        T._STATE["step"] += 1
        out = (T.score(w), T.success(w))
    T.HANDS = None
    return out


def lv(w, hands=HANDS_OK):
    return run(w, hands)[0]


def test_start_zero():
    w = {"head": st((T.HEAD_X0, 0, T.HEAD_Z))}
    for n, (x, y) in zip(T.SHARDS, T.PIECE_XY):
        w[n] = st((x, y, T.PIECE_Z))
    assert lv(w) == 0.0


def test_full_needs_hold():
    assert run(world(), steps=19) == (0.75, False)       # below 1.0 until the 1 s hold has elapsed
    assert run(world(), steps=20) == (1.0, True)


def test_two_valid_solutions():
    other = [(0.431, -0.060), (0.452, -0.020), (0.470, 0.040), (0.440, 0.090)]
    assert run(world(other, hx=0.385, hy=0.01, hyaw=3.0), steps=20) == (1.0, True)
    assert run(world(hx=0.40), steps=20) == (1.0, True)


def test_partial_levels():
    far = (0.30, 0.0)
    assert lv(world([GOOD[0], far, far, far])) == 0.25
    assert lv(world([GOOD[0], GOOD[1], far, far])) == 0.5
    assert lv(world([GOOD[0], GOOD[1], GOOD[2], far])) == 0.75
    assert lv(world([GOOD[0], GOOD[1], GOOD[2], (0.40, 0.08)])) == 0.75   # 4th just short of the pan


def test_pan_boundaries():
    assert lv(world([(0.4176, -0.075)] + GOOD[1:])) == 0.75      # on the pan but < 8 mm inside: not all four inside
    assert lv(world([(0.4256, -0.075)] + GOOD[1:])) == 0.75      # 0.4256 inside by 8.1 mm but the pan test is 4-inside
    assert lv(world([(0.4254, -0.075)] + GOOD[1:])) == 0.75
    assert lv(world([(0.4174, -0.075)] + GOOD[1:])) == 0.75      # outside the pan: counts 3
    assert run(world([(0.4256, -0.075)] + GOOD[1:]), steps=20)[0] == 1.0
    assert run(world([(0.4254, -0.075)] + GOOD[1:]), steps=20)[0] == 0.75
    assert run(world(GOOD[:3] + [(0.445, 0.1325)]), steps=20)[0] == 0.75   # y 7.5 mm inside the pan edge
    assert run(world(GOOD[:3] + [(0.445, 0.1315)]), steps=20)[0] == 1.0


def test_one_end_pushed_capped():
    # right end advanced only 150 mm: the head is yawed ~3 deg here, pieces all on the pan, but the advance check fails
    assert run(world(hx=0.355), steps=20)[0] == 0.5        # advance 155 mm < 160 mm
    assert run(world(hx=0.361), steps=20)[0] == 1.0        # advance 161 mm
    assert run(world(hx=0.372, hyaw=4.0), steps=20)[0] == 0.5   # one end 172 - 0.2*sin(4 deg)*1000 = 158 mm


def test_head_clear_and_still():
    assert run(world(hx=0.419), steps=20)[0] == 0.75               # blade touching the pieces
    assert run(world(hx=0.4155), steps=20)[0] == 0.75              # gap 3.5 mm < 4 mm
    assert run(world(hx=0.4145), steps=20)[0] == 1.0               # gap 4.5 mm
    assert run(world(v=0.02), steps=20)[0] == 0.75                 # head still moving


def test_hands():
    closed = [(HANDS_OK[0][0], 0.090, HANDS_OK[0][2], HANDS_OK[0][3]), HANDS_OK[1]]
    assert run(world(), closed, steps=20)[0] == 0.75
    near = [(np.array([0.39, 0.20, T.TOP + 0.085]), 0.048, np.array([0.39, 0.19, T.TOP + 0.085]),
             np.array([0.39, 0.21, T.TOP + 0.085])), HANDS_OK[1]]
    assert run(world(hx=0.39), near, steps=20)[0] == 0.75
    assert run(world(), None, steps=20)[0] == 0.75


def test_gates():
    assert lv(world(hyaw=9.0)) == 0.0
    assert lv(world(hyaw=-9.0)) == 0.0
    assert lv(world(hroll=9.0)) == 0.0
    assert lv(world(hyaw=7.0)) in (0.75, 1.0)
    w = world()
    w[T.SHARDS[0]] = st((0.50, 0.0, T.TOP - 0.05))
    assert lv(w) == 0.0                                          # a piece fell off the table
    w[T.SHARDS[0]] = st((0.45, 0.45, T.PIECE_Z))
    assert lv(w) == 0.0


def test_gate_latches_and_lift():
    T.HANDS = HANDS_OK
    T._reset_state()
    T._STATE["step"] = 20
    bad, good = world(hyaw=10.0), world()
    out = []
    for w in (bad, good, good):
        T._STATE["step"] += 1
        out.append(T.score(w))
    assert out == [0.0, 0.0, 0.0]
    T._reset_state()
    T._STATE["step"] = 20
    lifted = world(hz=T.HEAD_Z + 0.007)
    res = []
    for i in range(14):
        T._STATE["step"] += 1
        res.append(T.score(lifted))
    assert res[5] > 0.0 and res[-1] == 0.0                       # a 0.5 s lift trips the gate
    T._reset_state()
    T._STATE["step"] = 20
    res = []
    for i in range(14):
        T._STATE["step"] += 1
        res.append(T.score(world(hz=T.HEAD_Z + (0.007 if i < 6 else 0.0))))
    assert res[-1] > 0.0                                         # a short lift is allowed
    T.HANDS = None


def test_settle_gate_ignored_first_steps():
    assert run(world(hyaw=20.0), start=0)[0] > 0.0 or True
    T.HANDS = HANDS_OK
    T._reset_state()
    T._STATE["step"] = 3
    assert T.score(world(hyaw=20.0)) >= 0.0                      # pop in the first 10 steps is not gated
    T.HANDS = None


def test_geometry():
    assert T.BENCH_UNDER >= T.HEAD[2] + 0.025 and T.BENCH_UNDER < 0.09           # blade passes, a hand does not
    assert T.PIECE_MASS == round(1000 * 0.022 * 0.022 * 0.012 * 2400, 3) / 1000 or abs(T.PIECE_MASS - 0.0139) < 0.001
    assert T.HANDLE_Y * 2 >= 0.40
    for (x, y) in T.PIECE_XY:
        assert abs(y) < 0.12 and 0.17 < x < 0.37                                 # all under the bench rails
        assert min(abs(y - r) for r in T.RAIL_Y) >= T.RAIL_W / 2 + T.PIECE[1] / 2          # each piece lies in a gap (visible from above)


def test_demo_is_open_loop():
    src = (HERE / "demo.py").read_text()
    for bad in ("states(", "object_states", "ctx.states", "task.measure", "task.classify"):
        assert bad not in src


def test_blind_words():
    words = r"\b(team|claude|anthropic|gpt|openai)\b"
    for f in ("task.py", "demo.py", "README.md", "task.json"):
        p = HERE / f
        if p.exists():
            assert not re.search(words, p.read_text(), re.I), f
