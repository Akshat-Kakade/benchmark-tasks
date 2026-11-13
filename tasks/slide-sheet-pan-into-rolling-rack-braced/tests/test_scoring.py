"""Scoring boundaries, every partial level, zero gates, two valid solutions, blind words.
Run on the VM:  python -m pytest tests -q   (needs the yam_arena environment)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402

OPEN, CLOSED = 0.048, 0.099


def qy(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, math.sin(a), 0.0, math.cos(a)])


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def st(pos, quat=(0, 0, 0, 1.0), v=0.0, w=0.0):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=np.array(quat, float), lin_vel=np.array([v, 0, 0.0]),
                     ang_vel=np.array([w, 0, 0.0]), low=z, high=z)


def world(depth=T.FULL + 0.003, zb=T.SLOT_TOP[1], dy=0.0, tilt=0.0, rack_dx=0.0, rack_dy=0.0, rack_yaw=0.0, v=0.0,
          rack_v=0.0):
    """depth = front edge of the pan past the rack front; zb = lower face above the table."""
    tray_x = T.RX + depth - T.TD / 2
    return {"rack": st((T.RX + rack_dx, T.RY + rack_dy, T.RACK_BASE_Z), qz(rack_yaw), v=rack_v),
            "tray": st((tray_x, T.RY + dy, T.TOP + zb), qy(tilt), v=v), "counter": st((0, 0, 0))}


def G(gap=OPEN, qd=0.0):
    return {"L": (np.zeros(3), gap, qd), "R": (np.zeros(3), gap, qd)}


def run(seq, fresh=True):
    """seq: list of (world, grip). Returns the (score, success) of the last step."""
    if fresh:
        T._STATE.update(T._fresh())
    out = None
    for w, g in seq:
        T._STATE["step"] += 1
        out = T._update(w, g)
    return out


def settle(w, n=3):
    return [(world(), G())] * 16 + [(w, G())] * n


def hold(w, n=T.HOLD_STEPS + 1, g=None):
    return [(world(zb=T.COUNTER_T), G())] * 16 + [(w, g or G())] * n


def test_start_zero():
    assert run([(world(depth=-0.015, zb=T.COUNTER_T), G())] * 20) == (0.0, False)


def test_025_at_slot2_mouth():
    assert run(settle(world(depth=-0.03, zb=0.115)))[0] == 0.25
    assert run(settle(world(depth=-0.06, zb=0.115)))[0] == 0.0        # not at the mouth
    assert run(settle(world(depth=-0.03, zb=0.13)))[0] == 0.0         # 20 mm above the rib: not slot 2 height


def test_05_half_in_and_wrong_slot():
    assert run(settle(world(depth=0.13)))[0] == 0.5
    assert run(settle(world(depth=0.11)))[0] == 0.25
    assert run(settle(world(depth=T.FULL + 0.003, zb=T.SLOT_TOP[0])))[0] == 0.5   # wrong slot, fully in
    assert run(settle(world(depth=T.FULL + 0.003, zb=T.SLOT_TOP[2])))[0] == 0.5
    assert run(settle(world(depth=0.13, zb=T.SLOT_TOP[0])))[0] == 0.5


def test_075_full_but_hold_not_done():
    out = run(settle(world()))
    assert out == (0.75, False)
    assert run(settle(world(depth=T.FULL - 0.002)))[0] == 0.5       # 12 mm short of the stop


def test_100_only_after_hold_and_when_all_true():
    seq = hold(world(zb=T.SLOT_TOP[1] + 0.001), n=T.HOLD_STEPS - 1)
    assert run(seq) == (0.75, False)
    seq = hold(world(zb=T.SLOT_TOP[1] + 0.001), n=T.HOLD_STEPS)
    assert run(seq) == (1.0, True)


def test_hold_resets():
    w = world(zb=T.SLOT_TOP[1] + 0.001)
    seq = hold(w, n=15) + [(w, G(gap=CLOSED))] + [(w, G())] * 10
    assert run(seq)[1] is False
    seq = hold(w, n=15) + [(w, G(qd=0.2))] + [(w, G())] * 10
    assert run(seq)[1] is False
    seq = hold(world(zb=T.SLOT_TOP[1] + 0.001, v=0.05), n=30)
    assert run(seq)[1] is False                                       # pan still sliding


def test_not_resting_on_ribs_no_full():
    seq = hold(world(zb=T.SLOT_TOP[1] + 0.008), n=40)                   # 8 mm above the rib: 0.75 only
    assert run(seq) == (0.75, False)


def test_gate_rack_slid():
    for dx in (0.045, -0.045):
        assert run(hold(world(zb=T.SLOT_TOP[1] + 0.001, rack_dx=dx), n=40))[0] == 0.0
    assert run(hold(world(zb=T.SLOT_TOP[1] + 0.001, rack_dx=0.035), n=40))[1] is True


def test_gate_latches():
    w = world(zb=T.SLOT_TOP[1] + 0.001)
    seq = [(world(zb=T.COUNTER_T), G())] * 16 + [(world(rack_dx=0.06, zb=0.115, depth=0.1), G())] + [(w, G())] * 40
    assert run(seq) == (0.0, False)


def test_gate_rack_yaw_both_directions():
    for yaw in (9.0, -9.0):
        assert run(hold(world(zb=T.SLOT_TOP[1] + 0.001, rack_yaw=yaw), n=30))[0] == 0.0
    assert run(hold(world(zb=T.SLOT_TOP[1] + 0.001, rack_yaw=6.0), n=30))[1] is True


def test_gate_tray_tilt_and_fall():
    seq = [(world(zb=T.COUNTER_T), G())] * 16 + [(world(depth=-0.03, zb=0.115, tilt=16.0), G())] * 3
    assert run(seq)[0] == 0.0
    seq = [(world(zb=T.COUNTER_T), G())] * 16 + [(world(depth=-0.3, zb=0.0), G())] * 3     # fell to the table
    assert run(seq)[0] == 0.0
    seq = [(world(zb=T.COUNTER_T), G())] * 16 + [(world(depth=-0.03, zb=0.115, tilt=12.0), G())] * 3
    assert run(seq)[0] == 0.0 or run(seq)[0] == 0.25                    # 12 deg: no gate, not level (no 0.5/0.75)


def test_pan_off_centre():
    assert run(settle(world(dy=0.02)))[0] <= 0.5


def test_two_valid_solutions_same_score():
    """Left hand braces / right hand pushes, or the mirror: the scorer sees only the end state and both grippers open."""
    w = world(zb=T.SLOT_TOP[1] + 0.001)
    a = run(hold(w))
    b = run(hold(w, g={"L": (np.ones(3), OPEN, 0.0), "R": (-np.ones(3), OPEN, 0.0)}))
    assert a == b == (1.0, True)


def test_blind_words():
    here = pathlib.Path(__file__).resolve().parents[1]
    bad = re.compile("|".join(["ma" + "nas", "mani" + "pal", "ak" + "shat", "kak" + "ade", "cla" + "ude", "anthr" + "opic"]), re.I)
    for f in list(here.glob("*.py")) + list(here.glob("*.md")) + list(here.glob("*.json")):
        assert not bad.search(f.read_text()), f


def test_no_pose_reads_in_demo():
    src = (pathlib.Path(__file__).resolve().parents[1] / "demo.py").read_text()
    for w in ("object_states", "ctx.states", "._env"):
        assert w not in src, w
