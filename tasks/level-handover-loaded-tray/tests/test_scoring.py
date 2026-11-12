"""Scoring boundaries, every partial level, zero gate, two solutions, blind words. Run: python -m pytest tests -q (yam_arena env)."""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import task as T  # noqa: E402
from partkit import PartState  # noqa: E402


def qx(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def st(pos, quat, v=0.0):
    z = np.zeros(3)
    return PartState(pos=np.array(pos, float), quat_xyzw=quat, lin_vel=np.array([v, 0, 0.0]), ang_vel=z, low=z, high=z)


def world(v=0.0, y=T.PAD_L[1], x=T.PAD_L[0], lift=0.0, tilt=0.0, yaw=0.0, cup_dy=0.0, cup_tilt=(0, 0, 0), cup_off=(False, False, False)):
    qr = qz(yaw) if not tilt else qx(tilt)
    o = {"pad_left": st((0, 0, 0), qz(0)), "pad_right": st((0, 0, 0), qz(0)),
         "rack": st((x, y, T.RACK_Z + lift), qr, v)}
    rack = o["rack"]
    for i, (n, cy) in enumerate(zip(T.CUP_NAMES, T.CUP_Y)):
        lp = (0.0, cy + cup_dy, 0.012 + 0.0005)
        if cup_off[i]:
            lp = (0.0, cy, -0.05)
        o[n] = st(rack.to_world(lp), qx(cup_tilt[i]))
    return o


OPEN, CLOSED = 0.048, 0.069


def G(a=None, b=None, w=None, arm_v=0.0):
    """Gripper readings. a / b: 'hold' (TCP on the handle nearest the arm, fingers closed), 'open' (far away, open), or None."""
    rk = w["rack"]
    ya, yb = rk.to_world((0, T.HANDLE_Y, 0.028)), rk.to_world((0, -T.HANDLE_Y, 0.028))
    far = np.array([0.3, 0.0, 0.5])
    d = {}
    for tag, mode, handle in (("A", a, ya), ("B", b, yb)):
        d[tag] = (handle if mode == "hold" else far, CLOSED if mode == "hold" else OPEN, arm_v)
    return d


def run(seq):
    """seq: list of (world, a_mode, b_mode)."""
    memo = T._fresh()
    out = None
    for w, a, b in seq:
        out = T.classify(w, memo, G(a, b, w))
    return out, memo


def S(w, a="open", b="open"):
    return (w, a, b)


def path_to_far(tilt=0.0, y_mid=0.0):
    return [S(world(lift=0.0)), S(world(lift=0.07, y=0.4, tilt=0), "hold"), S(world(lift=0.07, y=0.10, tilt=tilt), "hold"),
            S(world(lift=0.07, y=y_mid, tilt=tilt), "hold", "hold"), S(world(lift=0.07, y=y_mid, tilt=tilt), "open", "hold"),
            S(world(lift=0.07, y=-0.35, tilt=tilt), "open", "hold")]


def place(y=T.PAD_R[1], **k):
    return S(world(y=y, lift=k.pop("lift", 0.0), **k))


def test_start_zero():
    (lv, ok, _), _ = run([S(world())])
    assert lv == 0.0 and not ok


def test_025_and_not_before_cross():
    (lv, _, _), _ = run([S(world(lift=0.07, y=0.30), "hold")])
    assert lv == 0.0
    (lv, _, _), _ = run([S(world(lift=0.07, y=0.10), "hold")])
    assert lv == 0.25
    (lv, _, _), _ = run([S(world(lift=0.03, y=0.10), "hold")])
    assert lv == 0.0                               # not lifted 40 mm


def test_025_needs_left_arm_hold():
    (lv, _, _), _ = run([S(world(lift=0.07, y=0.10), "open", "hold")])   # right gripper takes it at the start
    assert lv == 0.0


def test_05_verified_transfer_level():
    (lv, _, i), _ = run(path_to_far(tilt=5.0))
    assert lv == 0.5 and i["transfer"]
    (lv, _, _), _ = run(path_to_far(tilt=12.0))    # tilted > 8 while carried: stays at 0.25
    assert lv == 0.25


def test_no_transfer_without_second_hold():
    seq = [S(world(lift=0.07, y=0.4), "hold"), S(world(lift=0.07, y=-0.35), "hold")]     # left carries across alone
    (lv, _, i), _ = run(seq)
    assert lv == 0.25 and not i["transfer"]


def test_no_transfer_if_rack_rests_between():
    seq = [S(world(lift=0.07, y=0.1), "hold"), S(world(lift=0.0, y=0.0), "hold"),         # set down mid-way
           S(world(lift=0.07, y=0.0), "open", "hold"), S(world(lift=0.07, y=-0.3), "open", "hold")]
    (lv, _, i), _ = run(seq)
    assert lv == 0.0 and not i["transfer"]       # off both pads on the table: gate


def test_wrong_order_right_first():
    seq = [S(world(lift=0.07, y=0.4), "open", "hold"), S(world(lift=0.07, y=0.0), "hold", "open"),
           S(world(lift=0.07, y=-0.35), "open", "open")]
    (_, _, i), _ = run(seq)
    assert not i["transfer"]


def test_unknown_grippers_no_credit():
    memo = T._fresh()
    lv, _, _ = T.classify(world(lift=0.07, y=0.10), memo, None)
    assert lv == 0.0


def test_placed_after_unlevel_carry_stays_025():
    seq = path_to_far(tilt=12.0) + [place()]
    (lv, ok, _), _ = run(seq)
    assert lv == 0.25 and not ok


def test_075_placed_but_cups_shifted():
    seq = path_to_far() + [place(y=T.PAD_R[1] + 0.005, cup_dy=0.03)]
    (lv, ok, _), _ = run(seq)
    assert lv == 0.75 and not ok


def test_075_yaw_boundary():
    for yaw, want in ((6.0, 1.0), (10.0, 0.5)):
        (lv, _, _), _ = run(path_to_far() + [place(yaw=yaw)])
        assert lv == want, (yaw, lv)


def test_1_0_and_tolerances():
    (lv, ok, _), _ = run(path_to_far() + [place(y=T.PAD_R[1] + 0.012)])
    assert lv == 1.0 and ok
    (lv, ok, _), _ = run(path_to_far() + [place(y=T.PAD_R[1] + 0.02)])
    assert lv == 0.5 and not ok                    # 20 mm off: handed over but not placed


def test_hover_above_pad_is_not_seated():
    (lv, ok, _), _ = run(path_to_far() + [place(lift=0.008)])    # 8 mm above the pad, band-style pose
    assert lv == 0.5 and not ok
    (lv, ok, _), _ = run(path_to_far() + [place(lift=0.002)])
    assert lv == 1.0


def test_held_on_pad_not_full():
    memo = T._fresh()
    for w, a, b in path_to_far():
        T.classify(w, memo, G(a, b, w))
    w = world(y=T.PAD_R[1])
    lv, ok, _ = T.classify(w, memo, G(None, "hold", w))          # still gripped by B
    assert lv == 0.75 and not ok


def test_moving_rack_or_arm_not_full():
    memo = T._fresh()
    for w, a, b in path_to_far():
        T.classify(w, memo, G(a, b, w))
    w = world(y=T.PAD_R[1], v=0.1)
    assert T.classify(w, memo, G(None, None, w))[0] == 0.75
    w = world(y=T.PAD_R[1])
    assert T.classify(w, memo, G(None, None, w, arm_v=0.3))[0] == 0.75
    assert T.classify(w, memo, G(None, None, w))[0] == 1.0


def test_one_second_hold_never_early(monkeypatch):
    """score() returns 0.75 until the full condition has held for HOLD_STEPS steps."""
    memo_w = path_to_far() + [place()]
    seq = iter(memo_w)
    T._reset_state()
    cur = {}
    monkeypatch.setattr(T, "grip_state", lambda: cur["g"])
    out = []
    for i, (w, a, b) in enumerate(memo_w):
        T._STATE["step"] = i + 1
        cur["g"] = G(a, b, w)
        out.append((T.score(w), T.success(w)))
    assert out[-1] == (0.75, False)
    w = memo_w[-1][0]
    for k in range(T.HOLD_STEPS + 1):
        T._STATE["step"] += 1
        cur["g"] = G(None, None, w)
        res = (T.score(w), T.success(w))
        if k < T.HOLD_STEPS - 2:
            assert res == (0.75, False), k
    assert res == (1.0, True)


def test_gate_cup_fallen_or_off():
    (lv, _, _), _ = run(path_to_far() + [place(cup_tilt=(0, 90, 0))])
    assert lv == 0.0
    (lv, _, _), _ = run(path_to_far() + [place(cup_off=(False, True, False))])
    assert lv == 0.0


def test_gate_tilt_20_during_carry_latches():
    seq = [S(world(lift=0.07, y=0.4), "hold"), S(world(lift=0.07, y=0.1, tilt=22), "hold")] + path_to_far()[2:] + [place()]
    (lv, ok, _), _ = run(seq)
    assert lv == 0.0 and not ok


def test_gate_dropped_off_pad():
    seq = [S(world(lift=0.07, y=0.4), "hold"), S(world(lift=0.07, y=0.10), "hold"), S(world(lift=0.0, y=0.0))]
    (lv, _, _), _ = run(seq)
    assert lv == 0.0


def test_two_solutions():
    for mid in (0.0, 0.06):                        # handover at y = 0 or y = 0.06
        seq = path_to_far(y_mid=mid) + [place(y=T.PAD_R[1] - 0.004, x=T.PAD_R[0] + 0.006)]
        (lv, ok, _), _ = run(seq)
        assert lv == 1.0 and ok, mid


def test_reset_on_pad_between_grips_restarts_chain():
    seq = [S(world(lift=0.07, y=0.3), "hold"), S(world(lift=0.0), "hold"),               # put back on the left pad
           S(world(lift=0.07, y=0.2), "open", "hold"), S(world(lift=0.07, y=0.0), "open", "open")]
    (lv, _, i), _ = run(seq)
    assert lv == 0.0 and not i["transfer"]


def test_demo_reads_no_simulator_state():
    src = (pathlib.Path(__file__).resolve().parents[1] / "demo.py").read_text()
    for bad in ("states(", "object_states", "_env", "body_pos", "root_pos"):
        assert bad not in src, bad
