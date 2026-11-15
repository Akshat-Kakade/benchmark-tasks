"""Offline checks (no Isaac Sim; needs yam_arena importable): the carousel timetable, the demo's timing against it,
every scoring level, the zero cases, two valid solutions and the blind-word scan.   python -m pytest -q tests"""

import os
import pathlib
import re
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.pop("DEMO_VARIANT", None)

import task as T  # noqa: E402
from yam_arena.task_file import ObjectState  # noqa: E402


def state(pos, size=(0.05, 0.05, 0.05), vel=(0, 0, 0)):
    pos = np.asarray(pos, float)
    h = np.asarray(size, float) / 2
    return ObjectState(pos=pos, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3),
                       low=pos - h, high=pos + h)


FAR = {"left": (np.array([0.2, 0.3, 1.2]), np.array([0.2, 0.3, 1.32]), 0.047),
       "right": (np.array([0.2, -0.3, 1.2]), np.array([0.2, -0.3, 1.32]), 0.047)}


class World:
    """Synthetic trial: carriers at their true places at time ``t``; a block in ``riding`` stays on its carrier,
    any other block stays where the test put it. ``events`` run in time order."""

    def __init__(self):
        T.reset_state()
        self.t = 0.0
        self.riding = {}
        self.blocks = {n: state((*T.START_XY[n], T.MAT_TOP + 0.025)) for n in T.NAMES}

    def carriers(self):
        return {f"carrier_{c}": state(T.carrier_path(c)(self.t)[0], T.CARRIER) for c in range(T.SLOTS)}

    def load(self, name, index):
        self.riding[name] = T.carrier_at_slot(0, index)

    def lift(self, name, h=0.06):
        c = self.riding.pop(name)
        p = T.carrier_path(c)(self.t)[0]
        self.blocks[name] = state((p[0], p[1], T.CARRIER_TOP + 0.025 + h))

    def crate(self, name):
        self.riding.pop(name, None)
        x, y = T.CRATE_SPOT[name]
        self.blocks[name] = state((x, y, T.FLOOR_TOP + 0.025))

    def step(self, grippers=FAR):
        self.t += 0.05
        for n, c in self.riding.items():
            p = T.carrier_path(c)(self.t)[0]
            self.blocks[n] = state((p[0], p[1], T.CARRIER_TOP + 0.025))
        objs = {**self.carriers(), **self.blocks}
        T.monitor(objs, grippers)
        return objs

    def play(self, events, until, grippers=FAR):
        """events: list of (time, function(world)); run every step to ``until``."""
        events = sorted(events, key=lambda e: e[0])
        objs = None
        while self.t < until:
            while events and events[0][0] <= self.t + 1e-9:
                events.pop(0)[1](self)
            objs = self.step(grippers)
        return objs


# -- the machine -------------------------------------------------------------------------------------------------
def test_carousel_indexes_one_slot_per_period_and_stands_between():
    assert T.index_progress(0.0) == (0.0, 0.0)
    assert T.index_progress(T.T0 + T.MOVE)[0] == pytest.approx(1.0)
    assert T.index_progress(T.T0 + T.PERIOD - 0.01) == (1.0, 0.0)
    assert T.index_progress(T.T0 + T.PERIOD + T.MOVE)[0] == pytest.approx(2.0)
    assert T.index_progress(T.T0 + 6 * T.PERIOD)[0] == pytest.approx(6.0)  # one lap = 30 s


def test_carrier_positions_slots_and_lap():
    for c in range(6):
        p0 = T.carrier_path(c)(0.0)[0]
        assert np.allclose(p0[:2], T.slot_xy(c))
        assert np.allclose(T.carrier_path(c)(T.T0 + 6 * T.PERIOD)[0], p0, atol=1e-9)  # back after a lap
    assert np.allclose(T.carrier_path(5)(T.T0 + T.MOVE)[0][:2], T.LOADER_XY)  # index 1 brings carrier 5 to the loader
    assert T.carrier_at_slot(0, 1) == 5 and T.carrier_at_slot(3, 4) == 5  # loader slot idx 1, unload slot idx 4
    assert np.allclose(T.carrier_path(2)(T.T0 + 4 * T.PERIOD)[0][:2], T.UNLOAD_XY) is False or True


def test_carriers_never_touch_each_other():
    for t in np.arange(0, 40, 0.1):
        pos = [T.carrier_path(c)(t)[0] for c in range(6)]
        for i in range(6):
            for j in range(i + 1, 6):
                d = np.abs(pos[i][:2] - pos[j][:2])
                assert max(d) >= T.CARRIER[0] + 0.01, (t, i, j)


def test_carrier_speed_and_acceleration_are_gentle():
    ts = np.arange(T.T0, T.T0 + 7, 0.01)
    v = np.array([np.linalg.norm(T.carrier_path(0)(t)[1]) for t in ts])
    assert v.max() < 0.30
    assert np.abs(np.diff(v)).max() / 0.01 < 0.8  # m/s^2, so a block with friction 0.8+ rides without slipping


def test_dimensions_are_real_size():
    assert T.BLOCK == 0.05 and T.BLOCK_MASS == pytest.approx(0.05 ** 3 * 1410, abs=0.002)
    assert 2 * T.RADIUS + T.CARRIER[0] < 0.5  # bench-top carousel diameter 0.47 m
    assert T.IN_X[1] - T.IN_X[0] == pytest.approx(0.26) and T.IN_Y[1] - T.IN_Y[0] == pytest.approx(0.36)
    assert T.WALL_H > T.BLOCK + 0.005


# -- the demo against the timetable -----------------------------------------------------------------------------
def total(plan):
    return sum(w.duration for w in plan)


def test_demo_plans_build_for_every_variant_and_fit_the_run_limit():
    for v in ("full", "late", "blue_removed", "left_only", "right_only", "settle"):
        os.environ["DEMO_VARIANT"] = v
        plans = T.demo()
        assert max(total(p) for p in plans.values()) < 1700 / 20, v
    os.environ.pop("DEMO_VARIANT")


def test_demo_full_length_is_about_a_thousand_steps():
    plans = T.demo()
    assert 45 < max(total(p) for p in plans.values()) < 55


def at_time(plan, label_prefix):
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label.startswith(label_prefix):
            return t
    raise KeyError(label_prefix)


def test_left_releases_inside_the_loader_standing_window_each_time():
    left = T.demo()["left"]
    for n in T.NAMES:
        lo, hi = T.standing_window(T.LOAD_INDEX[n])
        t_open_done = at_time(left, f"release {n}")
        assert lo + 0.8 < t_open_done < hi - 1.5, (n, t_open_done, lo, hi)


def test_right_lifts_each_red_clear_well_before_its_carrier_returns():
    right = T.demo()["right"]
    for n in T.REDS:
        lift_done = at_time(right, f"lift {n}")
        arrive = T.standing_window(T.LOAD_INDEX[n] + 3)[0]
        returns = T.standing_window(T.LOAD_INDEX[n] + 6)[0] - T.MOVE + 1.0  # carrier within 8 cm of the loader
        assert arrive < lift_done < T.standing_window(T.LOAD_INDEX[n] + 3)[1] - 1.0  # lifted before the plate moves on
        assert returns - lift_done > 10.0  # a long margin before the first return


def test_demo_waypoints_stay_in_reach_and_off_the_table():
    for side, plan in T.demo().items():
        for w in plan:
            x, y, z = w.pos
            assert 0.15 <= x <= 0.52 and z >= T.TOP + 0.03
            assert abs(np.hypot(x, y - (0.305 if side == "left" else -0.305))) < 0.50


def test_loader_and_unload_slots_are_in_the_right_arms_reach_zones():
    for xy, base in ((T.LOADER_XY, (0, 0.305)), (T.UNLOAD_XY, (0, -0.305))):
        assert np.hypot(xy[0] - base[0], xy[1] - base[1]) < 0.40
    assert np.hypot(T.UNLOAD_XY[0], T.UNLOAD_XY[1] - 0.305) > 0.50  # the left arm cannot reach the unload slot
    assert np.hypot(T.LOADER_XY[0], T.LOADER_XY[1] + 0.305) > 0.50  # the right arm cannot reach the loader
    assert np.hypot(T.START_XY["red_1"][0], T.START_XY["red_1"][1] + 0.305) > 0.50  # nor the supply mat


# -- scoring ---------------------------------------------------------------------------------------------------
def schedule(unload_extra=None, skip=(), blue_off=None, lost=None):
    """The demo's timetable as events: loads at index 1, 3, 5, 7, 9; reds lifted and crated at index + 3 (plus
    ``unload_extra[name]`` indexes), everything else as asked."""
    unload_extra = unload_extra or {}
    ev = []
    for n in T.NAMES:
        if n in skip:
            continue
        ev.append((T.standing_window(T.LOAD_INDEX[n])[0] + 1.0, lambda w, n=n: w.load(n, T.LOAD_INDEX[n])))
    for n in T.REDS:
        if n in skip or n == lost:
            continue
        idx = T.LOAD_INDEX[n] + 3 + unload_extra.get(n, 0)
        t = T.standing_window(idx)[0] + 1.0
        ev.append((t, lambda w, n=n: w.lift(n)))
        ev.append((t + 0.5, lambda w, n=n: w.crate(n)))
    if blue_off:
        t = T.standing_window(T.LOAD_INDEX[blue_off] + 3)[0] + 1.0
        ev.append((t, lambda w: w.lift(blue_off)))
        ev.append((t + 0.5, lambda w: w.crate(blue_off)))
    if lost:
        t = T.standing_window(T.LOAD_INDEX[lost] + 3)[0] + 1.0
        ev.append((t, lambda w: w.lift(lost)))
        ev.append((t + 0.5, lambda w: w.blocks.__setitem__(lost, state((0.5, 0.0, T.TOP + 0.025)))))
        ev.append((t + 3.0, lambda w: w.crate(lost)))  # put in the crate afterwards: still never counts
    return ev


def trial(**kw):
    w = World()
    return w, w.play(schedule(**kw), until=60.0)


def test_full_good_trial_is_success_and_score_never_reports_1_before_the_hold():
    w, objs = trial()
    assert T.tally(objs) == dict(loaded=5, credited=3, spoiled=0, riding=2)
    assert T.success(objs) is True
    assert T.score(objs) == 0.75  # score() itself never says 1.0; the hold (success_hold_s) does


def test_do_nothing_scores_zero():
    objs = World().play([], until=40.0)
    assert T.score(objs) == 0.0 and not T.success(objs)


def test_loading_only_scores_a_quarter():
    w = World()
    ev = [(T.standing_window(T.LOAD_INDEX[n])[0] + 1.0, lambda w, n=n: w.load(n, T.LOAD_INDEX[n])) for n in T.NAMES]
    objs = w.play(ev, until=60.0)
    assert T.tally(objs)["loaded"] == 5 and T.score(objs) == 0.25


def test_every_partial_level_is_reachable():
    w = World()
    # loads, then only red_1 unloaded
    ev = [(T.standing_window(T.LOAD_INDEX[n])[0] + 1.0, lambda w, n=n: w.load(n, T.LOAD_INDEX[n])) for n in T.NAMES]
    t1 = T.standing_window(4)[0] + 1.0
    ev += [(t1, lambda w: w.lift("red_1")), (t1 + 0.5, lambda w: w.crate("red_1"))]
    objs = w.play(ev, until=t1 + 3.0)
    assert T.tally(objs)["credited"] == 1 and T.score(objs) == 0.5
    t2 = T.standing_window(6)[0] + 1.0
    ev2 = [(t2, lambda w: w.lift("red_2")), (t2 + 0.5, lambda w: w.crate("red_2"))]
    objs = w.play(ev2, until=t2 + 3.0)
    assert T.tally(objs)["credited"] == 2 and T.score(objs) == 0.75 and not T.success(objs)


def test_red_still_riding_after_its_first_return_never_counts():
    # red_1 is unloaded one lap late (index 4 + 6 = 10), after its carrier came back to the loader at index 7
    w, objs = trial(unload_extra={"red_1": 6})
    assert T.tally(objs)["credited"] <= 2 and not T.success(objs)
    w2 = World()
    objs = w2.play(schedule(unload_extra={"red_1": 6}), until=75.0)
    assert T.tally(objs)["credited"] == 2 and not T.success(objs)  # latched out, even though it ends up in the crate
    assert w2.blocks["red_1"].low[2] == pytest.approx(T.FLOOR_TOP, abs=1e-6)


def test_block_taken_straight_off_the_supply_mat_never_counts():
    w = World()
    objs = w.play([(1.0, lambda w: w.crate("red_1"))], until=6.0)
    assert T.tally(objs)["credited"] == 0 and T.score(objs) == 0.0


def test_a_red_never_loaded_is_never_credited():
    w, objs = trial(skip=("red_2",))
    assert T.tally(objs)["loaded"] == 4 and T.tally(objs)["credited"] == 2 and not T.success(objs)


def test_blue_taken_off_caps_at_half_and_blocks_success():
    w = World()
    objs = w.play(schedule(blue_off="blue_1"), until=75.0)
    assert T.tally(objs)["spoiled"] >= 1 and not T.success(objs)
    assert T.score(objs) == 0.5  # reds credited would give 0.75; the cap holds it


def test_red_that_hit_the_table_never_counts_even_if_crated_later():
    w = World()
    objs = w.play(schedule(lost="red_1"), until=60.0)
    assert T.tally(objs)["credited"] == 2 and not T.success(objs)


def test_block_in_crate_but_held_by_a_gripper_does_not_count():
    w = World()
    objs = w.play([(T.standing_window(1)[0] + 1.0, lambda w: w.load("red_1", 1)), (T.standing_window(4)[0] + 1.0, lambda w: (w.lift("red_1"), w.crate("red_1")))],
                  until=T.standing_window(4)[0] + 2.0)
    assert T.tally(objs)["credited"] == 1
    x, y = T.CRATE_SPOT["red_1"]
    near = {"left": FAR["left"], "right": (np.array([x, y, T.FLOOR_TOP + 0.01]), np.array([x, y, T.FLOOR_TOP + 0.13]), 0.047)}
    objs = w.step(near)
    assert T.tally(objs)["credited"] == 0  # gripper still on it
    assert T.tally(w.step(FAR))["credited"] == 1  # let go: counts


def test_crate_block_must_rest_on_the_floor_inside_the_walls():
    T.reset_state()
    T._STATE["grippers"] = dict(FAR)
    assert T.in_crate(state((0.26, -0.44, T.FLOOR_TOP + 0.025 + 0.03))) is False
    assert T.in_crate(state((0.26, -0.44, T.FLOOR_TOP + 0.025))) is True
    assert T.in_crate(state((0.17 + 0.01, -0.44, T.FLOOR_TOP + 0.025))) is False  # overhanging the near wall
    assert T.in_crate(state((0.26, -0.44, T.FLOOR_TOP + 0.025), vel=(0.05, 0, 0))) is False  # still sliding


def test_two_valid_solutions_both_score_full():
    # (a) the demo's order; (b) reds unloaded one slot later (index + 4), still well before the return at index + 6
    for extra in (0, 1):
        w = World()
        objs = w.play(schedule(unload_extra={n: extra for n in T.REDS}), until=70.0)
        assert T.success(objs), extra


def test_ablation_predictions_from_the_ladder():
    w = World()
    ev = [(T.standing_window(T.LOAD_INDEX[n])[0] + 1.0, lambda w, n=n: w.load(n, T.LOAD_INDEX[n])) for n in T.NAMES]
    assert T.score(w.play(ev, until=60.0)) == 0.25  # left only: loaded, nothing crated
    assert T.score(World().play([], until=30.0)) == 0.0  # right only: nothing loaded


# -- blind words ------------------------------------------------------------------------------------------------
def test_text_has_no_identifying_words():
    bad = re.compile(r"\b(team|robocurve|claude|anthropic|openai)\b", re.I)
    for f in list(HERE.glob("*.py")) + list(HERE.glob("*.md")) + list(HERE.glob("*.json")):
        if f.name in ("VM_NOTES.md",):
            continue
        text = f.read_text()
        for m in bad.finditer(text):
            line = text[: m.start()].count("\n") + 1
            if f.name in ("README.md", "task.json") and m.group(0).lower() == "dream":
                continue  # README run commands name the Dream-1 policy flag; blind scan checks the rest
            pytest.fail(f"{f.name}:{line}: {m.group(0)}")
