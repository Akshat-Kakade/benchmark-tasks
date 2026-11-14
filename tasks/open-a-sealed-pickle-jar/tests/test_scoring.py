"""Scoring boundaries, every partial level, zero gates, two valid solutions, lever geometry, blind words.
Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q   (needs the yam_arena environment)"""
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import geometry as g  # noqa: E402
import task as T  # noqa: E402
from scenekit import BodyState  # noqa: E402

TOP = g.TOP


def quat(axis, deg):
    b = math.radians(deg) / 2
    q = np.zeros(4)
    q["xyz".index(axis)] = math.sin(b)
    q[3] = math.cos(b)
    return q


def jar(tilt=0.0, dx=0.0, dy=0.0, v=0.0, w=0.0, z=TOP):
    z0 = np.zeros(3)
    return BodyState(pos=np.array([g.JAR_X + dx, g.JAR_Y + dy, z]), quat_xyzw=quat("x", tilt), lin_vel=np.array([v, 0, 0]),
                     ang_vel=np.array([0, 0, w]), low=z0, high=z0, joints={})


def lid_on_table(side=-1, r=0.17):
    return (g.JAR_X, g.JAR_Y + side * r, TOP + g.LID_H / 2)


def lid_on_jar():
    return (g.JAR_X, g.JAR_Y, TOP + g.LID_CZ)


def grips(near=False, open_=True, left_side=True):
    """Synthetic grips: left TCP 0.12 m behind the jar (or on it), right TCP 0.10 m above the lid."""
    gl = np.array([g.JAR_X, 0.12, TOP + 0.06]) if not near else np.array([g.JAR_X, 0.0, TOP + 0.06])
    gr = np.array([g.JAR_X, -0.17, TOP + 0.12])
    gap = 0.048 if open_ else 0.09
    return {"L": (gl, gap), "R": (gr, gap)}


def memo(**kw):
    m = T._fresh()
    m["step"] = 10
    m.update(kw)
    return m


def lv(j, lid, m=None, gr=None, speed=0.0):
    return T.classify(j, lid, speed, m if m is not None else memo(), gr if gr is not None else grips())


def test_do_nothing_is_zero():
    assert lv(jar(), lid_on_jar())[0] == 0.0


def test_levels():
    assert lv(jar(), lid_on_jar(), memo(popped=True))[0] == 0.25
    assert lv(jar(), lid_on_jar(), memo(popped=True, unlatched=True))[0] == 0.5
    off = (g.JAR_X, -0.06, TOP + 0.1)                 # in the air beside the jar
    assert lv(jar(), off, memo(popped=True, unlatched=True))[0] == 0.75
    assert lv(jar(), lid_on_table(), memo(popped=True, unlatched=True), speed=0.05)[0] == 0.75    # still moving
    assert lv(jar(), lid_on_table(), memo(popped=True, unlatched=True), gr=grips(near=True))[0] == 0.75
    assert lv(jar(), lid_on_table(), memo(popped=True, unlatched=True))[0] == 1.0


def test_lid_off_boundary():
    m = memo(popped=True, unlatched=True)
    on = (g.JAR_X, g.JAR_Y - (g.LID_OFF - 0.002), TOP + 0.1)
    off = (g.JAR_X, g.JAR_Y - (g.LID_OFF + 0.002), TOP + 0.1)
    assert lv(jar(), on, m)[0] == 0.5 and lv(jar(), off, m)[0] == 0.75


def test_lid_clearance_boundary():
    m = memo(popped=True, unlatched=True)
    near = lid_on_table(r=g.LID_CLEAR - 0.004)
    far = lid_on_table(r=g.LID_CLEAR + 0.004)
    assert lv(jar(), near, m)[0] == 0.75 and lv(jar(), far, m)[0] == 1.0


def test_lid_must_rest_on_table():
    m = memo(popped=True, unlatched=True)
    held = (g.JAR_X, -0.17, TOP + g.LID_H / 2 + 0.03)
    assert lv(jar(), held, m)[0] == 0.75


def test_jar_conditions():
    m = memo(popped=True, unlatched=True)
    assert lv(jar(tilt=12), lid_on_table(), m)[0] == 0.75
    assert lv(jar(tilt=8), lid_on_table(), m)[0] == 1.0
    assert lv(jar(dx=0.07), lid_on_table(), m)[0] == 0.75
    assert lv(jar(v=0.05), lid_on_table(), m)[0] == 0.75
    assert lv(jar(w=0.5), lid_on_table(), m)[0] == 0.75


def test_grippers_open_and_clear():
    m = memo(popped=True, unlatched=True)
    assert lv(jar(), lid_on_table(), m, gr=grips(open_=False))[0] == 0.75
    assert lv(jar(), lid_on_table(), m, gr={})[0] == 0.75


def test_spill_gate():
    m = memo(popped=True, unlatched=True)
    assert lv(jar(tilt=32), lid_on_table(), m)[0] == 0.0 and m["gate"]
    assert lv(jar(), lid_on_table(), m)[0] == 0.0                     # permanent
    m2 = memo(popped=True)                                           # sealed lid: the brine cannot spill
    assert lv(jar(tilt=32), lid_on_jar(), m2)[0] == 0.25 and not m2["gate"]
    assert lv(jar(z=TOP - 0.05), lid_on_jar(), memo(popped=True, unlatched=True))[0] == 0.0


def test_two_valid_solutions_both_full():
    """Lid set down on either side of the jar (either arm may hold the jar), both reach 1.0."""
    for side in (-1, 1):
        for gr in (grips(), {"L": grips()["R"], "R": grips()["L"]}):
            assert lv(jar(), lid_on_table(side=side), memo(popped=True, unlatched=True), gr=gr)[0] == 1.0


def test_unlatch_direction():
    assert T.loosened(math.radians(80)) and T.loosened(math.radians(95))
    assert not T.loosened(math.radians(79))
    assert not T.loosened(-math.radians(80)) and not T.loosened(-math.radians(170))     # tightening never unlatches
    spins = [0, 60, -50, -90, 20, 79, 30, 80]                                        # mixed back and forth: only the last reaches 80 the loosening way
    seen = [T.loosened(math.radians(x)) for x in spins]
    assert seen == [False] * 7 + [True]


def test_tightening_earns_at_most_pop():
    # a tightening turn leaves 'unlatched' False, so the best level is the pop alone
    m = memo(popped=True)
    assert not T.loosened(-math.radians(120))
    assert lv(jar(), lid_on_jar(), m)[0] == 0.25


def test_hold_is_20_steps():
    assert T.g.HOLD_STEPS == 20


def lever(tip, yaw_deg=90.0):
    """A lifter BodyState lying level, tip at ``tip`` (world), pointing along +y for yaw 90."""
    from scipy.spatial.transform import Rotation as Rot
    origin = np.asarray(tip, float) - np.array(g.TIP_LOCAL)
    q = Rot.from_euler("z", yaw_deg, degrees=True).as_quat()
    # rotate the tip offset: place the origin so that R @ TIP_LOCAL + origin == tip
    R = Rot.from_euler("z", yaw_deg, degrees=True).as_matrix()
    origin = np.asarray(tip, float) - R @ np.array(g.TIP_LOCAL)
    z0 = np.zeros(3)
    return BodyState(pos=origin, quat_xyzw=q, lin_vel=z0, ang_vel=z0, low=z0, high=z0, joints={})


def test_prise_geometry():
    j = jar()
    ok = (g.JAR_X, g.JAR_Y - 0.0335, TOP + 0.1065)
    assert not T.prise(j, lever(ok), 0.0)["ok"]                                       # blade under the edge but the lid not raised: hovering pays nothing
    assert T.prise(j, lever((g.JAR_X, g.JAR_Y - 0.0335, TOP + 0.1175)), 0.002)["ok"]  # lid edge raised 2 mm by the blade
    assert not T.prise(j, lever((g.JAR_X, g.JAR_Y - 0.0335, TOP + 0.1175)), 0.0005)["ok"]   # raised less than 1 mm
    assert not T.prise(j, lever((g.JAR_X, g.JAR_Y - 0.060, TOP + 0.1065)), 0.002)["ok"]    # tip outside the lid skirt (lid raised by something else)
    assert not T.prise(j, lever((g.JAR_X, g.JAR_Y - 0.0335, TOP + 0.16)), 0.002)["ok"]     # blade above the lid
    assert not T.prise(j, lever(ok, yaw_deg=-90), 0.002)["ok"]                         # pointing away
    assert not T.prise(j, lever(ok, yaw_deg=0), 0.002)["ok"]                           # sideways
    assert not T.prise(j, lever((g.JAR_X, g.JAR_Y - 0.0335, TOP + 0.05)), 0.002)["ok"]     # blade on the table


def test_demo_is_open_loop():
    src = (pathlib.Path(__file__).resolve().parents[1] / "demo.py").read_text()
    for bad in ("ctx.states", "object_states", ".states(", "pickle_jar\"].pos", "lever_tool\"].pos"):
        assert bad not in src


def test_tool_mass_consistent():
    steel = (125 * 24 * 2) * 1e-9 * 7850        # kg
    abs_ = 100 * 10 * 40 * 1e-9 * 1100
    assert abs(g.TOOL_MASS - (steel + abs_)) < 0.005
    assert 0.5 <= g.JAR_MASS <= 0.9
    glass = (2 * math.pi * g.BODY_R * g.BODY_H * 0.003 + math.pi * g.BODY_R ** 2 * 0.005 + 2 * math.pi * g.NECK_R * g.NECK_H * 0.003) * 2500
    assert abs(glass - g.GLASS_MASS) < 0.04, glass
    fill = math.pi * (g.BODY_R - 0.003) ** 2 * (g.FILL_TOP - 0.005) * 1100
    assert abs(fill - g.FILL_MASS) < 0.03, fill


def test_spill_angle():
    tan = 2 * g.HEADSPACE / (2 * (g.BODY_R - 0.003))
    assert math.degrees(math.atan(tan)) > math.degrees(g.SPILL_TILT)


_W = ["ma" + "nas", "mani" + "pal", "ma" + "he", "aks" + "hat", "kak" + "ade", "te" + "am", "clau" + "de", "anthro" + "pic", "robo" + "curve"]
BLIND = re.compile(r"\b(" + "|".join(_W) + r")\b", re.I)


def test_blind_words():
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in ("task.py", "README.md", "task.json", "demo.py", "geometry.py"):
        p = root / f
        if p.exists():
            assert not BLIND.search(p.read_text()), f
