"""Writes assets/rocker.usda: three rigid bodies (fixed base with the stops and the post, the beam on a horizontal
revolute pivot, the counterweight on a vertical revolute turntable on the beam). Mass, centre of mass and inertia are
computed here from the parts, so the beam's balance (a ballast against the weight at the front) is by design.

    python build_assets.py            # rewrites assets/rocker.usda and prints the balance numbers
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent

GREY, DARK = (0.55, 0.57, 0.62), (0.22, 0.23, 0.26)
ORANGE, BLUE, YELLOW = (0.95, 0.45, 0.08), (0.15, 0.35, 0.85), (0.98, 0.88, 0.05)
GREEN, RED, LIGHT = (0.10, 0.70, 0.25), (0.85, 0.12, 0.12), (0.80, 0.82, 0.86)


class Part:
    def __init__(self, name, kind, centre, size, mass, color, collide=True, axis="Z"):
        """kind box: size = half sizes; kind cyl: size = (radius, height)."""
        self.name, self.kind, self.c, self.size = name, kind, np.asarray(centre, float), size
        self.mass, self.color, self.collide, self.axis = mass, color, collide, axis

    def inertia(self):
        """Diagonal inertia about the part's own centre."""
        m = self.mass
        if self.kind == "box":
            a, b, c = (2 * np.asarray(self.size))
            return np.array([m / 12 * (b * b + c * c), m / 12 * (a * a + c * c), m / 12 * (a * a + b * b)])
        r, h = self.size
        side, ax = m * (3 * r * r + h * h) / 12, m * r * r / 2
        return {"X": np.array([ax, side, side]), "Y": np.array([side, ax, side]), "Z": np.array([side, side, ax])}[
            self.axis]

    def usda(self, pad):
        api = '(\n        prepend apiSchemas = ["PhysicsCollisionAPI"]\n    )' if self.collide else ""
        col = f"color3f[] primvars:displayColor = [({self.color[0]}, {self.color[1]}, {self.color[2]})]"
        t = tuple(float(v) for v in self.c)
        if self.kind == "box":
            s = tuple(2 * float(v) for v in self.size)
            body = (f"double size = 1\n        {col}\n        double3 xformOp:translate = {t}\n"
                    f"        float3 xformOp:scale = {s}\n"
                    '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            head = "Cube"
        else:
            body = (f'uniform token axis = "{self.axis}"\n        double radius = {self.size[0]}\n'
                    f"        double height = {self.size[1]}\n        {col}\n        double3 xformOp:translate = {t}\n"
                    '        uniform token[] xformOpOrder = ["xformOp:translate"]')
            head = "Cylinder"
        text = f'    def {head} "{self.name}" {api}\n    {{\n        {body}\n    }}\n'
        return "".join(pad + line + "\n" for line in text.splitlines())


def mass_props(parts):
    m = sum(p.mass for p in parts)
    com = sum(p.mass * p.c for p in parts) / m
    inertia = np.zeros(3)
    for p in parts:
        d = p.c - com
        inertia += p.inertia() + p.mass * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2,
                                                    d[0] ** 2 + d[1] ** 2])
    return m, com, inertia


def beam_parts(m_ballast):
    bh, tz = g.BEAM_HALF, g.BEAM_HALF[2]
    return [
        Part("bar", "box", (0, 0, 0), bh, g.M_BAR, GREY),
        Part("axle", "cyl", (0, 0, 0), (0.006, 0.09), 0.0, DARK, collide=False, axis="X"),
        Part("turntable", "cyl", (0, g.YT, tz + 0.003), (g.DISK_R, 0.006), g.M_DISK, LIGHT),
        Part("ballast", "box", (0, g.YL, tz + g.BAL_HALF[2]), g.BAL_HALF, m_ballast, DARK),
        Part("left_stem", "cyl", (0, g.YL, tz + 2 * g.BAL_HALF[2] + g.STEM_H / 2), (g.STEM_R, g.STEM_H),
             g.M_STEM_L, BLUE),
        Part("mark", "box", (0, sum(g.MARK_Y) / 2, tz + 0.001), (0.017, (g.MARK_Y[1] - g.MARK_Y[0]) / 2, 0.001),
             0.0, YELLOW, collide=False),
    ]


def weight_parts():
    return [
        Part("hub", "cyl", (0, 0, 0.004), (0.012, 0.008), 0.005, DARK),
        Part("arm", "box", (0, -0.030, 0.004), g.ARM_HALF, 0.015, DARK),
        Part("block", "box", (0, -g.RHO, g.W_HALF[2]), g.W_HALF, 0.150, ORANGE),
        Part("stem", "cyl", (0, -g.RHO, 2 * g.W_HALF[2] + g.STEM_H / 2), (g.STEM_R, g.STEM_H), 0.030, ORANGE),
    ]


def base_parts():
    top = g.stop_top()
    return [
        Part("plate", "box", (0, 0, 0.003), (0.045, 0.060, 0.003), 1.0, GREY),
        Part("pedestal", "box", (0, 0, 0.013), (0.020, 0.015, 0.013), 1.0, GREY),
        Part("green_stop", "box", (0, g.YL, top / 2), (0.02, 0.05, top / 2), 1.0, GREEN),
        Part("red_stop", "box", (0, -g.YL + 0.0, top / 2), (0.02, 0.05, top / 2), 1.0, RED),
        Part("post", "cyl", (g.POST_X, g.POST_Y, g.POST_TOP / 2), (g.POST_R, g.POST_TOP), 1.0, DARK),
    ]


def body(name, origin, parts, props, pad="        "):
    m, com, inertia = props
    cf = lambda v: "(" + ", ".join(f"{float(x):.7g}" for x in v) + ")"
    head = (f'    def Xform "{name}" (\n        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"'
            + (', "PhysicsArticulationRootAPI"' if name == "base" else "") + "]\n    )\n    {\n"
            f"        float physics:mass = {m:.6g}\n        point3f physics:centerOfMass = {cf(com)}\n"
            f"        float3 physics:diagonalInertia = {cf(inertia)}\n"
            "        quatf physics:principalAxes = (1, 0, 0, 0)\n"
            f"        double3 xformOp:translate = {cf(origin)}\n"
            '        uniform token[] xformOpOrder = ["xformOp:translate"]\n\n')
    return head + "\n".join(p.usda("    ") for p in parts) + "    }\n"


def revolute(name, b0, b1, p0, p1, axis, lo, hi, damping):
    cf = lambda v: "(" + ", ".join(f"{float(x):.7g}" for x in v) + ")"
    return (f'        def PhysicsRevoluteJoint "{name}" (\n'
            '            prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysicsJointStateAPI:angular"]\n        )\n'
            "        {\n"
            f"            float drive:angular:physics:damping = {damping}\n"
            "            float drive:angular:physics:maxForce = 100\n"
            "            float drive:angular:physics:stiffness = 0\n"
            '            uniform token drive:angular:physics:type = "force"\n'
            f'            uniform token physics:axis = "{axis}"\n'
            f"            rel physics:body0 = <{b0}>\n            rel physics:body1 = <{b1}>\n"
            f"            point3f physics:localPos0 = {cf(p0)}\n            point3f physics:localPos1 = {cf(p1)}\n"
            "            quatf physics:localRot0 = (1, 0, 0, 0)\n            quatf physics:localRot1 = (1, 0, 0, 0)\n"
            f"            float physics:lowerLimit = {lo}\n            float physics:upperLimit = {hi}\n        }}\n")


def balance():
    """Ballast mass so the beam body's moment about the pivot balances the weight at the front (y = YT)."""
    m_w = sum(p.mass for p in weight_parts())
    fixed = [p for p in beam_parts(0.0)]
    moment = sum(p.mass * p.c[1] for p in fixed)
    # weight at the front: its centre of mass sits at y = YT; total moment zero
    target = -m_w * g.YT
    return (target - moment) / g.YL


def build():
    m_ballast = balance()
    beam, weight, base = beam_parts(m_ballast), weight_parts(), base_parts()
    bp = mass_props(beam)
    wp = mass_props(weight)
    base_props = (5.0, np.zeros(3), np.array([0.01, 0.01, 0.01]))
    text = ('#usda 1.0\n(\n    defaultPrim = "rocker"\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n'
            '    upAxis = "Z"\n)\n\n'
            'def Xform "rocker"\n{\n    def Xform "Geometry"\n    {\n')
    text += body("base", (0, 0, 0), base, base_props)
    text += "\n" + body("beam", (0, 0, g.PZ), beam, bp)
    text += "\n" + body("weight", (0, g.YT, g.PZ + g.TURN_TOP), weight, wp)
    text += "    }\n\n    def Scope \"Physics\"\n    {\n"
    text += revolute("beam_pivot", "/rocker/Geometry/base", "/rocker/Geometry/beam", (0, 0, g.PZ), (0, 0, 0), "X",
                     -g.LIMIT_DEG, g.LIMIT_DEG, 0.08)
    text += "\n" + revolute("weight_turn", "/rocker/Geometry/beam", "/rocker/Geometry/weight",
                            (0, g.YT, g.TURN_TOP), (0, 0, 0), "Z", g.PSI_LIMITS_DEG[0], g.PSI_LIMITS_DEG[1], 0.01)
    text += ('\n        def PhysicsFixedJoint "root_joint"\n        {\n'
             "            rel physics:body0 = </rocker>\n            rel physics:body1 = </rocker/Geometry/base>\n"
             "            point3f physics:localPos0 = (0, 0, 0)\n            point3f physics:localPos1 = (0, 0, 0)\n"
             "            quatf physics:localRot0 = (1, 0, 0, 0)\n            quatf physics:localRot1 = (1, 0, 0, 0)\n"
             "        }\n    }\n}\n")
    out = HERE / "assets" / "rocker.usda"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    return m_ballast, bp, wp


if __name__ == "__main__":
    mb, bp, wp = build()
    m_w, com_w, _ = wp
    print(f"ballast {mb:.3f} kg; beam body {bp[0]:.3f} kg, com y {bp[1][1]*1000:.1f} mm z {bp[1][2]*1000:.1f} mm; "
          f"weight {m_w:.3f} kg com radius {abs(com_w[1])*1000:.1f} mm")
    # bias torque with the weight out (psi 0) and in (pi) at a level beam, N m (positive = left end down)
    for psi in (0.0, math.pi / 2, math.pi):
        y = g.YT + (-math.sin(psi + math.pi / 2)) * 0 + com_w[1] * math.cos(psi)
        tq = -9.81 * (bp[0] * bp[1][1] + m_w * y)
        print(f"psi {math.degrees(psi):5.0f}: moment about the pivot {tq:+.3f} N m")
    print(f"stop top {g.stop_top()*1000:.1f} mm, post top {g.POST_TOP*1000:.1f} mm, weight bottom at the front: "
          f"start {g.block_bottom(g.STOP)*1000:.1f}, level {g.block_bottom(0)*1000:.1f}, tipped "
          f"{g.block_bottom(-g.STOP)*1000:.1f} mm")
