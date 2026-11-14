"""Write the scene's USD assets from geometry.py.

    python build_assets.py          # needs pxr (Isaac Sim's Python), not the simulator

- assets/pickle_jar.usda: a floating articulation (the jar stands free on the table). The lid hangs off the jar on four joints:
  ``lid_x``, ``lid_y``, ``lid_z`` (prismatic; task.py keeps them locked until the lid is unlatched) and ``lid_spin`` (revolute about
  the lid's own axis, with a friction drive whose torque limit task.py sets: locked, then 0.4 N m, then free).
  The lid and the jar never collide with each other (one articulation), so the lid's solid cylinder may overlap the neck.
- assets/lever_tool.usda: the lever tool, one rigid body (steel blade with a hooked lip, plastic grip fin).

The jar asset's origin is the centre of the jar's base; the tool's is the fin centre on the blade underside.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import geometry as g

OUT = Path(__file__).resolve().parent / "assets"
GLASS = (0.10, 0.50, 0.22)
PICKLE = (0.36, 0.45, 0.14)
GOLD = (0.85, 0.70, 0.20)
DARK = (0.30, 0.22, 0.05)
STEEL = (0.72, 0.73, 0.76)
ORANGE = (0.95, 0.50, 0.08)


def stage_for(name):
    path = OUT / f"{name}.usda"
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def material(stage, path, color, roughness=0.5, opacity=1.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def phys_material(stage, path, static, dynamic):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return mat


def bind_phys(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)


def body(stage, path, mass, pos=(0.0, 0.0, 0.0), inertia=None):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    m = UsdPhysics.MassAPI.Apply(xf.GetPrim())
    m.CreateMassAttr(mass)
    if inertia is not None:
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
        m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    return xf


def cyl(stage, path, radius, height, z, color, collide=True, mat=None, torsional=0.0):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if torsional:
            c.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
            c.GetPrim().CreateAttribute("physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(torsional)
            c.GetPrim().CreateAttribute("physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(0.8 * torsional)
    if mat is not None:
        bind(c, mat)
    return c


def box(stage, path, centre, size, color, collide=True, mat=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*centre))
    c.AddScaleOp().Set(Gf.Vec3d(*size))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if mat is not None:
        bind(c, mat)
    return c


def joint(stage, path, kind, axis, body0, body1, pos0, limits=None, stiffness=0.0, damping=0.0, max_force=None):
    j = (UsdPhysics.PrismaticJoint if kind == "linear" else UsdPhysics.RevoluteJoint).Define(stage, path)
    j.CreateAxisAttr(axis)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    if limits is not None:
        j.CreateLowerLimitAttr(limits[0])
        j.CreateUpperLimitAttr(limits[1])
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), kind)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateTargetPositionAttr(0.0)
    if max_force is not None:
        drive.CreateMaxForceAttr(max_force)
    return j


def build_jar():
    stage, root = stage_for("pickle_jar")
    r = "/pickle_jar"
    prim = body(stage, f"{r}/jar", g.JAR_MASS).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxArticulationAPI")
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    glass = material(stage, f"{r}/Looks/glass", GLASS, roughness=0.15, opacity=0.85)
    pickles = material(stage, f"{r}/Looks/pickles", PICKLE, roughness=0.7)
    gold = material(stage, f"{r}/Looks/gold", GOLD, roughness=0.35)
    dark = material(stage, f"{r}/Looks/dark", DARK, roughness=0.5)
    glass_phys = phys_material(stage, f"{r}/Physics/glass", *g.JAR_MU)
    lid_phys = phys_material(stage, f"{r}/Physics/lid", *g.LID_MU)
    bind_phys(cyl(stage, f"{r}/jar/body", g.BODY_R, g.BODY_H, g.BODY_H / 2, GLASS, mat=glass, torsional=0.02), glass_phys)
    bind_phys(cyl(stage, f"{r}/jar/neck", g.NECK_R, g.NECK_H, g.BODY_H + g.NECK_H / 2, GLASS, mat=glass), glass_phys)
    cyl(stage, f"{r}/jar/pickles", g.BODY_R - 0.003, g.FILL_TOP - 0.005, 0.005 + (g.FILL_TOP - 0.005) / 2, PICKLE, collide=False, mat=pickles)
    # the lid on its chain of joints: x, y, z slides and the spin
    tiny = 1e-6
    for name in ("lid_x", "lid_y", "lid_z"):
        body(stage, f"{r}/{name}", 0.004, (0, 0, g.LID_CZ), inertia=tiny)
    body(stage, f"{r}/lid", g.LID_MASS, (0, 0, g.LID_CZ))
    bind_phys(cyl(stage, f"{r}/lid/shell", g.LID_R, g.LID_H, 0.0, GOLD, mat=gold, torsional=0.015), lid_phys)
    # two dark lug marks across the top, so the turn can be seen from above
    box(stage, f"{r}/lid/mark_a", (0, 0, g.LID_H / 2 + 0.0004), (0.060, 0.007, 0.0008), DARK, collide=False, mat=dark)
    box(stage, f"{r}/lid/mark_b", (0, 0, g.LID_H / 2 + 0.0004), (0.007, 0.060, 0.0008), DARK, collide=False, mat=dark)
    j = f"{r}/joints"
    joint(stage, f"{j}/lid_x", "linear", "X", f"{r}/jar", f"{r}/lid_x", (0, 0, g.LID_CZ), (0.0, 0.0), damping=0.5)
    joint(stage, f"{j}/lid_y", "linear", "Y", f"{r}/lid_x", f"{r}/lid_y", (0, 0, 0), (0.0, 0.0), damping=0.5)
    joint(stage, f"{j}/lid_z", "linear", "Z", f"{r}/lid_y", f"{r}/lid_z", (0, 0, 0), (0.0, 0.0), damping=0.5)
    joint(stage, f"{j}/lid_spin", "angular", "Z", f"{r}/lid_z", f"{r}/lid", (0, 0, 0), damping=100.0, max_force=g.SEAL_TORQUE)
    stage.GetRootLayer().Save()


def build_tool():
    stage, root = stage_for("lever_tool")
    r = "/lever_tool"
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(g.TOOL_MASS)
    steel = material(stage, f"{r}/Looks/steel", STEEL, roughness=0.3)
    abs_ = material(stage, f"{r}/Looks/abs", ORANGE, roughness=0.6)
    box(stage, f"{r}/blade", *g.BLADE, STEEL, mat=steel)
    box(stage, f"{r}/fin", *g.FIN, ORANGE, mat=abs_)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_jar()
    build_tool()
    for p in sorted(OUT.rglob("*.usda")):
        print(p.relative_to(OUT), p.stat().st_size, "bytes")
