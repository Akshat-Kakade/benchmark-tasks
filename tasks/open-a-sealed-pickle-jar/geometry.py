"""Geometry and mechanism numbers of the pickle-jar scene on the yam_arena station table, shared by build_assets.py, task.py,
demo.py and the tests. Pure Python (importable before Isaac Sim starts).

World frame (yam_arena's): x forward from the arms, y to the robot's left, z up, metres; heights are above the table top.
Arm bases are at (0, +0.305) (left) and (0, -0.305) (right).
"""

from __future__ import annotations

import math

TOP = 0.7475                              # yam_arena.task_file.TABLE_TOP["station"]

# -- the jar (origin: centre of its base on the table) ---------------------------------------------------------------------
JAR_X, JAR_Y = 0.22, 0.0
BODY_R, BODY_H = 0.039, 0.105             # glass body: 78 mm across, 105 mm tall to the shoulder ledge (a 16 oz / 500 mL pickle jar)
NECK_R, NECK_H = 0.0325, 0.017            # the neck above the shoulder: 65 mm across, 17 mm tall, glass top at z 0.122
GLASS_MASS = 0.25                         # 0.25 kg of glass (3 mm wall, 5 mm base, density 2.5 g/cm3)
FILL_MASS = 0.40                          # 0.40 kg of pickles and brine (0.36 L, density 1.1) fused to the jar: no slosh
JAR_MASS = GLASS_MASS + FILL_MASS         # 0.65 kg
JAR_MU = (0.45, 0.40)                     # glass on laminate
FILL_TOP = 0.095                          # brine surface height above the base (inner floor 5 mm + 0.36 L over a 36 mm inner radius)
HEADSPACE = 0.027                         # 27 mm from the brine surface to the glass top (0.122 - 0.095)
SPILL_TILT = math.radians(30)             # brine reaches the rim at tan = 2 * 27 mm / 72 mm = 0.75 (37 deg): gate at 30 deg

# -- the lid (a gold twist-off lug lid) -------------------------------------------------------------------------------------
LID_R, LID_H = 0.035, 0.021               # 70 mm across, 21 mm tall including its skirt; its bottom edge is 10 mm above the shoulder ledge
LID_Z0 = 0.115                            # lid bottom height when seated (the lug thread holds it up; the 8-10 mm gap under the skirt is where the lever goes)
LID_CZ = LID_Z0 + LID_H / 2               # lid centre height when seated
LID_MASS = 0.03                           # 0.3 mm tinplate, 70 mm disc + skirt + liner
LID_MU = (0.9, 0.8)

# -- the seal and the lug thread (task logic) -------------------------------------------------------------------------------
SEAL_TORQUE = 0.4                         # N m the lid resists once the seal is broken (thread friction + liner), as Coulomb friction
UNLATCH = math.radians(80)                # lug quarter turn in the LOOSENING direction: free after this much turn relative to the jar
LOOSEN_SIGN = 1.0                         # loosening = counter-clockwise seen from above (right-hand thread) = positive turn about the lid axis (+z up)
TIGHT_STOP = math.radians(1.1)            # the seated lid is given this stop in the tightening direction (a soft stop: the scorer, not the stop, is what makes tightening worthless)
SPIN_LIMIT = math.radians(95)             # before the unlatch the lid cannot turn further than this
RIM_R = (0.028, 0.038)                    # blade tip radial distance from the jar axis: at the 35 mm lid skirt (3 mm tolerance for the pose of a 2 mm blade), outside the 32.5 mm neck
RIM_Z = (BODY_H - 0.005, LID_Z0 + 0.002)  # tip underside height: on the shoulder ledge or up to the lid's (raised) bottom edge + 2 mm
POP_LIFT = 0.001                          # the lid edge must really be raised this much (lid_z joint) by the lever
POP_WINDOW = 0.003                        # the locked lid can move up this much (3 mm of play in the lug thread)
TOOL_AIM = math.radians(35)               # the tool must point at the jar axis within this
POP_STEPS = 5

# -- the lever tool (a "lid lifter": a thin steel blade, plastic grip fin). Origin: fin centre on the blade underside,
# +x toward the tip, tool lying flat on the table. -------------------------------------------------------------------------
BLADE = ((0.0125, 0.0, 0.001), (0.125, 0.024, 0.002))     # 125 x 24 x 2 mm steel plate, x from -0.050 to +0.075
FIN = ((0.0, 0.0, 0.022), (0.100, 0.010, 0.040))          # grip fin: 100 x 10 x 40 mm ABS, standing on the plate
TIP_LOCAL = (0.075, 0.0, 0.0)             # lowest front point of the blade
TAIL_LOCAL = (-0.050, 0.0, 0.0)           # rear end of the blade underside
TOOL_MASS = 0.091                         # steel 0.047 kg, ABS fin 0.044 kg
TOOL_X, TOOL_Y = 0.34, -0.14              # start pose: lying flat, tip toward +y (yaw +90 deg)
TOOL_YAW = math.pi / 2
FIN_Z = 0.022                             # fin centre height above the blade underside (pinch point)

# -- task end state ---------------------------------------------------------------------------------------------------------
LID_OFF = 0.050                           # lid centre this far from the jar axis counts as off the jar
LID_CLEAR = BODY_R + LID_R + 0.030        # lid at rest >= 30 mm from the jar footprint: centre >= 0.104 from the jar axis
LID_REST_Z = (LID_H / 2 - 0.004, LID_H / 2 + 0.004)       # lid centre height above the table when lying flat
JAR_TILT_OK = math.radians(10)
JAR_STAY = 0.060
OPEN_GAP = 0.058                          # finger-tip gap: 0.048 open, ~0.09 pinching
GRIP_CLEAR = 0.070                        # gripper TCP at least this far from the jar axis and from the lid centre (surface + 30 mm)
STILL_V = 0.005
HOLD_STEPS = 20
