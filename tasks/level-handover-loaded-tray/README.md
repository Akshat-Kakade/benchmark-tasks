# Pass a tray of filled cups from one arm to the other in mid-air, level, across a gap neither arm spans

A white rack carries three blue cups on the left grey pad. The right pad is 0.80 m away, beyond either arm's reach
(about 0.50 m from its own base), so one arm lifts the rack, passes it to the other arm near the middle of the table, and
the second arm sets it on the right pad. Core ability: a mid-air two-arm handover of a load that must stay level (cups
upright, tilt <= 8 degrees), with the grip swapped to the other handle. A tilted tray of filled cups spills the drinks.
Occupation: waiters/waitresses (O*NET 35-3031, about 2 million workers) and food servers (35-3023): carrying loaded trays and
switching hands with them is a regular core part of the job. The rack stands for the tray and is deliberately small (see limits).

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: open-loop scripted demo (fixed waypoints from the scene layout) (`DEMO_VARIANT` = `no_handover`, `early_release`, `tilt20`, `place_yaw15`, `mid_alt`, `settle`).
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid solutions, blind-word scan.
- `task.json`, `media/`.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1800 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1800
```
Runtime per trial: 1,800 steps (90 s simulated); the demo finishes in about 690 steps. No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Material | Mass | Pose |
|---|---|---|---|---|
| Pads | 130 (x) x 340 (y) x 3, grey, fixed | - | - | centres (0.30, +0.40) and (0.30, -0.40) |
| Rack (tray) | plate 90 x 230 x 12, white; handle block 60 x 40 x 40 at each short end (centres y +-135); 6 mm wells | friction 0.6 | 0.15 kg | centred on the left pad |
| Cups | 3 solid cylinders, dia 45 x 60 (espresso size), blue | friction 0.8 | 0.03 kg each (cup plus 30 mL water, lumped) | start centres x 0.30; y +0.345 / +0.400 / +0.455; z 0.7475 + 0.003 + 0.0125 (resting on the plate in the wells); rack start centre (0.30, +0.40), yaw 0 |

## Scoring (scorer revision 2)
A = left arm, B = right arm. An arm "holds" the rack when its TCP is within 50 mm of a handle centre and its two finger-tip bodies are
at least 58 mm apart (measured: 48 mm with the gripper open, 69 mm while pinching a 40 mm handle); both are read from the simulator's gripper links each step.
Zero (latched) if a cup spills (tipped > 20 deg or off the plate), the rack tilts > 20 deg while carried, or the rack sits on the table away from both pads.
- 0.25: A holds the rack, lifts it >= 40 mm off the pad with the cups upright and carries it past y = +0.15.
- 0.5: verified in-air transfer, in this order with the plate underside >= 20 mm above the pad throughout: A holds it, then B holds it,
  then A has let go while B still holds it. If the rack touches a pad in between, the chain restarts. Carry tilt never > 8 deg.
- 0.75: then the plate underside rests on the right pad (within 3 mm of the pad top, so not hovering or held above it), within 15 mm of
  the pad centre, 8 deg of yaw, tilt < 4 deg.
- 1.0: also all three cups within 15 mm of their wells, rack still (< 0.02 m/s, < 0.15 rad/s), both arms still (joint speed < 0.05 rad/s)
  and both grippers open and off the handles, all true for 20 consecutive steps (1 s). score() reports 0.75 until the 1 s has passed; it never returns 1.0 early.
Unknown gripper readings give no transfer credit. Two valid solutions: handover at y = 0 or at y = +0.06 (both 1.0).

## Cups, water and the spill check
The espresso-size cups (60 mm tall, 45 mm diameter) are filled with 30 mL of water (about 19 mm deep, 41 mm of freeboard), so a static spill needs a
tilt of about 60 deg; with slosh in a carried rack we count a spill at a cup tilted over 20 deg or a cup leaving the plate, and the rack
limit for full credit is 8 deg. The simulated cups are solid 30 g cylinders (no fluid); the sim spill check is exactly this cup-tilt rule,
so it matches the real rule. Real check: no water outside the cups at the end (paper towel under the rack).

## Real-world build and reset
A 3D-printed or laminate plate 90 x 230 x 12 mm with two 60 x 40 x 40 mm handle blocks and three 6 mm well rings, three
60 mm cups (each filled with 30 mL of water), two pads taped to the bench 0.80 m apart. Reset: cups in wells, rack on the left pad.

## Shortcuts and what stops them
- Sliding or throwing the rack across: pads are 0.80 m apart, no arm reaches both; a thrown rack drops (zero gate) or cups fall (zero).
- Carrying the rack across with one arm, or letting the right arm take it at the start pad: no credit above 0.25 without the verified A-then-B hold with the rack in the air (`no_handover`).
- Hovering above or holding the rack over the right pad: the rack must be seated (plate within 3 mm of the pad top), still, and released for 1 s.
- Releasing the left hand before the right holds the rack: it falls off any pad (zero gate; `early_release`).
- Carrying it tilted: tilt over 8 deg caps the score at 0.25, over 20 deg is zero.
- Gripping the plate edge instead of a handle is allowed if it stays level; handle height is free.
The demo must be re-measured on this revision (see the table); it reads exactly one thing from the simulator: each arm's own joint positions (the `state` observation, joint_pos), used to cancel tracking error. Every waypoint is computed from constants in task.py (pad centres, rack height, handle offsets); no object pose, contact or simulator state is read (a test greps demo.py for this). Object poses and gripper links are read only by the scorer.

## How this differs from nearby tasks
- #558 terminal-handover, #553 handover-mount, #617 chip hand-over, #100 scissor hand-off, #986 spring winding and #960
  seal segments pass a small part between hands as one step of an insertion/assembly; the scored end state is the part
  seated or wound. Here the object is a large, loaded rack, nothing is inserted, and the score is a level-hold constraint
  across the grip swap with the destination only reachable by the second arm.
- #1228 swap-tray-on-passing-cart and #1223 unload-rocking-tray move or tip a tray with a single hold or on a cart; this
  one passes it hand to hand in the air.
- #231 airlock pass-through moves a carrier through gated space; the reach gap here is geometric and nothing is a gate.
- #941 split spacer assembles around a shaft; unrelated mechanism.

## Measured in the simulator
The scorer was rewritten (revision 2); the rows below were measured on it. The earlier results (demo 1.0, Dream-1 0.0, ablations)
were measured on revision 1, which did not check which arm holds the rack and accepted a rack up to 12 mm above the pad, so they are not valid for this revision.
| Run (scorer revision 2) | Score |
|---|---|
| Demo (open loop; recorded final run) | 1.0 (success at step 831, run limit 1800) |
| `mid_alt` (handover at y = +0.06) | 1.0 (step 831) |
| `no_handover` (left carries to y = +0.15 and holds) | 0.25 |
| `early_release` (left lets go before the right holds) | 0.0 (rack falls) |
| `tilt20` (left tips the rack 20 deg while carrying) | 0.0 (tilt passed the 20 deg gate) |
| `place_yaw15` (set down 15 deg off) | 0.5 (carried and handed over, not placed within 8 deg) |
| Settle (hold only) | 0.0 |
| Real-size cups (75 x 95 mm, 0.2 kg each), same demo | 0.0 (not liftable; reverted, see simplifications) |
| Dream-1 run (final scorer, 1,800 steps) | 0.0 |

## What is simplified / not modelled
| Item | Real | Simulated |
|---|---|---|
| Cups | standard drinking cups, about 75 mm diameter x 95 mm, 150-200 mL of water, about 0.2 kg each | espresso-size cups, 45 mm x 60 mm, solid cylinders, 0.03 kg each (30 mL of water lumped in) |
| Tray | serving tray with a loaded mass of 0.6-1 kg | white rack, 0.15 kg empty, 0.24 kg with the three cups |
| Liquid | water slosh and spill | none; a spill is a cup tilted over 20 deg or off the plate |
| Hands | human hands | a 94 mm parallel gripper pinching a 40 mm handle |
We tried real-size cups (75 x 95 mm, 0.20 kg each, 0.75 kg total, handles 165 mm from the centre): the simulated pinch could not lift that load (demo score 0.0 through step 600), so this is the largest size that works with this gripper. The tray is a scaled-down stand-in for a real serving tray, and the task claims only the two-hand level handover, not full-size drink service.

## Honest deviations and limits
- The rack is 150 g, not the 250 g first planned: the pinch cannot hold a heavier rack against its torque about the closing axis.
- The handles are pinched across y (fingers open along y), the pads are at y = +-0.40, and cups stand in 6 mm wells (rack wells are 6 mm high rims of white boxes, 3 mm thick) (solid cylinders in the simulator; the spill rule is the tilt rule above).
- The 0.5 tier is judged from the left/right gripper link positions and finger gap against the handles, plus the rack pose.
- Arm-to-arm hand and body clearance was not measured; the hands are about 0.27 m apart at the handover.

Simulator notes: without wells the cups slid 12-22 mm during the carry.
