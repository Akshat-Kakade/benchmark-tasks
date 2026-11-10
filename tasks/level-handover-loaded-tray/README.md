# Hand a rack of sample cups from one arm to the other in mid-air, level, across a gap neither arm spans

A white rack carries three blue cups on the left grey pad. The right pad is 0.80 m away, beyond either arm's reach
(about 0.50 m from its own base), so one arm lifts the rack, passes it to the other arm near the middle of the table, and
the second arm sets it on the right pad. Core ability: a mid-air two-arm handover of a load that must stay level (cups
upright, tilt <= 8 degrees), with the grip swapped to the other handle. A tilted rack of open specimen cups spills
samples (laboratory contamination hazard; O*NET 29-2012, 31-9097).

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `no_handover`, `early_release`, `tilt20`, `place_yaw15`, `mid_alt`, `settle`).
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
| Rack | plate 90 x 230 x 12, white; handle block 60 x 40 x 40 at each short end (centres y +-135); 6 mm wells | friction 0.6 | 0.15 kg | centred on the left pad |
| Cups | 3 solid cylinders, dia 45 x 60, blue | friction 0.8 | 0.03 kg each | in the wells, plate y -55 / 0 / +55 |

## Scoring (held 1 s = 20 steps)
Zero if a cup is off the plate or tipped > 20 deg, the rack tilts > 20 deg while carried, or the rack sits on the table away from both pads (dropped).
- 0.25: rack lifted >= 40 mm, cups upright, carried past y = +0.15.
- 0.5: rack carried past y = -0.30 (beyond the left arm's reach, so the right hand has it) with tilt never > 8 deg while lifted.
- 0.75: then resting on the right pad within 15 mm of its centre and 8 deg of yaw.
- 1.0: also all three cups upright on the plate within 15 mm of their wells.
Everything is measurable with a ruler and a set square. Two valid solutions: handover at y = 0 or at y = +0.06 (both 1.0).

## Real-world build and reset
A 3D-printed or laminate plate 90 x 230 x 12 mm with two 60 x 40 x 40 mm handle blocks and three 6 mm well rings, three
60 mm cups (water-filled to 30 g for the spill check), two pads taped to the bench 0.80 m apart. Reset: cups in wells, rack on the left pad.

## Shortcuts and what stops them
- Sliding or throwing the rack across: pads are 0.80 m apart, no arm reaches both; a thrown rack drops (zero gate) or cups fall (zero).
- Releasing the left hand before the right holds the rack: it falls off any pad (zero gate; `early_release`).
- Carrying it tilted: tilt over 8 deg caps the score at 0.25, over 20 deg is zero.
- Gripping the plate edge instead of a handle is allowed if it stays level; handle height is free.
The scripted demo reads the true rack pose (closed loop); this proves feasibility, not perception.

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
Only `task.py` revision matters here. After the first ablation batch (`no_handover`, `early_release`, `tilt20`, `place_yaw15`,
`mid_alt`) the scoring was fixed (the 0.75 tier previously ignored level; tilt20 had wrongly scored 0.75). Only the demo and
`tilt20` were re-run on the final `task.py`; the other four rows were measured on the earlier revision and were NOT re-run, so
they are not vouched for on the final code (the offline tests cover the final scoring for those outcomes).
| Run | Score | task.py version |
|---|---|---|
| Demo | 1.0 (success at step 691, run 8 and again in final run d9; rack tilt peaked about 7 deg while carried) | final |
| `tilt20` (left tips the rack while carrying) | 0.25 (rack tilt peaked 17.5 deg, below the 20 deg zero gate, so it scored 0.25, not 0) | final |
| `no_handover` (left stops at y = +0.15, holds) | 0.25 | earlier revision, not re-run |
| `early_release` (left lets go at the middle) | 0.0 (rack dropped off both pads) | earlier revision, not re-run |
| `place_yaw15` (set down 15 deg off) | 0.5 | earlier revision, not re-run |
| `mid_alt` (handover at y = +0.06) | 1.0 | earlier revision, not re-run |
| Dream-1 run | 0.0 (1,800 steps) | final |
| Settle (hold only) | 0.0, nothing moved | earlier |

## Honest deviations and limits
- The rack is 150 g, not the 250 g first planned: the pinch cannot hold a heavier rack against its torque about the closing axis.
- The handles are pinched across y (fingers open along y), the pads are at y = +-0.40, and cups stand in 6 mm wells (solid cylinders, nothing can spill).
- The 0.5 tier uses object poses only (rack carried past y = -0.30 at tilt <= 8 deg while lifted); it does not measure the gripper-to-rack distance of the second hand.
- Arm-to-arm hand and body clearance was not measured; the hands are about 0.27 m apart at the handover.

Simulator notes: without wells the cups slid 12-22 mm during the carry.
