# Meet two free plates edge to edge in the air, level, with a 15 mm seam

A blue and an orange plate rest on low pedestals either side of a marked bench outline. Each arm pinches the handle block of
its plate; together they lift the plates and bring the two straight inner edges together above the outline, level, parallel,
with a small positive gap (no overlap, no touching) centred on the outline, and hold them. Core ability: aligning two
independently held rigid parts to a measured seam (gap, height step, yaw, midpoint) in free air. Real-world use: the fit
check of two prefabricated cover sections before fastening (O*NET 51-2092.00); a step or wide gap shows a bad pair.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `settle`, `one_arm`, `no_level`, `overlap`, `yaw15`, `gap10`, `gap22`, `mid_alt`).
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid seams, blind-word scan.
- `task.json`, `media/`.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 800 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 800
```
Runtime per trial: 800 steps (40 s simulated); the demo succeeds at about step 330. No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Mass | Pose |
|---|---|---|---|
| Bench outline | 190 (x) x 290 (y) x 2, white, fixed | - | centre (0.31, 0) |
| Pedestals | 100 x 60 x 20, grey, fixed | - | (0.31, +0.125) and (0.31, -0.125) |
| Blue plate | 140 (x) x 105 (y) x 12 plus a 60 x 40 x 50 handle block on its outer (+y) edge | 0.12 kg, friction 0.6 | centre (0.31, +0.125), on the left pedestal |
| Orange plate | mirrored, handle on the -y edge | 0.12 kg, friction 0.6 | centre (0.31, -0.125), on the right pedestal |

## Scoring (measured at the end, held 1 s = 20 steps; plate poses only)
Zero: a plate not held in the air (underside under 30 mm above the pedestal tops), tilt over 12 deg, overlap over 8 mm, or heights differing over 40 mm.
- 0.25: both plates held in the air.
- 0.5: seam gap (inner-edge distance) <= 65 mm and relative yaw <= 20 deg.
- 0.75: gap 0-35 mm, height step <= 20 mm, relative yaw <= 12 deg, edge midpoints within 40 mm along the seam.
- 1.0: gap 8-25 mm, step <= 12 mm, yaw <= 8 deg, tilt <= 8 deg, seam midpoint within 12 mm of y = 0, edge midpoints within 20 mm along the seam.
Two valid solutions: gap 10 mm or 20 mm (both 1.0; the 10 mm seam was also run in the simulator).

## Real-world build and reset
Two laminate or printed plates 140 x 105 x 12 mm with a 60 x 40 x 50 mm handle block glued to one long edge (one blue, one
orange), two 20 mm pedestals, a taped 190 x 290 mm outline. Reset: plates on the pedestals, handles outward. Scoring in the
real world is a feeler gauge and a straightedge for gap and step, a protractor for yaw.

## Shortcuts and what stops them
- Leaving a plate on its pedestal or the bench: fails the held-in-air gate (score 0), so one arm alone scores 0 (`one_arm`).
- Pushing one plate over the other: overlap over 8 mm is a zero gate; contact tilts the plates (`overlap` run scored 0).
- A wide or one-sided seam, a height step or a yaw error cap the score at 0.5/0.75 (`no_level`, `yaw15`).
- The seam midpoint is tied to the fixed outline (y = 0), so both plates must be placed, not only brought relatively together.
The scripted demo reads the true plate poses (closed loop); it proves feasibility, not perception.

## How this differs from nearby tasks
- #842 (overlapping guardrail splice) and #1067 / #872 (splice and bolt joints) end in an overlapped or fastened joint;
  here the plates must not overlap, nothing is pinned or fastened, and the score is a positive, parallel, level seam in free air.
- #1264 (duct ends with cleats) joins ends through cleats; here both parts are free bodies held by one hand each and the
  scored quantities are gap, step and yaw of two straight edges.
- The new capability tested: simultaneous closed-loop control of two free, hand-held bodies to a mm-level relative pose.

## Measured in the simulator
Demo run first try on the current `task.py` (no task.py change after it). All rows below are on that `task.py`;
`demo.py` changed once (the `no_level` variant: it first lowered the right plate 25 mm, which put it under the held-in-air
height and scored 0.0; it now raises the left plate 25 mm).
| Run | Score |
|---|---|
| Demo | 1.0 (success at step 329: gap 15 mm, step 1 mm, yaw 0.1 deg, midpoint 0 mm) |
| `gap10` (valid seam, 10 mm) | 1.0 (success at step 324) |
| `one_arm` (left only) | 0.0 |
| `no_level` (left plate 25 mm high, no correction) | 0.5 (step 25 mm) |
| `overlap` (target gap -20 mm) | 0.0 (contact tilted both plates about 17 deg) |
| `yaw15` (right plate turned 15 deg) | 0.5 (relative yaw 14.6 deg) |
| Settle, `gap22`, `mid_alt` | not run (offline tests cover the scoring) |
| Dream-1 run | 0.0 (800 steps) |

## Honest deviations and limits
- Plates are rigid boxes 12 mm thick (spec said 18 mm with a 50 mm tab); each has a 60 x 40 x 50 mm handle block so the
  pinch grip holds it without tilting (plate tilt stays under 1 deg). Plate mass 0.12 kg.
- Held-in-air height is 30 mm above the pedestal tops (reach limit of a straight-down grip at this lateral distance), not
  the spec's 0.815 m.
- Added to the 1.0 level: edge midpoints within 20 mm along the seam and tilt <= 8 deg.
- Arm-to-arm hand clearance was not measured; the handles are about 0.26 m apart at the seam.
