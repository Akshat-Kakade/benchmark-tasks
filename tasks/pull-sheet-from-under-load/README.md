# Pull a slip-sheet out from under a loaded block while the other hand holds the load

The instruction asks only for the outcome: a thin white sheet must come out from under an orange block without the block
moving, and the block must end resting on the table. The demo uses the left hand on the green tab and the right fist as a brace, but any arm and direction scores. Tests friction-shear extraction with the load held by a reaction force (not grasp-and-move). A dragged load
tips or lands off its pad: a crush and drop hazard when pulling slip-sheets or transit liners in warehouses and repair
shops. Category: Impact on jobs. Occupation: O*NET 53-7062 Laborers and Freight, Stock, and Material Movers, Hand (about
3 million US workers, far above the 500,000 mark). Why routine: breaking down and unloading pallets, cartons and totes is
the core of that job, and slip-sheets or liners between layers or under a load are removed by hand in every such breakdown,
many times a shift, while the other hand steadies the load. The stake: a load dragged out with the sheet tips or lands off
its pad on the worker; a robot that braces the load with one hand while the other pulls takes people out of that exposure.
The other occupations considered earlier (53-7051, 49-2011) were dropped: slip-sheet pulling is not a regular core activity there.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: OPEN-LOOP scripted demo (reads no simulator state; fixed waypoints from the nominal layout) (`DEMO_VARIANT` = `no_brace`, `no_pull`, `pinch`, `fast`).
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid solutions, blind-word scan.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 2000 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 2000
```
Runtime per trial: 2,000 steps (100 s of simulated time); the demo finishes in about 350 steps. No random seeds
(`object_jitter=0`, fixed layout).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Material | Mass | Pose |
|---|---|---|---|---|
| Sheet | 260 x 260 x 4 slab, white, rigid (laminate-like), plus tab 40 (x) x 60 (y) x 40 (z), green, rigid PVC-like, fixed to its +y edge (tab centre y 0.19, z 18 mm above the slab centre) | friction 0.3 | 0.15 kg (slab and tab together) | slab centre (0.30, 0.04) on the table |
| Block | 80 x 80 x 80 cube, orange, rigid (hollow plastic or wood-like) | friction 0.3 | 0.35 kg | centre (0.30, 0.0), standing on the sheet (bottom 4 mm above the table) |

## Scoring (final state; scored version 2, revised after review)
Block height is its lowest corner above the TABLETOP (4 mm while on the sheet, ~0 once the sheet is out), not relative to
its start pose. Zero if, after step 40, the block is ever tilted > 20 deg, moved > 80 mm horizontally, or its centre rose
> 10 mm above its start.
- 0.25: sheet pulled >= 60 mm (any direction).
- 0.5: sheet pulled >= 100 mm; or sheet clear but block moved 25-80 mm; or sheet clear but the block is suspended
  (bottom more than 3 mm above the tabletop, e.g. still held up by a hand): a suspended block never scores above 0.5.
- 0.75: sheet clear (footprint >= 5 mm from the block's), block seated on the table (bottom <= 3 mm) and moved 12-25 mm
  or tilted 5-20 deg; also shown while the full-credit conditions are true but the 1 s hold has not yet completed.
- 1.0: sheet clear, block seated (bottom <= 3 mm above the tabletop), moved <= 12 mm, tilted <= 5 deg, and block and sheet
  stay still (block 1 mm, sheet 2 mm) with all of that true for 20 consecutive steps (1 s). 1.0 is never returned before
  the full second; any lapse restarts the count.
The sheet is clear after about 135 mm of pull. Everything is measurable with a ruler and set square.

## Scale and start positions
Everything is metre-scale tabletop stock: sheet 260 x 260 x 4 mm, block 80 mm cube (350 g). Real slip-sheet loads are
heavier (a carton is 2-10 kg), but the YAM arm's reach and payload limit the size; the block is a scale model of a carton
and the sheet a scale model of a liner (stated in `fidelity`). Start poses (station frame, m, no rotation, object_jitter=0):
sheet slab centre (0.30, 0.04, 0.7495), tab centre (0.30, 0.19, 0.7675); block centre (0.30, 0.00, 0.7915).

## What the demo reads
Open loop. It reads nothing from the simulator and no camera: waypoints are fixed numbers computed from the nominal layout
(`task.py` constants) plus two offsets found in tuning (fist contact depth, pinch point 25 mm +y of the tab centre because
the hand's TCP is off its finger gap); the arms follow IK targets on their own kinematics.

## Real-world build and reset
A 4 mm laminate sheet with a taped-on 40 x 60 x 40 mm tab, a 350 g 80 mm block; reset by laying the sheet back under
the block on the marked spot. Forces are about 1-2 N, safe.

## Shortcuts and what stops them
- One-arm quick yank: the block drags with the sheet and trips the zero gate (measured: `fast` 0.0, `no_brace` 0.0).
- Lifting the block off, shoving it aside, tipping it: zero gate.
- Pushing the sheet out sideways with the fist: counts only if the block stays put, i.e. a valid extraction.
What the demo reads: nothing from the simulator. It is open loop: waypoints are computed once from the nominal layout in task.py (object_jitter=0) and played with IK on the arms' own kinematics. It works because the soft fist brace tolerates a few mm of error and the pull is a fixed 165 mm. demo.py contains no simulator-pose reads at all (a test greps for them); it uses no feedback beyond its own commanded joint targets from IK on the arms' kinematics.

## Measured in the simulator
All rows use the final scorer (version 2) and the final open-loop `demo.py` (cap 600 steps for demo runs, 2000 for Dream).
| Run | Score |
|---|---|
| Demo (fist brace, open loop) | 1.0 (success at step 332; sheet 156 mm out, block shift 4.8 mm, tilt 0, block bottom on the table) |
| `pinch` (right hand pinches the block's x sides instead: second valid solution) | 1.0 (step 375) |
| `no_brace` (left pulls alone) | 0.0 (block dragged past the zero gate) |
| `no_pull` / settle (brace only, sheet untouched) | 0.0 (no gate tripped) |
| `fast` (no brace, 150 mm/s pull) | 0.0 (block dragged past the zero gate) |
| Dream-1 run (1 run, 2000 steps) | 0.0 |

Notes: the demo needs two numbers found in tuning: the fist contact depth (TCP y = block face + 57.5 mm) and a 25 mm
+y offset of the pinch point from the tab centre (the hand's TCP is off its finger gap, seen in the wrist camera); the
fist also advances 27 mm toward -y over the pull because the sheet drags the block. A one-arm run cannot score: without
the brace the block is dragged with the sheet. The scorer reads object poses only (it cannot see grippers), so release of
the hands is not scored directly; the block and sheet must be still for 1 s, and the demo lifts both hands away.
