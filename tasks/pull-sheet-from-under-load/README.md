# Pull a slip-sheet out from under a loaded block while the other hand holds the load

Left hand pulls a thin white sheet out from under an orange block by its green tab; the right hand must keep the block
from moving. Tests friction-shear extraction with the load held by a reaction force (not grasp-and-move). A dragged load
tips or lands off its pad: a crush and drop hazard when pulling slip-sheets or transit liners in warehouses and repair
shops (O*NET 53-7051, 53-7062, 49-2011).

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `no_brace`, `no_pull`, `pinch`, `fast`).
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
| Sheet | 260 x 260 x 4 slab, white, plus tab 40 (x) x 60 (y) x 40 (z), green, on its +y edge | friction 0.3 | 0.15 kg | slab centre (0.30, 0.04) on the table |
| Block | 80 x 80 x 80, orange | friction 0.3 | 0.35 kg | centre (0.30, 0.0), standing on the sheet |

## Scoring (final state, held 1 s = 20 steps)
Zero if, after step 40, the block is ever tilted > 20 deg, moved > 80 mm horizontally, or lifted > 10 mm.
- 0.25: sheet pulled >= 60 mm in +y.
- 0.5: sheet pulled >= 100 mm, or sheet clear but block moved 25-80 mm.
- 0.75: sheet clear (its footprint >= 5 mm from the block's) and block moved 12-25 mm or tilted 5-20 deg.
- 1.0: sheet clear, block moved <= 12 mm, tilted <= 5 deg, resting on the table (<= 3 mm up), block and sheet still for 1 s.
The sheet is clear after about 135 mm of pull. Everything is measurable with a ruler and set square.

## Real-world build and reset
A 4 mm laminate sheet with a taped-on 40 x 60 x 40 mm tab, a 350 g 80 mm block; reset by laying the sheet back under
the block on the marked spot. Forces are about 1-2 N, safe.

## Shortcuts and what stops them
- One-arm quick yank: block drags, score <= 0.5 (to be measured, `fast`).
- Lifting the block off, shoving it aside, tipping it: zero gate.
- Pushing the sheet out sideways with the fist: counts only if the block stays put, i.e. a valid extraction.
The scripted demo reads the true block and sheet poses (closed loop); that proves feasibility, not perception.

## Measured in the simulator
| Run | Score |
|---|---|
| Settle (brace only, sheet untouched; `no_pull`) | 0.0, no gate; block shifts 7.6 mm from the 6 mm brace overshoot |
| Demo (fist brace) | 1.0 (success at step 290 of 348; sheet moved 135 mm, block shift 5 mm, tilt 0) |
| `pinch` | 1.0 (second valid solution) |
| `no_brace` | 0.0 (block dragged past the zero gate; predicted 0.25-0.5, the drag is larger than predicted) |
| `no_pull` | 0.0 |
| `fast` | 0.0 (no brace, 150 mm/s: block dragged past the zero gate) |
| Dream-1 run | 0.0 (2000 steps) |
