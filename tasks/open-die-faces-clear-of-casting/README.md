# Open both die faces clear of the casting

Two die halves stand on the table either side of a grey casting. Each half has a heavy grey base and a die face (blue on
the left, orange on the right) on a spring-returned slider, carried out to a grip post by a low bar. Both faces must be
drawn outward at the same time, to the green clearance marks, and held level, without disturbing the casting. A mould
operator separates both die faces before removing a finished part (O*NET 51-4072.00); uneven or partial opening can
scrape or trap the part. The core ability is a symmetric, simultaneous two-handed pull against two spring returns,
checked against a part that must stay put. No heat or adhesion is simulated.

Honest about the spring: each face is a prismatic slider of its base with a drive spring (60 N/m, modelled in the physics
engine, 5 N at 65 mm; the simulated arms stall around 7 N, so this is the strongest usable value). It is not a force
sensor and no force is scored. The casting is a free-standing light cylinder (0.12 kg) resting on the two bases, not a
fixed object; the bases are 3 kg with friction 1.0 and do not move under the 5 N reaction. The grip post is 20 mm wide
(a pinch the gripper holds reliably); an earlier 70 mm wide tab slipped out of the jaws.

## How this differs from nearby tasks
#865 closes a mould around a core, #411 holds one spring-closed die open, #982 and #789 open a lid or latch with both
hands, #1170 turns two faucet handles. Here both faces of one die are drawn apart together against their own spring
returns, to a measured equal clearance, around a part that must not move; one hand alone scores 0 because the second face
stays closed, and unequal travel is penalised by the symmetry limits.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `settle`, `one_arm`, `short`, `uneven`).
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid solutions, blind-word scan.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1600 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1600
```
Runtime per trial: up to 1,600 steps (80 s at 20 Hz); the demo succeeds at about step 250. No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Pose |
|---|---|---|
| Two bases, grey | 230 (x) x 155 (y) x 15 each, 3 kg, friction 1.0 | centres (0.29, +-0.0775) |
| Casting, dark grey cylinder | diameter 60, height 60, 0.12 kg | (0.29, 0) on the bases |
| Die faces, blue (left) / orange (right) | 140 (x) x 45 (y) x 55, slider 0.16 kg, spring 60 N/m, stroke 75 mm | inner edges 10-15 mm from the casting |
| Bar and grip post | bar 30 x 15 mm; post 40 (x) x 20 (y) x 60 | post at (0.29, +-0.205), pinched at z 0.85 |
| Clearance marks | green strips on each base | at 65 mm of face travel |
The posts are 0.41 m apart at rest and 0.55 m apart at full travel; the two hands never come close.

## Scoring (face slider travel and casting pose, read each step)
Zero if after step 40 the casting moves > 12 mm or tilts > 8 degrees, either base moves > 15 mm, or a face inner edge
reaches the casting. Otherwise the highest tier met by the current pose:
- 0.25: both faces out >= 20 mm. 0.5: both >= 40 mm and travel difference <= 30 mm.
- 0.75: both >= 55 mm and difference <= 20 mm.
- 1.0: both >= 65 mm, difference <= 12 mm, each inner face >= 35 mm from the casting surface, held for 1 s (the run then ends with success).
Two valid solutions: both faces at their outward stops (75 mm), or both held anywhere in 65-75 mm.
Travel is read from each slider relative to its own base, so dragging a base does not count.

## Real-world build and reset
Two sprung slides (about 60 N/m) on weighted bases, a plate-like face on each, a post per face, a free-standing cylinder
between, witness marks on the bases. Reset: let the faces return and stand the casting on its mark.

## Shortcuts and what stops them
- One arm only: the second face stays at rest, tier 0 (min travel < 20 mm).
- Stopping short: tiers cap at 0.5. Uneven pull: the 12 mm difference limit caps at 0.5-0.75.
- Shoving the casting or a base: zero gate.
The scripted demo reads true poses (closed loop); it proves feasibility, not perception.

## Measured in the simulator
| Run | Score |
|---|---|
| Settle (8 s hold, nothing touched) | 0.0, no gate, bases settle 3-7 mm outward in the first steps |
| Demo (both pinch the posts, pull in step to 72 mm, hold 1 s) | 1.0 (success at step 248; travel 72.4 / 65.2 mm at success; casting moved < 0.1 mm) |
| `one_arm` (left only, 72 mm) | 0.0 (right face never moves) |
| `short` (both stop at 45 mm) | 0.5 |
| `uneven` (72 mm and 50 mm) | 0.5 |
| Dream-1 run | 0.0 (1,600 steps) |
Notes: the first spawn dropped the casting on the table before the bases were turned into place and the bases then threw it
away, so `on_reset` also writes the casting pose. The right face reaches only 65.2 mm when the run ends one second after
the 65 mm threshold; the demo hold is closed loop and would still be creeping up.
