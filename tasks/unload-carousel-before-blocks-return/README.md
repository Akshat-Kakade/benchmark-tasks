# Unload a carousel: crate each red block before its carrier comes back round to the loader

Category: Impact on jobs. Two YAM arms flank a bench-top indexing carousel. The left arm loads five blocks from a supply
mat onto the carousel at the loading marker; the right arm must lift each of the three red blocks off the carousel into
a crate before the carrier it rides on comes back to the loader. The two blue blocks stay riding.

Instruction (24 words): *"Load the five blocks onto the carousel at the green marker. Crate each red block before it
returns to that marker; leave the blue ones riding."*

## What it tests and why

Machine-paced stations (packing tables, rotary assembly and inspection tables) run on a clock: parts go round on pallets
and the tender must clear each finished part before its pallet comes back to the loading station, where the next part
needs a free pallet. Done by packaging and filling machine operators and tenders (O*NET 51-9111.00; 371,600 employed,
BLS OEWS May 2023) and packers and packagers, hand (O*NET 53-7064.00; 645,210 employed, BLS OEWS May 2023): together about
1.0 million workers. Tending a machine and taking finished product off it, one part at a time, before the next one
arrives, is regular core work of both on packing lines (neither occupation alone is claimed to be a carousel job).
Here a returning carrier that still holds a finished (red) part is the failure: the loader needs a free pallet. A blue
block is an in-process part: it stays on its carrier and rides several laps, so a carrier coming back with a blue block
on it is normal. The task is a clock-paced, two-station version of this work: put each block on a free carrier, lift only
the red ones, and set them in the crate. A solver may read the carousel from the camera or count the machine's fixed
clock; the scripted demo does the latter.

Closest on the board and how this differs: "Unload fruit from a bin another robot arm is also unloading" (a shared bin,
no clock), the keep-a-station-stocked tasks such as `spring-pusher-lane-restock` (order of stock on a lane, no clock) and
hand-over tasks (an object changes hands). Here nothing is shared or handed over; the new elements are a machine clock
with a per-block deadline (the first return of the block's carrier to the loader) and per-block provenance (a block
counts only if it was seen on a carrier first), so taking a block straight off the supply mat scores 0.

## Files

- `task.py`: scene, per-step provenance monitor, `score()`, `success()`, scripted `demo()` with `DEMO_VARIANT` ablations.
- `motionkit.py`: driven bodies for yam_arena (copied unchanged from the merged task `push-rejects-off-moving-line`).
- `demo.py`: thin wrapper that runs the scripted demo (`python demo.py task.py out [--max-steps N] [--no-record]`).
- `check_reach.py`: runs every waypoint through the arm IK (worst error per waypoint, closest approach of the two hands).
- `tests/test_task.py`: offline tests (carousel timetable, demo timing against it, every scoring level, zero cases, two
  valid solutions, blind-word scan).

## Run commands (from this folder, after `source yam.env` on the VM)

```bash
# Dream-1 (one run, keep whatever it scores)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the five blocks onto the carousel at the green marker. Crate each red block before it returns to that marker; leave the blue ones riding." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/carousel-dream

# The scripted demo (expect 1.0 after about 990 steps)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the five blocks onto the carousel at the green marker. Crate each red block before it returns to that marker; leave the blue ones riding." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/carousel-demo
# or: python demo.py task.py demo_out
# ablations: DEMO_VARIANT=settle|left_only|right_only|late|blue_removed python demo.py task.py out --no-record

for log in ~/yam-eval/carousel-demo/*.json; do inspect-robots video "$log"; done
python -m pytest -q tests          # offline, needs yam_arena importable; python check_reach.py for the IK check
```

Runtime: the scripted demo is 49.8 s (about 990 steps at 20 Hz) of simulated time; the whole trial takes a few minutes
of wall time on the GPU VM. Run limit 1,800 steps (90 s). Seed: one fixed layout, `-E object_jitter=0`.

## Scene (station frame: x forward from the arm bases, y to the robot's left, z up, metres; table top z 0.7475)

| Part | Material | Size | Mass | Pose |
|---|---|---|---|---|
| 6 carrier plates | anodised aluminium plate, friction 0.8 static / 0.7 dynamic | 110 x 110 x 12 mm | driven by the machine | centres on a circle of radius 0.18 m about (0.34, 0.00), one per 60 degrees; 2 mm above the table |
| hub | dark painted steel | 100 x 100 x 30 mm | fixed | (0.34, 0.00) |
| green loading marker | painted strip | 120 x 15 x 2 mm | fixed | (0.34, 0.262), just outside the loader slot |
| 3 red blocks and 2 blue blocks | solid acetal (POM, 1.41 g/cm3) cubes | 50 mm | 176 g each | on the supply mat: red_1 (0.23, 0.37), red_2 (0.32, 0.37), red_3 (0.41, 0.37), blue_1 (0.275, 0.49), blue_2 (0.365, 0.49) |
| supply mat | rubber mat | 340 x 200 x 4 mm | fixed | centre (0.34, 0.44) |
| crate | polypropylene tray, 8 mm walls, 5 mm floor | inner 260 x 360 mm (x by y), walls 60 mm above the table (55 mm above its floor, 5 mm above the block tops) | fixed | inner x 0.17 to 0.43, y -0.62 to -0.26 |

Slots are numbered 0 (the loader, (0.34, +0.18)), 1 (0.50, +0.09), 2 (0.50, -0.09), 3 (the unload slot, (0.34, -0.18)),
4 (0.18, -0.09), 5 (0.18, +0.09); a carrier moves from slot k to k + 1 each index. Every 5.0 s the carousel indexes
one slot: 1.2 s of cosine-profile motion (peak 0.25 m/s) and 3.8 s standing; the first index starts at 5 s and the
machine never stops. A carrier is back at the loader after one lap, 30 s. A bench-top indexer of 0.47 m outer diameter
is a real machine size; 50 mm, 176 g blocks are small machined parts, everyday size.

## Why two arms (physical)

The two tending stations are fixed on either side of the machine: the loading station and the supply mat on the robot's
left, the unload slot and the crate on its right, with the carousel between the arms, as in a real two-tender line. Each
arm works its own station; the carousel stands still for only 3.8 s per 5 s index and does not wait. Nothing is handed
from arm to arm. See the measured one-arm runs below (left-only and right-only).

## Scoring

A per-step monitor follows every block for the whole trial:

- loaded: first step the block rests on a carrier (centre over the plate, bottom within 8 mm of its top) with both grippers
  clear (4 cm plus the finger opening). Never seen on a carrier: never counts, so a block taken straight off the supply
  mat scores 0 (it cannot even be set in the crate by the right arm: the mat is out of its reach).
- first return: the block's carrier is back within 8 cm of the loader slot after having been more than 12 cm away.
- late: the block is still on a carrier at or after its carrier's first return. Latched: never counts afterwards.
- lost: after loading, the block rests on the table, mat or floor outside the crate. Latched.
- credited (a red block): loaded, not late, not lost, and now resting in the crate: its bottom within 6 mm of the crate
  floor, inside the inner walls, still (under 1 cm/s) and let go (both grippers at least 4 cm plus the finger opening clear).
- spoiled (a blue block): after loading it is lifted more than 3 cm off its plate, lost, or in the crate. Latched.

| Score | Condition |
|---|---|
| 0 | fewer than 3 blocks loaded and no red credited |
| 0.25 | at least 3 blocks loaded onto the carousel |
| 0.5 | one red block credited |
| 0.75 | two red blocks credited (also what `score()` returns for the full condition, since the hold decides 1.0) |
| 1.0 | all five blocks loaded, all three red blocks credited, both blue blocks still riding on carriers, held for 1 s (20 steps) |

In physical terms: a red block that leaves its carrier and is not in the crate (it sits on the table, the mat or the
floor) is lost and never counts; any blue block taken off its carrier by more than 3 cm, or crated, or dropped is spoiled
and caps the score at 0.5. The instruction is goal-only and does not state these rules: "crate each red block" and "leave
the blue ones riding" imply the blue rule and that a red ends up in the crate, but not that a red set down on the table in
between is lost for good. That follows from the stated machine-paced goal (a part that left the pallet and is not in the
crate has not been handled as a finished part); it is a harsh rule and is disclosed here rather than in the instruction.

A spoiled blue block caps the score at 0.5. The score is never 1.0 before the 1 s hold (the framework ends the trial with
1.0 once `success()` has held for 1 s). Reds taken off at any index before the first return are fine (unit test: unloading
at the unload slot, or one slot later, both give 1.0).

Check on a real table: block on a plate, block in the crate resting on its floor, red block off the machine before its
plate returns: all visible to a person with a ruler and a stopwatch; the carousel clock is the machine's.

## Shortcuts and what stops them

| Shortcut | Result |
|---|---|
| do nothing | 0 |
| take a block straight off the supply mat into the crate | 0 (never seen on a carrier; also out of the right arm's reach) |
| load everything, never unload | 0.25 |
| unload each red after its carrier has come back to the loader | the red is latched late: never counts |
| crate the blue blocks too | capped at 0.5 |
| drop a red block on the table and pick it up later | lost: never counts |
| leave a gripper on a crated block | not released: not credited until it lets go |

## Measured in the simulator

| Run | Score | Note |
|---|---|---|
| settle (hold both arms, 300 steps) | 0.0 | blocks and carousel as designed |
| scripted demo | 1.0 | success after 993 steps (49.8 s), all five blocks loaded, red_1 crated 14 s before its carrier came back (back at the loader at step 714), the two blue blocks riding at the end |
| Dream-1 (1,800 steps) | 0.0 | neither arm touched a block |

Ablations (scripted, `DEMO_VARIANT`), each run once:

| Variant | Predicted | Measured | What happened |
|---|---|---|---|
| `settle`: both arms hold | 0.0 | 0.0 | carousel indexes, no block moves |
| `left_only`: the left arm loads all five blocks, the right arm holds | 0.25 | 0.25 | five loaded, no red crated; every red latched late at its carrier's return (steps 714, 914, 1114) |
| `right_only`: the right arm works the unload slot, the left holds | 0.0 | 0.0 | nothing is ever loaded, so nothing counts |
| `late`: reds loaded, but unloaded one lap (30 s) later | 0.25 | 0.25 | all three reds end in the crate yet are latched late: no credit |
| `blue_removed`: reds and blue_1 loaded, the right arm crates blue_1 as well | 0.5 | 0.5 | the three reds are credited (0.75) but the spoiled blue caps the score at 0.5 |
| block taken straight off the supply mat (unit test; also out of the right arm's reach) | 0.0 | 0.0 | never seen on a carrier |

One-arm necessity: the left arm alone cannot crate anything (the crate and the unload slot are beyond its reach), the right
arm alone cannot load (the supply mat and the loader are beyond its reach). It is a layout of two stations, as on a real
rotary table with two tenders; it is not enforced by a hidden rule. Neither a single arm nor a late unload scores above 0.25.

## Setup photo

`setup.png` is the first top-camera frame, upscaled 3x. The left arm's start pose hides the two blue blocks on the supply mat
(they are at y 0.49, behind the arm); they are visible from step 30 on and in the demo video.

## Demo

Open loop and timed against the carousel's fixed clock (first index at 5 s, about 1 s of slack); it reads nothing from the simulator: only the machine's
clock and the layout's fixed places (the demo uses no observations of the scene: only the arms' own joint states through the playback). The left arm lifts red_1, red_2, red_3, blue_1 and blue_2 in that order from the mat and sets each on the carrier standing in the
loader slot 1.3 s into the 3.8 s standing window after indexes 1, 3, 5, 7 and 9 (one block per 10 s); the right arm
lifts each red block off its carrier at the unload slot at indexes 4, 6 and 8, three indexes after its load and about 14 s
before the carrier comes back, and sets it in the crate. The carrier that carried red_1 returns to the loader at
about 36 s and is loaded with blue_1 right after. The arms are never closer than 0.365 m (TCP to TCP).

## What is simplified (real value in brackets)

- The carriers follow a prescribed indexing path exactly (driven bodies, like a servo-driven carousel): the track, the
  drive, the indexer's gearing and the inertia are not modelled (a real carousel has a track and a motor), and the
  carriers translate round the circle without turning (a real rotary table turns its pallets).
- The "finished" status is the block's colour (a real tender knows from the process).
- Blocks are 50 mm solid acetal cubes standing in for small machined or moulded parts; a real tender handles parts of
  many shapes.
- The loader is the other arm; a real station may be fed by hand or by a feeder.

## Real-world build and reset

Bench-top indexing rotary table of about 0.47 m outer diameter (or an indexing conveyor of six pallets) that indexes every
5 s; six aluminium pallets 110 x 110 x 12 mm; five 50 mm acetal cubes (three painted red, two blue) of 176 g; a rubber
supply mat 340 x 200 mm; a polypropylene parts crate of 360 x 260 mm inside with 55 mm walls. Reset: pallets at the six slots
with the loader slot at the marker, blocks on the mat at the marked positions, crate empty at its marks.
