# Push three loose cartridges flush with one two-handled pusher bar

Three yellow cartridges sit in three marked lanes. A long blue pusher bar lies square against their back faces and can only
be held by two orange handles on stems at its two ends (0.34 m apart). Both arms hold the bar and push all three cartridges
forward together until their front edges line up at the green stops. Core ability: moving one wide free tool with two hands
so that three independent loose workpieces end at the same depth (centre x 0.312-0.332 m, spread <= 12 mm) without leaving
their lanes. Real-world use: feeding a batch of cartridges into a fixture so none is left short-fed (O*NET 51-9198.00,
helpers of production workers: supplying materials). This is a pushing task, not an insertion task, and the pusher is a rigid bar.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `settle`, `probe`, `one_arm`, `no_yaw`).
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid end states, blind-word scan.
- `task.json`, `media/`.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 700 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 700
```
Runtime per trial: 700 steps (35 s simulated); the demo reaches 1.0 at about step 200 and ends near step 234. No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Mass | Pose |
|---|---|---|---|
| Lane strips | 200 (x) x 85 (y) x 2, white, fixed | - | x 0.27, y -0.105 / 0 / +0.105 |
| Stops | 20 x 85 x 20, green, fixed | - | front faces at x 0.360 in each lane |
| Cartridges (3) | 55 x 55 x 35, yellow | 0.09 kg, friction 0.45 / 0.60 / 0.45 (middle lane stickier) | x 0.220, one per lane |
| Pusher bar | 25 x 350 x 25, blue, with a 20 x 20 stem and a 40 x 35 x 25 orange handle block at y = +-0.170 (handle centre 0.10 above the table) | 0.22 kg, friction 0.2 | face centre x 0.178, flat against the cartridge backs |

## Scoring (measured at the end, perfect state held 1 s = 20 steps; cartridge and pusher poses only)
Zero gate: any cartridge tilts over 8 deg, is more than 15 mm off its lane centre (lane 85 mm vs cartridge 55 mm), or its front edge passes the stop face (x 0.361).
- 0.25: at least one cartridge advanced >= 40 mm.
- 0.5: all three advanced >= 60 mm.
- 0.75: all three centres x >= 0.300 and centre-x spread <= 25 mm.
- 1.0: every centre x 0.312-0.332, spread <= 12 mm, each y error <= 12 mm, each yaw <= 8 deg, and the pusher still square behind them (yaw <= 6 deg, pitch <= 8 deg, its face within 12 mm of the cartridge backs).
Two valid end states are unit-tested (all at x 0.318; and 0.317 / 0.322 / 0.327).

## Real-world build and reset
Three 55 x 55 x 35 mm blocks, a 350 mm bar with two handle posts, a taped lane layout with three stop blocks. Reset: bar square
against the blocks, blocks in the lane centres. Score with a straightedge across the three front faces and a ruler.

## Shortcuts and what stops them
- Poking each cartridge separately with a finger can reach 0.75 but not 1.0: the 1.0 level needs the pusher left square behind the cartridges.
- Pushing only the middle cartridge or sweeping one lane: the all-three tiers need every cartridge >= 60 mm.
- Over-pushing or a cartridge leaving its lane: zero gate.
- The task is not hard for a rigid bar held by both hands (the demo needs no yaw trimming, see `no_yaw`); the difficulty is the
  closed-loop two-hand carry of a low-friction bar at 25 mm/s and stopping at the right depth with the three lanes level.
The scripted demo reads the true poses (closed loop); it proves feasibility, not perception.

## How this differs from nearby tasks
- #638 (wide sled pushed home with both arms without racking it) is the closest: the same two-hand square-push mechanism. Here
  the load is three separate loose cartridges, and the score compares the final depth of each one (spread of three poses),
  their lane position and yaw, so the result depends on how three independent frictions (different per lane) are
  equalised, not on one rigid sled seating. This is only a modest difference and we state it plainly.
- #864 (holding back a spring pusher during restocking) and #478 / #1252 (loading magazines or pushers) involve springs or
  insertion; here nothing is spring-loaded and there is no insertion.

## Measured in the simulator
Demo run on the current `task.py` (no task.py change after any run listed; one build, one task.py version).
| Run | Score |
|---|---|
| Demo | 1.0 (success at about step 200; centres x 0.3146 / 0.3136 / 0.3126, y error <= 1.4 mm, pusher yaw 0.5 deg) |
| `settle` (arms hold still) | 0.0 |
| `no_yaw` (no yaw trim of the bar) | 1.0 (not needed: both hands hold the bar square; prediction of 0.5-0.75 did not hold) |
| `one_arm` (left arm alone, pushes handle +y) | 0.75 (final x 0.326 / 0.325 / 0.323, but one lane drifted 12.5 mm in y and the bar pitched 15 deg, so not 1.0). The bar did not yaw: it stayed square against the three cartridges, so one arm nearly succeeds, a weakness of this task. |
| Dream-1 run | 0.0 (700 steps; the two grippers reached the handles and the bar ended yawed, cartridges not advanced) |

## Honest deviations and limits
- No central grip guard (dropped as contrived); the handles are only at the two ends of the bar.
- Handles sit on 20 x 20 mm stems at 0.10 m above the table (spec 0.0675) so the fingers clear the bar; bar rests on the 2 mm lane strips.
- Lane walls are not physical: lanes are painted strips and the lane-exit gate is measured (15 mm); the stops are physical.
- 1.0 band: x 0.312-0.332 (spec 0.315-0.330), spread <= 12 mm (spec 10), plus a pusher-square condition not in the spec.
- Run limit 700 steps (spec 1,600). Lane friction differs (0.45 / 0.60 / 0.45).
