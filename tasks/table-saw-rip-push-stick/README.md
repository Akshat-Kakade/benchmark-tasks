# Rip a pine board on a running table saw (push stick, hold-down hand, guard and riving knife)

**Category:** Impact on jobs. Ripping stock on a table saw is routine carpenter work (O*NET 47-2031.00, about 700,000 US workers, BLS OEWS May 2023). The intact board is fed with a push stick so the hand stays clear of the blade, with the blade guard and riving knife in place. Two hands are needed because they do different jobs: one holds the board down on the table and tight to the fence against the cutting loads (which push it away from the fence and lift it), the other feeds it forward against the cutting resistance.

**Strengths.** Real-size build (600 x 1070 mm cast-iron saw table, 10 in blade, 3.2 mm kerf, 400 mm push stick, 19 x 89 x 220 mm pine board). The stock is one intact board; credit needs an actual kerf (feed with the board down and on the fence), so lifting the board over the blade or sliding it past scores 0. The scorer is an end-state ruler check (strip width, flat, still, hands clear). The scripted demo scores 1.0. Ablations (one arm only, bare hand, early release, lift-over) each fail.

**Contrast with other tasks.** #586 uses a relief block at a saw fence; #1245 is a paper guillotine with the hand kept clear. Here one workpiece is fed past the blade with a tool by one arm while the other arm holds it down and to the fence.

## What it tests
Two arms, two different jobs on one workpiece: the right arm feeds the board with a push stick (fork-heeled, so it straddles the blade and the riving knife) against a feed resistance, the left arm keeps the board down on the table and tight against the fence while the cut starts. The saw itself performs the cut, as in real life; the arms do what a carpenter's hands do (feed, hold down, hold to the fence), and because the embodiment has no cutting tool, the cut is machine-side code, disclosed below. The instruction gives the goal only; the score checks the hand-to-blade distance a carpenter is taught (100 mm, below 150 mm above the table).

Instruction: "Cut the pine board lengthwise along the fence so that a 40 mm strip comes off, then leave both pieces lying flat on the table."

## Scene (station frame: x forward, y left, z up, metres; table top z = 0.7475)
| Object | Size (mm) | Material, mass | Pose |
|---|---|---|---|
| saw table | 600 (x) x 1070 (y) x 30 (z) slab, x 0.120..0.720, y -0.62..0.45 | cast iron, with fence and blade one rigid body, 142.6 kg (cannot be shoved) | on the station table |
| rip fence | 30 x 1070 x 45 high, face at x = 0.185 | aluminium (in the 142.6 kg) | along the table |
| blade | 10 in (254 mm) disc drawn as two steps: 2.4 mm plate, 150 mm long at the table, 77 mm long at the top, 25 mm above the table, red teeth strip | steel | x = 0.2266 (centre of the kerf), y 0.105..0.255 |
| stop line | 4 mm red strip, flush, y = 0.005 (100 mm before the blade) | paint | across the table |
| riving knife | 2.8 x 30 x 22 mm plate in the kerf plane, 4 mm behind the blade's trailing teeth, 3 mm below the blade top | steel (in the saw body; no collision with the board, like the blade) | x 0.2266, y 0.259..0.289 |
| blade guard | overhead open frame 112 x 170 mm outside, 8 x 3 mm bars, 50 mm above the table (31 mm above the board, 25 mm above the blade top), on a 10 x 15 mm column 53 mm high beside the stock path | steel (in the saw body, collides) | frame x 0.2025..0.315, y 0.095..0.265; column x 0.305..0.315, y 0.19..0.205 |
| pine board (intact stock) | 89 x 220 x 19, ONE rigid body | pine 500 kg/m3, 0.186 kg, friction 0.30 | x 0.185..0.274 (against the fence), y -0.12..0.10 (front end 5 mm before the blade) |
| cut pieces (hidden until the cut) | strip 40 x 220 x 19 (0.0836 kg) and offcut 45.8 x 220 x 19 (0.0957 kg), 3.2 mm kerf between | pine | idle bodies lie hidden at x 0.55..0.65, y 0.28 on a free corner of the saw table; swapped in at the board's pose by the cut |
| push stick | forked heel (two blocks 20 x 26 mm tall, 14 mm gap for the blade), 400 mm bar 30 x 10 mm riding 3 mm above the blade top, upright grip fin 50 x 12 x 50 mm 330 mm behind the heel | pine, 0.115 kg, friction 0.5 | lying on the saw table at x 0.36, heel front face y -0.20 |

The 3.2 mm kerf (1/8 in) is the real saw kerf; the 89 mm board is a real 1x4. The board starts as one piece. The saw table is a real size benchtop/contractor saw with in- and outfeed.

## Scoring (end state plus kerf history; every number from task.py)
- Zero gate (latched, a real hazard): any gripper fingertip, gripper link or TCP within 100 mm in plan of the blade while below 150 mm above the saw table (the shop rule: hand 4 in from the blade, push stick inside 15 cm); the push stick touching the blade plate or the riving knife; stock falling or thrown off the table; stock kicked up more than 30 mm off the table (kickback).
- Zero for lift-over: if the board is ever more than 2 mm above the table, or tipped more than 3 degrees, while it overlaps the blade, the whole score is 0 (no credit for lifting stock over the blade, or for riding over it tipped up).
- Kerf length: the board's feed past the blade's leading edge, counted only on steps when the board is intact, lying flat on the table (<= 2 mm up, tilt <= 3 degrees) and against the fence (fence side within 8 mm, yaw <= 6 deg). Moving the board without that adds nothing. No kerf, no credit, whatever the final positions are.
- 0.25: kerf of at least 30 mm.
- The board is replaced by strip and offcut only when the kerf reaches 340 mm (the blade has run the board's length to within 30 mm).
- 0.5: the cut is complete (separated; both rear ends at least 12 mm past the blade's back edge) but a piece not lying flat, or the strip width at the blade centre off 40 mm by more than 1/4 in (6.4 mm).
- 0.75: complete, flat, width within 6.4 mm, but a gripper within 30 mm of a piece, a piece still moving, or width not within 3.2 mm.
- 1.0: complete, strip 40 +- 3.2 mm (1/8 in, a rough-cut tolerance) wide where the blade is, pieces flat (tilt <= 8 degrees) and still (<= 10 mm/s) on the saw table, both grippers >= 30 mm from the pieces and >= 100 mm from the blade (below 150 mm). All of this true for 20 consecutive steps (1 s) before 1.0 is returned. The push stick may lie anywhere.

## Design: what is code-driven, what is not modelled
- **The saw performs the cut; the cut is code (design choice).** A real carpenter's hands never cut the wood: the spinning blade does. The arms here do the carpenter's part (feed, hold down, hold to the fence) and the saw part is machine-side code, because the embodiment has no cutting tool. The stock is one rigid body, so it cannot come apart by contact. When the engaged kerf (above) has run the board's length, the board is replaced in one step, at its pose and velocity, by the strip and the offcut (3.2 mm of width is sawdust). The two piece bodies exist from the start but lie hidden on a free corner of the saw table. The blade and the riving knife have no collision with the board (a 0.4 mm clearance in a 3.2 mm kerf cannot be resolved by the solver); the scorer, not contact, decides what counts as cutting.
- **Feed resistance and loads are code.** While the board is engaged in the cut (front past the blade's leading edge, cut not finished) a persistent external force of 3.0 N acts on the board against the feed. A real rip of 19 mm pine needs about 25-60 N; the simulated arms stall near 7 N, so the resistance is scaled down to 3 N (5-10 % of the real force), on top of 0.6 N of sliding friction. The demo has to overcome it by pushing (the stick heel presses with a measured tracking error); it cannot free-slide. Alongside it, while the rear end is still more than 170 mm before the blade (the stretch a hold-down hand can cover while staying 100 mm from the blade), side 1.0 N away from the fence (about 3 % of a 30 N feed), lift 2.4 N (the board weighs 1.86 N, so an unheld board floats up and tips: the kickback tendency of rising rear teeth) and yaw torque 0.02 N m act on the board. These three loads are modelled choices, set so that an unheld board leaves the fence and tips; they are kept within what the arms can hold (a pad presses a few N), far below a human hand's force. The side load stops once the board is 12 mm off the fence. The 2 mm / 3 degree limits (and the 8 mm fence distance) in the kerf rule are scorer tolerances for 'board flat on the table and against the fence while over the blade'.
- **Blade guard and riving knife are modelled** at real size and position (table above), kept clear of the stick path: the guard (an open frame of 8 mm bars, so the stock stays visible) is above the stick bar and heel, the column stands beside the stock path, the stick's forked heel straddles the blade and the knife. The guard sits higher than a real one (50 mm above the table; a real guard rests on the stock) so the push stick passes under it. The riving knife and the blade have no collision with the board (their 0.4 mm clearance in the kerf cannot be resolved by the solver), so they are visible markers and the cut is judged by the scorer.
- **Feeding method:** push stick with forked heel, held by its grip fin, for the whole cut; hold-down pad (the closed left hand) on the board's top next to the fence for the first stretch, then lifted away before the 100 mm zone. The hands never come within 100 mm of the blade.
- **Not modelled:** blade rotation and teeth, sawdust, kickback dynamics, the pinch of the kerf closing, noise, vibration, the dust port.

**Limitations.** Coded cut and loads; small force magnitudes; the stick is gripped by a fin; fixed layout (object_jitter=0); guard higher than real.

## Files
`task.py` (scene, scoring, loads), `partkit.py` (shared compound-part helper, unchanged), `demo.py` (scripted demo), `tests/test_scoring.py`, `task.json`.

## Run
```
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "<instruction above>" --scorer task_score --max-steps 1500 --store-frames --log-dir <dir>
python demo.py task.py demo_out [--no-record]     # DEMO_VARIANT: settle, one_right, one_left, bare_hand, early_lift, lift_over
python -m pytest tests -q
```
Run limit 1,500 steps (75 s at 20 Hz). The demo takes 675 steps. Seeds: object_jitter=0, no randomness.

## The demo (open loop in the fixed layout)
It reads no object pose. Each arm's own joint positions (proprioception) are integrated into the command. Right arm: takes the push stick by its grip fin (fingers open along y, pinching the 12 mm fin), lifts it 7 mm, carries it to 30 mm behind the board, then feeds at 20 mm/s for the first 60 mm and 35 mm/s after, pushing against the 3 N feed resistance, until the board's rear end is 18 mm past the cut-complete line. Left arm: goes high over the saw (above 150 mm), lowers a closed fist onto the board's top next to the fence, presses (about 10 mm of commanded travel below contact, a few N), rides with the board for the stretch where the side/lift/yaw loads act, then lifts away and returns to its ready pose. Right arm releases and backs away 120 mm. The saw swaps the board for strip and offcut when the kerf reaches 340 mm.

## Real-world build and reset
A table saw with a 254 mm blade, a riving knife and a blade guard fitted, a rip fence set 40 mm from the blade's face, one 19 x 89 x 220 mm pine board (a 1x4 offcut, not pre-cut), a 400 mm push stick, a painted line 100 mm before the blade. Reset: a fresh board against the fence, front end 5 mm before the blade, stick at the right of the board, guard and knife in place. Measure the strip width at both ends with a ruler and the fingertip-to-blade distance against the painted line.

## Shortcuts and what stops each
1. Bare-hand push the whole way: gripper enters the 100 mm zone: score 0.
2. Lift the board over the blade: the board more than 2 mm above the table (or tipped over 3 degrees) while over the blade scores 0, and the lifted feed adds no kerf.
3. Slide the board past the blade without engaging (off the fence, tilted, up): adds no kerf, the board is never separated: no credit.
4. Fling the board: off the table or more than 30 mm up: 0.
5. Push the stick through the blade or knife: 0.
6. No hold-down hand: the loads push the board off the fence and lift it: see the ablations.
7. Hold-down hand riding into the zone: 0.

## Measured in the simulator
| Run | Score | What happened |
|---|---|---|
| settle (no action) | 0.0 | board at rest against the fence, nothing cut, no gate |
| demo (both arms) | 1.0 | 675 steps (of 1,500); kerf 340 mm, strip width 39.6-40.3 mm; held 1 s |
| right arm only (stick, no hold-down) | 0.0 | the unheld board tipped 3.7 degrees as its front reached the blade (lift load 2.4 N > its 1.86 N weight): lift-over rule, no cut credit |
| left arm only (bare closed hand pushes the rear) | 0.0 | board tipped at the blade entry (2.3 degrees): lift-over rule, kerf 7 mm |
| both arms, right pushes with a bare fist | 0.0 | fist within 99 mm of the blade: hand gate at step 274, kerf 2 mm |
| both arms, left lifts away after 10 mm | 0.0 | unheld board tipped 6.4 degrees over the blade: lift-over rule |
| lift-over bypass (board teleported 25 mm up while passing the blade; no arm can do this) | 0.0 | the lift-over rule latched as soon as the board overlapped the blade (step 413, kerf 8 mm); the score stays 0 at every step, no partial credit |
| Dream-1 run (final scene) | 0.0 | 1,500 steps, step limit hit; the arms moved but never took the stick or touched the board (last frame: board and stick untouched). A first wording that named the machine was refused by the service before any step and was not scored |
Runtime: the demo takes about 34 s of simulated time (675 steps at 20 Hz); run limit 1,500 steps.
