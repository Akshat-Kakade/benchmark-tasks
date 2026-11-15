# Cross-cut a pine plank on a running table saw (two push sticks, hood guard, riving knife)

**Category:** Impact on jobs. Cutting stock on a table saw is routine carpenter work (O*NET 47-2031.00, about 700,000 US workers, BLS OEWS May 2023). A plank is cut in two across its breadth by pushing both halves through the blade at the same time, one hand on each side of the blade, with push sticks so the hands stay clear, under a blade guard and with a riving knife in place. Two hands are needed because the blade's cutting reaction acts at the middle of the plank: pushed by one hand only, the plank turns about the blade instead of going through.

**Strengths.** Real-size cast-iron table and 10 in blade; one intact pine plank (19 x 140 mm, a real 1x6) that is cut by the saw in two; a shielding guard over the blade; a riving knife; one cutting resistance derived from a published feed force and scaled to the arms (not a tuned number); the cut front is a monotonic maximum, so strokes back and forth add nothing and lifting the plank over the blade scores 0; two push sticks and a 100 / 150 mm hand-to-blade rule; every ablation measured.

**Contrast with other tasks.** #586 uses a relief block at a saw fence; #1245 is a paper guillotine with the hand kept clear. Here one workpiece is pushed through a running blade by two arms at once, one on each side of the blade, and the physics (not a threshold) makes one arm fail.

## What it tests
Two arms pushing one workpiece through a blade at the same time from the two sides of the blade. The right arm pushes the right half's rear edge with a push stick, the left arm pushes the left half's rear edge, both forward (+x) through the blade at the same speed. The saw itself performs the cut, as in real life; the arms do what a carpenter's hands do (feed), and because the embodiment has no cutting tool, the cut is machine-side code, disclosed below. The instruction gives the goal only; the score checks the hand-to-blade distance a carpenter is taught (100 mm, below 150 mm above the table).

Instruction: "Cut the pine plank in two across its breadth, then leave both halves lying flat on the table."

## Scene (station frame: x forward = feed direction, y left, z up, metres; table top z = 0.7475; arm bases at x = 0, y = +-0.305)
| Object | Size (mm) | Material, mass | Pose |
|---|---|---|---|
| saw table | 810 (x) x 480 (y) x 30 (z) slab, x 0.09..0.90, y -0.24..0.24 | cast iron, with blade, knife and guard one rigid body, 85.4 kg (cannot be shoved) | on the station table, clear of the arm bases |
| blade | 10 in (254 mm) disc drawn as two steps: 2.4 mm plate, 150 mm long at the table, 77 mm long at the top, 25 mm above the table, red teeth strip | steel | in the plane y = 0 (between the arms), x 0.50..0.65 |
| riving knife | 2.8 mm thick, 30 mm long, 22 mm high plate in the kerf plane, 4 mm behind the blade's trailing teeth | steel (no collision with the plank, like the blade) | y = 0, x 0.654..0.684 |
| blade guard (shielding hood) | roof a grille of six slats with 8 mm gaps (a 14 mm gap right over the blade), 100 x 205 mm overall, and two side walls 3 mm thick, 50 mm above the table (the roof; 25 mm above the blade top), walls from 23 mm to 47 mm above the table so the plank passes under them with 4 mm clearance; hung from a 205 x 10 mm over-arm on a 10 x 10 x 50 mm column at the table's left edge. Drawn 25 % opaque where the renderer honours it; the grille gap lets the blade show from above | steel (in the saw body, collides) | hood x 0.49..0.695, y -0.05..0.05; over-arm x 0.68..0.695 to the column at y 0.23..0.24 |
| stop line | 4 mm red strip, flush, x = 0.40 (100 mm before the blade) | paint | across the table |
| pine plank (intact stock) | 19 x 140 x 440 (thickness x breadth x length), ONE rigid body, long side along y | pine 500 kg/m3, 0.585 kg, friction 0.30 | centred on the blade plane, x 0.355..0.495 (front edge 5 mm before the blade) |
| cut halves (hidden until the cut) | two 19 x 140 x 218.4 mm pieces, 3.2 mm kerf between | pine, 0.2905 kg each | idle bodies lie hidden at x 0.80, y +-0.12 on a free corner of the saw table; swapped in at the plank's pose by the cut |
| push sticks (2) | heel block 20 x 40 x 26 mm, 210 mm bar 30 x 10 mm, upright grip fin 50 x 12 x 50 mm 150 mm behind the heel | pine, 0.061 kg each, friction 0.5 | lying on the saw table at y = +-0.15, heel front face x 0.30 |

The 3.2 mm kerf (1/8 in) is the real saw kerf; the 19 x 140 mm plank is a real 1x6; the table is a job-site saw's size. The plank starts as one piece.

## Scoring (every number from task.py)
- Zero gate (latched, a real hazard): any gripper fingertip, gripper link or TCP within 100 mm in plan of the blade while below 150 mm above the saw table (the shop rule: hand 4 in from the blade, push stick inside 15 cm); a push stick touching the blade plate or the riving knife; stock falling or thrown off the table; stock kicked up more than 30 mm off the table (kickback).
- Zero for lift-over: if the plank is ever more than 2 mm above the table (the constant `LIFT_OVER`, one value used for the zero and for the kerf rule), or tipped more than 3 degrees, while it overlaps the blade, the whole score is 0.
- Cut front (kerf progress): the furthest the plank's front edge has been fed past the blade's leading edge, a **monotonic maximum**. It grows only by forward travel made on steps when the plank is intact, lying flat (<= 2 mm up, tilt <= 3 degrees), square to the blade (yaw <= 6 degrees) and centred on the blade plane (within 10 mm), by at most 10 mm per step, and only beyond the furthest point already cut. Strokes back and forth over the same region add nothing; moving the plank without those conditions adds nothing.
- 0.25: a cut front of at least 30 mm.
- The plank is replaced by the two halves only when the cut front reaches 170.5 mm (the plank's 140 mm breadth plus 36.5 mm for the blade's leading edge at the plank's top, minus 6 mm tolerance).
- 0.5: separated, both rear edges at least 12 mm past the blade's leading edge at the plank top (x >= 0.5485) but a half not lying flat, or the cut line's offset from the plank's midpoint (measured at the blade's leading edge, latched maximum over the cut) over 1/4 in (6.4 mm).
- 0.75: complete, flat, offset within 6.4 mm, but a gripper within 30 mm of a half, a half still moving, or offset not within 3.2 mm.
- 1.0: complete, cut line within 3.2 mm (1/8 in, a rough-cut tolerance) of the midpoint, halves flat (tilt <= 8 degrees) and still (<= 10 mm/s, <= 0.1 rad/s) on the saw table, both grippers >= 30 mm from the halves and >= 100 mm from the blade (below 150 mm). All of this true for 20 consecutive steps (1 s) before 1.0 is returned. The push sticks may lie anywhere.

## Design: what is code-driven, what is not modelled
- **The saw performs the cut; the cut is code (design choice).** A real carpenter's hands never cut the wood: the spinning blade does. The arms here do the carpenter's part (feed) and the saw part is machine-side code, because the embodiment has no cutting tool. The stock is one rigid body, so it cannot come apart by contact. When the cut front reaches the full breadth, the plank is replaced in one step, at its pose and velocity, by the two halves (3.2 mm of the plank's length is sawdust). The two half bodies exist from the start but lie hidden on a free corner of the saw table. The blade and the riving knife have no collision with the plank (a 0.4 mm clearance in a 3.2 mm kerf cannot be resolved by the solver); the scorer, not contact, decides what counts as cutting.
- **One cutting resistance.** While the plank is engaged with the blade (front past the blade's leading edge, cut not finished) a persistent external force of up to FEED_N acts on it against the feed, applied at the blade's leading edge on the blade plane; it ramps from 0 to FEED_N as the plank's forward speed goes from 0 to 5 mm/s (a cut resists the feed, it never pushes the plank back). FEED_N = 13 N x 19 mm / 30 mm = 8.2 N for both arms together (4.1 N each). Source and scaling: a hand-fed circular saw needs about 13-28 N of feed force when cross-cutting softwood (Kminiak and Gaff 2015, BioResources). I could not check the study's stock depth offline and ASSUME 30 mm; the cutting force is taken as proportional to the cut depth (the plank thickness, 19 mm), and the low end of the range (sharp blade, slow feed) is used. So the estimate for this plank is about 8-17 N and the simulation uses 8.2 N. The simulated arms stall near 7 N each, which is why the stock is a 19 mm softwood plank and not thicker or harder wood: **the embodiment's force limit shaped the stock choice.** A real hardwood or 38 mm plank would need more than two such arms can push. Friction adds 0.2 N. The demo overcomes the resistance by pushing (each stick heel presses with a measured tracking error); it cannot free-slide.
- **Why one arm fails: no tuned side, yaw or lift constants.** The torque of the resistance about the plank's centre comes only from where it acts: at the blade line, at the middle of the plank. With both sticks pushing evenly the torque cancels. With one stick pushing 150 mm off the blade line, the plank turns about the blade instead of going through (the plank yaws; once it is more than 6 degrees from square or 10 mm off the blade plane the feed no longer counts as cutting). The arms' own force limit adds to it: one arm alone has to supply all 8.2 N, above its stall force.
- **Blade guard and riving knife are modelled** (table above). The guard is a hood: a slatted roof above the blade's top (8 mm gaps, a 14 mm gap right over the blade so the blade shows in the video) and two walls beside it, so no hand or stick can reach the blade from above or the sides; a real guard is a solid clear shield, the grille is a visibility compromise. The plank enters under the walls (4 mm clearance) and the halves leave behind the knife. Feeding method: each push stick's heel pushes the plank's rear edge 150 mm off the blade plane (outside the hood and outside the 100 mm zone); the grip fins and the hands stay 150 mm or more from the blade line and the bars stay below the hood. The guard does not touch the plank; a real guard rests on the stock, this one is fixed 4 mm above it.
- **Miter guide / fence not modelled.** A real cross-cut holds the plank square with a miter gauge or a fence; here nothing does, the two sticks are what keep it square (that is the skill tested). A rear stop is not modelled either.
- **Not modelled:** blade rotation and teeth, sawdust, kickback dynamics, the pinch of the kerf closing, noise, vibration, the dust port.

**Limitations.** Coded cut and cutting resistance; the resistance is scaled to the arms and the stock depth behind the published range is an assumption; the stick is gripped by a fin; fixed layout (object_jitter=0); the guard is fixed above the stock.

## Files
`task.py` (scene, scoring, resistance), `partkit.py` (shared compound-part helper, unchanged), `demo.py` (scripted demo), `tests/test_scoring.py`, `task.json`.

## Run
```
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "<instruction above>" --scorer task_score --max-steps 1500 --store-frames --log-dir <dir>
python demo.py task.py demo_out [--no-record]     # DEMO_VARIANT: settle, one_right, one_left, shuffle, lift_over
python -m pytest tests -q
```
Run limit 1,500 steps (75 s at 20 Hz). The demo takes 445 steps. Seeds: object_jitter=0, no randomness.

## The demo (open loop in the fixed layout)
It reads no object pose. Each arm's own joint positions (proprioception) are integrated into the command. Both arms in parallel: take the push stick by its grip fin (fingers open along y, pinching the 12 mm fin), lift it 7 mm, carry it to 30 mm behind the plank's rear edge, then feed both sticks together at 20 mm/s for the first 60 mm and 35 mm/s after, pushing against the 8.2 N resistance, until the plank's rear edge is 12 mm further than the cut-complete line. Each arm releases and backs away. Reach was checked offline: every waypoint of both arms is within 0.45 m of the arm base with an IK error under 0.1 mm.

## Real-world build and reset
A table saw (job-site size) with a 254 mm blade, a riving knife and a blade guard fitted, a miter gauge or fence to hold the plank square, one 19 x 140 x 440 mm pine plank (a 1x6 offcut, not pre-cut), two push sticks, a painted line 100 mm before the blade. Reset: a fresh plank centred on the blade, front edge 5 mm before the blade, the sticks behind it, guard and knife in place. Measure both halves' lengths with a ruler and the fingertip-to-blade distance against the painted line.

## Shortcuts and what stops each
1. Bare hands pushing close to the blade: gripper inside the 100 mm zone below 150 mm: score 0.
2. Lift the plank over the blade: more than 2 mm up (or tipped over 3 degrees) while over the blade scores 0, and no kerf is counted.
3. Slide the plank past the blade without engaging (off the blade plane, turned, up): adds no cut front; the plank is never separated, no credit.
4. Shuffle the plank back and forth in the blade: the cut front is a monotonic maximum, the strokes add nothing.
5. One arm only: the plank turns about the blade and the cut does not complete (see the measurements).
6. Fling the plank: off the table or more than 30 mm up: 0.
7. Push a stick through the blade or knife: 0.

## Measured in the simulator
| Run | Score | What happened |
|---|---|---|
| settle (no action) | 0.0 | plank at rest centred on the blade plane, nothing cut, no gate |
| demo (both arms) | 1.0 | 445 steps (of 1,500); cut front reached 170.5 mm at step ~400; cut line within 1.4 mm of the midpoint; closest hand to the blade 117 mm; held 1 s |
| right arm only (left arm idle) | 0.0 | the plank swung about the right stick's contact point (yaw 70 degrees) instead of going through; cut front 4 mm, no credit |
| left arm only (right arm idle) | 0.0 | mirror image: yaw -69 degrees, cut front 3 mm |
| both arms shuffle the plank back and forth (8 strokes of 20 mm over the same region) | 0.0 | the cut front stays at 23 mm (below the 30 mm credit; summed travel would have been 320 mm, enough to finish the cut) |
| lift-over bypass (a test-only upward force of 9 N, 1.6 x the plank's weight, while it is engaged; no arm can do this) | 0.0 | the plank rode up at the blade entry (3.4 mm up at step 286): lift-over rule, no credit at any step |
| Dream-1 run (final scene) | 0.0 | 1,500 steps, step limit hit; one arm drifted over the saw, neither arm took a stick or touched the plank (last frame: plank and sticks untouched) |
Earlier builds of the same task (rip cut, two-piece stock) are not reported here. Runtime: the demo takes about 22 s of simulated time (445 steps at 20 Hz); run limit 1,500 steps.
