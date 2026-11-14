# Free a stuck pickle jar lid with a lifter tool, then twist it off while the other hand holds the jar

A 16 oz glass pickle jar (78 mm across, 122 mm tall, 0.65 kg with its pickles and brine) stands on the table with a stuck gold lug lid.
Beside it lies a steel lid lifter: a thin blade with an orange grip fin. The instruction is: **"Open the pickle jar and set the lid down."**
Most jars open by a plain twist; this one is stuck, the way a vacuum-packed jar is, and the usual fix is to slide a spoon, a jar key or a church
key under the lid edge and lift it until the seal lets go. Here the lid stays locked until the blade is under its edge and has raised the edge by
1 mm. After that the lid still turns only against about 0.4 N m, which is more than the jar's own grip on the table (about 0.08 N m), so a hand
that twists the lid spins the jar instead unless the other hand holds the jar. The robot has to use the tool, hold the jar with one arm, twist the
lid a quarter turn with the other, lift it off, set it down and let go of everything, without tipping the jar (the brine would spill). Core ability:
tool-use on a stuck closure followed by a twist reacted through a second hand (counter-torque).

## Importance
Opening a stuck jar is a firsthand everyday experience: most people have needed a spoon, a jar key or a second hand to free a vacuum-sealed lid. A
real vacuum-packed lid takes more than about 3 N m to break loose (an estimate from jar-opening torque studies, not measured here), past many people's grip
and past this gripper's about 0.6 N m slip limit, which is why the lever is needed. Category: Societal relatability. No occupation or worker count is claimed.

## Construct validity
- Real size and real mass: a 16 oz jar, 78 mm across and 122 mm tall (105 mm body, 17 mm neck of 65 mm); 0.25 kg of glass (3 mm wall, 5 mm base,
  2.5 g/cm3) plus 0.40 kg of pickles and brine (0.36 L at 1.1 g/cm3, 95 mm deep) = 0.65 kg. Lid 70 mm across, 21 mm tall, 0.03 kg of tinplate.
  Lever tool 0.091 kg: a 125 x 24 x 2 mm steel blade (0.047 kg) and a 100 x 10 x 40 mm ABS fin (0.044 kg).
- The physical parts are real PhysX: the jar stands free on the table (glass on laminate, friction 0.45) and really spins when a hand turns the
  lid against its friction and nothing holds the jar; the arms really hold 0.65 kg and react the turning torque; the lid is lifted and set down
  by a gripper pinch (it stays jointed to the jar, see below); the lever is a free rigid body gripped by its fin.
- One-arm runs (see the table) cannot open the jar: with no hand on it the jar turns with the lid (the lid reaches 0.25 at most).

## What is simplified / not modelled
- **The vacuum seal is not modelled as air. It is a latch in task logic.** The lid hangs off the jar on joints that are locked: three slides with 0.5 mm of play (3 mm upward
  play for the lid z slide) and a turn drive with a 30 N m friction limit. The lock is released by a physical event the robot causes: the lever blade tip (28-38 mm
  from the jar axis, i.e. at the 35 mm lid skirt edge (3 mm tolerance), at or below the lid's bottom edge, pointing at the jar axis within 35 degrees) holds the lid edge
  at least 1 mm up (the lid z joint reading) for 5 consecutive steps. Hovering the blade near the rim, or lifting the lid by hand, does not release it. The turn drive then resists
  with 0.4 N m of Coulomb friction and the lid cannot lift. Once it has turned 80 degrees relative to the jar in the loosening direction (a lug quarter turn; standard right-hand thread: loosening is counter-clockwise seen from above, positive turn about the lid axis; the sim's joint axis is +z up)
  it is unlatched and can be lifted and set on the table. The lever does not pull air in; the 1 mm raise stands in for it.
- **The lid is not a free rigid body.** It stays on the jar's joint chain (three slides and a turn) for the whole run: it is part of the jar's articulation, never
  collides with the jar, keeps the jar's tilt (it cannot tip or flip on its own) and, once unlatched, can be lifted and slid anywhere in the jar's frame and rests on the table. A tilted jar therefore
  tilts a lid lying on the table too; this only matters after a failure (the spill gate has already scored 0).
- **Torque numbers are estimates.** 30 N m (sealed) is a deliberately large friction limit: the real seal needs more than about 3 N m, and 30 N m is far above what the simulated wrist (7 N m) can give, so the lid
  cannot be turned before the lock is released. 0.4 N m (after the seal is broken) is an estimate for a loosened lug lid, in the 0.3-0.5 N m range; it was chosen because the gripper's pinch
  slipped on the lid at 0.6 N m (the lid stalled at 31-45 degrees) and 0.4 N m is still 5 times the jar's table friction (0.45 x 0.65 kg x 9.8 m/s2 x 26 mm = 0.075 N m), so one arm cannot turn the lid without spinning the jar.
- **The clockwise (tightening) stop is soft.** The seated lid has a 1 degree stop that way, but the wrist can still turn it clockwise (89 degrees in the `tighten` run); it never unlatches and never scores above 0.25. The scorer, not the stop, makes tightening worthless.
- **The demo turns the lid about 103 degrees** (the wrist sweeps 103 degrees) whereas a real lug lid needs about a quarter turn; the unlatch threshold is 80 degrees.
- **Pickles and brine are fused to the jar** as one 0.40 kg mass (no slosh, no fluid). Spill is geometric: with 27 mm of headspace the brine reaches the rim
  at a tilt of tan = 2 x 27 / 72, about 37 degrees; the gate is 30 degrees at any step once the lid is unlatched (a sealed jar spills nothing). Real test: weigh the
  jar before and after; at most 5 g lost.
- Glass is a rigid body; the lug thread, the safety button and the liner are not modelled (the lug quarter turn and the 0.4 N m are). The lid is held 10 mm above the shoulder ledge
  (the lugs) so that the blade fits under its edge; a real lid skirt sits lower. The lid's friction is 0.9 (tinplate against the rubber-like gripper pads; the jar's own friction is the real 0.45 for glass on laminate).
- The demo frees the lid with a straight vertical lift of the blade held under the lid edge (not a tilting lever); the release is the lid edge raised 1 mm, whatever raises it, so long as the blade tip is in place while the lid is up. The other hand lifting the lid while the blade sits under it would also release it: a known simplification.
- The blade is a plain flat plate, 2 mm thick; the prise is judged from the blade pose and the lid z joint, not from a contact force.

## Files
- `task.py`: scene, instruction, `success()`, `score()` and the seal/latch logic. `geometry.py`: all dimensions and mechanism numbers. `scenekit.py`: unchanged helper (USD bodies and articulations; the same helper as the merged child-resistant-cap task).
- `build_assets.py`: writes `assets/pickle_jar.usda` (jar and lid articulation) and `assets/lever_tool.usda` (needs `pxr`, Isaac Sim's Python). `assets/` holds the two generated files.
- `demo.py`: open-loop scripted demo (`DEMO_VARIANT` = `settle`, `hover`, `tighten`, `right_only`, `left_only`, `no_tool`, `pop_only`, `lid_not_released`).
- `tests/test_scoring.py`: scoring boundaries, every partial level, zero gates, two valid solutions, lever geometry, mass consistency, no pose reads in the demo, blind-word scan.
- `task.json`, `media/` (`demo_run.mp4`, `dream_run.mp4`, `setup.png`).

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Open the pickle jar and set the lid down." --scorer task_score --max-steps 1800 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1800            # scripted demo; DEMO_VARIANT=<name> for the ablations
```
Run limit 1,800 steps (90 s at 20 Hz). The scripted demo succeeds at step 1009 (50 s) and the run ends there. No random seeds
(`object_jitter=0`); the simulator is deterministic (two demo runs gave identical step counts). One demo run takes about 3 minutes of wall time including simulator start.

## Scene (metres, station frame, table top z = 0.7475; x forward, y left)
| Object | Size (mm) | Material | Mass | Start pose |
|---|---|---|---|---|
| Pickle jar | glass cylinder 78 across, 105 tall to the shoulder ledge, neck 65 across and 17 tall (glass top at 122); pickles and brine inside (95 mm deep) | clear green glass, friction 0.45 | 0.65 kg (0.25 glass + 0.40 contents) | base centre (0.22, 0), upright |
| Lid | cylinder 70 across, 21 tall (bottom at 115, 10 mm above the shoulder ledge, top at 136), two dark lug marks on top | gold tinplate, friction 0.9 | 0.03 kg | on the jar, axis on the jar's |
| Lever tool | steel blade 125 x 24 x 2 (x from -50 to +75 about the fin centre), ABS grip fin 100 x 10 x 40 (orange) on the blade | steel and ABS, friction 0.6 | 0.091 kg (steel 0.047 + ABS 0.044) | lying flat, fin centre (0.34, -0.14), tip toward +y |

Table top is the only other object. Nothing else is on the table; the lid has no drop zone, any clear spot on the table counts.

## Scoring
Scored on the end state; `score()` never returns 1.0 before the 1 s hold (it reports 0.75 meanwhile). Latched events: the seal break and the unlatch.
- 0 (latched): the jar tilts more than 30 degrees at any step once the lid is unlatched (brine at the rim), or its base drops more than 20 mm below the table.
- 0.25: seal broken: the lever blade tip 28-38 mm from the jar axis (under the lid skirt), at the shoulder ledge height or up to 2 mm above the lid's (raised) bottom edge, pointing at the jar axis within 35 degrees, with the lid edge raised at least 1 mm by it, for 5 steps. Hovering near the rim earns nothing.
- 0.5: lid unlatched (turned 80 degrees counter-clockwise seen from above relative to the jar: the loosening way; a tightening turn never unlatches) and still within 50 mm of the jar axis.
- 0.75: lid 50 mm or more from the jar axis but held, in the air, moving, or not yet at rest on the table.
- 1.0: lid unlatched and flat on the table (centre 6.5-14.5 mm above it), 104 mm or more from the jar axis (at least 30 mm clear of the jar's footprint), lid and jar speeds < 5 mm/s and jar turn rate < 0.1 rad/s, jar tilt at most 10 degrees and within 60 mm of its start, both finger-tip gaps at most 58 mm (open: 48 mm, pinching: about 90 mm) and both gripper TCPs at least 70 mm from the jar axis (surface plus 30 mm) and from the lid centre, for 20 consecutive steps. Contact with the tool, lid or jar by a gripper is never penalised.
Gripper state is read from the simulator's gripper links each step (the scorer may; the demo does not). Two valid solutions score 1.0 in the unit tests (the lid set down on either side of the jar, either arm holding the jar).

## Clarity: reading the setup picture
The green jar with the gold lid and its dark cross marks is in the middle; the orange-and-steel lever lying on the table beside it on the right arm's side, its blade pointing at the jar. There is nothing else on the table.

## Shortcuts and what stops them
- Twisting without the tool: the lid is locked (30 N m); score 0 (`no_tool`).
- Popping the seal and twisting with one arm: with no hand on the jar the jar turns with the lid and the lid never turns 80 degrees relative to it; the run ends at 0.25 (`right_only`; a slow twist was also tried on an earlier version of the code, not re-run on the final code).
- Holding the jar and doing nothing else: 0 (`left_only`).
- Lifting the jar by its sealed lid, or knocking the jar over to get at the lid: the lid does not come off; a tilt over 30 degrees once the lid is unlatched scores 0.
- Hammering or hovering with the tool: the release needs the blade under the lid edge and the lid edge really raised 1 mm for 5 steps.
- Turning the lid the wrong way (clockwise, tightening): the seated lid is given a 1 degree stop that way but the simulated stop is soft (in the `tighten` run the wrist turned it about 89 degrees clockwise anyway); a clockwise turn of any size never unlatches, so the score stays at what the seal release alone earns, 0.25 (`tighten`, below).
- Twisting the lid first and lifting it afterwards: the lid is locked until the blade has raised it, so it gains no turn; lifting the lid by hand without the blade does not release it.
- Keeping hold of the lid or the jar at the end: 0.75 (`lid_not_released`).

## Feasibility and demo observations
The demo is open loop: each waypoint is computed from constants in `geometry.py` (jar, tool and lid start positions and the set-down point). The only
feedback is each arm's own joint positions (the `state` observation), integrated into the command in free air to cancel tracking error. It reads no jar, lid or tool pose
(a unit test greps the demo for pose reads). Steps: the left arm closes on the jar body from the +y side (pads along x, hand tilted 30 degrees below horizontal);
the right arm picks the lever by its fin, slides the blade tip in on the ledge to 33.5 mm from the jar axis under the lid edge (aimed 6 mm toward the base in x, the measured shift of the held jar), lifts it 10 mm so that the lid edge goes up, holds 1 s; withdraws and puts the lever down; pinches the lid from above and turns its wrist 103 degrees (the lid unlatches against the jar held by the left arm); lifts the lid 30 mm, carries it to (0.22, -0.17),
lowers it, opens, and withdraws; the left arm opens last and retreats along its approach.

## How this differs from nearby tasks
- The merged child-resistant-cap task is a push-and-turn cap on a 80 g vial with no seal and no reaction on the vial worth holding; here the lid is locked
  until a separate tool breaks the seal, the jar is 0.65 kg and spins unless held, and the lid must also be lifted off and set down.
- Paint-tin lever-off tasks pry a press-fit lid straight up with a screwdriver: no turn and no holding hand.
- Rotary can opener tasks have one hand keep the handles; here the second hand's job is to react the lid torque through a glass jar on a low-friction table.

## Novelty
RLBench `open_jar` and Bi-DexHands `BottleCap` unscrew an unsealed lid with no seal, no tool and no reaction on the container. The merged child-resistant-cap task is a push-and-turn cap on a light vial, and open PR #922 (paint-tin lever) is a straight lid prise with no turn and no holding hand. Here the lid is stuck until a tool frees it, the jar is 0.65 kg and spins unless the second hand holds it, and the lid is then lifted off and set down.

## Real-world build and reset
A 16 oz (about 500 mL) glass jar filled to 95 mm with pickles and brine (0.65 kg), a hot-filled vacuum-sealed twist-off lug lid (70 mm, 4 lugs; the safety button
down), a flat steel lid lifter (125 x 24 x 2 mm) and an ABS fin. Reset: a new sealed jar on the laminate counter, tool flat beside it. Pass: lid on the counter and the
jar upright with at most 5 g of brine lost (weigh the jar before and after).

## Measured in the simulator
Final task.py, geometry.py, demo.py and scorer (the loosening-direction unlatch), `object_jitter=0`, deterministic. Basis column: **measured, final code** = run on the final code; **measured, earlier scorer** = run before the loosening-direction change; its path never reaches the changed unlatch test (the run never unlatches), so it is unaffected (traces show no unlatch and the lid z lock window unchanged); one popped-state limit differs by 1 degree.
Demo milestones: seal released at step 426 (the blade raised the lid edge 1 mm or more), lid unlatched at step 728 (turned 80 degrees relative to the jar, 97 degrees at most), success and end at step 1009 (50 s). Peak jar tilt 7.3 degrees (the gate is 30), the jar ends 4 mm from its start.
| Run | Score | Basis | Notes |
|---|---|---|---|
| Demo (open loop, `demo.py`; the wrist turns anticlockwise, the loosening way) | 1.0 | measured, final code | success at step 1009; video last frames: jar upright and open, lid on the table beside the right gripper, both grippers open |
| Dream-1 run (1,800 steps) | 0.0 | measured, earlier scorer | the right gripper came within 23 mm of the lid (step 324) and hovered around the jar, but the jar, lid and tool never moved (jar and tool poses unchanged, lid never lifted) |
| `settle` / do nothing (8 s hold) | 0.0 | measured, earlier scorer | everything at rest; the lid stays locked |
| `left_only` (hold the jar, nothing else) | 0.0 | measured, earlier scorer | |
| `hover` (everything up to the lever, but the blade is held under the lid edge without lifting it, then stop) | 0.0 | measured, earlier scorer | hovering near the rim earns nothing: the seal needs the lid edge raised 1 mm |
| `pop_only` (blade lifts the lid edge, then stop) | 0.25 | measured, earlier scorer | seal released at step 426 by the lid edge really being raised; no further credit |
| `no_tool` (hold the jar, twist the lid without the tool) | 0.0 | measured, earlier scorer | the locked lid turns 3.4 degrees at most; the sealed lid lifts and tips the jar (no gate while sealed) |
| `right_only` (lever and twist with the right arm only) | 0.25 | measured, final code | the jar turns about 120 degrees with the lid; the lid turns 1.2 degrees relative to it, then the arm lifts the jar by its lid |
| `lid_not_released` (everything, but the right hand keeps the lid) | 0.75 | measured, earlier scorer | lid unlatched and set down but still held |
| `tighten` (as the demo, but the wrist turns the lid clockwise, tightening, 103 degrees) | 0.25 | measured, final code | seal released at step 426; the lid turned 89 degrees clockwise (the 1 degree stop is soft) and never unlatched: no 0.5 or higher |
| swap arms (left arm the lever, right arm holds the jar) | not run | unit test only | `test_two_valid_solutions_both_full`: either arm holding the jar, lid on either side, scores 1.0 |
Dream-1 was run before the loosening-direction change and not repeated (the per-task run cap was reached); it never popped the seal or touched the jar, tool or lid, so the change cannot alter its score.
Tuning history for the redesigned (physical) release: the first demo (blade 1.5 mm above the ledge) stalled the blade at r 43 mm (it was pitched 3 mm nose-down) and scored 0; at 4-5 mm above the ledge the blade reached the lid edge but 10 mm off in x (the left grip pulls the jar 5-10 mm toward the base), which raised the lid but outside the pose window; shifting the blade 6 mm in x and widening the radial window to 28-38 mm (the tip sits within 3 mm of the 35 mm skirt, which a 2 mm blade pose allows) gave 1.0. Earlier tuning of the 0.4 N m seal torque (the pinch slipped at 0.6 N m) is described above.
