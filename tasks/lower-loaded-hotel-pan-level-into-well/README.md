# Lower a water-filled steel hotel pan level into a steam-table well and let go

A stainless half-size hotel pan (GN 1/2: 325 x 265 mm, 40 mm deep, 10 mm flange) filled with 20 mm of cold tap water (2.4 kg in total) sits on the table. A black
steam-table counter next to it has a pit that fits the pan body with 6 mm clearance per side. The arms must carry the pan by its
rim flange at its two short ends, keep it level, lower it into the pit until the flange rests on the counter all round, and let go. Core ability: a
two-handed level carry and a tight seating of a heavy open vessel. Tilting it so that water would overflow at its lowest rim corner (7.3 degrees across, 6.0 degrees along, less when combined) spills water (score 0); an
unaligned pan jams on the pit edge instead of seating.

## Importance
Restaurant cooks (O*NET 35-2014.00, about 1.4 million US workers per BLS) and food preparation workers (35-2021.00, about 0.9 million) set full pans into steam-table wells and hot-holding
wells at every service. A loaded half-size pan is 2-4 kg and the liquid is at 60 C or hotter; the real test below uses a 20 mm fill of cold tap water (2.4 kg in total), so a tilt, a drop onto the rim or
a pan left sitting on the edge is a scald hazard and a food-safety loss. Category: Impact on jobs.

## Construct validity
- Real size: GN 1/2 footprint 325 x 265 mm, 40 mm deep (a standard GN depth), 10 mm rim flange, a 2 mm wall, a flange-supported
  pan in a counter opening (the real way these pans are held) and a 2.4 kg carried mass (a real GN 1/2 40 mm pan is thin 0.8-1 mm steel, about 0.7 kg, plus 20 mm of cold tap water, 1.68 kg over the 261 x 321 mm inside).
- The arms really carry the full 2.4 kg (1.2 kg per arm); no mass was scaled down.
- The well is a counter slab with a pit that has the clearance a real well has (6 mm per side); the flange, not the floor, carries the pan.

## What is simplified / not modelled
- Liquid is not simulated: the water is a fused blue box inside the pan body (20 mm deep; "full pan" in the instruction means this 20 mm fill). Spill is
  geometric overflow at the lowest rim corner, checked every step and latched: the free surface stays horizontal and water leaves when a rim corner drops
  below it. With n the pan's up-normal the corner drop is 0.1325 m x |n_x/n_z| + 0.1625 m x |n_y/n_z|, and the pan spills when it exceeds the 17 mm of wall
  above the water (40 mm - 3 mm floor - 20 mm). Alone, tilt across the 265 mm width spills at 7.3 degrees and tilt along the 325 mm length at 6.0 degrees; a
  combination spills earlier (5 degrees across plus 4 degrees along drops the corner 23 mm). Slosh from acceleration is not modelled.
  Real-world equivalent: carry the pan over a weighed absorbent mat or catch tray; the spill allowed is at most 5 mL.
- Mass construction: the simulated pan is one 2.4 kg rigid body with 3 mm floor and 2 mm walls (by volume, 3 mm floor, 2 mm walls and the flange of stainless steel would be about 3.1 kg, so the simulated mass is set directly to 2.4 kg and is lighter than that steel; a real thin-steel pan is about 0.7 kg); the 2.4 kg is the total simulated carried mass (pan plus water), not a steel-only mass, and the water box does not add to it.
- The pan is one rigid body with square corners (real pans have rounded corners and stacking lugs); there are no handles or tabs: the arms hold the pan by its 10 mm x 3 mm rim flange, as a cook holds a handle-less pan (real pans may also have drop handles).
- The well is four fixed boxes standing on the table (40 mm high), not recessed into a heated bath; no water bath, no heat.
- Counter material is shown as black steel; friction is not important because the pan is seated, not slid.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: open-loop scripted demo (`DEMO_VARIANT` = `settle`, `spill`, `spill2`, `one_arm`, `tilt`, `jam`, `drop`, `no_release`, `stop_above`, `off_alt`, `off_alt2`).
- `tests/test_scoring.py`: scoring boundaries, every partial level, zero gates, two valid solutions, hold, blind-word scan.
- `task.json`, `media/` (`demo_run.mp4`, `dream_run.mp4`, `setup.png`).

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Move the full steel pan from the table into the black steam-table well and let go, so it sits in the well with its flange resting on the counter." --scorer task_score --max-steps 900 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 900            # scripted demo; DEMO_VARIANT=<name> for the ablations
```
Run limit 900 steps (45 s at 20 Hz). The demo finishes in 595 steps (30 s). No random seeds (`object_jitter=0`);
the simulator is deterministic.

## Scene (metres, station frame, table top z = 0.7475; x forward, y left)
| Object | Size (mm) | Material | Mass | Start pose |
|---|---|---|---|---|
| Hotel pan | floor 265 (x) x 325 (y) x 3; four walls 2 mm thick, 40 mm tall; rim flange 10 mm wide x 3 mm thick on the top of the walls (outside 285 x 345) | stainless steel (grey), friction 0.5 | 2.4 kg in total | centre (0.085, 0), underside on the table, long axis along y |
| Water | 261 x 321 x 20 box inside the pan, blue, part of the pan body (cold tap water, 1.68 kg real) | - | included in the 2.4 kg | inside the pan on the floor |
| Steam-table counter | four fixed strips 40 mm high around a pit 277 (x) x 337 (y): x-end strips 30 x 427 mm, side strips 277 x 45 mm | black steel | fixed | pit centre (0.40, 0); the front strip edge is 4 mm from the start flange |

The pan seated: flange underside on the counter top (z 0.7875), pan origin at z 0.7505, floor 3 mm above the table. Clearance of the
body in the pit: 6 mm per side in x and in y; the flange overlaps the counter by 4 mm per side.

## Scoring
Scored on the end pose; the zero gates are latched; the 1.0 condition has to hold for 20 consecutive steps (1 s) and `score()` never
returns 1.0 before that (it reports 0.75 meanwhile).
- 0 (latched): the water would overflow at any step (spill: the lowest rim corner drops more than 17 mm below the water level, corner drop = 0.1325 m x |n_x/n_z| + 0.1625 m x |n_y/n_z|; single-axis limits 7.3 degrees across, 6.0 degrees along), moves faster than 0.5 m/s (dropped onto the counter), or ends off the table (z < table - 20 mm).
- 0.25: held over the well: centre within 20 mm of the pit centre in x and y, every flange corner at least 12 mm above the counter top.
- 0.5: pan underside below the counter top (body in the pit) within 20 mm of the pit centre, flange not seated (a corner more than 2 mm proud).
- 0.75: all four flange-underside corners within 2 mm of the counter top, centre within 8 mm of the pit centre, tilt at most 2 degrees.
- 1.0: also both grippers open (finger-tip bodies 48 mm apart, against 84 mm while pinching the rim) and their TCPs at least 40 mm from the grip points (middle of each short-end flange), pan speed < 0.01 m/s and < 0.1 rad/s, all arm joint speeds < 0.05 rad/s, for 20 steps.
Gripper state is read from the simulator's gripper links each step (the scorer may; the demo does not). Two valid solutions score 1.0 in the unit tests (small offsets, either arm on either short end); in the simulator the demo with +3/+4 mm and -3/-4 mm offsets also scores 1.0.

## Clarity: reading the setup picture
The pan is the grey tray; the black strips are the counter; the pit is the gap in the middle of the black strips. The pan starts on the table with its long axis across the table (y).

## Shortcuts and what stops them
See the ablation table: one arm only tilts the pan (zero gate); lowering one side first tilts it; releasing it above the well drops it
(speed gate); carrying it over and holding it above the pit earns 0.25 only; keeping hold after seating earns at most 0.75 (0.5 in the simulator, see the table); a pan that is
not aligned rests on the rim (0.25 / 0.5, never above).

## Feasibility and demo observations
The demo is open loop: each waypoint is computed from constants in task.py (pan start, rim height, well position). The only
feedback is each arm's own joint positions (the `state` observation), integrated into the command to cancel steady tracking error. It
reads no object or simulator pose.

## How this differs from nearby tasks
- #1759 dock-a-laptop-level: both sides of a light rigid laptop are lifted and docked with a press. Here the load is a real-size 2.4 kg
  water-filled open pan carried level by its rim, with zero gates for spill (water overflow at the lowest rim corner) and for speed over 0.5 m/s (dropped), a
  seating into a 6 mm-per-side pit where a misaligned pan jams on the rim, and a required release (grippers open and clear, 1 s still).
- #1695 tart-ring lowering: a ring is lowered over a tart with no load or spill limit; nothing is seated on a flange.
- #1685 tip-full-pan tilts a pan on purpose; here any tilt is the failure.
- #649 steam-table well (arranging several pans) and #1363 steam-table lids are different objectives.

## Real-world build and reset
A stainless GN 1/2 x 40 mm pan (about 0.7 kg) filled with 20 mm of cold tap water (about 1.7 kg), 2.4 kg in total, carried over a weighed absorbent mat or catch tray (pass: at most 5 mL collected), a counter of 40 mm plywood strips with a 277 x 337 mm opening. Reset: pan on the table 4 mm from the counter, long axis along y.

## Measured in the simulator
Final task.py, demo.py and scorer (2.4 kg pan with 20 mm of water, lowest-rim-corner spill gate, no handles or tabs, grip on the 10 x 3 mm rim flange), `object_jitter=0`, deterministic. Every row below was run in the simulator with this final code (the demo row from the run that was recorded).
Demo carry: peak tilt across x 5.0 degrees, along y 0.3 degrees, peak corner drop 12.5 mm against the 17 mm freeboard (margin 4.5 mm); peak pan speed 0.09 m/s.
| Run | Score | Peak corner drop | Notes |
|---|---|---|---|
| Demo (open loop, `demo.py`) | 1.0 | 12.5 mm | 595 steps (30 s); the arms carry the full 2.4 kg |
| Dream-1 run (900 steps) | 0.0 | - | pan untouched; one run on the final tab-free scene (instruction unchanged) |
| `settle` / do nothing | 0.0 | 0 | pan rests at its start (no gate tripped) |
| `one_arm` (left arm only) | 0.0 | 34.9 mm | pan tilts about 10 degrees and falls: spill gate |
| `tilt` (left side lowered 50 mm less) | 0.0 | 36.3 mm | roll 8.9 degrees |
| `spill2` (left side 20 mm high during the carry) | 0.0 | 20.8 mm | roll 3.5 and pitch 5.6 degrees, each below its single-axis limit, together a spill |
| `spill` (left side raised 45 mm in the air, then the normal demo) | 0.0 | 26.2 mm | roll 7.4 degrees; the pan still ends seated, the gate is latched |
| `drop` (release over the well, no lowering) | 0.0 | 11.7 mm | speed gate (0.75 m/s) |
| `stop_above` (hold over the well) | 0.25 | 11.7 mm | |
| `no_release` (seat but keep holding) | 0.5 | 12.5 mm | the arms still holding keep the pan 14 mm proud and tilted 2 degrees, so it never counts as seated (this was 0.75 with the old tabs) |
| `jam` (target 15 mm toward the base) | 0.0 | 18.7 mm | pan jams on the rim and pitches 8.0 degrees |
| `off_alt` / `off_alt2` (+3/+4 mm, -3/-4 mm) | 1.0 / 1.0 | 12.2 / 12.7 mm | two valid solutions |

## Change history (reviewer feedback)
- Review 1 (spill detection): replaced the single 8-degree tilt rule by 20 mm of cold tap water, the per-corner overflow gate above and a real-world mat test (at most 5 mL); the fluid itself is not simulated.
- Review 2 (setup): the handle tabs are removed. The pan is carried by its own rim flange. Removing the tabs is the only change to the pan: size, 40 mm depth, 10 x 3 mm flange, 2.4 kg and water fill are as before. I had estimated beforehand that a rim-only grip of 2.4 kg would not hold; that estimate was too pessimistic: in the simulator the two-arm pinch on the flange (fingers across it) carries and seats the pan reliably (scored 1.0 with the flange raised 0, 6 and 12 mm; the final version has no raised rim).
