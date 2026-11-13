# Lower a loaded steel hotel pan level into a steam-table well and let go

A stainless half-size hotel pan (GN 1/2: 325 x 265 mm, 40 mm deep, 10 mm flange) full of food (3.0 kg) sits on the table. A black
steam-table counter next to it has a pit that fits the pan body with 6 mm clearance per side. The arms must carry the pan by its
two end tabs, keep it level, lower it into the pit until the flange rests on the counter all round, and let go. Core ability: a
two-handed level carry and a tight seating of a heavy open vessel. Tilting it over 8 degrees spills hot food (score 0); an
unaligned pan jams on the pit edge instead of seating.

## Importance
Restaurant cooks (O*NET 35-2014.00) and the kitchen staff working with them set full pans into steam-table wells and hot-holding
wells at every service. A full half-size pan is about 3 kg and the liquid is at 60 C or hotter, so a tilt, a drop onto the rim or
a pan left sitting on the edge is a scald hazard and a food-safety loss. Category: Impact on jobs.

## Construct validity
- Real size: GN 1/2 footprint 325 x 265 mm, 40 mm deep (a standard GN depth), 10 mm rim flange, a 2 mm wall, a flange-supported
  pan in a counter opening (the real way these pans are held) and a real 3.0 kg load (about 0.7 kg pan plus about 2.3 kg of food).
- The arms really carry the 3.0 kg: no mass was scaled down (a 1.5 kg share per arm).
- The well is a counter slab with a pit that has the clearance a real well has (6 mm per side); the flange, not the floor, carries the pan.

## What is simplified / not modelled
- The food is a fused box inside the pan body (30 mm deep, orange-brown): no slosh and no fluid. The spill rule is a tilt limit of
  8 degrees (real liquid depth 30 mm in a 265 x 325 mm pan spills at a few degrees of tilt plus slosh), checked every step.
- The pan is one rigid body with square corners (real pans have rounded corners and stacking lugs); handles are two upright
  orange tabs 24 x 20 x 45 mm on the short flanges (real pans have drop handles or no handle, and are held by the rim).
- The well is four fixed boxes standing on the table (40 mm high), not recessed into a heated bath; no water bath, no heat.
- Counter material is shown as black steel; friction is not important because the pan is seated, not slid.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: open-loop scripted demo (`DEMO_VARIANT` = `settle`, `one_arm`, `tilt`, `jam`, `drop`, `no_release`, `stop_above`, `off_alt`, `off_alt2`).
- `tests/test_scoring.py`: scoring boundaries, every partial level, zero gates, two valid solutions, hold, blind-word scan.
- `task.json`, `media/` (`demo_run.mp4`, `dream_run.mp4`, `setup.png`).

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 900 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 900            # scripted demo; DEMO_VARIANT=<name> for the ablations
```
Run limit 900 steps (45 s at 20 Hz). The demo finishes in 592 steps (30 s). No random seeds (`object_jitter=0`);
the simulator is deterministic.

## Scene (metres, station frame, table top z = 0.7475; x forward, y left)
| Object | Size (mm) | Material | Mass | Start pose |
|---|---|---|---|---|
| Hotel pan | floor 265 (x) x 325 (y) x 3; four walls 2 mm thick, 40 mm tall; rim flange 10 mm wide x 3 mm thick on the top of the walls (outside 285 x 345) | stainless steel (grey), friction 0.5 | 3.0 kg in total | centre (0.085, 0), underside on the table, long axis along y |
| Food | 261 x 321 x 30 box inside the pan, orange-brown, part of the pan body | - | included in the 3.0 kg | inside the pan on the floor |
| Handle tabs (2) | 24 (x) x 20 (y) x 45 (z), orange, on the short flanges, tops 45 mm above the flange | stainless steel, part of the pan body | included | centres y = +-0.1675 from the pan centre, x = 0 |
| Steam-table counter | four fixed strips 40 mm high around a pit 277 (x) x 337 (y): x-end strips 30 x 427 mm, side strips 277 x 45 mm | black steel | fixed | pit centre (0.40, 0); the front strip edge is 4 mm from the start flange |

The pan seated: flange underside on the counter top (z 0.7875), pan origin at z 0.7505, floor 3 mm above the table. Clearance of the
body in the pit: 6 mm per side in x and in y; the flange overlaps the counter by 4 mm per side.

## Scoring
Scored on the end pose; the zero gates are latched; the 1.0 condition has to hold for 20 consecutive steps (1 s) and `score()` never
returns 1.0 before that (it reports 0.75 meanwhile).
- 0 (latched): the pan tilts over 8 degrees at any step (spill), moves faster than 0.5 m/s (dropped onto the counter), or ends off the table (z < table - 20 mm).
- 0.25: held over the well: centre within 20 mm of the pit centre in x and y, every flange corner at least 12 mm above the counter top.
- 0.5: pan underside below the counter top (body in the pit) within 20 mm of the pit centre, flange not seated (a corner more than 2 mm proud).
- 0.75: all four flange-underside corners within 2 mm of the counter top, centre within 8 mm of the pit centre, tilt at most 2 degrees.
- 1.0: also both grippers open (finger-tip bodies 48 mm apart, against 84 mm while pinching a tab) and their TCPs at least 40 mm from the handle tabs, pan speed < 0.01 m/s and < 0.1 rad/s, all arm joint speeds < 0.05 rad/s, for 20 steps.
Gripper state is read from the simulator's gripper links each step (the scorer may; the demo does not). Two valid solutions score 1.0 in the unit tests (small offsets, either arm on either tab); in the simulator the demo with +3/+4 mm and -3/-4 mm offsets also scores 1.0.

## Clarity: reading the setup picture
Orange tabs are the handles; the black strips are the counter; the pit is the gap in the middle of the black strips. The pan starts on the table with its long axis across the table (y).

## Shortcuts and what stops them
See the ablation table: one arm only tilts the pan (zero gate); lowering one side first tilts it; releasing it above the well drops it
(speed gate); carrying it over and holding it above the pit earns 0.25 only; keeping hold after seating earns 0.75 only; a pan that is
not aligned rests on the rim (0.25 / 0.5, never above).

## Feasibility and demo observations
The demo is open loop: each waypoint is computed from constants in task.py (pan start, handle height, well position). The only
feedback is each arm's own joint positions (the `state` observation), integrated into the command to cancel steady tracking error. It
reads no object or simulator pose.

## How this differs from nearby tasks
- #1759 dock-a-laptop-level lifts a rigid laptop by both sides and docks it with a press; here the load is a heavy open vessel with
  a spill limit, the opening has a 6 mm clearance and the pan is released resting on its own flange.
- #1685 tip-full-pan tilts a pan on purpose and #1695 tart-ring lowering lowers a ring over a tart; neither seats a loaded pan
  level in a tight pit.
- #649 steam-table well (arrangement of several pans) and #1363 steam-table lids are different objectives.

## Real-world build and reset
A stainless GN 1/2 x 40 mm pan filled with 2.3 kg of water-based food (or sand with a lid), two welded 24 x 20 x 45 mm tabs on the short
flanges, a counter of 40 mm plywood strips with a 277 x 337 mm opening. Reset: pan on the table 4 mm from the counter, tabs along y.

## Measured in the simulator
All numbers are from runs in the simulator (object_jitter=0, deterministic). Peak pan tilt during the demo carry: 5.3 degrees (gate 8).
| Run | Score | Notes |
|---|---|---|
| Demo (open loop, `demo.py`) | 1.0 | success at step 583, run ends at 592; arms carry the full 3.0 kg |
| Dream-1 run (900 steps) | 0.0 | the pan was not touched |
| `settle` / do nothing | 0.0 | pan rests at its start, no gate |
| `one_arm` (left arm only) | 0.0 | pan tilts and drops (zero gate) |
| `tilt` (left side lowered 50 mm less) | 0.0 | roll reached 8.5 degrees during lowering: spill gate, although the pan ended seated |
| `drop` (release over the well, no lowering) | 0.0 | speed gate |
| `stop_above` (hold over the well) | 0.25 | |
| `no_release` (seat but keep holding) | 0.75 | |
| `jam` (15 mm toward the base) | 0.0 | pan jammed on the rim and pitched 8.2 degrees (gate) |
| `off_alt` / `off_alt2` (+3/+4 mm, -3/-4 mm) | 1.0 / 1.0 | two valid solutions |
`one_arm`, `tilt`, `drop`, `stop_above`, `no_release` and `jam` were measured before a 4 mm x-offset was added to the demo's carry target (the carried pan sits 4.6 mm behind the commanded x because it pitches about the pinch); `jam` therefore was 15 mm off, now about 11 mm. The demo, `off_alt` and `off_alt2` were measured on the final files. Without the offset a -4 mm alternative jammed (0.0).
