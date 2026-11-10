# Turn a T-head bracket out of a keyed housing and hold the two parts apart

A blue keyed housing (closed box, narrow slot between two front lips) holds the T-head of an orange bracket; the stem runs out
through the slot. The head (130 mm) is longer than the slot (110 mm), so a straight pull is blocked by the lips. Both brackets are
free rigid bodies. The left arm pinches the housing's back handle post and keeps it steady; the right arm pinches the orange handle
post, turns the bracket (about 80 deg in the demo) while its head passes the slot mouth, withdraws it, and both hold the parts apart
above their own marked outlines. Core ability: coordinated turn-then-withdraw of one free rigid part against another held part.
Real-world use: separating locked shipping hardware without bending it (O*NET 49-9041.00).

## How this differs from nearby tasks
- #1082 (separate linked S-hooks) has the same objective, separating two interlocked rigid parts. The mechanism here is different: the parts are
  not mutually looped; a keyed slot blocks a straight pull and the part must be reoriented by a measured turn while the other arm holds the housing
  to within 30 mm and 8 deg, then withdrawn through the slot and placed at a marked outline. The reviewer may still judge the ability close; we do not claim it is unrelated.
- #903 angles a single cross into a yoke: one part into a fixed seat, no second held free body. #165/#141 assemble slotted parts (insertion);
  this is the reverse, withdrawal after a turn. #1375 hooks a link onto something (engage), this disengages two held free bodies.
- #863 (pull a handle out through a slotted panel) also turns a T-head to fit a slot. There the panel is fixed and one part is pulled; here both parts are free rigid bodies, one arm must hold the housing steady (30 mm / 8 deg) while the other turns and withdraws the bracket, and both end held apart above marked outlines. Close in mechanism; we do not claim it is unrelated.

## Files
- `task.py` scene, instruction, `success()`, `score()`; `geom.py` pure box geometry, collision test and path planner (no simulator imports); `partkit.py` unchanged.
- `demo.py` closed-loop scripted demo. `DEMO_VARIANT`: `settle`, `one_arm`, `no_turn` (straight pull), `stop_early` (turn but do not leave), `keep_turn` (finish still turned).
- `ikscan.py` reach check of the whole planned path for both arms; `tests/` scoring and geometry tests; `task.json`, `media/`.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1500 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1500
```
Run limit 1,500 steps (75 s); the demo succeeds at step 517. No random seeds (`object_jitter=0`).

## Scene (m, station frame, table top 0.7475)
Housing: 0.30 x 0.18 x 0.044 box (floor, lid, walls; slot 0.110 wide, lips 10 mm thick, cavity 28 mm high) plus a 40 x 40 x 90 mm handle
post on its back face, 0.12 kg, centre (0.24, +0.10), slot toward -y, on a white outline. Orange bracket: 130 x 18 x 12 mm head
inside the cavity, 12 mm wide stem 150 mm long through the slot, 40 x 40 x 90 mm handle post at the stem end, 0.12 kg. A second white
outline (0.17 x 0.17) at (0.24, -0.18) marks where the orange head ends up. Handle posts are 90 mm tall so the finger bodies stay above the 44 mm housing.

## Scoring (every step; final state held 1 s = 20 steps)
Zero: either bracket origin under 20 mm above the table (not lifted), either tilt over 20 deg, either off the table area.
- 0.25: peak turn of the orange bracket relative to the housing >= 15 deg (peak is recorded while both are lifted).
- 0.5: peak turn >= 30 deg and head centre >= 0.12 m from the housing centre.
- 0.75: T-head fully out (all head corners beyond the lips' front face), >= 0.17 m apart, tilts <= 12 deg.
- 1.0: >= 0.20 m apart, tilts <= 8 deg, housing yaw <= 8 deg, housing within 30 mm of its outline, head within 30 mm of its outline.
Two valid solutions are tested: finish still turned 80 deg, or turned back to 0 (both 1.0; `keep_turn` also run in the simulator).

## Real-world build and reset
Printed or laminated housing (two plates, walls, lips) and a T-bolt-like bracket with a handle post; tape outlines. Reset: head in the
housing, stem through the slot, handle post upright on the table.

## Shortcuts and what stops them
- Straight pull: the 130 mm head cannot pass the 110 mm slot unturned (tested: blocked at 0 and 10 deg); `no_turn` scores 0.0.
- Lifting the head out over the lips: the housing has a closed lid 8 mm above the 12 mm head band, so there is no vertical exit.
- One arm: the orange bracket stays on the table, not lifted: 0.0 (`one_arm`).
- Turn without leaving: 0.25 (`stop_early`). Dropping or tilting either part: zero gate.
The scripted demo reads true poses (closed loop); it proves feasibility, not perception.

## Changes from the supplied spec (honest list)
- The supplied planar U-brackets cannot interlock in one plane (two U's with mouths facing each other either nest and slide apart, or their bars collide).
  The hook is now a keyed housing plus a T-head bracket; the idea (turn one part until it clears, then withdraw) is kept.
- Clearance: a rigid planner (`geom.plan`, box-intersection test over x, y, turn) finds a collision-free path with 10 mm margin on every box (not the 25 mm
  asked for: no path exists at 15 mm for this geometry; 12 mm works). Margin is verified in `tests/`. Housing 0.12 kg, orange 0.12 kg.
- The turn is needed because of the 130 vs 110 mm head/slot sizes: a slanted head can pass the slot from about 20-30 deg; the planned path with clearance
  turns the head while it passes the slot mouth (about 80 deg), so it is "turn while entering the mouth, then slide", not a pure turn in the cavity.
  The wrist/arm yaw does the whole turn (single step, no two-step turn needed; IK error 0.1 mm over the path).
- No overhead keeper; the lid closes the vertical exit. Thresholds 25/50 deg were lowered to 15/30 deg since the geometry is unblocked from about 20 deg.
- Held-in-air gate: 20 mm above the table (not 0.81 m).
- Outlines: housing on its own outline, orange head outline 0.28 m away.

## Measured in the simulator
All rows on the current `task.py` (not edited after the first demo; `demo.py` edited only before the first full run).
| Run | Score |
|---|---|
| Demo (closed loop) | 1.0 (success at step 517; tilts under 1.5 deg, peak turn 81 deg) |
| `keep_turn` (valid alternative, ends turned) | 1.0 (step 517) |
| `one_arm` | 0.0 |
| `no_turn` (straight pull) | 0.0 (head blocked by the lips; spec predicted 0-0.25) |
| `stop_early` (turn, no separation) | 0.25 |
| Settle (pure hold) | no pure settle run was done: the first run named `settle` actually executed the demo (260 steps). Rest tilt 2.5 deg and heights were read from the demo's first 100 steps; score is 0.0 until lifted |
| Dream-1 run | 0.0 (1,500 steps); the video shows only the arm approaching the parts, no grasp or turn |
