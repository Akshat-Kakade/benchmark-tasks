# Clock an open shell 90 degrees around a held shaft in the air

A blue shaft (45 mm cylinder with an upright handle stem and a black stripe) stands inside an orange open shell (a U of three
joined boxes, one wall white, a handle stem on a tab at its edge) with 20 mm clearance all round; a green bar lies on the table.
The left arm pinches the shaft's stem, the right arm the shell's handle; both lift 30 mm, then the right arm carries the shell
round the shaft through 90 deg clockwise (seen from above) until the white wall is parallel to the green bar, while the left arm
keeps the shaft at its position and heading. Core ability: rotating one free rigid body about another to a measured relative angle
while the inner body's pose is held fixed by the other hand. Real-world use: setting a service shell to an angle relative to a
registered shaft (O*NET 49-9041.00); turning both together gives the wrong relative angle.

## Files
`task.py` (scene, `success()`, `score()`), `partkit.py` (unchanged helper), `demo.py` (closed-loop scripted demo; `DEMO_VARIANT` =
`settle`, `one_arm`, `together`, `no_center`, `turn84`, `turn96`), `tests/` (scoring boundaries, every level, zero gates, two valid
solutions, geometry, blind-word scan), `task.json`, `media/`.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 600 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 450
```
Runtime per trial: 600 steps (30 s simulated); the demo succeeds at about step 371. No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Mass | Pose |
|---|---|---|---|
| Shaft | cylinder 45 dia x 70, handle stem 22 x 22 x 65 on top, black stripe on +x | 0.12 kg, friction 0.6 | axis (0.34, 0) |
| Shell | outer 140 x 140 x 35, opening 85 x 85 open towards +x, +y wall white, tab and handle stem 35 x 40 x 85 | 0.10 kg, friction 0.6 | opening centred on the shaft |
| Green bar | 10 x 140 x 3, fixed | - | x 0.425, y 0 |
Shaft-to-shell clearance 20 mm all round; the shell handle is 160 mm from the shaft axis. Both bodies start on the table.

## Scoring (measured at the end, held 1 s; two body poses only)
Zero: either body under 15 mm above its rest height (not held), shell touching the shaft, shaft moved over 30 mm, tilt over 12 deg.
- 0.25: shell turned >= 20 deg clockwise relative to the shaft.
- 0.5: >= 55 deg and shaft yaw change <= 20 deg.
- 0.75: 70-110 deg, shaft within 25 mm and 12 deg of its baseline (taken after step 10).
- 1.0: 78-102 deg, shaft within 15 mm and 8 deg, shell opening centre within 15 mm of the shaft axis, gap >= 8 mm, tilt <= 8 deg.
Two valid solutions: final turn 84 deg or 96 deg (both 1.0 in the offline tests; `turn84`/`turn96` were not run in the simulator).

## Real-world build and reset
A printed 45 x 70 mm shaft with a stem and a stripe, a printed three-wall shell with a handle tab, a taped bar. Reset: shaft
upright, shell round it, open side forward. Measure with a protractor between the white wall and the stripe.

## Shortcuts and what stops them
- Leaving the shaft on the table and turning only the shell: fails the "both held in the air" gate (`one_arm`: 0.0).
- Turning shaft and shell together: no relative angle (`together`: 0.0).
- Pushing the shell off-centre: the wall touches the shaft (zero gate) or concentricity fails (`no_center`: 0.75).
The demo reads true body poses (closed loop): it shows feasibility, not perception.

## How this differs from nearby tasks
- #941 assembles a split spacer around a captive shaft (a fit); here nothing is inserted, both parts are free and held in
  the air, and the score is the relative yaw of two concentric bodies.
- #990 carries a collar round a post (a path around a fixed post); here neither body is fixed, the other hand must hold the
  inner body's pose and the shell is turned about it in place.
- #1069 turns a long rod (one body, one rotation); here two free bodies are controlled at once to a relative angle.

## Measured in the simulator
Final `task.py`; the demo was run on it (8 simulator demo runs, 7 on earlier layouts, listed below).
| Run | Score |
|---|---|
| Settle (hold), earlier layout | 0.0 |
| Demo | 1.0 (success at step 371; turn 90 deg, shell centre 2 mm from the shaft axis, shaft drift 2 mm, yaw 0.01 deg) |
| `one_arm` (right arm only lifts the shell) | 0.0 (shaft stays on the table) |
| `together` (shaft turned with the shell) | 0.0 (an IK failure, not a clean ablation: the right arm could not follow, IK error 180 mm; no relative turn) |
| `no_center` (shell 16 mm off, no correction) | 0.75 |
| Dream-1 run | 0.0 (600 steps) |

## Honest deviations and limits
- Spec changes: shaft is a 70 mm cylinder with an upright stem on its axis (a sideways handle would be swept by the shell wall);
  both bodies start on the table and are lifted 30 mm (not 0.825 m); 1.0 band 78-102 deg, 15 mm, concentricity 15 mm; shell
  handle 160 mm from the shaft; shell opening is open on one side only; one 90 deg sweep (no regrasp needed).
- Earlier layouts failed: a shell handle 130 mm out put the two hands about 5 cm apart (hand contact, score 0); a shaft handle
  bar offset 100 mm from the axis tilted the shaft 20-30 deg in the pinch (0.0).
- `turn84` and `turn96` were not simulated (unit tests only); `settle` was run on an earlier layout only. Tests import `yam_arena`, so they run on the VM environment only.
