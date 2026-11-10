# Lower a closed frame over an upright post and seat it level

A blue rectangular frame (four joined bars with a 160 x 220 mm opening) rests on two low pedestals with an orange
handle stem on each of its long ends. A grey post stands in front of it, inside a green seating outline. Both arms pinch
the stems, lift the frame, carry its opening over the post and lower it level so it sits flat on the outline with the post
in the middle of the opening and touching no bar. Core ability: steering and lowering a closed rigid aperture over an
upright obstruction with two distant grips and no contact. Real-world use: fitting a protective or inspection frame around
equipment that cannot be moved (O*NET 49-9041.00, installation and maintenance of machinery); a frame that catches the
post cannot seat. The frame is one free rigid body (a rigid box assembly); nothing is pinned, hinged or fastened.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `settle`, `one_arm`, `no_center`, `stop_above`, `off_alt`, `off_alt2`, `yaw5`).
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid end poses, blind-word scan.
- `task.json`, `media/`.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1800 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1100
```
Runtime per trial: run limit 1,800 steps (90 s simulated); the demo succeeds at about step 465 (about 25 s). No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Mass | Pose |
|---|---|---|---|
| Frame | outer 260 (x) x 340 (y) x 16 thick bars; opening 160 x 220; a 40 x 35 x 86 handle stem at each long end (y = +-185, 70 mm above the bars) | 0.16 kg, friction 0.6 | underside centre (0.18, 0), on the pedestals |
| Pedestals | 30 x 30 x 85, grey, fixed, under the two stems | - | (0.18, +-0.185) |
| Post | cylinder, diameter 60, height 70, grey, fixed | - | (0.34, 0) |
| Seating outline | 270 x 350 x 8, green, fixed | - | (0.34, 0); frame seated: underside 0.7555 |
The opening leaves 50 mm (x) and 80 mm (y) each side of the post; the frame starts 15 mm above the post top.

## Scoring (measured at the end, held 1 s = 20 steps; frame pose only)
Zero gate: tilt over 12 deg; a bar reaches the post (underside below the post top while the post is not clear of every inner bar side); the frame below the post top but 60 mm or more off the post (fallen aside); or the post is not inside the opening.
- 0.25: post axis inside the frame opening (frame held above the post).
- 0.5: frame centre within 40 mm of the post axis and underside below 0.810 m (below the post top).
- 0.75: centre within 25 mm, underside within 20 mm of the outline (<= 0.7755), tilt <= 12 deg.
- 1.0: centre within 15 mm, underside within 12 mm of the outline (<= 0.7675), tilt <= 8 deg, post at least 8 mm from every inner bar side.
Two valid solutions end 1.0: centred, or offset 10 mm in x and/or y, or yawed 5 deg (unit tests only; not run in the simulator).

## Real-world build and reset
Laminate or printed bars joined into a 260 x 340 mm frame with two 70 mm handle stems; a 60 mm post glued in a base plate with a taped outline;
two 85 mm pedestals. Reset: frame on the pedestals. The real check is a feeler gauge around the post and a level on the frame.

## Shortcuts and what stops them
- Placing the frame beside the post fails the inside-the-opening gate (score 0).
- Dropping the frame crooked or on a bar fails the tilt and bar-on-post gates; a frame not carried over the post scores 0.
- One hand only: the frame swings about the single grip and never reaches the post (`one_arm`, 0.0).
- Skipping the centring correction: the front bar lands on the post (`no_center`, 0.0).
- The post is fixed and the frame is lowered to the outline, so it is not possible to score by moving the post or stopping above it (`stop_above`, 0.25).
The scripted demo reads the true frame pose (closed loop); it proves feasibility, not perception.

## How this differs from nearby tasks
- #841 seats a small casting ring over a crown (a short drop of a small hand-held ring); here a 340 mm frame with two distant handles passes a post with 50 to 80 mm clearance and both hands control its level.
- #739 sets a cover onto a hidden dowel (a pin search and press); here the post is visible, the aperture must clear it on the whole 85 mm descent, and the score is clearance, centre and tilt.
- #552 drops a rope eye over a bollard (a flexible loop); here the aperture is a rigid frame, so tilt and four-sided clearance matter.
- New capability: closed-loop two-hand tilt and position control of a rigid aperture past an obstruction.

## Measured in the simulator
All rows on the current `task.py` (it did not change after run 4; `demo.py` and `task.py` changed during runs 2-4, see deviations).
| Run | Score |
|---|---|
| Settle (explicit hold) | 0.0 |
| Demo | 1.0 (success at step 464; tilt under 0.5 deg, post clearance 50 mm) |
| `one_arm` (left hand only) | 0.0 |
| `no_center` (frame carried 70 mm too far forward, no correction) | 0.0 (bar on the post, tilt 15 deg) |
| `stop_above` (hold over the post, no lowering) | 0.25 |
| `off_alt`, `off_alt2`, `yaw5` | not run (offline tests cover the scoring) |
| Dream-1 run | 0.0 (1,800 steps; the frame stayed untouched on its pedestals) |

## Honest deviations and limits
- Spec geometry changed: post height 70 mm (spec 90), frame mass 0.16 kg (spec 0.25), frame handles are 86 mm stems, pedestals hold the frame 85 mm above the table, and the frame starts at x 0.18 (spec 0.22). Scoring heights are relative to the seated frame (underside 0.7555), not the spec's absolute numbers.
- The first demos failed (frame pitched 10-19 deg): the pedestals were under the frame's final footprint and touched its back bar. Moving the pedestals under the stems and the frame start back to x 0.18 fixed it; the pinch also uses fingers open along x, so the jaws resist pitch.
- The frame is lowered while the hands still hold it; the 1.0 hold counts while held on the outline.
- Arm-arm clearance: the hands are 0.37 m apart at the stems; not measured otherwise.
