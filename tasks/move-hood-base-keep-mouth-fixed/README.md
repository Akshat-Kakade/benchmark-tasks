# Move an extraction hood's mounting foot while its mouth stays over the work point

An orange mounting foot carries a blue telescoping arm that ends in a blue hood mouth, which hangs 4.7 mm above a white work
mark. A weak spring pulls the arm back toward the foot. One hand must hold the mouth over the work mark while the other hand
slides the foot about 7 cm toward the robot to a green line, so the arm extends and the mouth does not move. Moving the whole
assembly, or letting go of either part, uncovers the work point (O*NET 49-9041.00, maintenance of industrial equipment). The core
ability is holding one end of a telescoping assembly at a world pose while the other end is moved by a measured distance.

The scene measures mouth and foot positions only; no duct, airflow or force sensing is simulated.

## How this differs from nearby tasks
#933 and #187 extend a handle and #641 extends a rack foot: in each the telescoping part is the thing being moved. #1106 hangs a
telescoping rail from fixed hooks. #419 positions both ends of a gantry, so both ends move. Here one end must be held still in the
world (a measured drift limit of 12 mm) while the other end is translated 65-78 mm and the arm extends by the difference; a
policy that moves the whole assembly, or that releases the mouth, scores zero because the mouth follows the foot.

## Changes from the supplied spec (honest list)
- The assembly rests on the table (foot friction 0.1, mouth 4.7 mm above it) instead of floating at z 0.80, so the zero gate on mouth height is replaced by lift / tilt gates.
- Each part has a 20 mm wide, 80 mm tall grip post on a side tab (foot at y +0.11, mouth at y -0.11) so the gripper body clears the arm bar; the right hand is no longer near its reach limit (0.45 m). The scene is not shifted in x.
- The "green mark" is a thin green line just behind the foot's rear edge at the target (65-78 mm of travel), not a 0.13 x 0.10 box under the foot (the foot would slide over it).
- Slider along world +x via a -90 degree turn of the foot's body frame (partkit sliders run along local z), set by write_pose in on_reset; the foot settles 0.4x of its written x offset short, so it is written 44.2 mm forward to rest at x 0.270.
- Spring 40 N/m (modelled in the physics engine, about 2.8 N at full extension), hard stop 80 mm. Thresholds are as in the spec (d <= 12 mm, y error <= 12 mm); the 1.0 level also needs the state held for 1 s.
- The right hand's hover height was lowered to table top + 0.12 m so every waypoint is reachable (IK checked); the spring is a weak modelled one, not a real mechanism.
- The white mark is almost hidden under the mouth in the top camera (a 65 mm disc under a 60 x 75 mm mouth); its position is the mouth's start position.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `settle`, `no_mouth`, `no_foot`, `whole`, `stop50`, `probe`).
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid solutions, blind-word scan.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1500 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1500
```
Runtime per trial: up to 1,500 steps (75 s of simulated time); the demo succeeds at about step 412. No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Mass | Pose |
|---|---|---|---|
| Foot, orange | 120 (x) x 90 (y) x 35, friction 0.1; side tab 50 x 90 x 15; grip post 40 x 20 x 80 | 0.18 kg | centre (0.270, 0.0); post at y +0.11 |
| Arm, blue (slider along +x, 40 N/m, damping 5, stroke 80 mm) | bar 150 x 55 x 30 on top of the foot; mouth 60 x 75 x 61.5; side tab; grip post 40 x 20 x 80 | 0.12 kg | mouth centre (0.400, 0.0); post at y -0.11 |
| White work mark | disc d 65 mm, 2 mm thick | 3 kg | (0.400, 0.0) |
| Green line | 8 x 120 x 0.8 mm | 3 kg | x 0.135 (just behind the foot's rear edge at 71.5 mm of travel) |

## Scoring (read at the end; foot and mouth poses)
D = foot travel toward -x; d = mouth-centre distance from its start (baseline at step 15); extension = slider reading.
Zero if after step 40 the foot tilts > 10 degrees, lifts > 20 mm, or moves > 120 mm.
- 0.25: D >= 20 mm and d <= 45 mm.
- 0.5: D >= 40 mm and d <= 30 mm.
- 0.75: D >= 55 mm, d <= 20 mm, foot y error <= 20 mm.
- 1.0: D 65-78 mm, d <= 12 mm, y error <= 12 mm, extension >= 60 mm, tilt and yaw <= 8 degrees, all held for 1 s.

## Real-world build and reset
A steel foot plate on a low-friction pad, a telescoping tube on a slide with a light return spring and a hard stop, a hood mouth
at its end with a rigid grip post on each part; a printed mark under the mouth and a line on the table. Reset: slide the arm
back to rest and put the foot on its start marks.

## Shortcuts and what stops them
- Moving the whole assembly: the mouth follows the foot (d = D), no extension, 0.
- Extending the arm without moving the foot, or holding the mouth and not moving the foot: D stays near 0.
- Diagonal foot moves: y error limit. Releasing the mouth: the spring pulls it along with the foot.
The scripted demo reads true poses (closed loop); that proves feasibility, not perception.

## Measured in the simulator (task.py as submitted)
| Run | Score |
|---|---|
| Settle (8 s hold, nothing touched) | 0.0, no gate |
| Demo (right pinches mouth post, left pinches foot post and slides it at 20 mm/s) | 1.0 (success at step 412; D 71 mm, mouth drift <= 3 mm, extension 70 mm) |
| `no_mouth` (right hand never grips; left slides the foot) | 0.0 (mouth follows the foot, d 73 mm, extension about 1 mm) |
| `no_foot` (mouth held, foot never moves) | 0.0 |
| `whole` (both hands slide the assembly together) | 0.0 (D 70 mm, d 71 mm, no extension) |
| `stop50` (foot stops after 50 mm) | 0.5 |
| Dream-1 run | 0.0 (1,500 steps; arms hover near the posts, nothing gripped) |
Notes: the first demo run scored 1.0; the settle/pop tuning (POP_DX) was done in four settle runs before it. The predicted
`no_mouth` score was 0-0.5; it measured 0.0.
