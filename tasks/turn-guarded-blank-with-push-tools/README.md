# Turn a guarded blank with two push tools

A flat yellow blank (120 x 120 x 24 mm, white strip on its front edge) sits in a low grey guard (four posts plus a rail
frame, 250 x 250 mm clear, rails 92 mm above the table). The opening under the rails admits thin tools but not a gripper.
A blue and an orange push tool lie either side of it: a thin bar with a 40 mm wide flat paddle at one end and a tall handle
stem (with a small tail that balances the bar) at the other, outside the guard. Each hand pinches a stem; the paddles press
opposite faces of the blank near opposite corners, so the pair forms a turning couple and the blank turns clockwise until its
white edge points to the green mark (-45 deg) with its centre still between the four white ticks. Then both paddles back off.
Core ability: rotating an inaccessible free rigid part in place with two opposed tool contacts, closed loop on its measured
pose, while keeping it registered. Real-world use: reorienting a workpiece in a guarded machine pocket without putting a
hand in (O*NET 51-4041.00, positioning material for machine work).

## How this differs from nearby tasks
- #432 turns a wide board a quarter turn by handling it directly with the grippers; here no gripper may touch the blank or
  enter the guard, the blank is only reached through two tool contacts, and the scored quantity is a turn with a tight
  centre-drift band and tool withdrawal.
- #19 (merged) displaces a wire with one insulated stick: one contact, no turn, no registration.
- #474 lifts a cup with two sticks: the load leaves the table and there is no in-plane turn.

## Changes from the supplied spec (honest list)
Tip height fixed (paddle spans 748-768 mm, overlapping the blank's thickness, bottoms 0.5 mm off the table at rest, 6 mm
when held; the supplied 780 mm was above the blank). Paddles 40 mm wide, 20 mm tall, flat faces. Blank 24 mm thick, friction
0.2 (a single push slides it). Guard made explicit: posts at the corners, a rail frame (open in the middle so the top camera
sees the blank) 92 mm above the table, clear area 250 x 250 mm; handles stand outside at |y| about 0.205 m. Tools 0.08 kg
(not 0.10) with a stem and tail. Tool paddles start 20 mm from the faces. White dot replaced by four white ticks (a dot
would be hidden by the blank). Zero gate also trips if a handle stem is inside the guard footprint. Yaw bands, drift 15 mm
for 1.0, 30 mm for 0.75, 45 mm for 0.5 as in the notes.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `settle`, `one_tool`, `open_loop`, `no_withdraw`, `yaw30`, `yaw40`,
  `yaw50`, `yaw60`). `ikscan.py`: IK reach check along the whole motion.
- `tests/`: scoring boundaries, every level reachable, zero gates, two valid solutions, tool balance, blind-word scan.
- `task.json`, `media/`.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1800 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1800
```
Runtime per trial: run limit 1800 steps (90 s simulated); the demo succeeds at step 520 (26 s). No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
Blank centre (0.30, 0), 0.15 kg, friction 0.2. Blue tool paddle centre (0.211, +0.035), orange (0.389, -0.035), 0.08 kg
each, friction 0.5, handle stems at |y| 0.205 (about 0.25 m and 0.38 m from the arm bases). Guard posts at (0.30 +- 0.125,
+-0.125), rail frame underside at z 0.84. Marks: four white ticks 0.095 from the centre on the
axes, green angle line at -45 deg, 0.10-0.12 from the centre.

## Scoring (measured at the end, held 1 s)
Zero gate: blank tilt > 8 deg, a blank corner within the guard posts/rails (> 125 mm from the centre), centre > 80 mm off,
or a tool's handle stem inside the guard footprint (a hand in the guard). Otherwise: 0.25 turned >= 12 deg clockwise; 0.5 yaw
<= -25 deg and drift <= 45 mm; 0.75 yaw -33 to -57 deg and drift <= 30 mm; 1.0 yaw -37 to -53 deg, drift <= 15 mm, both
paddle centres > 78 mm from the blank centre along its axes (withdrawn), tilt <= 8 deg.

## Shortcuts and what stops them
- Gripping the blank: blocked by the rails/posts and the 94 mm opening vs the 120 mm blank; a hand in the guard trips the gate.
- One tool alone: the blank (friction 0.2) slides and turns: expect drift over 15 mm (see measured table).
- Moving the blank to put its edge under the mark: fails the drift band.
- Pushing past -53 deg: stays at 0.75 or 0.5.

## Real-world build and reset
Cut a 120 mm square blank from 24 mm board with a white strip on one edge, glaze the table patch to friction about 0.2
(laminate), two 3D-printed or wooden push tools (paddle 40 x 18 x 20 mm, 160 mm bar, 30 mm handle post), a grey 250 mm guard
frame on four posts. Reset: blank at the centre, edge strip at 0 deg, tools 20 mm from the faces.

## Measured in the simulator
All runs on the final `task.py` and `demo.py` (the first demo run, before the yaw trim was added to `demo.py`, scored 0.75: the blank overshot to -56 deg).
| Run | Result |
|---|---|
| Settle (hold, no motion) | 0.0, no gate tripped |
| Demo (closed loop on measured blank and tool poses) | 1.0 at step 520: yaw -46.3 deg, drift 2.7 mm, tilt 0.0, paddles 99 mm from the blank centre |
| Ablation `one_tool` (only the blue tool pushes, orange tool left on the table) | 0.0: the blank turned only 5 deg, drift 9 mm |
| Ablation `yaw30` (stop at -30 deg) | 0.5 |
| Ablation `no_withdraw` (paddles stay on the blank) | 0.75 (yaw -48.1, drift 1.7 mm, paddle centres 73 mm out) |
| Variant `open_loop` (tool-pose and centre trims off, blank-yaw trim off) | 1.0 at step 504: the two paddles hold the blank between them, so a plain scripted path also works |
| Dream-1 run (1800 steps) | 0.0 |

Notes: the one-tool prediction (0.25-0.75) was too optimistic; with the second tool lying passive on the far side the single push did not turn the blank past 12 deg, so no single-tool route to 1.0 was seen. The `open_loop` variant shows the task does not need hidden precision from feedback; the difficulty is holding both paddles on the faces with a 15 mm drift band. Not run: `yaw40`, `yaw50`, `yaw60`, `overlap`-type variants.
