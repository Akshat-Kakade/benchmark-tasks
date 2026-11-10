# Hold a spring-loaded flange joint open and slide a blind plate into the gap

## What it tests and why
Before a pipe is opened for maintenance, a solid blind (spectacle) plate is fitted between two flanges. The task needs
both hands at once: one hand pulls the lower flange down against its return spring and holds the gap open on feel
(the gap is only visible from the top camera as a thin dark slit), while the other lifts the plate off a tray, feeds it
in along +y with the stem out front, centres it on the green mark, and lets go; then the first hand eases the flange
back so the spring closes the joint on the plate (gap 10-14 mm). One core ability: holding a gap open against a spring
while blind-inserting a thin plate into it.

Honest modelling notes: the joint is rigid boxes. The lower flange is a slider with a spring drive (100 N/m) in the
physics engine. There is no thread, bolt or gasket; the plate is held by spring pressure and friction only. The plate is
fed along +y (stem out front) and the real-world spigot/pipe bore is dropped. Nothing is code-driven.

## How this differs from nearby tasks
- #283 (wedges under a pump base): a vertical shim under a load, nothing is held open; here a spring-held gap is
  spread and held by one hand while the other inserts.
- #1116 / #941 (seat a part into a slide or spacer): the opening is fixed; here the opening only exists while a hand
  holds it open against a return spring.
- #946 / #1286 (compress a spring): the spring is the goal; here it only provides the gap and the skill is blind insertion.
- #323 / #806 (lockout by lamps or hasp): logic/tag tasks; here a physical plate must sit in a physical gap.

## Files
`task.py` (scene, scoring), `partkit.py` (unchanged helper), `demo.py` (scripted two-arm demo, closed loop on measured
poses), `tests/` (31 offline tests), `task.json`.

## Run
    inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1000
    python demo.py task.py demo_out                 # scripted demo; DEMO_VARIANT=settle|no_left|no_right|no_release|offcentre
    PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q   # offline
Runtime per trial: 1,000 steps = 50 s of simulated time (demo finishes in about 610 steps, 30 s). Seeds: none, object_jitter=0.

## Scoring (score over the whole run, highest level kept)
- 0.25: gap >= 22 mm while the plate is held >= 12 mm above the tray.
- 0.5: plate inside the joint (its edge 60 mm past the flange edge, within the gap height) while the gap >= 14 mm.
- 0.75: plate centred within 6 mm, tilt <= 12 deg, level in the gap for 0.5 s with the flange still held >= 14 mm.
- 1.0: joint closed on the plate (gap 10-14 mm), centred, plate still for 1 s.
- Zero if the plate drops to the table, the joint base shifts > 10 mm, or the flange sits on its stop for > 1 s.
Rest state (nothing done) scores 0.0.

## Shortcuts and what stops them
Pushing the plate at a closed joint: the plate cannot enter and level 0.5 needs the gap open. Holding the flange open
forever: 1.0 needs the joint closed on the plate. Dropping the plate or yanking the joint: zero gate. Dragging the
joint: shift gate.

## Real-world build and reset
Two 130 mm square flange plates (3D-printed or aluminium) on a column, the lower one on a light spring (about 3 N at
30 mm), a yellow lug on the lower flange, a 12 mm plate with a stem and a knob, a white tray. Reset: let the flange
return to its stop and put the plate back on the tray.

## Measured in the simulator
| Run | Score |
|---|---|
| Rest, nothing moves (settle) | 0.0 |
| Scripted demo | 1.0 (606 steps) |
| no_left (plate pushed at a closed joint) | 0.25 |
| no_right (flange held open, no plate) | 0.0 |
| no_release (flange never allowed to close) | 0.75 |
| offcentre (plate centred 10 mm off) | 0.5 |
| Dream-1, one run, 1,000 steps | 0.0 |
The no_left (0.25) and no_right (0.0) values were not predicted exactly and are reported as found, without a full diagnosis.
The demo's worst IK error was 0.1 mm on both arms.

