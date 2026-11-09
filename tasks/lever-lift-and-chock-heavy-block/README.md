# Lever up a heavy block and chock it

Tests lever-and-fulcrum tool use: the right arm presses down the long end of a pry bar so a 2 kg block rises, the left arm
slides a U-shaped chock under the raised block, and the bar is then released so the block rests on the chock, not the bar.
A load left on the bar alone is a crush hazard (millwrights, O*NET 49-9044; automotive technicians, 49-3023).

## How this differs from nearby tasks
PR #830 (jack plus stands) and #928 (chock a rolling cradle) overlap in theme (raise or secure a heavy load). Here the distinct
capability is operating a passive lever with mechanical advantage: the right arm presses the long end of a bar (about 0.2 m arm
versus 0.07 m load arm) so the block rises, while the left arm inserts the chock and the bar is then released. It does not
involve a jack mechanism or a rolling object; it is related to those tasks, not unrelated.

## Files
task.py (scene, scoring), partkit.py (shared helper, unchanged), demo.py (scripted closed-loop demo), tests/ (unit tests), task.json.

## Run
    inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 2000
    python demo.py task.py demo_out [--max-steps 2000] [--no-record]     # DEMO_VARIANT= | no_chock | no_lever | palm
    PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q
Runtime per trial: 2,000 steps (100 s at 20 Hz). Seeds: object_jitter=0, no randomness.

## Scoring (read from object poses at the end; thresholds are ruler-measurable)
- 0.25 block raised 6 mm while the bar is pressed; 0.5 raised 9 mm and chock slid 80 mm; 0.75 chock fully under the raised block;
- 1.0 bar back within 2.5 deg of rest, block resting on the chock (2-7 mm above its start), tilt <= 5 deg, still 1 s.
- Zero gate: block tilt > 30 deg, block or chock carried > 20 cm, block raised > 6 mm with the bar at rest and no chock under it.

## Shortcuts
Lifting the block by hand trips the gate; pushing the chock in without lifting stalls at the block edge (chock 4 mm taller than
the gap); wedging with the bar still pressed fails the released-bar check.

## Real world
A 25 cm steel bar on a 22 mm fulcrum with two guide posts, 2 kg block, 3D-printed U chock. Reset: put the block on the bar's
platform, the chock on the left.

## Measured in the simulator
| Run | Score |
|---|---|
| Rest check (first 90 steps of the demo, no motion) | 0.0, no gate, bar 0.0 deg, lift 0.0 mm |
| Demo (514 steps, success true) | 1.0 |
| no_chock (press, no chock) | 0.25 |
| no_lever (chock only; the right arm never moves) | 0.0 |
| palm (open fingers press the bar) | 1.0 (first run with the 1.5 deg bar tolerance reached 0.75 because a block resting on the chock keeps the bar about 1.6 deg up; tolerance widened to 2.5 deg, rerun 1.0) |
| Dream-1 (one run, 2,000 steps, made with the 1.5 deg tolerance) | 0.0 |
