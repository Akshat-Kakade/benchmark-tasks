# Proof-pull a temporary anchor plate to its test load while holding the plate

A grey anchor plate on the table carries a yellow gauge carriage on a short rail; a spring pulls the carriage back and its
travel is the gauge reading. One hand must pull the yellow handle until the carriage is inside the green band (the proof
load), hold it there for one second and let it ease back, while the other hand holds the plate by its orange fin so the
reaction does not slide it. Temporary roof and fall-arrest anchors must be proof-loaded this way before a worker clips on
(O*NET 47-2181, 47-2211, 47-4099). The core ability is a target-load pull held against a braced base, not moving or
placing an object.

Honest about the gauge: the "force" is read from the travel of a spring-loaded carriage (80 N/m, a spring drive modelled
in the physics engine; about 4.4-5.6 N at 55-70 mm, the most the simulated arms hold reliably; a stiffer spring stalled
both arms). It is not a physical force sensor and no contact force is measured.

## How this differs from nearby tasks
#77 (preloaded spring scale) reads a scale and moves nothing against a reaction; #948 and #1110 (clutch / proof loads) set a
torque or calibration value; #1285 (ratchet strap taut) tightens until a strap is taut. Here the arm drives a spring gauge
to a target band, holds it for a second, releases it smoothly, and a second arm must resist the reaction through a
low-friction plate (the plate slides if the left arm does not hold it).

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo (`DEMO_VARIANT` = `no_left`, `no_right`, `no_hold`, `overload`, `settle`; `fist` is an unfinished alternative brace, not claimed).
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid solutions, blind-word scan.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1500 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1500
```
Runtime per trial: up to 1,500 steps (75 s of simulated time); the demo finishes in about 470 steps. No random seeds (`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Material | Mass | Pose |
|---|---|---|---|---|
| Plate | 120 (x) x 300 (y) x 15, grey | friction 0.1 | 0.8 kg | centre (0.30, -0.02) |
| Fin | 70 (x) x 20 (y) x 40, orange, on the plate's +y end | | | (0.30, 0.10) |
| Carriage | 50 (y) x 40 x 12 yellow slider between two rails along y, with a 40 x 20 x 40 yellow handle; spring 80 N/m, stroke 90 mm, damping 5 | friction 0.1 | 0.1 kg | handle at (0.30, -0.13) at rest, pulled toward +y |
| Scale | white strip beside the rail; green band 55-70 mm, red zone 80-90 mm | | | on the plate |
Rail, handle and fin lie on one line (x = 0.30), so the pull reaction gives no yaw torque on the plate (an off-line holder
let it yaw 16 degrees). The two hands work at least 16 cm apart (fin y +0.10, handle y -0.13 at rest, -0.07 in the band).

## Scoring (carriage travel and plate pose, read each step)
Zero if after step 40 the carriage ever passes 80 mm, or the plate moves > 15 mm, lifts > 10 mm or tilts > 20 degrees.
- 0.25: carriage pulled >= 20 mm.
- 0.5: carriage reached 55 mm (band entered).
- 0.75: carriage inside 55-70 mm continuously for 1.0 s with the plate moved <= 5 mm.
- 1.0: as 0.75, then the carriage is back at <= 10 mm within 3 s of leaving the band, plate moved <= 5 mm, carriage still.

## Real-world build and reset
A 200 x 140 mm steel plate on a low-friction pad, a spring-loaded slide with a printed scale and a handle, a fin on the
plate end for the holding hand. Reset: let the carriage return to rest, put the plate back on its marks.

## Shortcuts and what stops them
- Pulling with one arm lets the plate slide: plate-shift gate (zero).
- Slamming through the band: overload gate; the hold needs 1 s continuous inside the band.
- Flicking the handle: the spring returns it, no 1 s hold.
The scripted demo reads the true plate pose and carriage travel (closed loop); that proves feasibility, not perception.

## Measured in the simulator
| Run | Score |
|---|---|
| Settle (8 s hold, nothing touched) | 0.0, no gate, plate still |
| Demo (left pinches the fin, right pulls to 62 mm, holds 2.2 s, eases back) | 1.0 (success at step 466; plate moved <= 2.2 mm, peak carriage 61 mm) |
| `no_left` (plate unbraced) | 0.0 (plate slides, zero gate) |
| `no_right` (brace only) | 0.0 |
| `no_hold` (reaches the band, no hold) | 0.5 |
| `overload` (pull to 86 mm) | 0.0 |
| Dream-1 run | 0.0 (1,500 steps) |
Notes: after the on_reset placement the plate settles a few mm in its first steps, so scoring takes the plate baseline at
step 10. The second valid solution (left fist braced on the fin's +y face instead of a pinch) was tried once and not
made to work (the plate moved 12-22 mm before the fist took the load), so it is not claimed.
