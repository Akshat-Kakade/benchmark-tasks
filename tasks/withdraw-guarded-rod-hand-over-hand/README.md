# Withdraw a guarded sample rod uphill by exchanging grips hand over hand

A grey rod (540 x 20 x 20 mm, 0.30 kg) runs through three fixed box shrouds of a guide that rises 7.5 degrees toward +y. It
is free to slide (no spring, damping 8 N.s/m), so it rolls back down the guide at about 48 mm/s as soon as nobody holds
it. Between the shrouds the rod is bare in two openings, marked blue (right) and orange (left). The rod must be pulled
172 mm uphill until its red lower tip reaches the green band by the blue opening, and it has to stay there for a second.
A hand can only follow the rod while its fingers stay clear of the next shroud: about 80-85 mm (measured).
So the arms must pull by turns, hand over hand: the second hand takes the rod while the first still holds it
(stationary overlap), the first lets go and re-grips further down, and the rod never runs back. Real basis: withdrawing a
core sample, probe rod or tubing from a guide by hand over hand (O*NET 47-5013, 47-2061). The core ability is
exchanging load-bearing grips on one back-driving rod without losing the hold, not moving or placing an object.

## How this differs from nearby tasks
#558 (terminal handover) passes a free part from one hand to the other and does not have to keep a load from
running back; #538 (ratchet-held load) lowers a load against a pawl that holds it, here the only thing holding the rod is
a hand; #703 (relay-withdraw, title only) and #1101 (rod a duct, withdraw the rod) are about where a rod goes or
how far it is withdrawn, not about regripping a back-driving rod. Here the score is the rod travel held for one second and
the hands must overlap on the rod twice.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: unchanged helper.
- `demo.py`: closed-loop scripted demo on the measured rod travel. `DEMO_VARIANT` = `right_only`, `left_only`, `stop1`,
  `stop2`, `release`, `settle` (explicit hold, nothing touched), `probe`.
- `tests/`: scoring boundaries, every level reachable, zero gate, two valid solutions (RLR and LRL), blind-word scan.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "<INSTRUCTION in task.py>" --scorer task_score --max-steps 1500 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1500
```
Runtime per trial: up to 1,500 steps (75 s of simulated time); the demo succeeds at step 767. No random seeds
(`object_jitter=0`). The rod spawns at its upper stop and rolls down during the first 4 s (scoring starts once it was seen at
10 mm or less), so the first frames show it high on the guide.

## Scene (metres, station frame, table top 0.7475)
| Object | Size (mm) | Pose |
|---|---|---|
| Guide axis | three grey shrouds, 48 mm square, 32 mm bore, length 110 / 230 / 101 (axial u 0-110, 260-490, 640-741) | from (0.300, -0.367, table + 0.028) rising 7.5 deg toward +y, on three pillars, 8 kg |
| Rod | 540 x 20 x 20, grey, red 12 mm lower tip | slider of the guide, travel 185 mm, spring 0, damping 8, friction 0.8; lower tip at u = 5 mm at rest |
| Blue opening | blue strips beside the rod on the table, u 110-260 | right arm side |
| Orange opening | orange strips, u 490-640 | left arm side |
| Green band | green strips, u 177-200 (the tip is at u = 5 + s) | next to the blue opening |
Changes against the first sketch: the rod is the guide's own slider and the guide is turned in `on_reset` (sliders run along the
body's z), the openings are marked with strips on the table, and no grip collar was needed: the 20 mm rod is pinched
directly with fingers open along x. The guide settles about 11 mm down the slope in its first steps, so poses are
read from the measured guide pose (never nominal constants).

## Scoring (rod travel s in mm, read each step)
Each level needs s inside its range for 1 s (20 steps) with max - min <= 8 mm over that second; the best level reached
counts. Scoring starts once the rod has been seen at <= 10 mm.
- 0.25: s >= 30. 0.5: s >= 90. 0.75: s >= 160. 1.0: 172 <= s <= 185.5 (red tip at the green band).
- Zero if after step 40 the guide moves > 12 mm (from its settled pose at step 10) or tilts > 6 degrees.
A rod that is flicked up and rolls back never stays still for a second and scores nothing.

## Real-world build and reset
A short length of rail or tube with three guide sleeves tilted 7.5 degrees, a 0.3 kg rod with a 10 mm marker at its
tip, tape strips for the openings and the target band. Reset: let the rod run back to its stop.

## Shortcuts and what stops them
- One hand alone: the hand cannot follow the rod through the shroud, so it stalls near 80-90 mm (0.25 measured).
- Release at the target: the rod runs back at 48 mm/s, no 1 s hold.
- Pushing the guide or the rod through: guide shift/tilt gate; the rod's upper stop is 185 mm.
- Open hands: the rod rolls back to 0 (0.0).
The scripted demo reads the true rod travel and guide pose; that proves feasibility, not perception.

## Measured in the simulator
| Run | Score |
|---|---|
| Settle (explicit hold, hands open, nothing touched) = "open both grippers" | 0.0, no gate; rod rolled from the top stop to 0 by step 80 |
| Demo (right 0-58 mm, left 58-118 mm, right 118-178 mm, closed loop) | 1.0 (success at step 767; guide moved <= 0.4 mm after settling; IK error <= 0.1 mm) |
| `right_only` (one hand, pulls until the shroud stops it) | 0.25 (rod held at 76-83 mm) |
| `left_only` | 0.25 (rod held at 85 mm) |
| `stop1` (one stroke, hold at 58 mm) | 0.25 |
| `stop2` (two strokes, hold at 118 mm) | 0.5 |
| `release` (one stroke, then the hand opens) | 0.25 (the 1 s hold at 58 mm counts, then the rod rolls back) |
| Dream-1 run (1,500 steps) | 0.0 (rod never pulled, hit the step limit) |
Notes: no one-arm run reached 0.5; the prediction (<= 0.5) holds. The first one-arm versions of the demo wound their command
up against the shroud and dragged the guide (zero gate); the command is now capped (+20 mm over the target).
Both one-arm limits are measured, not designed: a finger on the next shroud stops the hand, so a different grip height
might reach a few mm more but not 172.
