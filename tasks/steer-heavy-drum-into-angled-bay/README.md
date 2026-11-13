# Turn a small steel drum on its side and roll it into an angled bay

A small (4 L, 1.2 kg) steel drum lies on its side in the aisle, its axis across the table (along y), free to roll. Ahead of it, turned
35 degrees to the left of straight ahead, is a bay made of two yellow rails and a red back bar. The task is to get the
drum into the bay and leave it resting against the back bar, hands clear. A rolling drum cannot be steered by pushing
it straight on: its two ends roll together, so a push from behind runs it straight on (or it just skids sideways
into a rail). To change its heading the two ends have to move in opposite senses at once (one end driven forward, the
other held or driven back), which takes both hands; then it rolls into the bay on its own heading.

The task is `task.py` for the `yam_arena` embodiment of yam-isaac-lab-arena, on the `station` scene, with
`partkit.py` (unchanged, builds the drum from a shell and two chime rings) next to it.

## Instruction given to the policy

> Get the steel drum into the angled bay between the two yellow rails, resting against the red back bar, and take your hands off it.

It states the goal and the end condition. It does not say how to turn the drum.

## Context: who does this and why

Warehouse and stock workers (laborers and freight, stock and material movers, hand, O*NET 53-7062.00; stockers and order
fillers, 53-7065.00) regularly turn barrels, drums and kegs on their sides and roll them into marked bays or rack positions
at the end of an aisle. The drum is rolled, not carried; it is turned by pushing its two ends in opposite senses, then rolled
straight into the bay until it meets the back stop. A drum that goes in crooked wedges against the bay posts and must be
handled again, and one let go on a slope runs away. We do not claim a worker count for this one task; it is a routine part
of this large occupation's handling work, not specialist maintenance. This benchmark uses a small drum (see below).

## How this differs from nearby tasks

- **`turn-parcel-into-bay` (open PR #424):** a flat 42 cm cardboard box is turned IN PLACE between two neighbour
  boxes and pushed lengthwise into a taped bay; it only slides (friction 1.2), and the score is gated on not touching
  the neighbours. Here the object is a free ROLLING cylinder: it keeps rolling when let go, a one-end push does not
  turn it, a turn needs a skid couple (one chime driven forward, the other pushed back) and the end state must be
  rolled-in against a back bar between rails, not slid.
- **#1158 (drum up a ramp, older):** moves a drum up an incline in a straight line; no heading change, no bay.
- **#1694 (puff pastry rolled on a pin):** the hands roll a pin over a sheet; the rolling object is the tool, not the load.
- **TK-HYNFZE (3 kg carton up onto a platform):** lifting and tipping over a step, no steering.

## Run it

From this folder, on the VM set up by the yam-isaac-lab-arena launchable (`source /data/IsaacLab-Arena/yam.env` first):

```bash
I="Get the steel drum into the angled bay between the two yellow rails, resting against the red back bar, and take your hands off it."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --max-steps 1000 --store-frames --log-dir ~/yam-eval/drum-dream

# Scripted demo (open loop), score and videos in demo_out/
python demo.py task.py demo_out --max-steps 1000
DEMO_VARIANT=right_only python demo.py task.py out_right --no-record      # ablations, see below
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q                       # scoring unit tests, no GPU
```

- **Run limit:** 1,000 steps (50 s at 20 Hz). The demo ends at about 800 steps.
- **Runtime per trial:** about 3-8 minutes of wall time on the shared L40S once Isaac Sim has started (depends on load).
- **Seeds:** none. Run with `-E object_jitter=0`; the start poses are fixed.

## What is in the scene (station frame: x forward, y left, z up, metres; table top z = 0.7475)

| Part | Material | Size | Mass | Start pose |
|---|---|---|---|---|
| Drum: steel shell (loose) | steel, friction 0.5 | 0.140 diameter x 0.270 long | 1.2 kg for the whole drum | axis along y, centre (0.231, -0.018), axle 0.075 above the table |
| Two chime rings on its ends | steel (dark grey), simulated as solid discs | 0.150 diameter x 0.015 wide each | no separate mass: the 1.2 kg is set once for the whole body | at the shell's ends, so the drum is 0.300 long and rolls on the two rings |
| Left and right rail (fixed) | yellow painted timber | 0.130 x 0.020 x 0.050 | fixed | along the bay axis (35 deg), centres 0.180 either side of it, from 0.030 before to 0.100 past the bay centre |
| Back bar (fixed) | red painted timber | 0.020 x 0.340 x 0.050 | fixed | across the bay, near face 0.076 past the bay centre |
| Heading band (fixed) | green tape | 0.120 x 0.020 x 0.001 | fixed | on the bay axis, visual only |

Bay centre (where the resting drum's centre ends up): (0.370, 0.080); the bay axis points 35 deg left of +x. The
rails' inner faces are 0.340 apart, so a drum 0.300 long has 20 mm to spare at each end. Mass check: a 0.8 mm steel shell of that size is 0.163 m2 x 0.8 mm x 7,850 kg/m3 = 1.0 kg. Real chime rings are thin hoops
(3 mm wall, about 0.1-0.17 kg each), giving a real empty drum of about 1.2-1.3 kg. The simulator models the rings as SOLID
15 cm x 1.5 cm discs; solid steel discs would weigh about 2 kg each, so their density is NOT steel: the body's total mass is
fixed at 1.2 kg and PhysX spreads it over the colliders (the drum behaves like a light solid cylinder).

Real-world build: a 4 L steel drum (140 mm x 270 mm shell, 0.8 mm sheet, two rolled 15 mm chime rings, empty, about 1.2 kg);
two 20 x 50 mm timber rails and a 20 x 50 mm timber bar screwed to a painted floor board, the 35 degree lane marked with tape. Reset: stand the drum on its side on the start mark with its
axis along the table's width.

## What is simplified / not modelled

- **Scale and mass:** the drum is a small 4 L steel drum (140 x 300 mm, 1.2 kg empty), not a heavy one: a real 200 L drum is 0.57 x 0.88 m and about 20 kg empty (200 kg full). It is made small so that both its ends are within the
  arms' reach (0.285 m between the hands). The arms' push forces (a few N) are those of a 1.2 kg drum, not of a 20 kg one.
- **No contents:** empty (no sloshing or moving mass). Its mass is spread like a solid cylinder in the simulator,
  not like a hollow shell.
- **Rolling resistance is added in code.** PhysX rolls a rigid cylinder with no resistance at all, and a hair of tilt
  in the table walked the free drum 0.25 m in 10 s. Once per control step the drum's speed is reduced by
  c_rr x g x dt with c_rr = 0.01 (steel on concrete), stopping it below that. It touches only the drum.
- **Chimes:** solid 15 mm discs (real ones are thin hoops); no rolled hoop profile, no bung, no chime lip detail. The shell is 5 mm smaller than the rings.
- **Floor:** the table top stands in for the concrete floor, painted line and bay boards are plain boxes.
- **Hands:** the closed gripper pushes the chime's rim; no grasping of the drum.

## Scoring (`score()`, `success()` in `task.py`; read from the simulator, never by the demo)

Everything is judged on the END state, except the zero gates. The drum's pose and its roll angle are read each step.

- **Zero gate (stays 0 for the rest of the trial):** the drum centre lifted more than 8 mm for more than 10 steps,
  its axis pitched more than 8 deg, the drum off the floor/table region, or a point on a drum end face inside a rail
  before the back bar has been touched (the drum hit a rail).
- **0.25:** the centre has moved at least 0.06 m from its start.
- **0.5:** moved at least 0.12 m and turned at least 15 deg toward the bay (yaw start +15 deg).
- **0.75:** centre inside the bay (between the rails and the bar) and yaw within 15 deg of 35 deg.
- **1.0:** centre within 20 mm of the bay centre, yaw 35 +/- 8 deg, the back bar touched, the drum rolled there (its
  roll angle about its own axis at least 70 % of travel / radius, so skidding it in cannot score 1.0), both grippers open
  (finger-tip gap <= 58 mm) and at least 120 mm from the drum's ends, and the drum still (1.5 mm and 0.5 deg over the
  window); all of it true for 20 consecutive steps (1 s). Score 1.0 is never returned earlier.

The scorer reads the drum's pose, the arms' finger bodies and nothing else. Either direction of approach and either arm
order reach 1.0 if the end state is right (unit tests `test_two_valid_solutions`, `test_yaw_tolerance_8`).

## The scripted demo (`demo.py`)

OPEN LOOP: it reads only the layout constants of `task.py` and each arm's own joint angles (a small integrator trims
each hand onto its planned point; the arms hold 15 mm steady errors otherwise). It does not read the drum's pose.
Both hands close and go to the drum's two chimes: the right hand behind its chime, the left in front of its chime. Over 8 s
the right hand pushes its chime forward and the left pushes its chime back: a couple that turns the drum 35 deg in place
(the planned arcs of both hand points follow the chime's rim). The left hand opens, lifts over the drum and comes down
behind its chime; both hands then push forward 0.17 m along the 35 deg heading at the same speed over 8 s, the drum rolls
into the bay against the back bar, the hands open, back off and lift. Observations used: the layout constants and
`joint_pos` of both arms.

## Measured in the simulator

| Run | Score | What it shows |
|---|---|---|
| Settle (both hands held at rest, 220 steps) | 0.0, no gate | with rolling resistance on, the drum stays within 1 mm of its start (without it, it walked 0.25 m in 10 s) |
| Demo (full, 800 steps) | **1.0** (twice: dev run and recorded run) | final: centre 14 mm from the bay centre, yaw 35.0 deg, bar touched, rolled 1.81 rad for 0.176 m of travel (0.135 m of roll against 0.123 m needed), hands open and clear |
| left_only (left hand does its part alone) | 0.0 | the drum is pushed back and yawed to -44 deg, a hand pitches it past 8 deg (gate) |
| right_only | 0.5 | it yaws only to 16 deg and arrives 41 mm off the bay centre (the left end is free) |
| alt (right hand alone, walks between the two chimes) | 0.25 | no turn (yaw 2 deg); the right arm cannot even reach the left chime (IK clamped 122 mm), so the walk is partly cut by reach |
| equal (no couple, both hands push straight) | 0.25 | runs straight, yaw 5 deg, no turn |
| half_turn (couple turns 17.5 deg, then straight push) | 0.25 (peak 0.5) | ends at yaw 12 deg, outside the 15 deg band of the bay heading |
| no_release (hands stay on the chimes) | 0.75 | drum in the bay, but the 1.0 tier needs the grippers open and clear |
| lifted_carry | not run | the lift gate is covered by the unit tests only (`test_lift_gate`); a one-hand lift of the drum is not reachable here |
| Dream-1 (one run, 1,000 steps) | 0.0 | its right hand went to the drum's right end and its left hand to the left side, but the drum was neither turned nor moved into the bay |

Worst IK position error of the demo: left hand 20 mm (during the lift over the drum, at height), right hand 0.1 mm.
Arm-to-arm clearance in the plan is at least 0.285 m between the two TCPs.

## Shortcuts and what stops them

- Pushing the drum straight (equal speeds): runs straight on past the bay mouth; tiers stop at 0.25 (no turn).
- One hand alone, or one hand walking between the two ends: cannot hold one end while the other moves (see table).
- Skidding or lifting the drum in: the roll-consistency requirement and the lift/pitch gates.
- Hitting a rail first: zero gate; the bar must be touched before a rail.
- Leaving the hands on it: the 1.0 tier needs both grippers open and clear.
