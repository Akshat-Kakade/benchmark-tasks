# Tip the rocker so its counterweight can pass the stop

A grey inspection rocker: a 34 cm beam on a pivot at its middle, with a ballast block and a blue handle at its left
end and, near its right end, a round turntable that carries an orange counterweight. The beam rests with its right
end down on a red stop. A dark post stands in front of the turntable, exactly where the counterweight would swing
past it, so the weight cannot be turned to the yellow mark on the beam while the beam sits there. Tipping the beam
(left end down onto the green stop) lifts the turntable about 4 cm and lets the weight swing over the post. The weight
is then on the other side of the pivot's balance point: the beam is left-heavy and stays on the green stop.

The task: tip the beam, and while it is tipped turn the counterweight round to the yellow mark, then let go of both
parts and leave the rocker resting on the green stop. Real-world basis: rebalancing a pivoted inspection fixture
(a rocker, a balance beam, a tilt arm) by moving its eccentric counterweight past an obstacle without taking it off.

Why two hands: the beam falls back as soon as the tipping hand lets go (the weight is still on the wrong side of the
balance point), and the counterweight cannot be turned until the beam is tipped. So one part must be held in a
temporary position while the other is moved, and only then do both settle into the new stable state.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the rocker object, the monitor, `success()` and `score()` |
| `geometry.py` | Dimensions and the forward kinematics of the beam and the weight (pure NumPy) |
| `build_assets.py` | Writes `assets/rocker.usda` from `geometry.py`: masses, centre of mass and inertia are computed, so the balance is by design |
| `assets/rocker.usda` | The rocker as a USD articulation: fixed base, beam on a revolute pivot, counterweight on a revolute turntable |
| `scenekit.py` | USD bodies and articulations for yam_arena task files (as in lock-before-loading; adds a per-joint drive setting) |
| `demo.py` | The scripted full-score demo (closed loop on the measured beam and weight angles) and the ablation variants |
| `ikscan.py` | Offline reach check of every demo waypoint with the yam_arena kinematics |
| `tests/test_task.py` | Unit tests: geometry, every score level, the zero gate, the success state |

`partkit.py` is not used: the pivots are real revolute joints in a USD asset (see "How the pivots are modelled").

## How the pivots are modelled

Both pivots are real PhysX revolute joints in `assets/rocker.usda`, simulated by the engine; nothing is moved by code.
The task file only reads the two joint angles (beam about x, weight about the vertical) to score. Choices that matter:

- Linked bodies of an articulation do not collide with each other, so the beam's two stops are the joint limit of
  the beam pivot (+8 and -8 degrees), drawn as a red and a green block whose tops meet the beam's end at the limit.
  The post is part of the fixed base and the weight is not linked to it, so the weight really collides with the post.
- The turntable joint has friction (PhysX joint friction coefficient 0.3, damping 0.01). Without it the counterweight
  rolled by itself to the low side of the tilted turntable as soon as the beam was tipped, and one arm tipping was
  enough (measured, run 3). With it the weight stays where it is put, also on the tilted beam.
- The beam pivot has damping 0.08 N m s/rad and no friction. The ballast (0.20 kg) is chosen so the beam is balanced
  when the weight is at the front: weight turned out, the beam is right-heavy (0.093 N m); turned in, left-heavy.

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Tip the rocker beam over so its left end rests on the green stop; while it is tipped, turn the orange counterweight round to the yellow mark. Let go of both handles and leave the rocker resting on the green stop." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/rocker-dream
```

Run limit 2,400 steps (2 minutes at 20 Hz). A trial takes about 2 to 4 minutes of wall time on the L40S; the scripted
demo needs 640 steps (32 s of simulated time). There is no randomness: `object_jitter=0`, one fixed start state.

## Scripted demo and ablations

```bash
python demo.py task.py demo_out                       # full score; writes demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=tip_release python demo.py task.py out   # or --variant tip_release / no_tip / one_arm / left_only / settle
```

1. **Tip.** The left arm pinches the blue handle (round, 3.2 cm) at mid-height and presses the left end down, closed
   loop on the measured beam angle: the command runs ahead of the measured angle at 10 deg/s, never more than 16 deg
   ahead (the arm is soft), down to the stop angle plus 8 deg, and it keeps pressing.
2. **Turn the weight.** The right arm pinches the orange handle with its fingers open along the direction of travel
   and swings the weight round the turntable axis at 22 deg/s, closed loop on the measured weight angle (it waits
   when the weight lags by more than 15 deg), until the weight is 183 deg round. The wrist turns with the weight.
3. **Let go.** The right arm opens and lifts away, then the left arm does. The beam stays on the green stop.

## Scoring

The scorer runs after every control step (`SCORE_OVER = "end"`); latches are marked.

- **0** if the counterweight is forced through the post (its bottom more than 6 mm below the post top while it is
  within 55 to 125 degrees of the turntable). The physics normally prevents this; it is the zero gate.
- **0.25** (latched) once the beam has been tipped far enough for the weight to clear the post by 8 mm.
- **0.5** (latched) once, after that, the weight has been swung past 100 degrees.
- **0.75** while the weight is within 8 degrees of the yellow mark (180 deg) and the beam is within 5 degrees of the
  green-stop angle, but a gripper is still near the rocker or the beam is still moving.
- **1.0** (success, held 2 s): that pose, the beam at rest (below 8 deg/s), no gripper within 6 cm of a handle, the
  beam or the weight block (a gripper counts as holding a handle within 4.5 cm with its fingers less than 6.5 cm apart),
  and no violation.

Weight within 8 degrees of the mark and beam within 5 degrees of the green stop (the spec's 8 degrees was tightened for
the beam to 5 because the beam's whole travel is only 16 degrees).

## Shortcuts and what stops them

| Shortcut | Result |
|---|---|
| Doing nothing | 0.0 |
| Turn the weight without tipping (`no_tip`) | The post blocks the weight at about 63 degrees: 0.0 (measured) |
| Tip, then let go before turning the weight (`tip_release`) | The beam falls back to the red stop: 0.25 (measured) |
| Tip and hold, nobody turns the weight (`left_only`) | The turntable friction holds the weight at the start: 0.25 (measured) |
| One arm lifts the orange handle to tip the beam and turns the weight (`one_arm`) | One scripted attempt scored 0.25: the beam tipped, then dropped back to level as the weight turned, and the post stopped the weight at 63 degrees |
| Shove the weight through the post | Rigid contact; the zero gate flags any deeper than 6 mm |

The demo knows the answer; it is a feasibility proof, not a bound on difficulty.

## Measured in the simulator

| Run | Score | Notes |
|---|---|---|
| Scripted demo | 1.0 | 637 steps (32 s); beam -8.0 deg, weight 183.2 deg at the end |
| `tip_release` | 0.25 | beam back at +6.3 deg |
| `no_tip` | 0.0 | weight stopped at 63 deg by the post |
| `left_only` | 0.25 | |
| `one_arm` | 0.25 | right arm tipped the beam by the orange handle, the beam fell back to -0.3 deg, weight stopped at 63 deg |
| Dream-1, one run, 2,400 steps | 0.0 | the arms reached the handles but the rocker never reached the target pose |

Rest poses (settle run, nothing touching the rocker): beam at +8.0 deg on the red stop, weight at 0 deg; the weight
stays at 0 deg for the whole run.

## Real-world build and reset

Build: an aluminium beam (4 x 34 cm) on an 8 mm axle through a pedestal, with a steel ballast block at one end and a
turntable on a 6 mm shaft near the other, a friction washer on the turntable shaft (a light detent), an orange 150 g
block on a 6 cm arm for the counterweight, a 2 cm dowel post, and a red and a green rubber stop. Pick the ballast so the
beam is balanced with the weight at the front. Reset: turn the weight out toward the right end (against its pin), let
the beam settle on the red stop, check the post is upright.

## How this differs from nearby tasks

Nearest on the board: the ballast-on-gimbal task (#1128), #1376, #361 and the hinged-platform task (#46). Those move a
load or level a platform about one joint. Here two coupled joints must be moved in a fixed order with one held in a
temporary position: the weight can only pass the post while the beam is tipped, and the beam only stays tipped after
the weight has passed. The score is a measured end state of both joints.

## Honest limits

- The pivots are the engine's joints, the stops are joint limits, and the demo is closed loop on the measured angles.
- The turntable friction (0.3) is a tuned PhysX coefficient, not a measured real value.
- The one-arm ablation is one scripted attempt, not a proof that one arm cannot do it.
- Handles are 3.2 cm round stems (closed fingers grip 3 cm), tall enough that the fingers stay clear of the post.

Hands are 0.25 to 0.35 m apart throughout (handles at y = +0.15 and y = -0.20 to -0.10), far over the 30 mm
arm-arm clearance. IK reach (`ikscan.py`): every demo waypoint under 0.1 mm for both arms.
