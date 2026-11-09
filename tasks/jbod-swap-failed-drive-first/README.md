# Replace two flagged drives in a top-loading drive drawer without losing the array

A grey top-loading drive drawer holds four drives that form one RAID 5 array. The lamp in front of slot 2 is red (the
drive has failed) and the lamp in front of slot 3 is amber (that drive still works but is about to fail). The robot must
replace both with blue spares from a white tray on the left and put the removed drives in a black returns bin on the
right. Two rules come from the array, not from the geometry:

1. **Failed drive first.** RAID 5 survives one missing member. With slot 2 already failed, pulling the amber drive (or
   either healthy drive) leaves only two working members: the data is lost.
2. **Wait for the rebuild.** A spare seated in an empty slot rebuilds for 8 s (white lamp blinking at 1 Hz) and only
   then counts as a working member (green). Pulling the amber drive before the green lamp loses the data.

Hot-swapping drives in top-loading enclosures (60-bay JBODs in AI data centres) is one of the most frequent remote-hands
jobs, and "pulled a second drive during a rebuild" is a classic data-loss incident (O*NET 15-1231 Computer Network
Support Specialists; 49-2011 Computer, Automated Teller, and Office Machine Repairers). The task tests whether a robot
sequences physical work around an asynchronous machine process instead of parallelising greedily.

**Why two arms.** The spares tray is at y = +0.36 to +0.46 m and the returns bin at y = -0.42 m: each is within reach of
only one arm (the right arm cannot reach the tray, the left cannot reach the bin), while both flagged slots (y = -0.05
and +0.05) are reachable by both. The left arm brings spares in and the right arm carries drives out. The tempting
bimanual shortcut, both arms swapping both slots at once, is exactly what destroys the array.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the hidden array state machine, lamps, `success()` and `score()` |
| `partkit.py` | Rigid parts built from several boxes (the two-colour drives) and the step/reset hooks that drive the lamps |
| `demo.py` | The scripted full-score demo and its ablations (`DEMO_VARIANT`) |
| `tests/test_task.py` | Offline tests: every score level, zero gates, two valid solutions, ablation logic, blind words |

## Run a trial

In the Isaac Lab-Arena environment (`source /data/IsaacLab-Arena/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "In the grey top-loading drive drawer, the drive in the red-lit slot has failed and the drive in the amber-lit slot is about to fail: replace both with the blue spares from the white tray on the left, and put the removed drives in the black returns bin on the right. The four drives form one RAID 5 array, so never have two of its slots without a working drive: wait for a replaced slot's white rebuilding lamp to turn green before pulling another drive." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/jbod-dream
```

Run limit: 2,400 steps (2 minutes at 20 Hz).

## Scripted demo

```bash
python demo.py task.py demo_out                          # full score
DEMO_VARIANT=parallel     python demo.py task.py abl_par  # left arm pulls the amber drive while the right carries the red one out
DEMO_VARIANT=simultaneous python demo.py task.py abl_sim # both arms pull slots 2 and 3 at the same moment
DEMO_VARIANT=amber_first python demo.py task.py abl_amb  # amber drive pulled first
DEMO_VARIANT=no_wait     python demo.py task.py abl_nw   # the left arm, after seating the first spare, pulls the amber drive next door
DEMO_VARIANT=failed_only python demo.py task.py abl_fo   # only the failed drive is replaced
DEMO_VARIANT=settle      python demo.py task.py settle   # explicit hold, nothing moves
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
```

The right arm pulls the red-lit drive straight up, carries it out in front of the drawer and sets it in the bin. Only
when it has left the drawer does the left arm, waiting in front of the drawer with a spare, bring the spare over slot 2,
correct its position from the spare's measured pose, lower it in and let go. When slot 2's lamp turns green the right
arm pulls the amber drive to the bin and the left arm seats the second spare. Only one arm is ever over the drawer:
over adjacent slots (10 cm apart) the two gripper housings would collide. The demo reads the slot state from the task
(the same state the lamps show), so it is a feasibility proof, not a policy.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). A drive is *out* once
its bottom is more than 15 mm above its slot floor (or it has left its slot's footprint); it is *seated* when its centre
is inside a slot's inner footprint, its bottom within 5 mm of the floor and it tilts at most 10 degrees. A slot is
working while its original healthy or amber drive is in it, or once a seated spare has finished its 8 s rebuild.

- **0** if at any time fewer than 3 slots were working (data loss: all lamps turn steady red), if a healthy drive
  (slot 1 or 4) was ever pulled out, or if at the end any drive is outside a slot, a tray pocket and the bin (lying on
  the table or the drawer, or still held).
- **0.25** the failed drive is in the returns bin (centre inside the bin's inner footprint, lowest corner below the rim).
- **0.5** plus a spare seated in slot 2 and rebuilt (green).
- **0.75** plus the amber drive, pulled only after slot 2 was rebuilt, in the returns bin.
- **1.0** plus a spare seated and rebuilt in slot 3, all four lamps green, and all gripper points of both arms more than
  60 mm from the drawer; held for 1 s.

Either spare may go to either slot and the removed drives may lie anywhere in the bin: both are tested to score 1.0.

## Shortcuts and what stops them

| Shortcut | What stops it |
|---|---|
| Both arms swap both slots in parallel | Pulling the amber drive while slot 2 is failed or empty leaves 2 working: data loss, 0 |
| Pull the amber drive first | Same rule: 0 |
| Pull the amber drive before the green lamp | Slot 2 is still rebuilding, not working: 0 |
| Drop the removed drives on the table | End-state gate: every drive must be in a slot, pocket or the bin |
| Replace only the failed drive | 0.5 |

## Timing margins

| Event | Hidden timer | Scripted demo |
|---|---|---|
| Slot 2 rebuild | 8.0 s after the spare is seated (step 397 in the demo) | green at step 557 |
| Amber drive pulled (demo) | after the green lamp | about step 760, 10 s after green: the right arm waits for the left to fetch spare 2 and leave the drawer |
| Amber drive pulled (`no_wait`) | before the green lamp | 7.0 s after seating: data loss, 1.0 s before the rebuild would have finished |

With the arms sequenced so that only one is over the drawer, the right arm cannot reach slot 3 within 8 s of the
first spare being seated; the early pull that the rule catches is the greedy one by the arm already at the drawer
(`no_wait`) or a second arm working in parallel (`parallel`).

## Runtime and seeds

The demo reaches success at step 1,361 (68 s); about 6 minutes wall time on an L40S shared with other runs, plus
about 1 minute for Isaac Sim to start. A full 2,400-step policy trial takes about 10-15 minutes. The layout is fixed
(`-E object_jitter=0`); there are no seeds.

## Measured in the simulator

| Run | Score | Notes |
|---|---|---|
| settle (explicit hold, 8 s) | 0.0 | no gate tripped; every drive at rest, 0 mm off its floor, 0 deg tilt; lamps green, red, amber, green (slots 1-4) |
| demo | 1.0 | levels 0.25 at step 365, 0.5 at 557, 0.75 at 1,149, 1.0 at 1,342, success at 1,361; worst IK error 1.13 mm over all waypoints (offline check); closest approach between the two arms' gripper points 266 mm |
| parallel (predicted 0.0) | 0.0 | data loss at about step 300: the amber drive came out while slot 2 was empty |
| simultaneous (not predicted) | 0.0 | the two grippers came within 71 mm over adjacent slots; neither drive came out, so no data loss was recorded; it scores 0 because nothing was done |
| amber_first (predicted 0.0) | 0.0 | data loss at about step 200 |
| no_wait (predicted 0.0) | 0.0 | data loss 7.0 s after the first spare was seated |
| failed_only (predicted 0.5) | 0.5 | held to the end of the 900-step run |
| Dream-1 (2,400 steps) | 0.0 | moved both arms around the spare tray and the bin; no drive was pulled and the lamps never changed |

In the demo the drives are released 4 mm above the bin floor. The failed drive tipped over and lies flat in the bin,
and the amber drive ended leaning about 50 degrees against the bin wall, with its centre inside the bin and its lowest
corner 24 mm above the drawer floor level, well below the rim. The scoring only requires that a removed drive's centre
is inside the bin's inner footprint and its lowest corner is below the rim, so a toppled or leaning drive counts as in
the bin, as it would for a technician dropping a drive into a returns tote.

The geometry was changed from the first design for reach. Lifting a 110 mm drive out of a 70 mm slot needed a
straight-down tool height of 0.94 m over the far-side slot, but the arm reaches only about 0.87 m there. The drives are
therefore 90 mm tall in 40 mm deep slots, and the drawer sits at x 0.26 m.

## Fidelity

Drives, drawer, tray and bin are rigid boxes. The array (health, rebuild timer, data loss) is computed in code from the
measured drive poses; the lamps are small loose boxes moved by code each step between the panel and a hidden spot.
No contact-force cap is enforced in simulation.

## Real-world build and reset

Drawer, slot walls, lamp panel, tray and bin: 3D-printed PLA or folded sheet metal to the dimensions in `task.py`. The
drives are 150 g printed blocks (40 x 90 x 90 mm, top 10 mm silver). The lamps are four RGB LEDs behind diffusers,
driven by a microcontroller that reads a reed or IR presence switch at the bottom of each slot and runs the same state
machine. Reset: put drives 1-4 back in slots 1-4, the two spares in the tray, empty the bin, reset the controller.
