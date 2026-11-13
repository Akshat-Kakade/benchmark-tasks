# Slide a loaded sheet pan into the second slot of a rolling rack without letting the rack roll

Instruction: "Put the full baking tray into the second slot of the rolling rack without letting the rack move."
(The brace is not stated; the rack-movement rule is stated as an outcome.) Run limit 900 steps (45 s at 20 Hz). Category: Impact on jobs.

## Importance
Cooks, restaurant (O*NET 35-2014.00, about 1.4 million US workers per BLS (May 2023: 1,412,350), one of the largest kitchen occupations) load and unload sheet pans into wheeled speed racks
every shift. The rack rolls away from the push of the pan, so a cook steadies the rack with one hand and slides the pan in
with the other. A policy that only pushes the pan shoves the rack away and leaves the pan half in.

## How this differs from nearby tasks
Closest on the board: #512 (unload a tray from a pass shelf: pulling out, no rolling base), #1660 (bike onto a bus rack),
#1223 (rocking tray) and the slip-sheet tasks (brace a load, pull a sheet). Here the new part is bracing a free-rolling base
against the push of the load while the load is lifted to the right slot and slid in.

## Construct validity (every noun in the task has geometry)
All positions in m in the station frame (x forward, y left, table top z = 0.7475), part sizes in mm.
| Part | Geometry | Material | Mass | Start pose |
|---|---|---|---|---|
| Rack (one rigid body) | base plate 256 x 404 x 15, raised 10 mm on 4 sprung caster feet (40 x 40 x 10, friction 0.01); two side panels 250 x 5 x 205, 354 apart inside; rear stop panel 6 x 364 x 205 at 250 deep; two front posts 20 x 20 x 240; three pairs of ledge ribs (25 wide, 6 thick, 250 long) with top faces 55, 110, 165 above the table, on light spring sliders with friction 0.45 | aluminium | simulated 6 kg (same parts in solid aluminium: about 8 kg) | open front at x = 0.34, centred y = -0.03 |
| Counter | plate 270 x 360 x 15, fixed | grey plastic | fixed | centre x 0.2045, y -0.03 |
| Pan (quarter sheet pan) | floor 241 (x) x 330 (y), 20 high walls 3 thick, folded grab handle 8 x 80 x 36 mm (x, y, z) on the rear wall, ten cookie blocks 40 x 40 x 12 | sheet metal, friction 0.45; walls modelled 3 mm for collision (real pan 1 mm aluminium ~0.28 kg, or stainless ~0.65 kg) | simulated 0.35 kg with load | on the counter, centred y -0.03, front edge 15 mm before the rack |

Pan side clearance in the rack 12 mm per side, 13 mm of the floor rests on each rib; 55 mm slot pitch.

## What is simplified / not modelled
- A real speed rack is about 1.8 m tall, 20-30 kg on 100 mm casters; here a 0.24 m, three-slot, 6 kg segment (the arms reach
  about 0.2 m above the table); the 6 kg is the simulated mass, not computed from the geometry. Casters are four sprung feet with friction 0.01 (rolling resistance), not wheels.
- Slot pitch 55 mm (real 75-90 mm); ribs are flat ledges with no flare.
- The simulated pan has 3 mm collision walls and 0.35 kg total mass (that mass was simulated) (a real 1 mm pan is ~0.28 kg aluminium or ~0.65 kg stainless, plus cookies), cookies are rigid blocks, no heat. The grab handle stands in for the
  rolled rim a cook pushes on (a pinch on the 3 mm bare wall slipped).
- The demo is close to the gate: the rack moved at most 34 mm of the 40 mm allowed (the left jaws' close shifts the free rack about 18 mm). Results repeat exactly with object_jitter 0 and seed 42, so this is stable in the simulator but has little margin.
- The scorer reads object poses and the gripper links (finger gap, joint speeds) from the simulator; the demo does not.

## Scoring (end state; hazard gates latch)
0 if at any step the rack has moved 40 mm or turned/tipped 8 degrees, the pan tilts 15 degrees or falls. Otherwise:
0.25 pan at slot-2 height (within 10 mm of the rib) at the rack mouth; 0.5 at least 120 mm into slot 2 and level (<= 5
deg), or fully in another slot; 0.75 fully in slot 2 (front edge within 10 mm of the stop), centred within 12 mm, level;
1.0 as 0.75 with the pan on the ribs (within 4 mm), both grippers open, pan, rack and both arms still, for 20 consecutive
steps (1 s). A rack that ends where it stood but rolled 40 mm at any moment scores 0. Either arm may brace or push.

## Run
    source /data/IsaacLab-Arena/yam.env      # on the simulator machine
    PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q        # 15 offline scoring tests
    python demo.py task.py demo_out --max-steps 900            # scripted demo, writes demo_*_cam.mp4
    DEMO_VARIANT=one_arm python demo.py task.py out --no-record  # ablations (see below)
    inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --instruction "Put the full baking tray into the second slot of the rolling rack without letting the rack move." \
      --scorer task_score --max-steps 900 --store-frames --log-dir runs/dream
Runtime: the demo takes 736 steps (37 s of simulated time), about 5-15 minutes wall clock on a shared GPU. Seed: object_jitter 0, environment seed 42 (runs repeat exactly).

## Demo (open loop)
`demo.py` plays a fixed script from the layout constants. Observations used: only each arm's own measured joint angles
(the command is nudged so the measured gripper pose tracks the plan); no object poses. Left hand pinches the rack's front-left
post and holds; right hand pinches the pan's grab handle, lifts to rib height with the hand pitched 6 degrees nose-up (the
pan sags in the pinch), slides in 100 mm in open air, settles onto the ribs while sliding on, pushes to the stop, opens, lifts away; the left hand opens last and lifts away.

## Shortcuts and what stops them
Pushing with one arm: the rack rolls (friction 0.01 feet, ribs drag the pan) and the 40 mm gate fires. Wrong slot: scores
0.5 at most. Half-in and leaving a hand closed cannot reach 1.0. Dropping or tilting the pan: gate.

## Real-world build and reset
Aluminium shelf rack on unlocked swivel casters (or a small cart with a rib set), a quarter sheet pan (24 x 33 cm) loaded with
ten ~35 g cookies, a counter at pan height. Reset: park the rack at the marked spot, pan on the counter.

## Measured in the simulator
| Run | Score | Note |
|---|---|---|
| Demo (both arms) | 1.0 | 736 steps; rack moved at most 34 mm |
| one_arm (right hand only) | 0.0 | rack rolled 86 mm, gate |
| slot1 (pan carried to slot 1) | 0.5 | wrong slot |
| half_in | 0.25 (the released half-in pan tilted 14 degrees; predicted 0.5) | |
| no_release (left hand never opens) | 0.75 (open-gripper condition fails) | |
| do nothing | 0.0 | |
| Dream-1 run | 0.0 (900 steps, pan never lifted; run with the earlier wording "...second slot of the rolling rack, and leave the rack where it stands.") | |
