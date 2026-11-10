# Press a push-pin clip into a springy panel while the other arm backs it from underneath

Tests one ability: applying a sustained counter-force on the far face of a compliant panel (holding it to <= 3 mm) while
the other arm presses a detent fastener home. Pressing an unsupported panel only sags it (0.5 N/mm seats), so one arm alone
cannot seat the clip. Real-world relevance: door-card and bumper trim clips are pressed in over a backed panel
(O*NET 49-3023, 51-2098).

## Files
`task.py` (scene, scoring), `partkit.py` (shared part helper, unchanged), `demo.py` (scripted closed-loop demo and ablations),
`tests/` (scoring unit tests), `task.json`.

## Run
    inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
      -E cam_width=320 -E cam_height=176 --instruction "<see task.json>" --scorer task_score --max-steps 1800 --store-frames
    python demo.py task.py demo_out                  # scripted demo (DEMO_VARIANT=right_only|left_only|both_no_press)
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests

Run limit 1,800 steps (90 s). The demo takes about 735 steps (37 s). Object jitter 0 (one fixed layout, no seeds).

## Scene (metres, station frame, table top 0.7475)
- Heavy base (4 kg): two rails along y at x = 0.30 +- 0.085, each carrying two silver seats (30 mm squares, k = 125 N/m each,
  0.5 N/mm in total, 20 mm travel), at y = +-0.07.
- Grey panel 200 x 200 x 8 mm, 0.1 kg, on the seats, with a 34 mm square hole at (0.30, -0.02).
- Red clip: 30 mm stem with a 44 mm cap in the hole, prismatic, detent 0.6 N/mm, 10 mm travel (about 6 N to seat).
- Blue paddle (0.2 kg): fork whose prongs straddle the clip stem, on a 300 mm stem with an upright post to grip, lying at y = +0.18.

## Scoring (read from simulator joints and poses; no hidden state)
sag = rest height of the panel minus its height (rest measured in the settle run); press = clip travel.
- 0.25: press >= 3 mm at any time.
- 0.5: paddle touching the panel's underside within 40 mm of the clip axis while sag <= 8 mm.
- 0.75: press >= 9 mm while sag <= 3 mm.
- 1.0: that for 10 steps (0.5 s), then the paddle is clear of the panel (fork >= 20 mm away) and the panel is back at rest
  +- 4 mm and still for 1 s.
- Zero (after step 40): panel bottomed (sag > 18 mm) with the clip pressed, panel or base moved > 40 mm, panel tilted > 20 degrees, paddle lifted > 20 cm.

## Real-world build and reset
Two foam or coil-spring seats under a plastic trim panel on a rigid base, a spring-loaded push clip in a hole (a plunger with a
spring), and a flat fork-shaped backing tool on a handle. Reset: lay the paddle on the table, let the clip spring back.

## Shortcuts and what stops them
- Pressing alone: the panel sags 12 mm at 6 N, sag > 3 mm so no credit above 0.25 (ablation 0.25).
- Jamming the paddle anywhere else: it has to be within 40 mm of the clip axis for 0.5.
- Forcing the clip through with the panel on its stops: zero gate.
- Leaving the paddle under the panel: it has to be withdrawn and the panel back at rest.
- The simulator clip has no latch: it springs back when released, so "seated" is the recorded 10 mm travel held 0.5 s under support.

## Measured in the simulator
| Run | Score |
|---|---|
| Scripted demo (both arms) | 1.0 (735 steps) |
| Ablation: right arm only (no paddle) | 0.25 |
| Ablation: left arm only (paddle, no press) | 0.5 |
| Ablation: both arms, right does not press | 0.5 |
| Dream-1 run | 0.0 |

Note: while the clip is pressed the paddle slides 3 to 4 cm along y in the left hand's grip, but stays under the panel.

## Deviations from a real clip job
Spring rates are deliberately softer (about half of a stiff trim panel's seat stiffness and a light detent) so the arms can press and back reliably; the panel sag figures are therefore larger than for real trim. The clip has no latch. The paddle slides 3 to 4 cm along y in the gripper during the press.

## How this differs from nearby tasks
Other board tasks use a second hand to hold or take up a strap (child-seat press with an anchor strap, #1291), to back a workpiece against drill breakout (#1260), or to place support pins under a board (#1037). Here the second arm must insert a thin fork under a compliant, spring-seated panel and keep its sag within 3 mm while the first arm presses a detent clip through it, then withdraw the fork and let the panel return. The scored quantity is the sag/press coupling, which none of those tasks measures.
