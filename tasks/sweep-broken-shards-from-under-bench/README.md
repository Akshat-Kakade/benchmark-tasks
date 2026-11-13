# Sweep broken ceramic pieces from under a low bench into the pan with a two-handled squeegee

A cup has smashed under a low bench. Four ceramic pieces lie under the overhang, out of reach of a hand and too sharp to
pick up. A blue floor squeegee lies in front of the bench: the middle of its head goes under the overhang, so the only
places to hold it are the two orange handle blocks at its two ends (0.46 m apart), one per arm. Both arms advance the head
together so that its straight edge collects all four pieces and leaves them on the green pan, then release it and keep
still. Core ability: driving one wide tool with two separated pushes so that it stays square while it collects several
loose pieces. The scorer reads where the pieces end up.

Done by: janitors and cleaners (O*NET 37-2011.00), who sweep up broken crockery and spilled debris with push brooms and wide
push heads (floor squeegee or broom heads) under counters and benches; the fragments are sharp and the gap is too low for a hand.
A dustpan is simplified to a flat pan.

## Files
- `task.py`: scene, instruction, `success()`, `score()`. `partkit.py`: helper, unchanged.
- `demo.py`: open-loop scripted demo (`DEMO_VARIANT` = `settle`, `probe`, `left_only`, `right_only`, `half_stroke`,
  `three_pieces_only`, `no_release`, `poke`). `tests/`: scoring boundaries, every partial level, zero gate, two valid end
  states, demo-is-open-loop grep, blind-word scan. `task.json`, `media/`.

## Instruction given to the policy

> A cup has smashed under the bench. Get every piece of it out from under the bench and into the green pan, then let go of the squeegee and leave it and the pieces still.

## Run
```
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "A cup has smashed under the bench. Get every piece of it out from under the bench and into the green pan, then let go of the squeegee and leave it and the pieces still." --scorer task_score --max-steps 1200 \
  --store-frames --log-dir runs/trial
python demo.py task.py demo_out --max-steps 1200
```
Runtime per trial: run limit 1200 steps (60 s simulated); the demo scores 1.0 at about step 380 (19 s). No random seeds
(`object_jitter=0`).

## Scene (metres, station frame, table top 0.7475, x forward, y left)
| Object | Size (mm) | Mass / material | Start pose |
|---|---|---|---|
| Squeegee head | blade 30 (x) x 420 (y) x 30 (z), blue, with a 40 x 40 x 60 orange handle block on the table at each end | 1.0 kg in total (30 x 30 mm steel box-section, 2.5 mm wall, 0.42 m = 0.91 kg, plus the two hollow sockets); friction 0.35 | blade centre x 0.200, y 0; handle block centres y = +-0.230 |
| Bench | four rails 230 x 30 x 20 (y = -0.105, -0.035, 0.035, 0.105: 40 mm gaps), a 20 x 240 x 20 cross bar at x 0.16, two 20 x 20 posts at x 0.16, y = +-0.11; wood colour, fixed | fixed | rails x 0.15-0.38, underside 60 mm above the table (the 30 mm blade passes, a hand does not) |
| Pieces (4) | 22 x 22 x 12 each, cream | 14 g each (22 x 22 x 12 mm of glazed earthenware, 2.4 g/cm3); friction 0.4 | (0.300, -0.070), (0.285, 0.004), (0.315, -0.004), (0.305, 0.070): all under the rails, three gaps (two pieces share the middle gap, 30 mm apart in x) |
| Pan | floor 75 x 280, level with the table, plus green walls 12 mm high on both sides (10 mm thick) and the far end; fixed | fixed | floor x 0.4175-0.4925, y +-0.14; the sweep side is open |
Table friction is the simulator default; the blade and the pieces use the values above.

## Scoring (measured at the end; the full state held 1 s = 20 steps; poses and the two gripper states)
Zero gate (latched): the head tilts > 8 deg, yaws > 8 deg at any step, any piece leaves the table, or the blade bottom
rises > 5 mm for more than 0.5 s (the first 10 steps are ignored for the spawn settle).
- 0.25 / 0.5 / 0.75: one / two / three piece centres on the pan at the END (a piece dragged back off the pan drops the tier). At most 0.5 unless both handle blocks have advanced >= 160 mm (read from the head's two end positions).
- 1.0: all four piece centres >= 8 mm inside the pan edge, none under the bench, the blade at least 4 mm clear of the pieces, both handle blocks advanced >= 160 mm, both grippers open (finger gap <= 58 mm) and >= 40 mm from the squeegee, all objects still (speed <= 12 mm/s) for 20 consecutive steps.
Two valid end states (different piece spreads, head offset and yaw) are unit-tested to score 1.0.

## Demo (open loop)
Observations used: none from the scene. The script is a fixed list of waypoints built from the task constants (handle
block positions, pan position) and plays them through IK; each arm's own `joint_pos` is integrated into its command so the
hand follows its commanded path (proprioception only, no object poses, no camera). Steps: approach and pinch both handle
blocks (fingers open along x, pad pressing the block from behind), advance both hands together 0.215 m at 30 mm/s, hold
0.5 s, pull back 25 mm so the blade is clear of the pieces, open both grippers, lift 0.12 m, hold still.

## Real-world build and reset
A 0.42 m steel-section squeegee head with two end sockets (or a floor squeegee with its handle removed and two grab blocks),
a low shelf or bench with a 60 mm gap, four ceramic pieces (about 2 cm) placed under it, a marked pan area. Reset: head
square in front of the bench, pieces in the listed places. Score with a ruler.

## Shortcuts and what stops them
- One arm alone: the blade yaws past 8 deg (zero gate) in both one-arm runs.
- Moving the head only a little: pieces stay under the bench (`half_stroke`).
- Missing a piece: the full tier needs all four (`three_pieces_only` was measured at 0.0 because the sideways slide tripped the yaw gate). Leaving the grippers closed, or the head on the pieces: capped at 0.75 (`no_release`).
- Reaching under the bench with a gripper: the rails are 60 mm off the table, so the hand body cannot enter; a lone arm also cannot meet the 160 mm two-handle advance condition. (The `poke` ablation was measured on an earlier bench layout, not re-run.)
- Doing nothing: 0.0 (`settle`).

## How this differs from nearby tasks
- #145 (pan held while a brush sweeps) and #1189 (carton jam): there the second arm holds a fixed container or clears a jam; here both arms move the same free tool and the target is loose pieces that start under an overhang.
- #419 (guided beam squared) and our own earlier two-arm pushes of a rigid bar (#638 style): the bar there is the load; here the tool carries four independent loose pieces, so the score is an end-state count of what ended up on the pan, with the tool released and still.
- The squeegee middle is under the bench, so the two ends are the only grips: one hand cannot keep the blade square (measured).

## What is simplified / not modelled
- The pan floor is level with the table and has walls only on its two sides and far end (the sweep side is open): a 0.3-0.5 mm step already stopped the pieces in the simulator. The floor is not always visible in the camera (it is coplanar with the table); the walls mark it.
- Head mass: the simulated squeegee is 1.0 kg; a typical light squeegee is about 0.45 kg. The bimanual necessity depends on this mass, on the pan walls and on the rail bench: at 0.45 kg one arm alone scored 0.75 (the pinched hand held the blade square). The heavier blade is a steel box-section (0.91 kg) plus sockets.
- Pieces are rigid 12 mm thick boxes; real shards are 3-5 mm thick and irregular, and there is no dust.
- No rubber blade: the blade is a rigid box with friction 0.35 (a real rubber edge grips more).
- The bench is four fixed rails with a cross bar on two posts at the near end, a cantilever; a real bench has legs, which would be outside the sweep.
- The gap under the bench is 60 mm (a real gap is often 80-150 mm); it is kept low so a hand cannot go under it.
- Handle blocks are rigid and fused to the head; a real handle socket has play.

## End-frame legibility
The pan floor is flush with the table, so in the 320x176 camera frames only the green pan walls are visible; the pieces that
end on the pan are a few pixels each. The pieces under the rails are visible from above through the gaps.

## Measured in the simulator
All rows on the current `task.py` (rail bench, pan walls, head 1.0 kg), run limit 1200, `object_jitter=0`.
| Run | Score |
|---|---|
| Demo | 1.0 (success at step 382; head yaw 0.35 deg, advance 189 / 192 mm, IK error 0.09 mm) |
| `settle` | 0.0 |
| `left_only` | 0.0 (head yaw 9.8 deg: zero gate) |
| `right_only` | 0.0 (head yaw 12 deg: zero gate) |
| `half_stroke` | 0.0 (pieces stay under the bench) |
| `three_pieces_only` | 0.0 (the sideways slide yawed the head 14 deg: zero gate; prediction was <= 0.75) |
| `no_release` | 0.75 (all four pieces on the pan; grippers still closed so no 1.0) |
| `poke` | 0.0, measured on an earlier bench layout (solid plate); not re-run on the final geometry |
| Dream-1 run | 0.0 (1200 steps; one arm reached the squeegee, nothing was swept) |
