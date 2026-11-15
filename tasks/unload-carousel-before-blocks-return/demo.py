"""Scripted demo for the carousel task: runs the task file's demo() through the yam_scripted policy.

    python demo.py task.py demo_out [--max-steps 1800] [--no-record]

DEMO_VARIANT=full (default) | settle | left_only | right_only | late | blue_removed selects the ablations described in
the README; each is read by task.demo(). Writes the eval log (and, unless --no-record, the per-camera MP4s) to the
output folder. The demo is open loop: it reads nothing from the simulator; its timing is the carousel's fixed
timetable and its places are the layout's fixed places.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

INSTRUCTION = ("Load the five blocks onto the carousel at the green marker. Crate each red block before it returns "
               "to that marker; leave the blue ones riding.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="task.py")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    cmd = ["inspect-robots", "run", "--embodiment", "yam_arena", "--policy", "yam_scripted", "-E", f"task={a.task}",
           "-E", "object_jitter=0", "-E", "control_hz=20", "-E", "cam_width=320", "-E", "cam_height=176",
           "--instruction", INSTRUCTION, "--scorer", "task_score", "--max-steps", str(a.max_steps),
           "--log-dir", str(out)]
    if not a.no_record:
        cmd.append("--store-frames")
    r = subprocess.run(cmd)
    if r.returncode == 0 and not a.no_record:
        for log in sorted(out.glob("*.json")):
            subprocess.run(["inspect-robots", "video", str(log)])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
