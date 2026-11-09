"""Lets the tests run without Isaac Sim: if yam_arena is not installed, a minimal stand-in for yam_arena.task_file
(TABLE_TOP, Box, ObjectState) is registered. On the VM the real package is used."""

import pathlib
import sys
import types
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import yam_arena.task_file  # noqa: F401
except ImportError:
    tf = types.ModuleType("yam_arena.task_file")
    tf.TABLE_TOP = {"station": 0.7475, "tabletop": 0.75}

    @dataclass(frozen=True)
    class Box:
        name: str
        pos: tuple
        color: tuple = (0.7, 0.7, 0.7)
        mass: float | None = None
        yaw: float = 0.0
        size: tuple = (0.04, 0.04, 0.04)

    @dataclass(frozen=True)
    class ObjectState:
        pos: np.ndarray
        quat_xyzw: np.ndarray
        lin_vel: np.ndarray
        ang_vel: np.ndarray
        low: np.ndarray
        high: np.ndarray

    tf.Box, tf.ObjectState = Box, ObjectState
    pkg = types.ModuleType("yam_arena")
    pkg.task_file = tf
    sys.modules["yam_arena"], sys.modules["yam_arena.task_file"] = pkg, tf
