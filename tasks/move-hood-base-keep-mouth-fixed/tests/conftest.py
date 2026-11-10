"""Use the real yam_arena when installed (the VM); otherwise a minimal stand-in so the scoring tests run anywhere."""
import pathlib
import sys
import types
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import yam_arena.task_file  # noqa: F401
except ImportError:
    import numpy as np

    pkg = types.ModuleType("yam_arena")
    tf = types.ModuleType("yam_arena.task_file")
    tf.TABLE_TOP = {"station": 0.7475}

    @dataclass(frozen=True)
    class Box:
        name: str
        pos: tuple
        size: tuple
        mass: float | None = None
        color: tuple = (0.5, 0.5, 0.5)
        yaw: float = 0.0

    @dataclass(frozen=True)
    class ObjectState:
        pos: np.ndarray
        quat_xyzw: np.ndarray
        lin_vel: np.ndarray
        ang_vel: np.ndarray
        low: np.ndarray
        high: np.ndarray

    tf.Box, tf.ObjectState = Box, ObjectState
    pkg.task_file = tf
    sys.modules["yam_arena"], sys.modules["yam_arena.task_file"] = pkg, tf
