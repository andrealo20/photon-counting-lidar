"""Check that the ctypes declarations still match the C structs.

The Python tools pass structures across the boundary by memory layout, not by
name. If a field is added in the wrong place, or a type is not the width it
appears to be, ctypes reads whatever happens to sit at the offset it
computed. Nothing crashes and nothing complains; the sweeps simply produce
numbers that look plausible and are wrong.

This builds nothing. It runs the layout probe that CMake already built and
compares its output, entry by entry, with what ctypes says about the
declarations in plidar.py.

Usage::

    python tools/check_bindings.py
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import plidar as P

ROOT = Path(__file__).resolve().parent.parent

STRUCTS = [
    ("plidar_irf", P.Irf),
    ("plidar_scene", P.Scene),
    ("plidar_detector", P.Detector),
    ("plidar_estimate", P.Estimate),
    ("plidar_crb_result", P.CrbResult),
    ("plidar_rng", P.Rng),
]


def find_probe() -> Path:
    names = ["plidar_layout_probe", "plidar_layout_probe.exe"]
    candidates = [p for name in names for p in ROOT.glob(f"build*/**/{name}")]
    if not candidates:
        raise FileNotFoundError(
            f"layout probe not found under {ROOT}. Configure and build with "
            "PLIDAR_SHARED on, which is the default."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def from_c(probe: Path) -> dict[str, int]:
    out = subprocess.run([str(probe)], check=True, capture_output=True, text=True)
    values = {}
    for line in out.stdout.splitlines():
        key, value = line.rsplit(" ", 1)
        values[key] = int(value)
    return values


def from_python() -> dict[str, int]:
    values = {}
    for name, cls in STRUCTS:
        values[name] = ctypes.sizeof(cls)
        for field, _type in cls._fields_:
            values[f"{name}.{field}"] = getattr(cls, field).offset
    return values


def main() -> int:
    c_side = from_c(find_probe())
    py_side = from_python()

    problems = []
    for key in sorted(set(c_side) | set(py_side)):
        in_c = c_side.get(key)
        in_py = py_side.get(key)
        if in_c is None:
            problems.append(f"{key}: declared in plidar.py, absent from the C probe")
        elif in_py is None:
            problems.append(f"{key}: present in C, absent from plidar.py")
        elif in_c != in_py:
            problems.append(f"{key}: C says {in_c}, ctypes says {in_py}")

    for problem in problems:
        print(f"mismatch: {problem}")
    print(f"{len(c_side)} layout entries checked, {len(problems)} mismatched")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
