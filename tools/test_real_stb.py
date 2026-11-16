"""Checks for the parts of real_stb.py that read the recorded sample: the
intrinsics parser, the inverse of the published extrinsic, the per pixel
Livox reference and the summary of a fitted response. Inputs small enough
to work out by hand, so no dataset and no library are needed."""

from __future__ import annotations

import sys

import numpy as np

import real_stb as R

# The layout of the dataset's config.yaml, line endings included. Each block
# also holds the tangential coefficients p1 and p2, which share their names
# with the blocks.
CONFIG = ("common:\r\n  dt_ps: 750\r\n  image_sizes:\r\n    - [256, 192]\r\n    \r\n"
          "datasets:\r\n"
          "  p1:\r\n    cx: 130.5\r\n    cy: 97.25\r\n    fx: 121.0\r\n    fy: 120.0\r\n"
          "    k1: -0.25\r\n    k2: 0.0625\r\n    p1: 0.0\r\n    p2: 0.0\r\n"
          "  p2:\r\n    cx: 1.0\r\n    cy: 2.0\r\n    fx: 3.0\r\n    fy: 4.0\r\n"
          "    k1: 5.0\r\n    k2: 6.0\r\n    p1: 0.0\r\n    p2: 0.0")


def test_parse_intrinsics_reads_one_block():
    assert R.parse_intrinsics(CONFIG) == {"cx": 130.5, "cy": 97.25, "fx": 121.0, "fy": 120.0,
                                          "k1": -0.25, "k2": 0.0625, "p1": 0.0, "p2": 0.0}
    assert R.parse_intrinsics(CONFIG, "p2")["fx"] == 3.0
    for text in (CONFIG.replace("  p1:", "  p3:"),            # no such block
                 CONFIG.replace("    fy: 120.0\r\n", ""),      # a key missing
                 CONFIG.replace("    p2: 0.0\r\n  p2:", "    p2: 0.1\r\n  p2:")):
        try:
            R.parse_intrinsics(text)
        except ValueError:
            continue
        raise AssertionError(f"accepted {text!r}")


def test_to_sensor_inverts_the_extrinsic():
    # A reflection, det -1, like the published matrix: S = (1, 2, 3) goes to
    # R S + t = (3, 1, -2) + (0.5, 0, -1).
    rot = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0.0]])
    ext = np.column_stack([rot, [0.5, 0.0, -1.0]])
    assert np.allclose(R.to_sensor(np.array([[3.5, 1.0, -3.0]]), ext), [[1, 2, 3]])
    # The published rows fall short of unit length, as here, and the
    # transpose then shrinks two coordinates by 0.15 percent.
    short = np.array([[0, 0.05, 0.998], [1, 0, 0], [0, -0.998, 0.05]])
    ext = np.column_stack([short, [0.5, 0.0, -1.0]])
    s = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    assert np.allclose(R.to_sensor(s @ short.T + ext[:, 3], ext), s, rtol=0, atol=1e-12)


def test_livox_reference_groups_points_by_pixel():
    # Unit focal lengths, no distortion and no transform: (x, y, z) lands on
    # column rint(x/z) and row rint(y/z), at range |(x, y, z)|.
    pts = np.array([[0, 0, 1], [0, 0, 4], [0, 0, 2], [0, 0, 10], [0, 0, 3],
                    [9, 6, 3], [6, 4, 2],  # column 3, row 2
                    [0, 0, -1],            # behind the sensor
                    [-1, 0, 1.0]])         # left of column 0
    data = {"livox": pts, "extrinsic": np.column_stack([np.eye(3), np.zeros(3)]),
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "k1": 0.0, "k2": 0.0}}
    med, iqr, cnt = R.livox_reference(data, 0, 0)
    assert med.shape == iqr.shape == cnt.shape == (R.WIDTH * R.HEIGHT,)
    assert (med[0], iqr[0], cnt[0]) == (3.0, 2.0, 5)  # ranges 1 2 3 4 10
    # Ranges sqrt(56) and sqrt(126): the median halfway, quartiles a quarter in.
    a, b = np.sqrt(56.0), np.sqrt(126.0)
    p = 2 * R.WIDTH + 3
    assert np.isclose(med[p], (a + b) / 2) and np.isclose(iqr[p], (b - a) / 2) and cnt[p] == 2
    assert cnt.sum() == 7 and np.isnan(med[cnt == 0]).all()
    med, _, cnt = R.livox_reference(data, 1, 2)  # one column right, two rows down
    assert med[2 * R.WIDTH + 1] == 3.0 and cnt[2 * R.WIDTH + 1] == 5 and cnt[0] == 0


def test_response_shape_of_a_triangle():
    # Nodes 0 1 2 1 0, one node per bin: half height at nodes 1 and 3, 10 and
    # 90 percent at 0.2 and 1.8 on the way up.
    fwhm, rise, _ = R.response_shape((np.array([0, 1, 2, 1, 0.0]), -1.5 * R.BIN_WIDTH,
                                      R.BIN_WIDTH))
    assert np.isclose(fwhm, 2.0) and np.isclose(rise, 1.6), (fwhm, rise)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:  # first failure ends the run
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
            return 1
    print(f"{len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
