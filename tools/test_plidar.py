"""Checks for the Python bindings of the tabulated response and the first
photon model. Plain asserts, numpy and the standard library only."""

from __future__ import annotations

import gc
import sys

import numpy as np

from plidar import SIM_FIRST_PHOTON, Plidar


def _gauss_scene(signal: float, background: float = 1e6):
    return Plidar.scene(period=100e-9, bin_width=50e-12, t0=37.3e-9,
                        signal=signal, background=background, efficiency=0.3,
                        irf=Plidar.gaussian(150e-12))


def test_table_scene_outlives_its_irf():
    lib = Plidar()
    irf = lib.table(np.array([0.0, 5.0, 0.0]), -1e-9, 1e-9)
    sc = Plidar.scene(period=100e-9, bin_width=0.25e-9, t0=30e-9, signal=1.0,
                      background=0.0, efficiency=1.0, irf=irf)
    before = lib.rate(sc, sc.t0)
    del irf
    gc.collect()
    junk = [np.full(3 * 3, 7.0) for _ in range(64)]  # noqa: F841
    assert sc._keep is not None
    assert np.array_equal(lib.rate(sc, sc.t0), before)


def test_first_photon_bindings_round_trip():
    lib = Plidar()
    sc = _gauss_scene(2.0 / 0.3)
    h = lib.analytic(sc, 1e6, SIM_FIRST_PHOTON)
    est = lib.mle_first_photon(sc, h, 1e6, 1.0, np.zeros(4 * sc.nbins))
    assert abs(est.t0 - 37.3e-9) < 1e-3 * sc.bin_width

    # With 1e6 Hz of background the first photon bound sits 1.1% above the
    # ideal one, so the limit is only checked where pile up is negligible.
    weak = _gauss_scene(1e-4, 1e4)
    a = lib.crb_first_photon(weak, 1e6, 1.0, np.zeros(2 * weak.nbins))
    b = lib.crb(weak, 1e6, np.zeros(2 * weak.nbins))
    assert abs(a.var_tof_joint / b.var_tof_joint - 1.0) < 2e-3


def test_short_scratch_is_refused():
    lib = Plidar()
    sc = _gauss_scene(2.0 / 0.3)
    h = lib.analytic(sc, 1e6, SIM_FIRST_PHOTON)
    try:
        lib.mle_first_photon(sc, h, 1e6, 1.0, np.zeros(4 * sc.nbins - 1))
    except ValueError:
        return
    raise AssertionError("short scratch was accepted")


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
