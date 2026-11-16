"""Checks for the method that reads the cycle count and the response out of
recorded histograms. Plain asserts and numpy; every histogram comes from the
library, through the calls the method makes on the real data."""

from __future__ import annotations

import sys

import numpy as np

import stb_method as S
from plidar import SIM_FIRST_PHOTON, Plidar

BIN = 750e-12      # one bin of the sensor, and the unit of the fixture
NBINS = 672
CYCLES = 7000.0


def _scene(irf, signal: float, t0: float = 300 * BIN):
    return Plidar.scene(period=NBINS * BIN, bin_width=BIN, t0=t0,
                        signal=signal, background=0.0, efficiency=1.0, irf=irf)


def _l1(got, true) -> float:
    """L1 distance between two tabulated densities, both at unit area.

    got is used as fit_response returns it. true is moved so that the half
    height point of its rising edge sits at t = 0, the origin fit_response
    gives the recovered density, so a wrong t_first counts as a difference.
    """
    s, t_first, dt = got
    t = t_first + dt * np.arange(s.size)
    s0, _, dt0 = true
    t0 = dt0 * (np.arange(s0.size) - S.rising_edge(s0))
    x = np.linspace(min(t[0], t0[0]), max(t[-1], t0[-1]), 20001)
    a = np.interp(x, t, s / np.trapezoid(s, t), left=0.0, right=0.0)
    b = np.interp(x, t0, s0 / np.trapezoid(s0, t0), left=0.0, right=0.0)
    return float(np.trapezoid(np.abs(a - b), x))


def test_rising_edge_interpolates_between_bins():
    assert S.rising_edge(np.array([0, 0, 1, 2, 3, 4, 4, 4.0])) == 3.0
    assert S.rising_edge(np.array([0, 0, 1, 3, 4.0])) == 2.5
    assert np.isnan(S.rising_edge(np.array([3, 4, 1.0])))  # edge runs into bin 0


def test_coates_rows_marks_refused_rows():
    mu = S.coates_rows(Plidar(), np.array([[1, 2, 3], [5, 5, 5]]), 10.0)
    assert np.isfinite(mu[0]).all() and np.isnan(mu[1]).all()  # cycles run out


def test_split_groups_keeps_every_count():
    rng = np.random.default_rng(1)
    hist = rng.poisson(40.0, NBINS).astype(np.uint64)  # what lib.simulate returns
    groups = S.split_groups(hist, 4, rng)
    assert groups.shape == (4, NBINS) and groups.dtype.kind == "i"
    assert np.array_equal(groups.sum(axis=0), hist)

    first = [S.split_groups(np.array([1000]), 4, rng)[0, 0] for _ in range(2000)]
    sigma = np.sqrt(1000 * 0.25 * 0.75 / 2000)
    assert abs(np.mean(first) - 250.0) < 3 * sigma


def test_estimate_cycles_finds_the_true_count():
    lib = Plidar()
    irf = lib.table(*S.flat_top(BIN))
    rng = np.random.default_rng(2)
    hists = []
    for total in (2250, 3750, 5250, 6200):  # the middle of each search class
        signal = -np.log1p(-total / CYCLES)
        for t0 in rng.uniform(60, 560, 50) * BIN:  # any phase within a bin
            sc = _scene(irf, signal, t0)
            hists.append(np.rint(lib.analytic(sc, CYCLES, SIM_FIRST_PHOTON)))
    hists = np.array(hists)
    m, curve = S.estimate_cycles(lib, hists, hists.sum(axis=1))
    assert curve.shape == S.M_GRID.shape
    assert abs(m - CYCLES) <= 100, (m, curve)
    # A real dip: without Coates the curve is flat and its argmin is noise.
    assert curve.min() < 0.5 * min(curve[0], curve[-1]), curve


def test_fit_response_recovers_the_density():
    lib = Plidar()
    true = S.flat_top(BIN)
    sc = _scene(lib.table(*true), 1.0)
    rng = np.random.default_rng(3)
    # Signal 1 with no background is what Coates returns on noiseless data.
    mu = np.array([lib.rate(sc, t0) for t0 in rng.uniform(60, 560, 400) * BIN])
    edges = np.array([S.rising_edge(r) for r in mu])
    got = S.fit_response(mu, edges, BIN)
    assert _l1(got, true) < 0.02, _l1(got, true)


def test_fit_response_is_exact_on_grid_phases():
    # With every phase on the 1/8 grid and the true shifts as edges the model
    # holds exactly, so only the penalty separates the fit from the truth.
    # The check above cannot see a double integration: an average of the
    # aligned windows is only 0.0044 away from the fixture in L1.
    lib = Plidar()
    true = S.flat_top(BIN)
    sc = _scene(lib.table(*true), 1.0)
    rng = np.random.default_rng(4)
    shift = rng.integers(60, 560, 200) + rng.integers(0, 8, 200) / 8
    mu = np.array([lib.rate(sc, x * BIN) for x in shift])
    got = S.fit_response(mu, shift, BIN, smooth=1e-6)
    assert _l1(got, true) < 1e-4, _l1(got, true)


def test_is_single_return_tells_one_return_from_two():
    lib = Plidar()
    clean = _scene(lib.table(*S.flat_top(BIN, tail=0.0)), 1.0)
    tailed = _scene(lib.table(*S.flat_top(BIN)), 1.0)
    t0 = 200.3 * BIN
    # Both pulses of the pair carry the tail. The second sits on the tail of
    # the first, so it holds the highest bin and the edge lands on it.
    pair = lib.rate(tailed, t0) + lib.rate(tailed, t0 + 30 * BIN)
    cases = {"clean single pulse": (lib.rate(clean, t0), True),
             "two equal pulses 30 bins apart": (pair, False),
             "single pulse with the 4 percent tail": (lib.rate(tailed, t0), True)}
    for name, (mu, single) in cases.items():
        assert S.is_single_return(mu, S.rising_edge(mu)) == single, name


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
