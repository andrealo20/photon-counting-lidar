"""Checks for the method that reads the cycle count and the response out of
recorded histograms and measures the estimators on groups of them. Plain
asserts and numpy; every histogram comes from the library, through the calls
the method makes on the real data."""

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
    assert S.l1_distance(got, true) < 0.02, S.l1_distance(got, true)


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
    assert S.l1_distance(got, true) < 1e-4, S.l1_distance(got, true)


def test_fit_response_does_not_depend_on_the_pixel_count():
    # Ten copies of the same pixels carry the same shape, so the fit must not
    # move. Weights that counted pixels would weaken the penalty tenfold.
    lib = Plidar()
    sc = _scene(lib.table(*S.flat_top(BIN)), 1.0)
    rng = np.random.default_rng(5)
    mu = np.array([lib.rate(sc, t0) for t0 in rng.uniform(60, 560, 100) * BIN])
    edges = np.array([S.rising_edge(r) for r in mu])
    once = S.fit_response(mu, edges, BIN)[0]
    tiled = S.fit_response(np.tile(mu, (10, 1)), np.tile(edges, 10), BIN)[0]
    assert np.allclose(once, tiled, rtol=1e-9, atol=1e-12), np.abs(once - tiled).max()


class _Spy:
    """Hands every call on to the library and keeps a copy of each array it
    was given."""

    def __init__(self, lib):
        self.lib, self.arrays = lib, []

    def __getattr__(self, name):
        method = getattr(self.lib, name)

        def call(*args):
            self.arrays += [a.copy() for a in args if isinstance(a, np.ndarray)]
            return method(*args)
        return call


def test_coates_refusal_is_a_counted_failure():
    lib = Plidar()
    irf = lib.table(*S.flat_top(BIN))
    tmpl = _scene(irf, 1.0)  # the estimators read only its grid and response
    k, t0 = 4, 300.4 * BIN
    quota = CYCLES / k  # the trials Coates gets for one group
    # A bright return records in the first bins of the pulse. Scaled to quota
    # counts, every trial of the group records, so Coates runs out at the
    # group's last count; the first photon model only needs counts <= M.
    bright = lib.analytic(_scene(irf, 3.6, t0), CYCLES, SIM_FIRST_PHOTON)
    group = np.rint(bright * quota / bright.sum())
    group[np.argmax(group)] += quota - group.sum()
    est = S.estimate_all(lib, tmpl, group, CYCLES, 1 / k, np.zeros(4 * NBINS))
    assert np.isnan(est["matched_coates"]) and np.isnan(est["coates_poisson"]), est
    assert np.isfinite(est["first_photon"]), est

    # The same group as one of the k of a pixel whose other groups Coates
    # accepts. split_groups is swapped for one that returns these groups, so
    # precision meets exactly that group.
    rng = np.random.default_rng(6)
    weaker = np.rint(lib.analytic(_scene(irf, 2.0, t0), CYCLES, SIM_FIRST_PHOTON))
    groups = np.vstack([group, rng.binomial(weaker.astype(np.int64), 1 / k,
                                            (k - 1, NBINS))]).astype(np.int64)
    assert (groups[1:].sum(axis=1) < quota).all()
    whole = groups.sum(axis=0)
    split, S.split_groups = S.split_groups, lambda hist, k, rng: groups
    try:
        rows, _ = S.precision(lib, tmpl, whole[None], whole.sum(keepdims=True), CYCLES, k, rng)
    finally:
        S.split_groups = split
    seen = [r for r in rows if r["groups"]]
    assert {r["estimator"]: r["failures"] for r in seen} == {
        "argmax": 0, "centroid": 0, "matched_coates": 1, "coates_poisson": 1,
        "first_photon": 0}, seen
    assert all(r["pixels"] == 1 for r in seen), seen  # 3 groups left still count


def test_each_estimator_is_judged_against_its_own_whole_estimate():
    # A spike 8 bins after the edge holds the fullest bin of the whole
    # histogram and of every group. The argmax agrees with itself, so it does
    # not fail, though it sits more than FAIL_BINS from the first photon t0.
    lib = Plidar()
    irf = lib.table(*S.flat_top(BIN))
    tmpl = _scene(irf, 1.0)
    k, t0 = 4, 300.4 * BIN
    rng = np.random.default_rng(8)
    base = np.rint(lib.analytic(_scene(irf, 0.5, t0), CYCLES, SIM_FIRST_PHOTON))
    groups = rng.binomial(base.astype(np.int64), 1 / k, (k, NBINS))
    groups[:, 308] += 60
    whole = groups.sum(axis=0)
    edge = lib.mle_first_photon(tmpl, whole.astype(np.float64), CYCLES, 1.0,
                                np.zeros(4 * NBINS)).t0
    assert 308.5 * BIN - edge > S.FAIL_BINS * BIN, edge / BIN
    split, S.split_groups = S.split_groups, lambda hist, k, rng: groups
    try:
        rows, _ = S.precision(lib, tmpl, whole[None], whole.sum(keepdims=True), CYCLES, k, rng)
    finally:
        S.split_groups = split
    argmax = [r for r in rows if r["groups"] and r["estimator"] == "argmax"]
    assert argmax[0]["failures"] == 0 and argmax[0]["pixels"] == 1, argmax


def test_precision_excludes_pixels_above_the_cycle_count():
    lib = Plidar()
    irf = lib.table(*S.flat_top(BIN))
    # M under the top of the classes, so that M and M + 1 counts sit inside
    # one and only the cycle count can set those pixels aside.
    m = 6800.0
    assert any(lo <= m < m + 1 < hi for lo, hi in S.CLASSES)
    rng = np.random.default_rng(7)
    hists = np.array([np.rint(lib.analytic(_scene(irf, s, t0), m, SIM_FIRST_PHOTON))
                      for s, t0 in zip(rng.uniform(0.3, 3.0, 20),
                                       rng.uniform(60, 400, 20) * BIN)]).astype(np.int64)
    # One pixel holds a count more than there were cycles and another exactly
    # as many, in bins no other pixel reaches, so any array of either handed
    # to the library shows there. The first photon model would take the
    # second, so only the rule total >= M keeps it out.
    hists[7] = 0
    hists[7, 600:608] = 850
    hists[7, 608] = 1
    hists[11] = 0
    hists[11, 620:628] = 850
    spy = _Spy(lib)
    rows, excluded = S.precision(spy, S.template_scene(irf, NBINS, BIN), hists,
                                 hists.sum(axis=1), m, 4, rng)
    assert excluded["at_or_above_cycles"] == 2, excluded
    assert spy.arrays and not any(a[600:].any() for a in spy.arrays if a.size == NBINS)
    assert sum(r["pixels"] for r in rows if r["estimator"] == "first_photon") == 18, rows


def test_is_single_return_tells_one_return_from_two():
    lib = Plidar()
    clean = _scene(lib.table(*S.flat_top(BIN, tail=0.0)), 1.0)
    tailed = _scene(lib.table(*S.flat_top(BIN)), 1.0)
    t0 = 200.3 * BIN
    # Both pulses of the pair carry the tail. The second sits on the tail of
    # the first, so it holds the highest bin and the edge lands on it. A
    # weaker second return leaves the edge on the first, so it has to be
    # caught after the edge.
    pair = lib.rate(tailed, t0) + lib.rate(tailed, t0 + 30 * BIN)
    weaker = lib.rate(tailed, t0) + 0.5 * lib.rate(tailed, t0 + 30 * BIN)
    cases = {"clean single pulse": (lib.rate(clean, t0), True),
             "two equal pulses 30 bins apart": (pair, False),
             "a second return at half height 30 bins later": (weaker, False),
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
