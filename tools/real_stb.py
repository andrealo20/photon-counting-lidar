"""Precision of the depth estimators on single photon histograms, against
the first photon bound, with the cycle count and the response read out of
the histograms themselves.

The pixels are split in two. The even ones calibrate the method: the cycle
count M is where the flux classes agree on one shape, and the response is
fitted to the even pixels that hold one return. The odd ones measure it:
each histogram is split into groups, every estimator runs on every group,
and the spread over the groups is set against the first photon bound for
one group (stb_method.precision).

--synthetic runs the same steps on pixels the library simulates, with a
known M and a known response, and checks the method against them:

  * M within 3 percent of the truth;
  * the fitted response within 0.05 of the true one in L1;
  * the first photon ratio of spread to bound between 0.90 and 1.15 in
    every flux class;
  * the first photon spread over the groups of one split within 10 percent
    of its spread over groups thinned from independent histograms. The
    groups of a split share the cycles of one acquisition, and this is how
    much that weighs.

The bands were fixed before the method saw any recorded data and are not
moved to make a run pass. The other estimators are printed without a band:
the bound holds for unbiased estimators, and some of them are not.

Usage::

    python tools/real_stb.py --synthetic [--seed 1]

Exits 1 when a check fails.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import stb_method as S
from plidar import Plidar

BIN_WIDTH = 750e-12  # the sensor's bin
NBINS = 672

# The synthetic truth: totals from about 1550 to 6800, every flux class.
CYCLES = 7000
PIXELS = 3000
SIGNAL = (0.25, 3.6)  # photons per cycle, log-uniform
FIRST_BIN, LAST_BIN = 60, 560
INDEPENDENT = 400  # evaluation pixels for the split against independent groups


def calibrate(lib: Plidar, hists: np.ndarray, totals: np.ndarray, bin_width: float):
    """M and the response from the calibration pixels, as (M, the collapse
    curve over M_GRID, the response as fit_response returns it, the number
    of single return pixels it was fitted to)."""
    cycles, curve = S.estimate_cycles(lib, hists, totals)
    mu = S.coates_rows(lib, hists, cycles)
    edges = np.array([S.rising_edge(r) for r in mu])
    single = np.array([S.is_single_return(r, e) for r, e in zip(mu, edges)])
    return cycles, curve, S.fit_response(mu[single], edges[single], bin_width), int(single.sum())


def print_table(rows: list[dict], excluded: dict) -> None:
    print(f"{'class':>11}  {'estimator':<15}{'pixels':>7}{'groups':>8}{'failed':>8}"
          f"{'share':>8}{'ratio':>8}{'bound mm':>10}")
    for r in rows:
        share = r["failures"] / r["groups"] if r["groups"] else float("nan")
        print(f"{r['class_lo']:>5}-{r['class_hi']:<5}  {r['estimator']:<15}{r['pixels']:>7}"
              f"{r['groups']:>8}{r['failures']:>8}{share:>8.1%}{r['ratio']:>8.3f}"
              f"{r['bound_mm']:>10.2f}")
    print("excluded: " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in excluded.items()))


def split_over_independent(lib: Plidar, tmpl, scenes, hists: np.ndarray, cycles: float,
                           k: int, rng: np.random.Generator, sim) -> float:
    """Pooled first photon sd over the k groups of one split of each
    histogram, over the pooled sd over k groups each thinned from a new
    independent histogram of the same scene. A pixel enters only when every
    one of its 2k estimates is within FAIL_BINS of the true t0."""
    det = Plidar.detector(first_photon_only=True)
    work = np.zeros(4 * tmpl.nbins)
    keep = 1.0 / k
    pooled = np.zeros(2)
    for sc, h in zip(scenes, hists):
        fresh = [lib.simulate(sc, det, CYCLES, sim)[0].astype(np.int64) for _ in range(k)]
        sides = [S.split_groups(h, k, rng), rng.binomial(fresh, keep)]
        t = np.array([[S.estimate_all(lib, tmpl, g, cycles, keep, work)["first_photon"]
                       for g in np.ascontiguousarray(side, dtype=np.float64)]
                      for side in sides])
        if (np.abs(t - sc.t0) <= S.FAIL_BINS * tmpl.bin_width).all():
            pooled += t.var(axis=1, ddof=1)
    return float(np.sqrt(pooled[0] / pooled[1]))


def synthetic(seed: int) -> int:
    start = time.perf_counter()
    lib = Plidar()
    true = S.flat_top(BIN_WIDTH)  # unit = one bin
    irf = lib.table(*true)
    rng = np.random.default_rng(seed)
    sim = lib.seed(seed, 0)
    det = Plidar.detector(first_photon_only=True)  # no dead time, no afterpulsing
    signal = np.exp(rng.uniform(np.log(SIGNAL[0]), np.log(SIGNAL[1]), PIXELS))
    t0 = rng.uniform(FIRST_BIN, LAST_BIN, PIXELS) * BIN_WIDTH
    scenes = [Plidar.scene(period=NBINS * BIN_WIDTH, bin_width=BIN_WIDTH, t0=t,
                           signal=s, background=0.0, efficiency=1.0, irf=irf)
              for s, t in zip(signal, t0)]
    hists = np.array([lib.simulate(sc, det, CYCLES, sim)[0] for sc in scenes]).astype(np.int64)
    totals = hists.sum(axis=1)

    # Even pixels calibrate and odd pixels measure, as the columns of the
    # recorded data. The truth is read only to compare.
    cycles, _, response, single = calibrate(lib, hists[0::2], totals[0::2], BIN_WIDTH)
    l1 = S.l1_distance(response, true)
    print(f"M {cycles:.0f} (true {CYCLES}); response fitted to {single} single return "
          f"pixels, L1 {l1:.4f} from the true one")

    tmpl = S.template_scene(lib.table(*response), NBINS, BIN_WIDTH)
    rows, excluded = S.precision(lib, tmpl, hists[1::2], totals[1::2], cycles,
                                 S.K_GROUPS, rng)
    print_table(rows, excluded)

    picked = [p for p in range(1, PIXELS, 2) if totals[p] < cycles][:INDEPENDENT]
    independent = split_over_independent(lib, tmpl, [scenes[p] for p in picked],
                                         hists[picked], cycles, S.K_GROUPS, rng,
                                         lib.seed(seed, 1))

    first = [r["ratio"] for r in rows if r["estimator"] == "first_photon"]
    checks = [
        (f"|M/{CYCLES} - 1| = {abs(cycles / CYCLES - 1):.3f} <= 0.03",
         abs(cycles / CYCLES - 1) <= 0.03),
        (f"response L1 {l1:.4f} <= 0.05", l1 <= 0.05),
        ("first photon ratio in [0.90, 1.15] in every class: "
         + ", ".join(f"{r:.3f}" for r in first), all(0.90 <= r <= 1.15 for r in first)),
        (f"first photon sd, split over independent, {len(picked)} pixels: "
         f"{independent:.3f} in [0.90, 1.10]", 0.90 <= independent <= 1.10),
    ]
    for name, ok in checks:
        print(("pass  " if ok else "FAIL  ") + name)
    print(f"ran in {time.perf_counter() - start:.0f} s")
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--synthetic", action="store_true",
                    help="run the method on simulated pixels and check it")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if not a.synthetic:
        ap.error("pass --synthetic")
    return synthetic(a.seed)


if __name__ == "__main__":
    sys.exit(main())
