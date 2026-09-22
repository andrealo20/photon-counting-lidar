"""Depth error against photon budget, measured against the bound.

This is the headline experiment. The scene is held fixed, the number of
signal photons collected is swept over four decades, and each estimator is
run over many independent realisations of the same scene. What comes out is
three curves and one bound:

  * above a threshold the maximum likelihood error sits on the Cramer Rao
    bound, and the precision is far finer than the width of the instrument
    response;

  * below it the likelihood occasionally has a larger maximum on a
    background fluctuation, the estimate lands metres away, and the root
    mean square error leaves the bound behind. Reporting the error of the
    realisations that did not do that, separately from how often it
    happened, says considerably more than one averaged number.

Usage::

    python tools/sweep_photons.py --trials 400 --sbr 1.0

Results land in docs/data/photons.csv, which tools/make_figures.py turns
into the plot in the README.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from plidar import Plidar, PlidarError, tof_to_range

PERIOD = 100e-9      # 10 MHz repetition, just under 15 m unambiguous
BIN_WIDTH = 50e-12
T0 = 37.3e-9         # 5.59 m
EFFICIENCY = 0.3
SIGMA = 60e-12       # jitter of the avalanche and the timing chain
TAU = 140e-12        # diffusion tail
CYCLES = 1_000_000

# An estimate further out than this is not a noisy version of the right
# answer, it is a different peak. The split is made at twenty times the width
# of the response, which is far outside anything the bound would allow and
# well inside the period.
OUTLIER_TOF = 20.0 * math.sqrt(SIGMA * SIGMA + TAU * TAU)


def attempt(fn) -> float:
    """Run an estimator, treating a refusal as a lost realisation.

    With one photon in the whole histogram the centroid has nothing to take
    a mean of and says so rather than returning a number. That is the same
    kind of failure as landing on the wrong peak and is counted the same
    way, because a rangefinder that returns nothing and one that returns a
    distance from the wrong surface are both wrong.
    """
    try:
        return fn().t0 - T0
    except PlidarError:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--sbr", type=float, default=1.0,
                    help="signal photons over background photons per cycle")
    ap.add_argument("--points", type=int, default=13)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "docs" / "data" / "photons.csv")
    args = ap.parse_args()

    lib = Plidar()
    irf = Plidar.emg(SIGMA, TAU)
    detector = Plidar.detector(first_photon_only=True)

    budgets = np.unique(np.round(
        np.logspace(0, 4, args.points)).astype(int))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for budget in budgets:
        signal_per_cycle = budget / (CYCLES * EFFICIENCY)
        background_per_cycle = signal_per_cycle / args.sbr
        scene = Plidar.scene(
            period=PERIOD,
            bin_width=BIN_WIDTH,
            t0=T0,
            signal=signal_per_cycle,
            background=background_per_cycle / (EFFICIENCY * PERIOD),
            efficiency=EFFICIENCY,
            irf=irf,
        )
        lib.check(scene)

        scratch = np.zeros(2 * scene.nbins, dtype=np.float64)
        crb = lib.crb(scene, float(CYCLES), scratch)

        errors = {"centroid": [], "matched": [], "mle": []}
        rng = lib.seed(args.seed + int(budget), 0)

        for _ in range(args.trials):
            hist, _recorded = lib.simulate(scene, detector, CYCLES, rng)
            counts = hist.astype(np.float64)

            errors["centroid"].append(attempt(
                lambda: lib.centroid(scene, counts,
                                     3.0 * math.hypot(SIGMA, TAU))))
            errors["matched"].append(attempt(
                lambda: lib.matched(scene, counts, scratch)))
            errors["mle"].append(attempt(
                lambda: lib.mle(scene, counts, float(CYCLES), scratch)))

        detections = (signal_per_cycle + background_per_cycle) * EFFICIENCY
        for name, values in errors.items():
            err = np.asarray(values)
            inliers = np.abs(err) < OUTLIER_TOF  # nan compares false, as wanted
            rows.append({
                "photons": int(budget),
                "sbr": args.sbr,
                "estimator": name,
                "trials": args.trials,
                "detections_per_cycle": detections,
                "rmse_mm": (
                    tof_to_range(float(np.sqrt(np.mean(err ** 2)))) * 1e3
                    if np.isfinite(err).all() else float("nan")),
                "inlier_rmse_mm": (
                    tof_to_range(float(np.sqrt(np.mean(err[inliers] ** 2)))) * 1e3
                    if inliers.any() else float("nan")),
                "bias_mm": tof_to_range(float(np.mean(err[inliers]))) * 1e3
                if inliers.any() else float("nan"),
                "outlier_fraction": float(1.0 - inliers.mean()),
                "crb_mm": tof_to_range(math.sqrt(crb.var_tof_joint)) * 1e3,
            })
        mle_row = rows[-1]
        print(f"{budget:6d} photons  "
              f"bound {mle_row['crb_mm']:8.4f} mm  "
              f"mle {mle_row['inlier_rmse_mm']:8.4f} mm inlier  "
              f"{mle_row['outlier_fraction'] * 100:5.1f}% lost")

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
