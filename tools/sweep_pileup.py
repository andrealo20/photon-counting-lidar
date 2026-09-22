"""How far pile up shortens a range, and how much of it Coates gives back.

No noise is involved. The histogram is the exact expected one under the
first photon rule, so every number below is the systematic error alone, with
the statistical part removed. That is the right way to look at a bias: a
longer integration does not reduce it.

Usage::

    python tools/sweep_pileup.py

Writes docs/data/pileup.csv and prints the table the README quotes.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from plidar import SIM_FIRST_PHOTON, Plidar, tof_to_range

PERIOD = 100e-9
BIN_WIDTH = 50e-12
T0 = 37.3e-9
EFFICIENCY = 0.3
SIGMA = 60e-12
TAU = 140e-12
CYCLES = 1e7

# Signal photons per cycle and background rate, walking from a rate a careful
# operator would accept up to one nobody would.
POINTS = [
    (0.05, 1.0e6),
    (0.15, 3.0e6),
    (0.50, 5.0e6),
    (1.00, 8.0e6),
    (2.00, 1.0e7),
    (4.00, 1.5e7),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "docs" / "data" / "pileup.csv")
    args = ap.parse_args()

    lib = Plidar()
    irf = Plidar.emg(SIGMA, TAU)
    rows = []

    print(f"{'det/cycle':>10} {'raw bias':>12} {'corrected':>14}")
    for signal, background in POINTS:
        scene = Plidar.scene(period=PERIOD, bin_width=BIN_WIDTH, t0=T0,
                             signal=signal, background=background,
                             efficiency=EFFICIENCY, irf=irf)
        lib.check(scene)
        scratch = np.zeros(2 * scene.nbins, dtype=np.float64)

        distorted = lib.analytic(scene, CYCLES, SIM_FIRST_PHOTON)
        raw = lib.mle(scene, distorted, CYCLES, scratch)

        # Coates takes a recorded histogram and returns rates per cycle, so
        # the cycles go back on before the estimator sees it.
        as_counts = np.rint(distorted).astype(np.uint64)
        restored = lib.coates(as_counts, int(CYCLES)) * CYCLES
        fixed = lib.mle(scene, restored, CYCLES, scratch)

        detections = 1.0 - math.exp(
            -EFFICIENCY * (signal + background * PERIOD))
        rows.append({
            "signal_per_cycle": signal,
            "background_per_s": background,
            "detections_per_cycle": detections,
            "raw_bias_mm": tof_to_range(raw.t0 - T0) * 1e3,
            "corrected_bias_mm": tof_to_range(fixed.t0 - T0) * 1e3,
        })
        print(f"{detections:10.3f} {rows[-1]['raw_bias_mm']:+9.3f} mm "
              f"{rows[-1]['corrected_bias_mm']:+11.5f} mm")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
