"""Turn the sweep results into the figures the README shows.

Reads docs/data/*.csv, writes docs/images/*.png. Nothing is computed here
beyond what is already in the csv files, so a figure and the numbers quoted
next to it cannot disagree.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
IMAGES = ROOT / "docs" / "images"

LABELS = {
    "centroid": "centroid",
    "matched": "matched filter",
    "mle": "maximum likelihood",
}
COLOURS = {"centroid": "#c2410c", "matched": "#0369a1", "mle": "#15803d"}


def read(path: Path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def photons_figure(rows, out: Path) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["estimator"]].append(r)
    for values in by.values():
        values.sort(key=lambda r: int(r["photons"]))

    # Kept small on purpose. A figure that renders at 900 px in a README
    # fills the column and pushes everything else off the first screen, so
    # the canvas is sized for the width it will actually be shown at and the
    # text is scaled to stay legible there.
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    bound = by["mle"]
    ax.plot([int(r["photons"]) for r in bound],
            [float(r["crb_mm"]) for r in bound],
            "k--", lw=1.4, label="Cramer Rao bound", zorder=5)

    for name, values in by.items():
        x = [int(r["photons"]) for r in values]
        ax.plot(x, [float(r["inlier_rmse_mm"]) for r in values], "-o",
                ms=4, lw=1.6, color=COLOURS[name], label=LABELS[name])
        ax2.plot(x, [100.0 * float(r["outlier_fraction"]) for r in values],
                 "-o", ms=4, lw=1.6, color=COLOURS[name])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("depth error, mm", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Depth error against photon budget, signal to background 1",
                 fontsize=10)
    ax.tick_params(labelsize=9)

    ax2.set_xscale("log")
    ax2.set_xlabel("detected signal photons", fontsize=10)
    ax2.set_ylabel("lost, %", fontsize=10)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.tick_params(labelsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--images", type=Path, default=IMAGES)
    args = ap.parse_args()

    photons = args.data / "photons.csv"
    if photons.exists():
        photons_figure(read(photons), args.images / "photons.png")
    else:
        print(f"{photons} not found, run tools/sweep_photons.py first")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
