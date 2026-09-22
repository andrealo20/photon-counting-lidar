"""Check that the tables in the README still match the data they came from.

A table edited by hand, or left behind when a sweep was rerun, is the easiest
way for a repository to end up making a claim its own code does not support.
This reads the two result tables out of README.md and compares every number
against docs/data/*.csv. It needs neither the library nor a build.

Usage::

    python tools/check_readme.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DATA = ROOT / "docs" / "data"

# The README quotes two decimals, so a disagreement below half of the last
# place quoted is rounding rather than drift.
TOL_MM = 0.006
TOL_PCT = 0.06

ESTIMATOR_COLUMNS = ["mle", "matched", "centroid"]


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def value_and_lost(cell: str) -> tuple[float, float | None]:
    """Parse "5.98 mm (0.5% lost)" or "19.62 mm"."""
    m = re.match(r"^(-?[\d.]+)\s*mm(?:\s*\(([\d.]+)%\s*lost\))?$", cell)
    if not m:
        raise ValueError(f"cannot read {cell!r}")
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


def load(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def check_estimator_table(text: str, problems: list[str]) -> int:
    rows = load("photons.csv")
    checked = 0

    for line in text.splitlines():
        if not line.startswith("| ") or " mm |" not in line:
            continue
        parts = cells(line)
        if len(parts) != 5 or not parts[0].isdigit():
            continue

        photons = parts[0]
        matching = {r["estimator"]: r for r in rows if r["photons"] == photons}
        if not matching:
            fail(problems, f"README quotes {photons} photons, csv has no such row")
            continue

        bound, _ = value_and_lost(parts[1])
        csv_bound = float(matching["mle"]["crb_mm"])
        if abs(bound - csv_bound) > TOL_MM:
            fail(problems,
                 f"{photons} photons: bound {bound} against {csv_bound:.4f}")
        checked += 1

        for column, estimator in zip(parts[2:], ESTIMATOR_COLUMNS):
            quoted, lost = value_and_lost(column)
            row = matching[estimator]
            actual = float(row["inlier_rmse_mm"])
            if abs(quoted - actual) > TOL_MM:
                fail(problems,
                     f"{photons} photons, {estimator}: {quoted} against "
                     f"{actual:.4f}")
            actual_lost = 100.0 * float(row["outlier_fraction"])
            if lost is None:
                if actual_lost > TOL_PCT:
                    fail(problems,
                         f"{photons} photons, {estimator}: README shows no "
                         f"losses, csv has {actual_lost:.2f}%")
            elif abs(lost - actual_lost) > TOL_PCT:
                fail(problems,
                     f"{photons} photons, {estimator}: {lost}% lost against "
                     f"{actual_lost:.2f}%")
            checked += 1
    return checked


def check_pileup_table(text: str, problems: list[str]) -> int:
    rows = load("pileup.csv")
    checked = 0

    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        parts = cells(line)
        if len(parts) != 3 or not parts[0].endswith("%"):
            continue

        detections = float(parts[0].rstrip("%"))
        candidates = [r for r in rows
                      if abs(100.0 * float(r["detections_per_cycle"])
                             - detections) < 0.05]
        if len(candidates) != 1:
            fail(problems,
                 f"README quotes {parts[0]} detections, csv has "
                 f"{len(candidates)} matching rows")
            continue
        row = candidates[0]

        quoted, _ = value_and_lost(parts[1])
        actual = float(row["raw_bias_mm"])
        if abs(quoted - actual) > TOL_MM:
            fail(problems, f"{parts[0]}: bias {quoted} against {actual:.4f}")
        checked += 1

        corrected = abs(float(row["corrected_bias_mm"]))
        if corrected > 1e-3:
            fail(problems,
                 f"{parts[0]}: corrected bias {corrected:.5f} mm is no longer "
                 "below a micrometre, the README says it is")
        checked += 1
    return checked


def main() -> int:
    text = README.read_text(encoding="utf-8")
    problems: list[str] = []

    checked = check_estimator_table(text, problems)
    checked += check_pileup_table(text, problems)

    if checked == 0:
        print("no tables found in the README, the parser or the README moved")
        return 1
    for problem in problems:
        print(f"mismatch: {problem}")
    print(f"{checked} values checked, {len(problems)} mismatched")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
