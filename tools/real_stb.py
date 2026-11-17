"""Precision of the depth estimators on single photon histograms, against
the first photon bound, with the cycle count and the response read out of
the histograms themselves.

The pixels are split in two. The even ones calibrate the method: the cycle
count M is the ceiling of their totals (stb_method.cycles_from_totals), and
the response is fitted to the even pixels that hold one return. The M at
which their flux classes agree on one shape after Coates is printed next
to it as a diagnostic. The odd ones measure the method:
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

--data runs the same steps on one recorded sample of SP-TransientBench,
which tools/fetch_stb.py downloads to a folder outside the repository. The
even columns of the sensor calibrate and the odd columns measure. A pixel
enters when the dataset has an offset for it (under 10 m), its total is
over 1500, its fullest bin lies between 26 and 639 and its total is under
M. An odd pixel must also hold a single return on its whole histogram after
Coates: the bound is the bound of one return, and on a pixel with two
surfaces the spread would measure how wrong that model is. Every pixel set
aside is counted. The Livox points recorded with the sample are a second,
coarser reference: the registration of the two sensors limits it to
centimetres. The derived files go to --out and carry the dataset's
licence, CC BY-NC-SA 4.0. --cycles-only stops once M is known and writes
nothing.

Usage::

    python tools/real_stb.py --synthetic [--seed 1]
    python tools/real_stb.py --data DIR --sample K [--cycles-only] [--out docs/real] [--seed 1]

Exits 1 when a check fails.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

import stb_method as S
from plidar import Plidar, tof_to_range

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIN_WIDTH = 750e-12  # the sensor's bin
NBINS = 672

# The recorded sensor: WIDTH by HEIGHT pixels, pixel (u, v) on row
# v*WIDTH + u of the histogram file. Its columns split the pixels in two.
WIDTH, HEIGHT = 256, 192
MAX_OFFSET = 10.0  # m; the offset map holds 1000 where it has no value
MIN_TOTAL = 1500
PEAK_BINS = (26, 639)  # the fullest bin, both ends included
# The dataset turns the fullest bin p into a range as (p - 17) c dt/2 - offset.
PEAK_ORIGIN = 17
SHIFTS = range(-8, 9)  # pixels the Livox projection may move either way
# A reference is compact on a pixel with at least 4 points whose quartiles
# lie closer than 5 cm, so that one surface fills the pixel.
COMPACT_POINTS, COMPACT_IQR = 4, 0.05
TAIL_BINS = (10, 20, 30)  # where the response's tail is reported, after its edge
BAND_ROWS = 24

LABELS = {"argmax": "fullest bin", "centroid": "centroid",
          "matched_coates": "matched filter on Coates",
          "coates_poisson": "Coates then Poisson", "first_photon": "first photon"}
COLOURS = {"argmax": "#737373", "centroid": "#c2410c", "matched_coates": "#0369a1",
           "coates_poisson": "#7c3aed", "first_photon": "#15803d"}

# The synthetic truth: totals from about 1550 to 6800, every flux class.
CYCLES = 7000
PIXELS = 3000
SIGNAL = (0.25, 3.6)  # photons per cycle, log-uniform
FIRST_BIN, LAST_BIN = 60, 560
INDEPENDENT = 400  # evaluation pixels for the split against independent groups


def single_returns(lib: Plidar, hists: np.ndarray, cycles: float):
    """Coates through the library on every row with the given cycles, the
    rising edge of each row and whether it holds a single return. A row at
    or above the cycles comes back NaN and is not a single return."""
    mu = S.coates_rows(lib, hists, cycles)
    edges = np.array([S.rising_edge(r) for r in mu])
    return mu, edges, np.array([S.is_single_return(r, e) for r, e in zip(mu, edges)])


def calibrate(lib: Plidar, hists: np.ndarray, totals: np.ndarray, bin_width: float):
    """M and the response from the calibration pixels, as (M, the collapse
    curve over M_GRID, the response as fit_response returns it, the number
    of single return pixels it was fitted to). M is the ceiling of the
    totals; the collapse curve is a diagnostic."""
    cycles = S.cycles_from_totals(totals)
    curve = S.estimate_cycles(lib, hists, totals)[1]
    mu, edges, single = single_returns(lib, hists, cycles)
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
    cycles, curve, response, single = calibrate(lib, hists[0::2], totals[0::2], BIN_WIDTH)
    l1 = S.l1_distance(response, true)
    print(f"M {cycles:.0f} from the totals (true {CYCLES}; collapse minimum "
          f"{S.M_GRID[np.argmin(curve)]}, a diagnostic); response fitted to {single} "
          f"single return pixels, L1 {l1:.4f} from the true one")

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


def parse_intrinsics(text: str, name: str = "p1") -> dict[str, float]:
    """The numbers of one block of the dataset's config.yaml.

    A block is a line "name:" followed by deeper indented "key: number"
    lines. A few lines do here what a YAML parser would, and the tools
    install none. The projection uses fx, fy, cx, cy, k1 and k2 and has no
    tangential term, so a block without them, or with p1 or p2 other than
    0, is refused.
    """
    lines = text.splitlines()
    head = [i for i, line in enumerate(lines) if line.strip() == f"{name}:"]
    if not head:
        raise ValueError(f"config.yaml has no block {name!r}")
    indent = len(lines[head[0]]) - len(lines[head[0]].lstrip())
    out = {}
    for line in lines[head[0] + 1:]:
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        key, _, value = line.partition(":")
        out[key.strip()] = float(value)
    if not {"fx", "fy", "cx", "cy", "k1", "k2"} <= out.keys() or out.get("p1") or out.get("p2"):
        raise ValueError(f"block {name!r} lacks fx, fy, cx, cy, k1 or k2, or has "
                         f"tangential distortion: {out}")
    return out


def to_sensor(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Livox points, one per row, in the frame of the single photon sensor.

    The published 3 by 4 matrix [R | t] takes the sensor to the Livox, P =
    R S + t, so S = R^-1 (P - t). R is a reflection, det close to -1, but two
    of its rows fall short of unit length by 0.08 percent: its transpose is
    not its inverse and would shorten two coordinates by 0.15 percent.
    """
    return np.linalg.solve(extrinsic[:, :3], (points - extrinsic[:, 3]).T).T


def load_sample(root: Path, sample: int) -> dict:
    """One sample of the DepthEstimate track and the calibration it needs.

    hist: (49152, 672) int64 counts, row v*256 + u. offsets: (49152,) in m,
    valid under MAX_OFFSET. livox: (N, 3) points in m, without the null
    rows and the points 30 m or more away. intrinsics: the "p1" block of
    config.yaml. extrinsic: (3, 4), sensor to Livox. The files are read as
    numbers only, and one of the wrong size is refused.
    """
    root = Path(root)
    path = root / "DepthEstimate" / "Histogram" / f"{sample}.txt"
    hist = np.fromfile(path, dtype=np.int64, sep=" ")
    if hist.size != WIDTH * HEIGHT * NBINS:
        raise ValueError(f"{path}: {hist.size} counts, expected {WIDTH * HEIGHT} rows of {NBINS}")
    offsets = np.loadtxt(root / "config" / "offset.txt")
    if offsets.shape != (WIDTH * HEIGHT,):
        raise ValueError(f"offset.txt: shape {offsets.shape}, expected ({WIDTH * HEIGHT},)")
    # X, Y and Z are columns 8 to 10 of the Livox export, in m.
    livox = np.loadtxt(root / "DepthEstimate" / "gt" / f"{sample}.csv", delimiter=",",
                       skiprows=1, usecols=(8, 9, 10), ndmin=2)
    norm = np.linalg.norm(livox, axis=1)
    ext = np.loadtxt(root / "config" / "final_extrinsic_sp_to_livox.txt", comments="#")
    if ext.shape != (4, 4) or not np.array_equal(ext[3], [0, 0, 0, 1]):
        raise ValueError(f"the extrinsic is not 4 by 4 with a last row 0 0 0 1: {ext}")
    return {"hist": hist.reshape(-1, NBINS), "offsets": offsets,
            "livox": livox[(norm > 0.0) & (norm < 30.0)],
            "intrinsics": parse_intrinsics((root / "config" / "config.yaml")
                                           .read_text(encoding="utf-8")),
            "extrinsic": ext[:3]}


def livox_reference(data: dict, du: int, dv: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel, the median range of the Livox points that land on it, the
    distance between their quartiles and their number, with the projection
    moved du columns right and dv rows down.

    The points go into the sensor frame (to_sensor), and those in front of
    it are projected with the radial distortion of the intrinsics and
    rounded to the nearest pixel. Quantiles interpolate between order
    statistics, as numpy.percentile does. NaN where no point lands.
    """
    k = data["intrinsics"]
    s = to_sensor(data["livox"], data["extrinsic"])
    s = s[s[:, 2] > 0.0]
    x, y = s[:, 0] / s[:, 2], s[:, 1] / s[:, 2]
    r2 = x * x + y * y
    d = 1.0 + k["k1"] * r2 + k["k2"] * r2 * r2
    u = np.rint(k["fx"] * x * d + k["cx"]).astype(int) + du
    v = np.rint(k["fy"] * y * d + k["cy"]).astype(int) + dv
    ok = (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    pix, rng = v[ok] * WIDTH + u[ok], np.linalg.norm(s[ok], axis=1)
    order = np.lexsort((rng, pix))
    pix, rng = pix[order], rng[order]
    first = np.flatnonzero(np.diff(pix, prepend=-1))
    n = np.diff(np.append(first, pix.size))

    def quantile(f: float) -> np.ndarray:
        at = first + f * (n - 1)
        lo = np.floor(at).astype(int)
        hi = np.minimum(lo + 1, first + n - 1)
        return rng[lo] + (at - lo) * (rng[hi] - rng[lo])

    med, iqr = np.full(WIDTH * HEIGHT, np.nan), np.full(WIDTH * HEIGHT, np.nan)
    cnt = np.zeros(WIDTH * HEIGHT, dtype=int)
    med[pix[first]] = quantile(0.5)
    iqr[pix[first]] = quantile(0.75) - quantile(0.25)
    cnt[pix[first]] = n
    return med, iqr, cnt


def compact(iqr: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    return (cnt >= COMPACT_POINTS) & (iqr < COMPACT_IQR)  # NaN compares false


def residual_stats(r: np.ndarray) -> tuple[int, float, float, float]:
    """Over the finite values of r, in m: how many, their median, 1.4826
    times their median absolute deviation from it (the standard deviation of
    a normal sample) and the share within 5 cm of the median."""
    r = r[np.isfinite(r)]
    m = float(np.median(r))
    d = np.abs(r - m)
    return r.size, m, float(1.4826 * np.median(d)), float(np.mean(d < 0.05))


def find_shift(data: dict, sel: np.ndarray, ranges: np.ndarray) -> tuple[int, int, float, int]:
    """The du, dv of SHIFTS at which ranges - offset - reference spreads
    least over the pixels of sel with a compact reference, as (du, dv, that
    spread, the pixels it is taken over)."""
    best = (np.inf, 0, 0, 0)
    for du in SHIFTS:
        for dv in SHIFTS:
            med, iqr, cnt = livox_reference(data, du, dv)
            ok = sel & compact(iqr, cnt)
            n, _, spread, _ = residual_stats(ranges[ok] - data["offsets"][ok] - med[ok])
            if spread < best[0]:
                best = (spread, du, dv, n)
    return best[1], best[2], best[0], best[3]


def response_shape(response) -> tuple[float, float, np.ndarray]:
    """Width at half height and 10 to 90 percent rise of a response as
    fit_response returns it, in bins, and its density TAIL_BINS after the 50
    percent edge, as shares of the peak."""
    s, _, dt = response
    per_bin = BIN_WIDTH / dt
    up = S.rising_edge(s)
    # The falling edge is the rising edge of the reversed response.
    fwhm = (s.size - 1 - S.rising_edge(s[::-1]) - up) / per_bin
    rise = (S.rising_edge(s, 0.9) - S.rising_edge(s, 0.1)) / per_bin
    tail = np.interp(up + np.array(TAIL_BINS) * per_bin, np.arange(s.size), s) / s.max()
    return fwhm, rise, tail


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}")


def shapes_figure(before: np.ndarray, after: np.ndarray, cycles: float, out: Path) -> None:
    """Mean shape of each flux class, aligned on its rising edge, as
    recorded and after Coates with the M the classes agree on."""
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9), sharey=True)
    x = np.arange(before.shape[1]) - 6  # stb_method.class_shapes starts 6 bins early
    colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(S.CLASSES)))
    for ax, shapes, title in ((axes[0], before, "as recorded"),
                              (axes[1], after, f"after Coates, M = {cycles:.0f}")):
        for (lo, hi), s, c in zip(S.CLASSES, shapes, colours):
            ax.plot(x, s, lw=1.4, color=c, label=f"{lo}-{hi}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("bins from the rising edge", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("share of the window", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, title="photons per pixel", title_fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def precision_figure(rows: list[dict], out: Path) -> None:
    """Spread over bound against the mean photons of each flux class, one
    line per estimator."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="first photon bound")
    for name in S.ESTIMATORS:
        r = [x for x in rows if x["estimator"] == name]
        ax.plot([x["mean_photons"] for x in r], [x["ratio"] for x in r], "-o", ms=4, lw=1.6,
                color=COLOURS[name], label=LABELS[name])
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10], labels=["1", "2", "3", "5", "10"])
    ax.set_xlabel("recorded photons per pixel, class mean", fontsize=10)
    ax.set_ylabel("spread over bound", fontsize=10)
    ax.set_title(f"Spread over {S.K_GROUPS} groups of each pixel, against the bound "
                 "for one group", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=9)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def print_cycles(cycles: float, totals: np.ndarray, curve: np.ndarray) -> None:
    """M from the totals, after the collapse curve and its minimum, which
    are a diagnostic."""
    low = S.M_GRID[np.argmin(curve)]
    print("distance between the class shapes after Coates, against M (a diagnostic):")
    for m, d in zip(S.M_GRID, curve):
        print(f"  {m:5d}  {d:.4f}" + ("  <- collapse minimum" if m == low else ""))
    print(f"M {cycles:.0f}, from the totals of the valid even pixels (99.9th percentile "
          f"{np.percentile(totals, 99.9):.0f}); collapse minimum {low}, a diagnostic")


def real(root: Path, sample: int, out: Path, cycles_only: bool, seed: int) -> int:
    start = time.perf_counter()
    lib = Plidar()
    data = load_sample(root, sample)
    hist, offsets = data["hist"], data["offsets"]
    totals, peak = hist.sum(axis=1), hist.argmax(axis=1)
    u = np.arange(hist.shape[0]) % WIDTH
    even, odd = u % 2 == 0, u % 2 == 1
    ok_offset = offsets < MAX_OFFSET
    ok_total = totals > MIN_TOTAL
    ok_peak = (peak >= PEAK_BINS[0]) & (peak <= PEAK_BINS[1])
    base = ok_offset & ok_total & ok_peak
    print(f"sample {sample}: {hist.shape[0]} pixels, {data['livox'].shape[0]} Livox points")

    cal = base & even
    if cycles_only:
        print("even pixels by search class: " + ", ".join(
            f"{lo}-{hi} {int((cal & (totals >= lo) & (totals < hi)).sum())}"
            for lo, hi in S.SEARCH_CLASSES))
        print_cycles(S.cycles_from_totals(totals[cal]), totals[cal],
                     S.estimate_cycles(lib, hist[cal], totals[cal])[1])
        print(f"ran in {time.perf_counter() - start:.0f} s")
        return 0

    # Even columns: M and the response.
    cycles, curve, response, fitted = calibrate(lib, hist[cal], totals[cal], BIN_WIDTH)
    print_cycles(cycles, totals[cal], curve)
    fwhm, rise, tail = response_shape(response)
    print(f"response fitted to {fitted} even single return pixels: width at half height "
          f"{fwhm:.2f} bins ({fwhm * BIN_WIDTH * 1e9:.2f} ns), 10 to 90 percent rise "
          f"{rise:.2f} bins, density " + ", ".join(
              f"{100 * t:.1f}%" for t in tail)
          + " of the peak at " + ", ".join(str(b) for b in TAIL_BINS) + " bins after the edge")

    # Odd columns: the pixels the precision is measured on.
    under = totals < cycles
    ev = base & under & odd
    _, _, single = single_returns(lib, hist[ev], cycles)
    pix = np.flatnonzero(ev)[single]
    print(f"{'pixels set aside':<34}{'even':>7}{'odd':>7}")
    for name, out_mask in (("offset 10 m or more", ~ok_offset),
                           (f"total {MIN_TOTAL} or less", ok_offset & ~ok_total),
                           (f"fullest bin outside {PEAK_BINS[0]}-{PEAK_BINS[1]}",
                            ok_offset & ok_total & ~ok_peak),
                           ("total at or above M", base & ~under)):
        print(f"  {name:<32}{int((out_mask & even).sum()):>7}{int((out_mask & odd).sum()):>7}")
    print(f"  {'not a single return':<32}{int((cal & under).sum()) - fitted:>7}"
          f"{int(ev.sum()) - pix.size:>7}")
    print(f"{'left':<34}{fitted:>7}{pix.size:>7}")

    tmpl = S.template_scene(lib.table(*response), NBINS, BIN_WIDTH)
    rows, excluded = S.precision(lib, tmpl, hist[pix], totals[pix], cycles, S.K_GROUPS,
                                 np.random.default_rng(seed))
    for r in rows:
        t = totals[pix][(totals[pix] >= r["class_lo"]) & (totals[pix] < r["class_hi"])]
        r["mean_photons"] = float(t.mean()) if t.size else float("nan")
    print_table(rows, excluded)

    # The Livox reference. The shift is chosen on the even columns with the
    # fullest bin, in the dataset's own conversion to range, which the offset
    # map is calibrated for. The other estimators place t0 on the rising
    # edge, a constant away from that conversion, so each one's median is
    # removed and only its spread compares.
    fullest = tof_to_range((peak - PEAK_ORIGIN) * BIN_WIDTH)
    du, dv, spread, n = find_shift(data, cal & under, fullest)
    print(f"Livox projection moved {du} columns and {dv} rows: spread of the fullest bin "
          f"{100 * spread:.1f} cm over {n} even pixels with a compact reference")
    med, iqr, cnt = livox_reference(data, du, dv)
    near = pix[compact(iqr[pix], cnt[pix])]
    ref = offsets[near] + med[near]
    work = np.zeros(4 * NBINS)
    est = [S.estimate_all(lib, tmpl, np.ascontiguousarray(hist[p], dtype=np.float64), cycles,
                          1.0, work) for p in near]
    res = {name: (fullest[near] if name == "argmax"
                  else tof_to_range(np.array([e[name] for e in est]))) - ref
           for name in S.ESTIMATORS}
    band = near // WIDTH // BAND_ROWS
    livox_rows = []
    print(f"{'estimator':<16}{'pixels':>7}{'median cm':>11}{'spread cm':>11}{'within 5 cm':>13}"
          + "".join(f"{f'{b}-{b + BAND_ROWS - 1}':>9}" for b in range(0, HEIGHT, BAND_ROWS)))
    removed = {}
    for name in S.ESTIMATORS:
        count, m, sd, within = residual_stats(res[name])
        if name != "argmax":
            removed[name], m = m, 0.0
        centred = res[name] - removed.get(name, 0.0)
        bands = [100 * np.nanmedian(centred[band == b]) if np.isfinite(centred[band == b]).any()
                 else float("nan") for b in range(HEIGHT // BAND_ROWS)]
        print(f"{name:<16}{count:>7}{100 * m:>11.1f}{100 * sd:>11.1f}{100 * within:>12.0f}%"
              + "".join(f"{x:>9.1f}" for x in bands))
        livox_rows.append([name, count, f"{100 * m:.2f}", f"{100 * sd:.2f}", f"{100 * within:.1f}"])
    print("medians removed, cm: " + ", ".join(f"{k} {100 * v:.1f}" for k, v in removed.items()))
    print("the bands are medians over rows of the sensor, in cm")

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "collapse.csv", ["cycles", "distance"],
              [[int(m), f"{d:.5f}"] for m, d in zip(S.M_GRID, curve)])
    write_csv(out / "precision.csv",
              ["class_lo", "class_hi", "estimator", "pixels", "groups", "failures", "ratio",
               "bound_mm", "mean_photons"],
              [[r["class_lo"], r["class_hi"], r["estimator"], r["pixels"], r["groups"],
                r["failures"], f"{r['ratio']:.4f}", f"{r['bound_mm']:.4f}",
                f"{r['mean_photons']:.0f}"] for r in rows])
    write_csv(out / "livox.csv", ["estimator", "pixels", "median_cm", "spread_cm", "within5_pct"],
              livox_rows)

    # The shapes the collapse judges, over every class, at the M found.
    shown = cal & under
    raw = hist[shown].astype(np.float64)
    before = S.class_shapes(raw, np.array([S.rising_edge(r) for r in raw]), totals[shown],
                            S.CLASSES)
    mu, edges, _ = single_returns(lib, hist[shown], cycles)
    after = S.class_shapes(mu, edges, totals[shown], S.CLASSES)
    print(f"distance between the {len(S.CLASSES)} class shapes: "
          f"{S.shape_distance(before):.3f} as recorded, {S.shape_distance(after):.3f} after Coates")
    shapes_figure(before, after, cycles, out / "shapes.png")
    precision_figure(rows, out / "precision.png")
    print(f"ran in {time.perf_counter() - start:.0f} s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true",
                      help="run the method on simulated pixels and check it")
    mode.add_argument("--data", type=Path, help="folder that tools/fetch_stb.py filled")
    ap.add_argument("--sample", type=int, help="sample of the DepthEstimate track, with --data")
    ap.add_argument("--cycles-only", action="store_true",
                    help="with --data, stop once M is known and write nothing")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "real",
                    help="where --data writes its files")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.synthetic:
        return synthetic(a.seed)
    if a.sample is None:
        ap.error("--data needs --sample")
    return real(a.data, a.sample, a.out, a.cycles_only, a.seed)


if __name__ == "__main__":
    sys.exit(main())
