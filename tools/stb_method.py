"""The cycle count and the instrument response, read out of recorded
histograms alone.

A pixel of the sensor records at most one photon per laser cycle, and the
number of cycles M is not in the data. Coates undoes that pile up only with
the right M. Too large an M under-corrects and too small an M over-corrects,
and either way the corrected shape of a return still depends on how bright
the pixel is. So M is taken as the value at which pixels of
different flux agree on one shape (estimate_cycles). With M known, the
corrected rates of pixels that hold one return are aligned on their rising
edges and fitted with a piecewise linear density, which becomes the table
the library uses for the response (fit_response).

Every number this module produces about a histogram (Coates, rates, analytic
histograms) comes from the library. numpy does the bookkeeping, the
alignment, the least squares fit and the statistics.
"""

from __future__ import annotations

import numpy as np

from plidar import Plidar, PlidarError

# Flux classes, by total recorded counts per pixel. The search for M stops at
# 6400 so that every pixel it uses has cycles left at the smallest M tried.
CLASSES = [(1500, 3000), (3000, 4500), (4500, 6000), (6000, 6500), (6500, 7000)]
SEARCH_CLASSES = [(1500, 3000), (3000, 4500), (4500, 6000), (6000, 6400)]
M_GRID = np.arange(6500, 9001, 100)

# Weight of the second difference penalty in fit_response, to be fixed on the
# synthetic data before the method sees the real data. The data term counts
# pixels, so one value smooths less as the number of pixels grows.
SMOOTH = 10.0


def flat_top(unit: float, oversample: int = 8,
             tail: float = 0.04) -> tuple[np.ndarray, float, float]:
    """The flat topped response of tests/fixture_table.h, as (samples,
    t_first, dt) in seconds, ready for Plidar.table.

    With x = t/unit the density is r(x) * [f(x) + tail * (1 - f(x)) *
    exp(-max(x - 9, 0)/8)]. r rises from 10 to 90 percent in 5 units with its
    half height at x = 0, f ends a top about 9 units wide, and the last term
    is the long tail at about 4 percent of the peak. The nodes sit every
    1/oversample of a unit over [-6, 40], with both ends at zero. tail = 0
    gives the same pulse without its tail.
    """
    x = -6.0 + np.arange(46 * oversample + 1) / oversample
    r = 1.0 / (1.0 + np.exp(-x / 1.138))
    f = 1.0 / (1.0 + np.exp(x - 9.0))
    s = r * (f + tail * (1.0 - f) * np.exp(-np.maximum(x - 9.0, 0.0) / 8.0))
    s[0] = s[-1] = 0.0
    return s, -6.0 * unit, unit / oversample


def rising_edge(mu: np.ndarray, frac: float = 0.5) -> float:
    """Fractional index at which the rising edge of mu crosses frac times
    its maximum.

    The values sit at the integer indices and are joined by straight lines.
    The crossing is searched leftwards from the first maximum, so a later
    return of the same height does not move it. NaN when the edge runs into
    index 0 without crossing, which includes a row of NaN or of zeros.
    """
    m = int(np.argmax(mu))
    level = frac * mu[m]
    below = np.flatnonzero(mu[:m] < level)
    if below.size == 0:
        return float("nan")
    i = below[-1]
    return float(i + (level - mu[i]) / (mu[i + 1] - mu[i]))


def coates_rows(lib: Plidar, hist: np.ndarray, cycles: float) -> np.ndarray:
    """Coates through the library on every row of hist, in detected photons
    per cycle per bin. A row the library refuses, because the cycles run out
    inside it, comes back as NaN."""
    h = np.ascontiguousarray(hist, dtype=np.float64)
    mu = np.full(h.shape, np.nan)
    for p, row in enumerate(h):
        try:
            mu[p] = lib.coates_d(row, cycles)
        except PlidarError:
            pass
    return mu


def class_shapes(mu: np.ndarray, edges: np.ndarray, totals: np.ndarray,
                 classes, before: int = 6, after: int = 34) -> np.ndarray:
    """Mean shape of each flux class, aligned on the rising edge.

    A pixel contributes the window [floor(e) - before, floor(e) + after) of
    its rates, scaled to sum 1, to the class its total falls in; a class is
    the mean of those windows, and a row of NaN when it has none. Pixels
    without an edge, or whose window leaves the histogram, are skipped.
    """
    width = before + after
    ok = np.isfinite(edges)
    start = np.floor(np.where(ok, edges, 0.0)).astype(int) - before
    ok &= (start >= 0) & (start + width <= mu.shape[1])
    rows = np.flatnonzero(ok)
    win = mu[rows[:, None], start[rows, None] + np.arange(width)]
    win = win / win.sum(axis=1, keepdims=True)
    t = totals[rows]
    shapes = np.full((len(classes), width), np.nan)
    for c, (lo, hi) in enumerate(classes):
        sel = (t >= lo) & (t < hi)
        if sel.any():
            shapes[c] = win[sel].mean(axis=0)
    return shapes


def shape_distance(shapes: np.ndarray) -> float:
    """Mean L1 distance of the class shapes from their mean, over the
    classes that have pixels."""
    s = shapes[np.isfinite(shapes).all(axis=1)]
    return float(np.abs(s - s.mean(axis=0)).sum(axis=1).mean())


def estimate_cycles(lib: Plidar, hist: np.ndarray, totals: np.ndarray,
                    grid=M_GRID, classes=SEARCH_CLASSES) -> tuple[float, np.ndarray]:
    """The M of the grid at which the Coates corrected class shapes are
    closest to each other, and the distance at every M of the grid.

    Only pixels with a total under the top of the classes enter, 6400 for
    SEARCH_CLASSES. Every one of them then has cycles left at the smallest
    M of the grid, so Coates is defined for all of them at every M and the
    pixel set does not change along the curve.
    """
    keep = totals < max(hi for _, hi in classes)
    h, t = hist[keep], totals[keep]
    curve = np.empty(len(grid))
    for j, m in enumerate(grid):
        mu = coates_rows(lib, h, float(m))
        edges = np.array([rising_edge(r) for r in mu])
        curve[j] = shape_distance(class_shapes(mu, edges, t, classes))
    return float(grid[np.argmin(curve)]), curve


def is_single_return(mu: np.ndarray, edge: float) -> bool:
    """Whether a pixel holds one return, of the kind the response model
    describes.

    mu is the pixel's Coates corrected rate, in detected photons per cycle
    per bin: a one dimensional float array of nbins values. edge is the
    fractional bin index of the 50 percent point of its rising edge, as
    rising_edge returns it. Returns True when the pixel holds a single
    return, which fit_response may average into the response, and False
    otherwise, including when the edge or any rate is NaN.

    A single return leaves nothing above 10 percent of its peak outside the
    response: no bin before it, and no 5 bin average after its flat top. A
    second surface closer than about 14 bins (1.6 m) merges with the first
    and is not caught.
    """
    if np.isnan(edge):
        return False
    limit = 0.1 * mu.max()
    i = int(edge)
    # The response starts about 6 bins before its 50 percent edge. Before it
    # almost every cycle is still listening, so Coates is precise bin by bin.
    before = mu[:max(i - 6, 0)]
    # From 14 bins on, past the top and its fall, only the response's own 4
    # percent tail remains. A bright pixel has few cycles left to measure it
    # with, so the rates there are averaged over 5 bins before the test.
    after = np.convolve(mu, np.ones(5) / 5, mode="valid")[i + 14:]
    return bool((before < limit).all() and (after < limit).all())


def _hat_primitive(x: np.ndarray, h: float) -> np.ndarray:
    """Integral up to x of a hat of unit height and half width h, with x
    measured from the centre of the hat."""
    u = np.clip(x, -h, h)
    return np.where(u <= 0.0, (u + h) ** 2, 2.0 * h * h - (h - u) ** 2) / (2.0 * h)


def fit_response(mu: np.ndarray, edges: np.ndarray, bin_width: float,
                 oversample: int = 8, span: tuple[int, int] = (-6, 40),
                 smooth: float = SMOOTH) -> tuple[np.ndarray, float, float]:
    """The response density, fitted to pixels aligned on their rising edges.

    Averaging the aligned windows on a grid finer than the bin would return
    the response already integrated over one bin, and the library would
    integrate it a second time. So the density is fitted. The unknowns are
    its values f_k at the nodes tau_k = span[0] + k/oversample, in bins from
    the edge, with the first and the last held at zero. Each pixel's phase,
    e - floor(e), is rounded to the nearest 1/oversample, and the windows of
    one phase, scaled to sum 1, are averaged into y_phi. The model is

        y_phi[i] = sum_k f_k * (integral of hat_k over bin i shifted by phi)

    solved by least squares weighted by the number of pixels in each phase,
    plus smooth times the squared norm of the second differences of f. The
    penalty is what holds down the oscillations one bin of integration
    cannot see.

    Negative values are then set to zero, and the area that removes is
    printed. t_first puts the half height point of the rising edge of the
    density at t = 0. Returns (samples, t_first, dt) in seconds, ready for
    Plidar.table; mu and edges are the caller's selection of pixels.
    """
    h = 1.0 / oversample
    lo, hi = span
    tau = lo + h * np.arange(1, (hi - lo) * oversample)  # the free nodes
    offsets = np.arange(lo, hi + 1)  # every bin the support can reach
    ok = np.isfinite(edges)
    base, phase = np.divmod(np.rint(np.where(ok, edges, 0.0) * oversample).astype(int),
                            oversample)
    ok &= (base + lo >= 0) & (base + hi < mu.shape[1])
    rows = np.flatnonzero(ok)
    win = mu[rows[:, None], base[rows, None] + offsets]
    win = win / win.sum(axis=1, keepdims=True)

    blocks, rhs = [], []
    for m in range(oversample):
        sel = phase[rows] == m
        if sel.any():
            # Bin i of the window spans [i - phi, i + 1 - phi) from the edge.
            left = (offsets - m * h)[:, None] - tau
            a = _hat_primitive(left + 1.0, h) - _hat_primitive(left, h)
            w = np.sqrt(sel.sum())
            blocks.append(w * a)
            rhs.append(w * win[sel].mean(axis=0))
    k = tau.size
    second = np.eye(k, k, -1) - 2.0 * np.eye(k) + np.eye(k, k, 1)
    blocks.append(np.sqrt(smooth) * second)
    rhs.append(np.zeros(k))
    f = np.linalg.lstsq(np.vstack(blocks), np.concatenate(rhs), rcond=None)[0]

    removed = np.abs(f[f < 0.0]).sum()
    f = np.maximum(f, 0.0)
    print(f"fit_response: negative values set to zero, {removed / f.sum():.2e} "
          "of the area removed")
    samples = np.concatenate(([0.0], f, [0.0]))
    dt = bin_width / oversample
    return samples, -rising_edge(samples) * dt, dt


def split_groups(hist: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Every count of hist sent to one of k groups at random, as an integer
    array of shape (k, nbins). Each group is the histogram thinned with
    keep 1/k, and the groups add up to hist exactly."""
    # multinomial refuses to cast the uint64 counts lib.simulate returns
    return rng.multinomial(np.asarray(hist, dtype=np.int64), [1 / k] * k).T
