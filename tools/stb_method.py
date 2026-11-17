"""The cycle count and the instrument response, read out of recorded
histograms alone, and the precision of the depth estimators on them.

A pixel of the sensor records at most one photon per laser cycle, and the
number of cycles M is not in the data. Coates undoes that pile up only with
the right M. Too large an M under-corrects and too small an M over-corrects,
and either way the corrected shape of a return still depends on how bright
the pixel is. No pixel can record more photons than there were cycles, so M
is read from the ceiling of the totals (cycles_from_totals). The value at
which pixels of different flux agree on one shape after Coates
(estimate_cycles) depends on which pixels the scene holds, and is kept as a
diagnostic. With M known, the corrected rates of pixels that hold one
return are aligned on their rising edges and fitted with a piecewise linear
density, which becomes the table the library uses for the response
(fit_response).

A recorded pixel is one acquisition, so the spread of an estimator cannot be
seen on it directly. Splitting its counts at random into K_GROUPS groups
gives K_GROUPS histograms thinned with keep 1/K_GROUPS, and the spread of an
estimator over them is its precision on one group, set against the first
photon bound for that keep (precision). What is measured is the precision
on one group, which holds 1/K_GROUPS of the counts.

Every number this module produces about a histogram (Coates, rates, analytic
histograms, every estimate except the argmax, bounds) comes from the
library. numpy does the argmax, the bookkeeping, the alignment, the least
squares fit, the splitting and the statistics.
"""

from __future__ import annotations

import numpy as np

from plidar import Irf, Plidar, PlidarError, Scene, tof_to_range

# Flux classes, by total recorded counts per pixel. The search for M stops at
# 6400 so that every pixel it uses has cycles left at the smallest M tried.
CLASSES = [(1500, 3000), (3000, 4500), (4500, 6000), (6000, 6500), (6500, 7000)]
SEARCH_CLASSES = [(1500, 3000), (3000, 4500), (4500, 6000), (6000, 6400)]
M_GRID = np.arange(6500, 9001, 100)

# Weight of the second difference penalty in fit_response, fixed on the
# synthetic pixels of real_stb.py --synthetic before the method saw recorded
# data: the smallest value on a grid of quarter decades at which the fitted
# response stays within half the band of that check (L1 0.025) for seeds 1
# to 8. A smaller penalty lets more noise into the fit, a larger one rounds
# off more of a real shape. Each phase is weighted by its share of the
# pixels, so the penalty acts the same however many pixels enter the fit.
SMOOTH = 10.0 ** -2.25

ESTIMATORS = ("argmax", "centroid", "matched_coates", "coates_poisson", "first_photon")
K_GROUPS = 4
# A group estimate further than this from the same estimator's estimate on
# the whole histogram has found a different answer, and counts as a failure.
FAIL_BINS = 5.0


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


def cycles_from_totals(totals) -> float:
    """M as the smallest multiple of 100 at or above the 99.9th percentile
    of the totals.

    A pixel records at most one photon per cycle, so no total can exceed M
    and the ceiling of the totals bounds M from below. On a sensor whose
    brightest pixels detect in nearly every cycle the totals pile up just
    under it. The ceiling measures the instrument, while the shape collapse
    of estimate_cycles moves with the pixels a scene holds. The percentile
    leaves out the rare pixel far above the rest: one reaches 8424 in
    sample 1.
    """
    return float(np.ceil(np.percentile(totals, 99.9) / 100.0) * 100.0)


def estimate_cycles(lib: Plidar, hist: np.ndarray, totals: np.ndarray,
                    grid=M_GRID, classes=SEARCH_CLASSES) -> tuple[float, np.ndarray]:
    """The M of the grid at which the Coates corrected class shapes are
    closest to each other, and the distance at every M of the grid. A
    diagnostic next to cycles_from_totals: where the classes agree depends on
    the scene.

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

    solved by least squares with each phase weighted by its share of the
    pixels, plus smooth times the squared norm of the second differences of
    f. The penalty is what holds down the oscillations one bin of integration
    cannot see. Weighting by shares rather than counts keeps its strength the
    same however many pixels enter the fit.

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
            w = np.sqrt(sel.mean())
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


def l1_distance(got, true) -> float:
    """L1 distance between two tabulated densities, both at unit area.

    got is used as fit_response returns it. true is moved so that the half
    height point of its rising edge sits at t = 0, the origin fit_response
    gives the recovered density, so a wrong t_first counts as a difference.
    """
    s, t_first, dt = got
    t = t_first + dt * np.arange(s.size)
    s0, _, dt0 = true
    t0 = dt0 * (np.arange(s0.size) - rising_edge(s0))
    x = np.linspace(min(t[0], t0[0]), max(t[-1], t0[-1]), 20001)
    a = np.interp(x, t, s / np.trapezoid(s, t), left=0.0, right=0.0)
    b = np.interp(x, t0, s0 / np.trapezoid(s0, t0), left=0.0, right=0.0)
    return float(np.trapezoid(np.abs(a - b), x))


def template_scene(irf: Irf, nbins: int, bin_width: float) -> Scene:
    """The scene the estimators take as their template: the time grid and
    the response. They read neither its t0 nor its amplitudes, which are
    placeholders, and efficiency 1 leaves the amplitudes they fit in
    detected photons."""
    return Plidar.scene(period=nbins * bin_width, bin_width=bin_width, t0=0.0,
                        signal=1.0, background=0.0, efficiency=1.0, irf=irf)


def _t0(fit) -> float:
    try:
        return fit().t0
    except PlidarError:
        return float("nan")


def estimate_all(lib: Plidar, tmpl: Scene, hist: np.ndarray, cycles: float,
                 keep: float, work: np.ndarray) -> dict[str, float]:
    """t0 in seconds from every estimator of ESTIMATORS, NaN where the
    library refuses.

    hist is a recorded histogram, keep 1, or one group of a split, keep
    1/k, as contiguous float64 counts. cycles is M, the cycles of the whole
    acquisition, and work holds at least 4*nbins doubles.

    argmax is the centre of the fullest bin, and the centroid takes the raw
    counts over three standard deviations of the response either side of
    it. A group keeps each count with probability keep, so on average it is
    the record of M*keep cycles at the same rates: the matched filter and
    "Coates then Poisson" correct it with M*keep trials, and both give NaN
    when its counts reach M*keep and Coates refuses. The matched filter also
    gives NaN when its correlation peaks on the first or the last candidate,
    for a return at or beyond the reach of its template. The first photon
    fit takes the counts as they are, with M and keep.
    """
    out = {"argmax": float((np.argmax(hist) + 0.5) * tmpl.bin_width),
           "centroid": _t0(lambda: lib.centroid(tmpl, hist,
                                                3.0 * np.sqrt(lib.variance(tmpl.irf)))),
           "matched_coates": float("nan"), "coates_poisson": float("nan")}
    trials = cycles * keep
    try:
        mu = lib.coates_d(hist, trials)
    except PlidarError:
        pass
    else:
        out["matched_coates"] = _t0(lambda: lib.matched(tmpl, mu, work))
        out["coates_poisson"] = _t0(lambda: lib.mle(tmpl, mu * trials, trials, work))
    out["first_photon"] = _t0(lambda: lib.mle_first_photon(tmpl, hist, cycles, keep, work))
    return out


def precision(lib: Plidar, tmpl: Scene, hists: np.ndarray, totals: np.ndarray,
              cycles: float, k: int, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """Spread of every estimator over the k groups of each pixel, against
    the first photon bound for one group, by flux class.

    hists holds one histogram of integer counts per row, totals their sums,
    cycles is M and tmpl comes from template_scene. A pixel whose total
    reaches M is outside the first photon model, which allows one record per
    cycle, and is set aside before the library sees it. So is a pixel whose
    total falls outside CLASSES, or whose whole histogram the first photon
    fit or its bound refuses. The second value returned counts each kind.

    For every other pixel the first photon fit of the whole histogram, keep
    1, gives t0, the signal and the background, and from them the bound v_p
    for one group, keep 1/k. Every estimator also runs on the whole
    histogram, then the pixel is split with split_groups and every estimator
    runs on every group. A group estimate that is NaN, or further than
    FAIL_BINS bins from the same estimator's estimate on the whole
    histogram, is a failure. NaN is a refusal of the library, mapped by
    estimate_all, such as Coates once the cycles run out or the matched
    filter once the return lies beyond the reach of its template. Where the
    library refuses the whole histogram, every group fails. Each estimator
    is judged against its own estimate because a biased one sits at a
    constant offset from the first photon t0, as the argmax does on a flat
    top, and that offset is not a failure. A pixel counts for an estimator
    when at least 2 of its groups are left, and s_p^2 is their variance
    (ddof 1).

    One row per class and estimator, in the order of CLASSES and
    ESTIMATORS: pixels is the pixels that count, groups the groups tried (k
    for each pixel of the class with a bound), failures the failed ones
    among them, ratio sqrt(sum s_p^2 / sum v_p) and bound_mm the median of
    sqrt(v_p) as a distance in mm, both over the pixels that count and NaN
    when there are none.
    """
    totals = np.asarray(totals)
    keep = 1.0 / k
    tol = FAIL_BINS * tmpl.bin_width
    work = np.zeros(4 * tmpl.nbins)
    cls = np.full(totals.shape, -1)
    for c, (lo, hi) in enumerate(CLASSES):
        cls[(totals >= lo) & (totals < hi)] = c
    over = totals >= cycles
    excluded = {"at_or_above_cycles": int(over.sum()),
                "outside_classes": int((~over & (cls < 0)).sum()),
                "no_whole_fit": 0}
    # A copy of the template carries each pixel's fit into the bound.
    bound = Scene.from_buffer_copy(tmpl)
    bound._keep = getattr(tmpl, "_keep", None)
    tried = np.zeros(len(CLASSES), dtype=int)
    failed = np.zeros((len(CLASSES), len(ESTIMATORS)), dtype=int)
    s2 = [[[] for _ in ESTIMATORS] for _ in CLASSES]
    v = [[[] for _ in ESTIMATORS] for _ in CLASSES]

    for p in np.flatnonzero(~over & (cls >= 0)):
        whole = np.ascontiguousarray(hists[p], dtype=np.float64)
        try:
            fit = lib.mle_first_photon(tmpl, whole, cycles, 1.0, work)
            bound.t0, bound.signal, bound.background = fit.t0, fit.signal, fit.background
            var = lib.crb_first_photon(bound, cycles, keep, work).var_tof_joint
        except PlidarError:
            excluded["no_whole_fit"] += 1
            continue
        c = cls[p]
        tried[c] += k
        ref = estimate_all(lib, tmpl, whole, cycles, 1.0, work)
        groups = np.ascontiguousarray(split_groups(hists[p], k, rng), dtype=np.float64)
        est = [estimate_all(lib, tmpl, g, cycles, keep, work) for g in groups]
        for j, name in enumerate(ESTIMATORS):
            t = np.array([e[name] for e in est])
            ok = np.abs(t - ref[name]) <= tol  # NaN compares false
            failed[c, j] += k - ok.sum()
            if ok.sum() >= 2:
                s2[c][j].append(np.var(t[ok], ddof=1))
                v[c][j].append(var)

    rows = []
    for c, (lo, hi) in enumerate(CLASSES):
        for j, name in enumerate(ESTIMATORS):
            vp = np.array(v[c][j])
            rows.append({
                "class_lo": lo, "class_hi": hi, "estimator": name,
                "pixels": vp.size, "groups": int(tried[c]), "failures": int(failed[c, j]),
                "ratio": float(np.sqrt(np.sum(s2[c][j]) / vp.sum())) if vp.size else float("nan"),
                "bound_mm": float(np.median(tof_to_range(np.sqrt(vp)))) * 1e3
                if vp.size else float("nan"),
            })
    return rows, excluded
