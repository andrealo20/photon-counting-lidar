#include <math.h>

#include "plidar/estimate.h"

/* Golden section constant, (sqrt(5) - 1)/2. */
#define PLIDAR_INV_PHI 0.61803398874989484820

/* Floor on the fitted background, in counts per bin. It exists only so the
 * logarithm stays finite when a scene has no background at all; it is many
 * orders below any count the estimator will ever see. */
#define PLIDAR_B_FLOOR 1e-12

/* Half width of the IRF template in bins. The variance of the EMG already
 * carries the tail, so the same number of standard deviations on each side
 * covers a shape that is not symmetric. */
static size_t tmpl_half(const plidar_scene *t, size_t nbins)
{
    const double s = sqrt(plidar_irf_variance(&t->irf));
    double h = ceil(PLIDAR_TEMPLATE_SIGMAS * s / t->bin_width);

    /* Clamped before the conversion, not after. A response wide against the
     * bin, or a bin width close to zero, makes this ratio larger than any
     * size_t, and converting a double that does not fit is undefined rather
     * than merely wrong. Callers whose template would not fit are turned
     * away by the size checks that follow. */
    if (!(h >= 1.0)) {
        h = 1.0;
    }
    if (!(h < (double)nbins)) {
        h = (double)nbins;
    }
    return (size_t)h;
}

/*
 * IRF mass in each bin of a window of 2h+1 bins centred on the candidate
 * bin. Entry j covers [(j - h - 0.5)w, (j - h + 0.5)w) relative to the
 * candidate time of flight, so entry h is the bin the return sits in.
 */
static plidar_status_t build_template(const plidar_scene *t, size_t h, double *out)
{
    const double start = -((double)h + 0.5) * t->bin_width;
    return plidar_irf_bin(&t->irf, 0.0, start, t->bin_width, 2u * h + 1u, out);
}

/* Weighted sum of the histogram under a window centred on bin s. */
static double weighted(const double *counts, const double *weight, size_t h, size_t s)
{
    const size_t w = 2u * h + 1u;
    double acc = 0.0;

    for (size_t j = 0; j < w; j++) {
        acc += counts[s + j - h] * weight[j];
    }
    return acc;
}

/* Bin whose window maximises the weighted sum. Candidates are restricted to
 * those whose window lies entirely inside the histogram, so a return within
 * a template half width of either end is out of reach. That is a real
 * limitation of a finite record, not an implementation shortcut. */
static size_t scan_max(const double *counts, size_t nbins, const double *weight,
                       size_t h, double *best_out)
{
    size_t best = h;
    /* -HUGE_VAL rather than -INFINITY: the latter is a float in C99, and
     * widening it to a double is a promotion that clang refuses under
     * -Wdouble-promotion. HUGE_VAL is a double to begin with. */
    double best_val = -HUGE_VAL;

    for (size_t s = h; s + h < nbins; s++) {
        const double v = weighted(counts, weight, h, s);
        if (v > best_val) {
            best_val = v;
            best = s;
        }
    }
    if (best_out != NULL) {
        *best_out = best_val;
    }
    return best;
}

/* Sub bin offset from a parabola through the three samples around the peak,
 * in units of one bin, clamped to the half bin the parabola can justify. */
static double parabolic(double lo, double mid, double hi)
{
    const double denom = lo - 2.0 * mid + hi;
    double delta;

    if (fabs(denom) < 1e-300) {
        return 0.0;
    }
    delta = 0.5 * (lo - hi) / denom;
    if (delta > 0.5) {
        delta = 0.5;
    }
    if (delta < -0.5) {
        delta = -0.5;
    }
    return delta;
}

static void fill_result(plidar_estimate *out, double t0)
{
    out->t0 = t0;
    out->range = plidar_tof_to_range(t0);
}

plidar_status_t plidar_est_centroid(const plidar_scene *tmpl, const double *counts,
                                    size_t nbins, double half_window,
                                    plidar_estimate *out)
{
    size_t peak = 0u;
    size_t lo, hi, hw, nwin;
    double best = -HUGE_VAL;
    double total = 0.0, win_sum = 0.0, bg = 0.0;
    double num = 0.0, den = 0.0;

    if (tmpl == NULL || counts == NULL || out == NULL || nbins == 0u) {
        return PLIDAR_ERR_ARG;
    }
    if (!(half_window > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }

    out->signal = 0.0;
    out->background = 0.0;
    out->loglik = 0.0;
    out->iterations = 0u;

    for (size_t i = 0; i < nbins; i++) {
        total += counts[i];
        if (counts[i] > best) {
            best = counts[i];
            peak = i;
        }
    }

    {
        /* Same reasoning as in tmpl_half: clamp while it is still a double.
         * A window wider than the record is simply the whole record. */
        double w = ceil(half_window / tmpl->bin_width);
        if (!(w >= 1.0)) {
            w = 1.0;
        }
        if (!(w < (double)nbins)) {
            w = (double)nbins;
        }
        hw = (size_t)w;
    }
    lo = (peak > hw) ? (peak - hw) : 0u;
    hi = (peak + hw < nbins - 1u) ? (peak + hw) : (nbins - 1u);
    nwin = hi - lo + 1u;

    for (size_t i = lo; i <= hi; i++) {
        win_sum += counts[i];
    }
    /* Everything outside the window is taken to be background. With the
     * window at three standard deviations this leaks a little of the tail
     * into the background level, which makes the estimate slightly worse and
     * is exactly what a system that does not model the tail would do. */
    if (nbins > nwin) {
        bg = (total - win_sum) / (double)(nbins - nwin);
    }

    for (size_t i = lo; i <= hi; i++) {
        const double v = counts[i] - bg;
        const double centre = ((double)i + 0.5) * tmpl->bin_width;
        num += v * centre;
        den += v;
    }
    if (!(den > 0.0)) {
        return PLIDAR_ERR_NO_CONVERGE;
    }

    fill_result(out, num / den);
    return PLIDAR_OK;
}

plidar_status_t plidar_est_matched(const plidar_scene *tmpl, const double *counts,
                                   size_t nbins, double *scratch,
                                   plidar_estimate *out)
{
    plidar_status_t st;
    size_t h, s;
    double c_lo = 0.0, c_mid = 0.0, c_hi = 0.0, delta;

    if (tmpl == NULL || counts == NULL || scratch == NULL || out == NULL) {
        return PLIDAR_ERR_ARG;
    }
    h = tmpl_half(tmpl, nbins);
    if (nbins < 2u * h + 3u) {
        return PLIDAR_ERR_ARG;
    }

    out->signal = 0.0;
    out->background = 0.0;
    out->loglik = 0.0;
    out->iterations = 0u;

    st = build_template(tmpl, h, scratch);
    if (st != PLIDAR_OK) {
        return st;
    }

    s = scan_max(counts, nbins, scratch, h, &c_mid);

    /* The parabola needs a neighbour on each side. At the very edge of the
     * searchable range there is none, so the estimate stays on the bin
     * centre rather than extrapolating. */
    if (s > h && s + h + 1u < nbins) {
        c_lo = weighted(counts, scratch, h, s - 1u);
        c_hi = weighted(counts, scratch, h, s + 1u);
        delta = parabolic(c_lo, c_mid, c_hi);
    } else {
        delta = 0.0;
    }

    fill_result(out, ((double)s + 0.5 + delta) * tmpl->bin_width);
    return PLIDAR_OK;
}

/*
 * Profile log likelihood at a given time of flight and amplitude pair, plus
 * the window sums the amplitude update needs. The window follows t0, so the
 * split between "inside" and "outside" is recomputed for each candidate.
 */
typedef struct {
    double loglik;
    double signal_counts; /* sum of N_i * w_i over the window */
    double win_counts;    /* sum of N_i over the window */
    double mass;          /* sum of g_i over the window */
} plidar_fit_terms;

static plidar_status_t eval_at(const plidar_scene *t, const double *counts,
                               size_t nbins, double t0, double amp_a, double amp_b,
                               double total, double *g, size_t h,
                               plidar_fit_terms *terms)
{
    const size_t width = 2u * h + 1u;
    plidar_status_t st;
    double centre_bin;
    size_t s;
    double acc = 0.0, sig = 0.0, win = 0.0, mass = 0.0;

    centre_bin = floor(t0 / t->bin_width);
    if (centre_bin < (double)h) {
        centre_bin = (double)h;
    }
    if (centre_bin > (double)(nbins - h - 1u)) {
        centre_bin = (double)(nbins - h - 1u);
    }
    s = (size_t)centre_bin;

    st = plidar_irf_bin(&t->irf, t0, ((double)(s - h)) * t->bin_width,
                        t->bin_width, width, g);
    if (st != PLIDAR_OK) {
        return st;
    }

    for (size_t j = 0; j < width; j++) {
        const double n = counts[s + j - h];
        const double rate = amp_a * g[j] + amp_b;
        const double share = (rate > 0.0) ? (amp_a * g[j] / rate) : 0.0;

        acc += n * log(rate) - amp_a * g[j];
        sig += n * share;
        win += n;
        mass += g[j];
    }

    /* Outside the window the model is flat background, so its contribution
     * collapses to two terms and does not need a pass over the histogram. */
    acc += (total - win) * log(amp_b) - (double)nbins * amp_b;

    terms->loglik = acc;
    terms->signal_counts = sig;
    terms->win_counts = win;
    terms->mass = mass;
    return PLIDAR_OK;
}

plidar_status_t plidar_est_mle(const plidar_scene *tmpl, const double *counts,
                               double cycles, size_t nbins, double *scratch,
                               plidar_estimate *out)
{
    plidar_status_t st;
    size_t h, width, s_mf, s0;
    double *tpl, *work;
    double total = 0.0, mass = 0.0;
    double amp_a, amp_b, t0;
    double lo, hi;
    uint32_t iters = 0u;

    if (tmpl == NULL || counts == NULL || scratch == NULL || out == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (!(cycles > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    h = tmpl_half(tmpl, nbins);
    width = 2u * h + 1u;
    if (nbins < 2u * width) {
        return PLIDAR_ERR_ARG;
    }
    tpl = scratch;
    work = scratch + width;

    st = build_template(tmpl, h, tpl);
    if (st != PLIDAR_OK) {
        return st;
    }
    for (size_t i = 0; i < nbins; i++) {
        total += counts[i];
    }
    for (size_t j = 0; j < width; j++) {
        mass += tpl[j];
    }
    if (!(total > 0.0) || !(mass > 0.0)) {
        return PLIDAR_ERR_NO_CONVERGE;
    }

    /* Stage one: a matched filter, used only to place the window that the
     * first amplitude guess is read off. */
    s_mf = scan_max(counts, nbins, tpl, h, NULL);
    {
        double plain = 0.0;
        for (size_t j = 0; j < width; j++) {
            plain += counts[s_mf + j - h];
        }
        amp_b = (nbins > width) ? ((total - plain) / (double)(nbins - width))
                                : (total / (double)nbins);
        if (!(amp_b > PLIDAR_B_FLOOR)) {
            amp_b = PLIDAR_B_FLOOR;
        }
        amp_a = (plain - (double)width * amp_b) / mass;
        if (!(amp_a > 0.0)) {
            amp_a = plain / mass;
        }
        if (!(amp_a > 0.0)) {
            amp_a = 1.0 / mass;
        }
    }

    /* Stage two: scan the whole period with the statistic the likelihood
     * reduces to once the amplitudes are fixed. The logarithms depend only
     * on the template, so they are taken once here rather than at every
     * candidate, which is what makes a full scan affordable. */
    for (size_t j = 0; j < width; j++) {
        work[j] = log1p((amp_a / amp_b) * tpl[j]);
    }
    s0 = scan_max(counts, nbins, work, h, NULL);
    t0 = ((double)s0 + 0.5) * tmpl->bin_width;

    /* Stage three: coordinate ascent. Expectation maximisation moves the two
     * amplitudes with the time of flight held still, golden section moves
     * the time of flight with the amplitudes held still, and both steps only
     * ever increase the likelihood. */
    {
        const double tol = PLIDAR_MLE_TOL_BINS * tmpl->bin_width;
        plidar_fit_terms terms;
        double last = 0.0;

        for (uint32_t round = 0u; round < 50u; round++) {
            for (uint32_t em = 0u; em < 50u; em++) {
                double next_a, next_b;

                st = eval_at(tmpl, counts, nbins, t0, amp_a, amp_b, total, work, h,
                             &terms);
                if (st != PLIDAR_OK) {
                    return st;
                }
                next_a = terms.signal_counts / terms.mass;
                next_b = (total - terms.signal_counts) / (double)nbins;
                if (!(next_b > PLIDAR_B_FLOOR)) {
                    next_b = PLIDAR_B_FLOOR;
                }
                if (!(next_a > 0.0)) {
                    next_a = PLIDAR_B_FLOOR;
                }
                iters++;
                if (fabs(next_a - amp_a) <= 1e-12 * amp_a &&
                    fabs(next_b - amp_b) <= 1e-12 * amp_b) {
                    amp_a = next_a;
                    amp_b = next_b;
                    break;
                }
                amp_a = next_a;
                amp_b = next_b;
            }

            /* Bracket two bins each way. The coarse scan already put t0
             * within half a bin of a local maximum, and the amplitude update
             * does not move it further than that. */
            lo = t0 - 2.0 * tmpl->bin_width;
            hi = t0 + 2.0 * tmpl->bin_width;
            {
                double x1 = hi - PLIDAR_INV_PHI * (hi - lo);
                double x2 = lo + PLIDAR_INV_PHI * (hi - lo);
                double f1, f2;

                st = eval_at(tmpl, counts, nbins, x1, amp_a, amp_b, total, work, h,
                             &terms);
                if (st != PLIDAR_OK) {
                    return st;
                }
                f1 = terms.loglik;
                st = eval_at(tmpl, counts, nbins, x2, amp_a, amp_b, total, work, h,
                             &terms);
                if (st != PLIDAR_OK) {
                    return st;
                }
                f2 = terms.loglik;

                while (hi - lo > tol) {
                    if (f1 > f2) {
                        hi = x2;
                        x2 = x1;
                        f2 = f1;
                        x1 = hi - PLIDAR_INV_PHI * (hi - lo);
                        st = eval_at(tmpl, counts, nbins, x1, amp_a, amp_b, total,
                                     work, h, &terms);
                        if (st != PLIDAR_OK) {
                            return st;
                        }
                        f1 = terms.loglik;
                    } else {
                        lo = x1;
                        x1 = x2;
                        f1 = f2;
                        x2 = lo + PLIDAR_INV_PHI * (hi - lo);
                        st = eval_at(tmpl, counts, nbins, x2, amp_a, amp_b, total,
                                     work, h, &terms);
                        if (st != PLIDAR_OK) {
                            return st;
                        }
                        f2 = terms.loglik;
                    }
                }
            }

            {
                const double moved = fabs(0.5 * (lo + hi) - t0);
                t0 = 0.5 * (lo + hi);
                st = eval_at(tmpl, counts, nbins, t0, amp_a, amp_b, total, work, h,
                             &terms);
                if (st != PLIDAR_OK) {
                    return st;
                }
                last = terms.loglik;
                if (moved <= tol) {
                    break;
                }
            }
        }

        out->loglik = last;
    }

    fill_result(out, t0);
    out->signal = amp_a / (cycles * tmpl->efficiency);
    out->background = amp_b / (cycles * tmpl->efficiency * tmpl->bin_width);
    out->iterations = iters;
    return PLIDAR_OK;
}
