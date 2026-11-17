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
    double h;

    if (t->irf.kind == PLIDAR_IRF_TABLE) {
        /* The support is finite and known, so cover all of it. */
        const double last = t->irf.t_first + (double)(t->irf.n - 1u) * t->irf.dt;
        h = ceil(fmax(fabs(t->irf.t_first), fabs(last)) / t->bin_width);
    } else {
        h = ceil(PLIDAR_TEMPLATE_SIGMAS * sqrt(plidar_irf_variance(&t->irf)) /
                 t->bin_width);
    }

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

    st = build_template(tmpl, h, scratch);
    if (st != PLIDAR_OK) {
        return st;
    }

    s = scan_max(counts, nbins, scratch, h, &c_mid);

    /* A peak on the first or the last candidate has no neighbour on one
     * side, and the correlation may still be climbing past it: the return
     * lies at or beyond the reach of the template. The edge of the search
     * would be reported wherever the return was, so nothing is. */
    if (s == h || s + h + 1u == nbins) {
        return PLIDAR_ERR_DOMAIN;
    }
    c_lo = weighted(counts, scratch, h, s - 1u);
    c_hi = weighted(counts, scratch, h, s + 1u);
    delta = parabolic(c_lo, c_mid, c_hi);

    out->signal = 0.0;
    out->background = 0.0;
    out->loglik = 0.0;
    out->iterations = 0u;
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

/*
 * The first photon model. With mu_i = a g_i(t0) + b photons per cycle in bin
 * i and S_i the rate summed over the bins before i, a cycle records in bin i
 * with probability P_i = exp(-S_i) (1 - exp(-mu_i)) and records nothing with
 * probability exp(-S_n). Counts kept with probability q are multinomial over
 * the bins and the n_0 = M - N cycles left without a kept record, N being
 * the total, so
 *
 *     L = N log q - sum_i n_i S_i + sum_i n_i log(1 - exp(-mu_i))
 *         + n_0 log(1 - q + q exp(-S_n))
 *
 * Outside the window g is zero, which splits sum_i n_i S_i in two: b times
 * sum_i i n_i, one number for the whole histogram, and a times the template
 * mass ahead of each count, which is the mass before its bin for a count in
 * the window and all of it for a count after the window. Counts outside the
 * window then enter through prefix sums only, and every evaluation is one
 * pass over the window, as in the Poisson fit.
 *
 * A bin with no counts adds nothing but its share of S, so its logarithm is
 * not taken and b = 0 stays finite. The empty category is skipped the same
 * way when every cycle recorded.
 */
typedef struct {
    const double *counts;
    const double *before; /* before[i]: counts in the bins before i, nbins + 1 entries */
    size_t nbins;
    size_t width;
    double base;   /* N log q */
    double moment; /* sum of i n_i */
    double none;   /* n_0 */
    double keep;   /* q */
} plidar_fp_data;

/* Gradient and Hessian of L in (a, b), and the Hessian once more without the
 * empty category, the one term that is not concave. */
typedef struct {
    double ga, gb;
    double haa, hab, hbb;
    double caa, cab, cbb;
} plidar_fp_slope;

/* L at (a, b) for the window that starts at bin lo, with the template in g.
 * The gradient and Hessian come out of the same pass when slope is given. */
static double fp_loglik(const plidar_fp_data *d, size_t lo, const double *g, double a,
                        double b, plidar_fp_slope *slope)
{
    const double *n = d->counts + lo;
    const double after = d->before[d->nbins] - d->before[lo + d->width];
    const double outside = d->before[lo] + after;
    const double nb = (double)d->nbins;
    double mass = 0.0, ahead = 0.0, acc = 0.0;
    double ga = 0.0, gb = 0.0, caa = 0.0, cab = 0.0, cbb = 0.0, bend = 0.0;

    for (size_t j = 0; j < d->width; j++) {
        ahead += n[j] * mass;
        mass += g[j];
        if (n[j] > 0.0) {
            const double mu = a * g[j] + b;
            const double hit = -expm1(-mu);

            acc += n[j] * log(hit);
            if (slope != NULL) {
                /* The first derivative of n log(hit) in mu, and the second
                 * with its sign turned. */
                const double r = n[j] * exp(-mu) / hit;
                const double c = r / hit;

                ga += r * g[j];
                gb += r;
                caa += c * g[j] * g[j];
                cab += c * g[j];
                cbb += c;
            }
        }
    }
    ahead += mass * after;
    acc += d->base - b * d->moment - a * ahead;

    if (outside > 0.0) {
        const double hit = -expm1(-b);

        acc += outside * log(hit);
        if (slope != NULL) {
            const double r = outside * exp(-b) / hit;

            gb += r;
            cbb += r / hit;
        }
    }

    if (d->none > 0.0) {
        const double s = a * mass + nb * b;
        double dl = -1.0, d2l = 0.0;

        if (d->keep < 1.0) {
            /* log(1 - q + q exp(-s)) and its first two derivatives. The
             * second is positive: this term is convex in s. */
            const double u = d->keep * exp(-s);
            const double den = (1.0 - d->keep) + u;

            acc += d->none * log1p(d->keep * expm1(-s));
            dl = -u / den;
            d2l = u * (1.0 - d->keep) / (den * den);
        } else {
            /* At q = 1 the term is exactly -s, which stays finite where
             * exp(-s) would underflow. */
            acc -= d->none * s;
        }
        ga += d->none * dl * mass;
        gb += d->none * dl * nb;
        bend = d->none * d2l;
    }

    if (slope != NULL) {
        slope->ga = ga - ahead;
        slope->gb = gb - d->moment;
        slope->caa = -caa;
        slope->cab = -cab;
        slope->cbb = -cbb;
        slope->haa = bend * mass * mass - caa;
        slope->hab = bend * mass * nb - cab;
        slope->hbb = bend * nb * nb - cbb;
    }
    return acc;
}

/* The Newton direction for (a, b), from the Hessian when it is negative
 * definite and from its concave part when it is not. */
static void fp_direction(const plidar_fp_slope *s, double a, double b, double *da,
                         double *db)
{
    double aa = s->haa, ab = s->hab, bb = s->hbb;

    if (!(aa < 0.0 && aa * bb > ab * ab)) {
        aa = s->caa;
        ab = s->cab;
        bb = s->cbb;
    }
    if (aa < 0.0 && aa * bb > ab * ab) {
        const double det = aa * bb - ab * ab;

        *da = (ab * s->gb - bb * s->ga) / det;
        *db = (ab * s->ga - aa * s->gb) / det;
        if (b > 0.0 || *db > 0.0) {
            return;
        }
    }
    /* b stays where it is, at zero with the step pointing below it, or with
     * a curvature too flat to solve for both, and a moves alone. */
    {
        const double c = (s->haa < 0.0) ? s->haa : s->caa;

        *db = 0.0;
        *da = (c < 0.0) ? -s->ga / c : -0.5 * a;
    }
}

/*
 * The amplitudes at a fixed time of flight. The Poisson fit has a closed form
 * update for them and this model has none. An expectation maximisation over
 * the photons the first photon rule hides would keep the Poisson form, but
 * at the fluxes this fit is for most photons are hidden, and the convergence
 * of EM slows with the share that is missing. So it is Newton's method on
 * (a, b), with the gradient and Hessian that come out of the same pass as L
 * and the step halved until L does not go down. With two unknowns a step is a
 * 2 by 2 solve, and started from the amplitudes of a nearby t0, as the search
 * below starts it, it takes one to four steps.
 *
 * Everything in L is concave in (a, b) except the empty category, which with
 * q < 1 is convex in S_n. Where that makes the Hessian indefinite the term is
 * left out of the curvature, which leaves the curvature of a concave function
 * and a direction that still climbs.
 *
 * b = 0 is a legitimate answer, so b is not fitted through its logarithm. A
 * step that would take b below zero is cut short to land on zero, and the
 * halving brings it back inside if L is lower there. At b = 0, with the step
 * pointing below it, b stays at zero and Newton runs on a alone.
 *
 * Returns the updates taken, and L at the amplitudes it leaves.
 */
static uint32_t fp_amplitudes(const plidar_fp_data *d, size_t lo, const double *g,
                              double *amp_a, double *amp_b, double *loglik)
{
    double a = *amp_a, b = *amp_b;
    double here = fp_loglik(d, lo, g, a, b, NULL);
    uint32_t taken = 0u;

    /* At b = 0, a count the template stopped covering when t0 moved makes L
     * minus infinity, and there is no slope to climb from there. */
    if (!(here > -HUGE_VAL)) {
        b = fmax(b, PLIDAR_B_FLOOR);
        here = fp_loglik(d, lo, g, a, b, NULL);
    }

    for (uint32_t it = 0u; it < 50u; it++) {
        plidar_fp_slope s;
        double da, db, step = 1.0, gained = -1.0;
        int to_zero = 0;

        (void)fp_loglik(d, lo, g, a, b, &s);
        fp_direction(&s, a, b, &da, &db);

        /* The gain the quadratic model promises. Below the rounding of L
         * there is nothing left to find, and a line search would chase it. */
        if (!(0.5 * (s.ga * da + s.gb * db) > 1e-12 * fabs(here))) {
            break;
        }
        if (b + db < 0.0) {
            step = b / -db;
            to_zero = 1;
        }
        for (int k = 0; k < 60; k++) {
            const double na = a + step * da;
            const double nb = to_zero ? 0.0 : b + step * db;

            if (na > 0.0) {
                const double there = fp_loglik(d, lo, g, na, nb, NULL);

                if (there >= here) {
                    gained = there - here;
                    a = na;
                    b = nb;
                    here = there;
                    break;
                }
            }
            step *= 0.5;
            to_zero = 0;
        }
        if (!(gained >= 0.0)) {
            break;
        }
        taken++;
        if (gained <= 1e-12 * fabs(here)) {
            break;
        }
    }
    *amp_a = a;
    *amp_b = b;
    *loglik = here;
    return taken;
}

/* The profile likelihood at t0: the window placed as eval_at places it, and
 * the amplitudes refitted there from the values they come in with. */
static plidar_status_t fp_profile(const plidar_scene *t, const plidar_fp_data *d, size_t h,
                                  double t0, double *g, double *amp_a, double *amp_b,
                                  uint32_t *iters, double *loglik)
{
    double centre = floor(t0 / t->bin_width);
    size_t lo;
    plidar_status_t st;

    if (centre < (double)h) {
        centre = (double)h;
    }
    if (centre > (double)(d->nbins - h - 1u)) {
        centre = (double)(d->nbins - h - 1u);
    }
    lo = (size_t)centre - h;
    st = plidar_irf_bin(&t->irf, t0, (double)lo * t->bin_width, t->bin_width, d->width, g);
    if (st != PLIDAR_OK) {
        return st;
    }
    *iters += fp_amplitudes(d, lo, g, amp_a, amp_b, loglik);
    return PLIDAR_OK;
}

plidar_status_t plidar_est_mle_first_photon(const plidar_scene *tmpl, const double *counts,
                                            double cycles, double keep, size_t nbins,
                                            double *scratch, plidar_estimate *out)
{
    plidar_status_t st;
    plidar_fp_data d;
    size_t h, width, s_mf, s0;
    double *tpl, *work, *before;
    double total, moment = 0.0, mass = 0.0;
    double amp_a, amp_b, t0, last = 0.0;
    uint32_t iters = 0u;

    if (tmpl == NULL || counts == NULL || scratch == NULL || out == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (!(cycles > 0.0) || !isfinite(cycles) || !(keep > 0.0) || !(keep <= 1.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    h = tmpl_half(tmpl, nbins);
    width = 2u * h + 1u;
    if (nbins < 2u * width) {
        return PLIDAR_ERR_ARG;
    }
    tpl = scratch;
    work = scratch + width;
    before = scratch + 2u * width;

    before[0] = 0.0;
    for (size_t i = 0; i < nbins; i++) {
        if (!(counts[i] >= 0.0) || !isfinite(counts[i])) {
            return PLIDAR_ERR_DOMAIN;
        }
        before[i + 1u] = before[i] + counts[i];
        moment += (double)i * counts[i];
    }
    total = before[nbins];
    /* Every count is the record of a different cycle. */
    if (total > cycles) {
        return PLIDAR_ERR_DOMAIN;
    }
    if (!(total > 0.0)) {
        return PLIDAR_ERR_NO_CONVERGE;
    }

    st = build_template(tmpl, h, tpl);
    if (st != PLIDAR_OK) {
        return st;
    }
    for (size_t j = 0; j < width; j++) {
        mass += tpl[j];
    }
    if (!(mass > 0.0)) {
        return PLIDAR_ERR_NO_CONVERGE;
    }

    d.counts = counts;
    d.before = before;
    d.nbins = nbins;
    d.width = width;
    d.base = total * log(keep);
    d.moment = moment;
    d.none = cycles - total;
    d.keep = keep;

    /* Stage one: the matched filter places the window, and the share of
     * cycles that recorded gives the total rate per cycle,
     * T = -log(1 - N/(q M)). When every cycle recorded nothing bounds T, so
     * half a cycle is kept back. The counts outside the window give the
     * background its share of T and the signal has the rest. */
    s_mf = scan_max(counts, nbins, tpl, h, NULL);
    {
        const double kept = keep * cycles;
        const double plain = before[s_mf + h + 1u] - before[s_mf - h];
        double rate = log(kept / fmax(kept - total, 0.5));

        if (!(rate > 0.0)) {
            rate = total / kept;
        }
        amp_b = rate * ((total - plain) / total) / (double)(nbins - width);
        if (!(amp_b > PLIDAR_B_FLOOR)) {
            amp_b = PLIDAR_B_FLOOR;
        }
        amp_a = (rate - (double)nbins * amp_b) / mass;
        if (!(amp_a > 0.0)) {
            amp_a = rate / mass;
        }
    }

    /* Stage two: every candidate at fixed amplitudes. With the template at
     * the bin centre, candidates differ only by a weighted sum over their
     * window and by the counts before it, each of which was spared the
     * masking of the whole template, so the weights are taken once and the
     * scan costs what the Poisson one does. b is at the floor or above here,
     * which keeps every weight finite. */
    {
        const double lb = log(-expm1(-amp_b));
        double cum = 0.0, best = -HUGE_VAL;

        for (size_t j = 0; j < width; j++) {
            work[j] = log(-expm1(-(amp_a * tpl[j] + amp_b))) - lb + amp_a * (mass - cum);
            cum += tpl[j];
        }
        s0 = h;
        for (size_t s = h; s + h < nbins; s++) {
            const double v = amp_a * mass * before[s - h] + weighted(counts, work, h, s);
            if (v > best) {
                best = v;
                s0 = s;
            }
        }
    }
    t0 = ((double)s0 + 0.5) * tmpl->bin_width;

    /*
     * Stage three: golden section on t0 in two bins each way, with the
     * amplitudes refitted at every t0 it tries, repeated around the new t0
     * until t0 moves less than the tolerance.
     *
     * The refit is where this departs from the Poisson fit, which holds the
     * amplitudes still for a whole section. Under pile up t0 and a are
     * coupled: a brighter return records earlier, and a later t0 undoes it.
     * Alternating one section with one amplitude fit then closes the gap by
     * a constant factor per round, the squared correlation of t0 with the
     * amplitudes in the bound, 0.42 at keep 1 and 0.67 at keep 1/4 for the
     * Monte Carlo scene of the tests. On their noiseless histograms it
     * stopped up to four tolerances short, with the signal off by 2e-4 of
     * itself. With the refit the section climbs the profile
     * likelihood, the maximum over the amplitudes at each t0, whose maximum
     * is the joint one, and the second round finds nothing left to move. A
     * refit also lets b leave zero wherever the window no longer covers a
     * count, so the section never meets the minus infinity that L has there
     * at b = 0.
     */
    {
        const double w = tmpl->bin_width;
        const double tol = PLIDAR_MLE_TOL_BINS * w;

        for (uint32_t round = 0u; round < 50u; round++) {
            double lo_t = t0 - 2.0 * w, hi_t = t0 + 2.0 * w;
            double x1 = hi_t - PLIDAR_INV_PHI * (hi_t - lo_t);
            double x2 = lo_t + PLIDAR_INV_PHI * (hi_t - lo_t);
            double a1 = amp_a, b1 = amp_b, a2 = amp_a, b2 = amp_b;
            double f1, f2, moved;

            st = fp_profile(tmpl, &d, h, x1, work, &a1, &b1, &iters, &f1);
            if (st != PLIDAR_OK) {
                return st;
            }
            st = fp_profile(tmpl, &d, h, x2, work, &a2, &b2, &iters, &f2);
            if (st != PLIDAR_OK) {
                return st;
            }
            /* Each new point starts from the amplitudes of the point next to
             * it that the bracket keeps. */
            while (hi_t - lo_t > tol) {
                if (f1 > f2) {
                    hi_t = x2;
                    x2 = x1;
                    f2 = f1;
                    a2 = a1;
                    b2 = b1;
                    x1 = hi_t - PLIDAR_INV_PHI * (hi_t - lo_t);
                    st = fp_profile(tmpl, &d, h, x1, work, &a1, &b1, &iters, &f1);
                } else {
                    lo_t = x1;
                    x1 = x2;
                    f1 = f2;
                    a1 = a2;
                    b1 = b2;
                    x2 = lo_t + PLIDAR_INV_PHI * (hi_t - lo_t);
                    st = fp_profile(tmpl, &d, h, x2, work, &a2, &b2, &iters, &f2);
                }
                if (st != PLIDAR_OK) {
                    return st;
                }
            }

            amp_a = (f1 > f2) ? a1 : a2;
            amp_b = (f1 > f2) ? b1 : b2;
            moved = fabs(0.5 * (lo_t + hi_t) - t0);
            t0 = 0.5 * (lo_t + hi_t);
            st = fp_profile(tmpl, &d, h, t0, work, &amp_a, &amp_b, &iters, &last);
            if (st != PLIDAR_OK) {
                return st;
            }
            if (moved <= tol) {
                break;
            }
        }
    }

    fill_result(out, t0);
    out->signal = amp_a / tmpl->efficiency;
    out->background = amp_b / (tmpl->efficiency * tmpl->bin_width);
    out->loglik = last;
    out->iterations = iters;
    return PLIDAR_OK;
}
