#include <math.h>

#include "plidar/irf.h"

/* C99 does not promise M_PI and friends, and the build asks for strict C99,
 * so the constants live here. */
#define PLIDAR_SQRT1_2      0.70710678118654752440
#define PLIDAR_LOG_2PI      1.83787706640934548356
#define PLIDAR_INV_SQRT_2PI 0.39894228040143267794
#define PLIDAR_FWHM_PER_SIG 2.35482004503094938202 /* 2*sqrt(2*ln 2) */

/* Standard normal distribution function. */
static double phi_cdf(double x)
{
    return 0.5 * erfc(-x * PLIDAR_SQRT1_2);
}

/* Left of this the Mills expansion is used instead of erfc. */
#define PLIDAR_MILLS_X (-30.0)

/*
 * Logarithm of the bracket in Mills' ratio expansion,
 *
 *     Phi(x) = phi(x)/(-x) * ( 1 - 1/x^2 + 3/x^4 - 15/x^6 + 105/x^8 - ... )
 *
 * valid for x well below zero. At x = -30 the first term dropped is 10395/x^12,
 * which is 2e-14 relative, and the series is asymptotic so that is also the
 * error bound.
 */
static double log_mills_series(double x)
{
    static const double coef[5] = {1.0, 3.0, 15.0, 105.0, 945.0};
    const double z2 = x * x;
    double zp = z2;
    double s = 1.0;

    for (int k = 0; k < 5; k++) {
        const double term = coef[k] / zp;
        s += ((k % 2) == 0) ? -term : term;
        zp *= z2;
    }
    return log(s);
}

/*
 * log of the standard normal distribution function.
 *
 * The EMG needs Phi at arguments that run far into the left tail, where the
 * value underflows long before the expression it belongs to does. Taking the
 * logarithm keeps the whole thing in range, but only if the logarithm itself
 * is computed without first forming the underflowed value.
 *
 * Above the threshold, erfc is exact enough and nowhere near its own
 * underflow. Below it, the expansion above is used.
 */
static double log_phi_cdf(double x)
{
    if (x > PLIDAR_MILLS_X) {
        return log(phi_cdf(x));
    }
    return -0.5 * x * x - 0.5 * PLIDAR_LOG_2PI - log(-x) + log_mills_series(x);
}

plidar_status_t plidar_irf_init_gaussian(plidar_irf *irf, double sigma)
{
    if (irf == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (!(sigma > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    /* Assigned whole, so the fields of the other kinds come out zero and a
     * table pointer left over from an earlier use cannot survive. */
    *irf = (plidar_irf){.kind = PLIDAR_IRF_GAUSSIAN, .sigma = sigma};
    return PLIDAR_OK;
}

plidar_status_t plidar_irf_init_emg(plidar_irf *irf, double sigma, double tau)
{
    if (irf == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (!(sigma > 0.0) || !(tau > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    *irf = (plidar_irf){.kind = PLIDAR_IRF_EMG, .sigma = sigma, .tau = tau};
    return PLIDAR_OK;
}

plidar_status_t plidar_irf_init_table(plidar_irf *irf, const double *samples,
                                      size_t n, double t_first, double dt,
                                      double *work)
{
    double area = 0.0;

    if (irf == NULL || samples == NULL || work == NULL || n < 2u) {
        return PLIDAR_ERR_ARG;
    }
    if (!(dt > 0.0) || !isfinite(dt) || !isfinite(t_first)) {
        return PLIDAR_ERR_DOMAIN;
    }
    if (samples[0] > 0.0 || samples[n - 1u] > 0.0) {
        return PLIDAR_ERR_DOMAIN;
    }
    for (size_t k = 0; k < n; k++) {
        if (!(samples[k] >= 0.0) || !isfinite(samples[k])) {
            return PLIDAR_ERR_DOMAIN;
        }
        area += samples[k];
    }
    /* With both ends at zero the trapezoidal rule is dt times the plain sum.
     * An area that overflowed would normalise every sample to zero. */
    area *= dt;
    if (!(area > 0.0) || !isfinite(area)) {
        return PLIDAR_ERR_DOMAIN;
    }

    {
        double *f = work;
        double *left = work + n;
        double *right = work + 2u * n;

        for (size_t k = 0; k < n; k++) {
            f[k] = samples[k] / area;
        }
        left[0] = 0.0;
        for (size_t k = 1; k < n; k++) {
            left[k] = left[k - 1u] + 0.5 * dt * (f[k - 1u] + f[k]);
        }
        /* Summed from the right end, so that the mass beyond a node in the
         * far tail is made of tail terms only, rather than one minus a
         * number within rounding of one. */
        right[n - 1u] = 0.0;
        for (size_t k = n - 1u; k > 0u; k--) {
            right[k - 1u] = right[k] + 0.5 * dt * (f[k - 1u] + f[k]);
        }
    }

    *irf = (plidar_irf){.kind = PLIDAR_IRF_TABLE, .table = work, .n = n,
                        .t_first = t_first, .dt = dt};
    return PLIDAR_OK;
}

double plidar_sigma_from_fwhm(double fwhm)
{
    return fwhm / PLIDAR_FWHM_PER_SIG;
}

/*
 * The factor tau*pdf(t), which both the pdf and the cdf are built from.
 * Assembled in the exponent so that the diverging exp and the vanishing Phi
 * never exist separately. The result is bounded by 1, because the EMG pdf
 * cannot exceed the peak of the exponential factor.
 */
static double emg_tail_term(const plidar_irf *irf, double t)
{
    const double v = irf->sigma / irf->tau;
    const double a = t / irf->sigma;
    const double x = a - v;

    if (x > PLIDAR_MILLS_X) {
        return exp(0.5 * v * v - t / irf->tau + log_phi_cdf(x));
    }

    /*
     * Once x is deep in the left tail the exponent is a difference of two
     * quantities that are both close to v^2/2, and adding them only to
     * subtract them again throws away every digit they have in common. By
     * v = 1e5, which is a tail time constant five orders below the jitter,
     * the answer has lost its fifth significant figure; by v = 1e8 it is
     * wrong by forty percent, and past v = 1e154 the term overflows to a not
     * a number. None of that is physical, but a sweep that drives tau
     * towards zero to watch the response become Gaussian walks straight into
     * it.
     *
     * The cancellation is exact and can be done on paper. Substituting the
     * expansion for log Phi and using a*v = t/tau,
     *
     *   v^2/2 - t/tau + log Phi(x)
     *     = v^2/2 - a*v - x^2/2 - log(sqrt(2 pi)) - log(-x) + log(S)
     *     = -a^2/2 - log(sqrt(2 pi)) - log(-x) + log(S)
     *
     * which has nothing large in it at all.
     */
    return exp(-0.5 * a * a - 0.5 * PLIDAR_LOG_2PI - log(-x) +
               log_mills_series(x));
}

/* Time of the last node of a table. */
static double table_end(const plidar_irf *irf)
{
    return irf->t_first + (double)(irf->n - 1u) * irf->dt;
}

/*
 * The cell of a table that holds t, and the fraction u of the way across it,
 * for t strictly inside the support. The index is clamped in floating point,
 * before the conversion, because a t just short of the last node can round
 * onto it and because converting a double past the range of size_t is
 * undefined.
 */
static size_t table_cell(const plidar_irf *irf, double t, double *u)
{
    const double x = (t - irf->t_first) / irf->dt;
    const size_t last = irf->n - 2u;
    const size_t k = (x < (double)last) ? (size_t)x : last;
    const double frac = x - (double)k;

    *u = (frac < 1.0) ? frac : 1.0;
    return k;
}

static double table_pdf(const plidar_irf *irf, double t)
{
    const double *f = irf->table;
    size_t k;
    double u;

    if (!(t > irf->t_first) || !(t < table_end(irf))) {
        return 0.0;
    }
    k = table_cell(irf, t, &u);
    return (1.0 - u) * f[k] + u * f[k + 1u];
}

/*
 * Mass of a table on one side of t: to the left for the distribution
 * function, to the right for the survival function. In cell k, with f(t)
 * the interpolated density, the left mass is the mass left of node k plus
 * the trapezoid from node k to t, and the right mass is the mass right of
 * node k + 1 plus the trapezoid from t to node k + 1. Each is built only
 * from terms on its own side, which is what lets the right one keep its
 * relative precision in the far tail.
 */
static double table_mass(const plidar_irf *irf, double t, int right)
{
    const double *f = irf->table;
    const double *left_of = f + irf->n;
    const double *right_of = f + 2u * irf->n;
    size_t k;
    double u, ft;

    if (!(t > irf->t_first)) {
        return right ? 1.0 : 0.0;
    }
    if (!(t < table_end(irf))) {
        return right ? 0.0 : 1.0;
    }
    k = table_cell(irf, t, &u);
    ft = (1.0 - u) * f[k] + u * f[k + 1u];
    if (right) {
        return right_of[k + 1u] + 0.5 * irf->dt * (1.0 - u) * (ft + f[k + 1u]);
    }
    return left_of[k] + 0.5 * irf->dt * u * (f[k] + ft);
}

/*
 * First and second moments of a table about the point c, integrated cell by
 * cell in closed form. On a cell whose midpoint lies m from c, the density
 * is fm + df*w/dt, with fm and df the mean and the difference (right minus
 * left) of its two node values and w the offset from the midpoint, so
 *
 *     integral of (t - c)   f = dt * (m*fm + dt*df/12)
 *     integral of (t - c)^2 f = dt * (m^2*fm + m*dt*df/6 + dt^2*fm/12)
 */
static void table_moments(const plidar_irf *irf, double c, double *m1,
                          double *m2)
{
    const double *f = irf->table;
    const double dt = irf->dt;
    double s1 = 0.0, s2 = 0.0;

    for (size_t k = 0; k + 1u < irf->n; k++) {
        const double m = irf->t_first + ((double)k + 0.5) * dt - c;
        const double fm = 0.5 * (f[k] + f[k + 1u]);
        const double df = f[k + 1u] - f[k];

        s1 += m * fm + dt * df / 12.0;
        s2 += m * m * fm + m * dt * df / 6.0 + dt * dt * fm / 12.0;
    }
    *m1 = dt * s1;
    *m2 = dt * s2;
}

double plidar_irf_pdf(const plidar_irf *irf, double t)
{
    if (irf == NULL) {
        return 0.0;
    }
    if (irf->kind == PLIDAR_IRF_TABLE) {
        return table_pdf(irf, t);
    }
    if (irf->kind == PLIDAR_IRF_GAUSSIAN) {
        const double a = t / irf->sigma;
        return PLIDAR_INV_SQRT_2PI * exp(-0.5 * a * a) / irf->sigma;
    }
    return emg_tail_term(irf, t) / irf->tau;
}

double plidar_irf_cdf(const plidar_irf *irf, double t)
{
    double c;

    if (irf == NULL) {
        return 0.0;
    }
    if (irf->kind == PLIDAR_IRF_GAUSSIAN) {
        c = phi_cdf(t / irf->sigma);
    } else if (irf->kind == PLIDAR_IRF_TABLE) {
        c = table_mass(irf, t, 0);
    } else {
        /* cdf(t) = Phi(t/sigma) - tau*pdf(t). The two terms are close when t
         * is far to the left, where both are negligible anyway, so the loss
         * of relative precision there does not reach the bin masses. */
        c = phi_cdf(t / irf->sigma) - emg_tail_term(irf, t);
    }
    if (c < 0.0) {
        return 0.0;
    }
    if (c > 1.0) {
        return 1.0;
    }
    return c;
}

/*
 * Survival function, 1 - cdf, computed so that it never subtracts two
 * numbers near one.
 *
 * For the Gaussian it is erfc read from the other side. For the EMG,
 * cdf = Phi(a) - T with T the tail term, so 1 - cdf = Phi(-a) + T, and both
 * of those are positive and small together in the right tail.
 *
 * This is what makes far tail bin masses mean anything. Taking a bin mass as
 * cdf(hi) - cdf(lo) works while the values are away from one, and stops
 * working the moment the tail mass drops below the spacing of doubles near
 * one: both ends round to the same number and the mass comes out as exactly
 * zero. Meanwhile the derivative of that same mass is a difference of two
 * densities, which keeps its precision for another fifteen orders. A bin
 * with zero mass and a nonzero derivative is not a rounding nuisance, it is
 * an inconsistency, and the Fisher information divides by that mass.
 */
static double irf_sf(const plidar_irf *irf, double t)
{
    double s;

    if (irf->kind == PLIDAR_IRF_GAUSSIAN) {
        s = phi_cdf(-t / irf->sigma);
    } else if (irf->kind == PLIDAR_IRF_TABLE) {
        s = table_mass(irf, t, 1);
    } else {
        s = phi_cdf(-t / irf->sigma) + emg_tail_term(irf, t);
    }
    if (s < 0.0) {
        return 0.0;
    }
    if (s > 1.0) {
        return 1.0;
    }
    return s;
}

double plidar_irf_mean(const plidar_irf *irf)
{
    if (irf == NULL) {
        return 0.0;
    }
    if (irf->kind == PLIDAR_IRF_TABLE) {
        double m1, m2;
        table_moments(irf, 0.0, &m1, &m2);
        return m1;
    }
    return (irf->kind == PLIDAR_IRF_GAUSSIAN) ? 0.0 : irf->tau;
}

double plidar_irf_variance(const plidar_irf *irf)
{
    if (irf == NULL) {
        return 0.0;
    }
    if (irf->kind == PLIDAR_IRF_TABLE) {
        /* About the mean rather than the origin, so that a response far
         * from t = 0 does not lose its width to the difference of two large
         * second moments. */
        double m1, m2;
        table_moments(irf, plidar_irf_mean(irf), &m1, &m2);
        return m2 - m1 * m1;
    }
    if (irf->kind == PLIDAR_IRF_GAUSSIAN) {
        return irf->sigma * irf->sigma;
    }
    return irf->sigma * irf->sigma + irf->tau * irf->tau;
}

double plidar_irf_peak_bound(const plidar_irf *irf)
{
    double gauss_peak;

    if (irf == NULL) {
        return 0.0;
    }
    if (irf->kind == PLIDAR_IRF_TABLE) {
        double peak = 0.0;
        for (size_t k = 0; k < irf->n; k++) {
            if (irf->table[k] > peak) {
                peak = irf->table[k];
            }
        }
        return peak;
    }
    gauss_peak = PLIDAR_INV_SQRT_2PI / irf->sigma;
    if (irf->kind == PLIDAR_IRF_GAUSSIAN) {
        return gauss_peak;
    }
    return (gauss_peak < 1.0 / irf->tau) ? gauss_peak : 1.0 / irf->tau;
}

plidar_status_t plidar_irf_bin(const plidar_irf *irf, double t0,
                               double t_start, double width, size_t nbins,
                               double *out)
{
    if (irf == NULL || out == NULL || nbins == 0u) {
        return PLIDAR_ERR_ARG;
    }
    if (!(width > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }

    /*
     * Bins left of the centre are differenced on the distribution function,
     * bins right of it on the survival function. Either way the two values
     * being subtracted are small together, which is what keeps the relative
     * precision of the result. One evaluation per edge still, with the bin
     * that straddles the centre paying for the changeover.
     */
    {
        double lo = t_start - t0;
        int right = (lo > 0.0);
        double prev = right ? irf_sf(irf, lo) : plidar_irf_cdf(irf, lo);

        for (size_t i = 0; i < nbins; i++) {
            const double hi = t_start + (double)(i + 1u) * width - t0;
            double mass;

            if (right) {
                const double here = irf_sf(irf, hi);
                mass = prev - here;
                prev = here;
            } else if (hi > 0.0) {
                const double here = irf_sf(irf, hi);
                mass = (1.0 - here) - prev;
                prev = here;
                right = 1;
            } else {
                const double here = plidar_irf_cdf(irf, hi);
                mass = here - prev;
                prev = here;
            }
            out[i] = (mass > 0.0) ? mass : 0.0;
        }
    }
    return PLIDAR_OK;
}

plidar_status_t plidar_irf_bin_dt0(const plidar_irf *irf, double t0,
                                   double t_start, double width, size_t nbins,
                                   double *out)
{
    if (irf == NULL || out == NULL || nbins == 0u) {
        return PLIDAR_ERR_ARG;
    }
    if (!(width > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }

    double prev = plidar_irf_pdf(irf, t_start - t0);
    for (size_t i = 0; i < nbins; i++) {
        const double edge = t_start + (double)(i + 1u) * width;
        const double here = plidar_irf_pdf(irf, edge - t0);
        out[i] = prev - here;
        prev = here;
    }
    return PLIDAR_OK;
}
