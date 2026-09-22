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
    irf->kind  = PLIDAR_IRF_GAUSSIAN;
    irf->sigma = sigma;
    irf->tau   = 0.0;
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
    irf->kind  = PLIDAR_IRF_EMG;
    irf->sigma = sigma;
    irf->tau   = tau;
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

double plidar_irf_pdf(const plidar_irf *irf, double t)
{
    if (irf == NULL) {
        return 0.0;
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
    return (irf->kind == PLIDAR_IRF_GAUSSIAN) ? 0.0 : irf->tau;
}

double plidar_irf_variance(const plidar_irf *irf)
{
    if (irf == NULL) {
        return 0.0;
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
