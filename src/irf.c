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

/*
 * log of the standard normal distribution function.
 *
 * The EMG needs Phi at arguments that run far into the left tail, where the
 * value underflows long before the expression it belongs to does. Taking the
 * logarithm keeps the whole thing in range, but only if the logarithm itself
 * is computed without first forming the underflowed value.
 *
 * Above x = -30, erfc is exact enough and nowhere near its own underflow.
 * Below it, Mills' ratio expansion is used; at that threshold the first term
 * dropped is around 1e-12 relative.
 */
static double log_phi_cdf(double x)
{
    if (x > -30.0) {
        return log(phi_cdf(x));
    }

    static const double coef[5] = {1.0, 3.0, 15.0, 105.0, 945.0};
    const double z2 = x * x;
    double zp = z2;
    double s = 1.0;

    for (int k = 0; k < 5; k++) {
        const double term = coef[k] / zp;
        s += ((k % 2) == 0) ? -term : term;
        zp *= z2;
    }
    return -0.5 * z2 - 0.5 * PLIDAR_LOG_2PI - log(-x) + log(s);
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
    return exp(0.5 * v * v - t / irf->tau + log_phi_cdf(a - v));
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

    double prev = plidar_irf_cdf(irf, t_start - t0);
    for (size_t i = 0; i < nbins; i++) {
        const double edge = t_start + (double)(i + 1u) * width;
        const double here = plidar_irf_cdf(irf, edge - t0);
        const double mass = here - prev;
        out[i] = (mass > 0.0) ? mass : 0.0;
        prev = here;
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
