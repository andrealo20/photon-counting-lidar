#include <math.h>

#include "plidar/coates.h"

plidar_status_t plidar_coates_invert_d(const double *hist, double cycles,
                                       double *mu, size_t nbins)
{
    double remaining = cycles;

    if (hist == NULL || mu == NULL || nbins == 0u) {
        return PLIDAR_ERR_ARG;
    }
    if (!(cycles > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }

    for (size_t i = 0; i < nbins; i++) {
        const double n = hist[i];
        const double after = remaining - n;

        if (!(n >= 0.0) || after < 0.0) {
            return PLIDAR_ERR_DOMAIN;
        }
        if (after <= 0.0) {
            /* No cycle survived past this bin, so the data carry no
             * information about any later one and no finite rate explains
             * this one. */
            return PLIDAR_ERR_SATURATED;
        }

        /* log(R_i / R_{i+1}) written as log1p(N_i / R_{i+1}): the ratio is
         * within 1e-6 of one in the regime these systems run at, and the
         * plain logarithm would throw away most of the digits. */
        mu[i] = log1p(n / after);
        remaining = after;
    }
    return PLIDAR_OK;
}

plidar_status_t plidar_coates_invert(const uint64_t *hist, uint64_t cycles,
                                     double *mu, size_t nbins)
{
    double remaining = (double)cycles;

    if (hist == NULL || mu == NULL || nbins == 0u) {
        return PLIDAR_ERR_ARG;
    }
    if (cycles == 0u) {
        return PLIDAR_ERR_DOMAIN;
    }

    for (size_t i = 0; i < nbins; i++) {
        const double n = (double)hist[i];
        const double after = remaining - n;

        if (after <= 0.0) {
            return PLIDAR_ERR_SATURATED;
        }
        mu[i] = log1p(n / after);
        remaining = after;
    }
    return PLIDAR_OK;
}

plidar_status_t plidar_coates_forward(const double *mu, double cycles,
                                      double *hist, size_t nbins)
{
    double cum = 0.0;

    if (mu == NULL || hist == NULL || nbins == 0u) {
        return PLIDAR_ERR_ARG;
    }
    if (!(cycles > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }

    for (size_t i = 0; i < nbins; i++) {
        if (!(mu[i] >= 0.0)) {
            return PLIDAR_ERR_DOMAIN;
        }
        hist[i] = cycles * exp(-cum) * (-expm1(-mu[i]));
        cum += mu[i];
    }
    return PLIDAR_OK;
}
