/**
 * @file irf.h
 * @brief Instrument response function of the detector and timing chain.
 *
 * The IRF is the arrival time distribution the system would record for a
 * return of zero duration. Everything this library can resolve is set by its
 * shape, so it is worth modelling properly rather than assuming a Gaussian
 * and moving on.
 *
 * A single photon avalanche diode does not have a symmetric response. The
 * core is close to Gaussian, from the jitter of the avalanche build up and
 * of the timing electronics, but carriers generated outside the
 * multiplication region reach it by diffusion and arrive late. That adds an
 * exponential tail on the right. The standard model for the pair is the
 * exponentially modified Gaussian, a Gaussian of width sigma convolved with
 * an exponential of time constant tau:
 *
 *     pdf(t) = (1/tau) * exp(v^2/2 - t/tau) * Phi(t/sigma - v),  v = sigma/tau
 *     cdf(t) = Phi(t/sigma) - tau * pdf(t)
 *
 * with Phi the standard normal distribution function. The second line is not
 * a convenience, it is an identity: the two terms that would otherwise
 * appear in the derivative of the cdf cancel exactly. The tests use it as a
 * consistency check between the two routines.
 *
 * That tail is the reason a centroid estimator is biased late and a matched
 * filter is not, which is one of the results this repository is built to
 * show, so it has to be in the forward model from the start.
 *
 * Time is in seconds throughout, and t is measured relative to the arrival
 * of an ideal return.
 */
#ifndef PLIDAR_IRF_H
#define PLIDAR_IRF_H

#include <stddef.h>

#include "plidar/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PLIDAR_IRF_GAUSSIAN = 0, /**< symmetric, the idealisation */
    PLIDAR_IRF_EMG      = 1  /**< Gaussian core with a diffusion tail */
} plidar_irf_kind;

typedef struct {
    plidar_irf_kind kind;
    double sigma; /**< s, standard deviation of the Gaussian core */
    double tau;   /**< s, time constant of the diffusion tail, EMG only */
} plidar_irf;

plidar_status_t plidar_irf_init_gaussian(plidar_irf *irf, double sigma);
plidar_status_t plidar_irf_init_emg(plidar_irf *irf, double sigma, double tau);

/** Convert a Gaussian full width at half maximum to its standard deviation. */
double plidar_sigma_from_fwhm(double fwhm);

double plidar_irf_pdf(const plidar_irf *irf, double t);
double plidar_irf_cdf(const plidar_irf *irf, double t);

/** Mean of the distribution: 0 for the Gaussian, tau for the EMG. */
double plidar_irf_mean(const plidar_irf *irf);
/** Variance: sigma^2 for the Gaussian, sigma^2 + tau^2 for the EMG. */
double plidar_irf_variance(const plidar_irf *irf);

/**
 * An upper bound on the pdf, which the Monte Carlo simulator needs as the
 * envelope rate for thinning.
 *
 * For the Gaussian it is the exact peak. For the EMG there is no closed form
 * mode, but a convolution cannot exceed the peak of either factor times the
 * mass of the other, and both factors are normalised, so the bound is the
 * smaller of the Gaussian peak and the exponential peak. Using a bound
 * rather than the true maximum costs a few rejected candidates and costs
 * nothing in correctness.
 */
double plidar_irf_peak_bound(const plidar_irf *irf);

/**
 * Probability mass of the IRF in each bin of a uniform time grid, for a
 * return centred at t0.
 *
 * Bin i covers [t_start + i*width, t_start + (i+1)*width). The mass is taken
 * as the difference of two cdf values rather than the pdf at the midpoint,
 * so the result is exact for any bin width and the masses of a grid covering
 * the whole line sum to one.
 *
 * Values are clamped at zero: in the far tail the two cdf values are equal
 * to working precision and their difference can come out very slightly
 * negative.
 */
plidar_status_t plidar_irf_bin(const plidar_irf *irf, double t0,
                               double t_start, double width, size_t nbins,
                               double *out);

/**
 * Derivative of the binned mass with respect to t0, which is what the
 * Fisher information needs. Equal to pdf(lo - t0) - pdf(hi - t0) by the
 * fundamental theorem of calculus, with no finite differencing involved.
 */
plidar_status_t plidar_irf_bin_dt0(const plidar_irf *irf, double t0,
                                   double t_start, double width, size_t nbins,
                                   double *out);

#ifdef __cplusplus
}
#endif

#endif /* PLIDAR_IRF_H */
