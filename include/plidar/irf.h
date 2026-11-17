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
    PLIDAR_IRF_EMG      = 1, /**< Gaussian core with a diffusion tail */
    PLIDAR_IRF_TABLE    = 2  /**< piecewise linear density from samples */
} plidar_irf_kind;

typedef struct {
    plidar_irf_kind kind;
    double sigma;        /**< s, standard deviation of the Gaussian core, Gaussian and EMG only */
    double tau;          /**< s, time constant of the diffusion tail, EMG only */
    const double *table; /**< table only: 3*n doubles, caller owned, see plidar_irf_init_table */
    size_t n;            /**< table only: number of nodes */
    double t_first;      /**< table only: s, time of node 0 */
    double dt;           /**< table only: s, node spacing */
} plidar_irf;

plidar_status_t plidar_irf_init_gaussian(plidar_irf *irf, double sigma);
plidar_status_t plidar_irf_init_emg(plidar_irf *irf, double sigma, double tau);

/**
 * A response given as samples of its density, for a shape neither model
 * above describes, such as one measured on the instrument.
 *
 * Node k sits at t_first + k*dt and holds samples[k]. Between nodes the
 * density is linear and outside the first and last node it is zero. The
 * samples can be in any units: they are normalised here to unit area by the
 * trapezoidal rule, which is exact for a piecewise linear density.
 *
 * Both end samples must be zero. That keeps the density continuous, and the
 * continuity is what makes plidar_irf_bin_dt0, which takes the density at the
 * two edges of a bin, the exact derivative of the mass plidar_irf_bin gives
 * for that bin. A density that jumped at the end of its support would put a
 * kink in the masses that no density value describes.
 *
 * work holds 3*n doubles and must not overlap samples. It receives the
 * normalised density at the nodes in [0, n), the mass left of each node in
 * [n, 2n), and the mass right of each node in [2n, 3n), the last summed from
 * the right end. plidar_irf_bin differences the bins right of the centre on
 * the survival function, and summing from that end keeps the far tail a sum
 * of small terms with its relative precision intact. The centre is t = 0, so
 * the origin of the table belongs inside the bulk of the response, for
 * instance at the half height point of the rising edge.
 *
 * work becomes irf->table and must outlive irf and every copy of it: a scene
 * passed by value shares the table rather than duplicating it. samples is not
 * referenced after the call.
 *
 * irf is written only on success, and work only once every check has passed.
 * Null pointers and n < 2 give PLIDAR_ERR_ARG. A dt that is not positive and
 * finite, a t_first that is not finite, a sample that is negative or not
 * finite, a nonzero end sample, or an area that is zero or overflows give
 * PLIDAR_ERR_DOMAIN.
 */
plidar_status_t plidar_irf_init_table(plidar_irf *irf, const double *samples, size_t n,
                                      double t_first, double dt, double *work);

/** Convert a Gaussian full width at half maximum to its standard deviation. */
double plidar_sigma_from_fwhm(double fwhm);

double plidar_irf_pdf(const plidar_irf *irf, double t);
double plidar_irf_cdf(const plidar_irf *irf, double t);

/**
 * Mean of the distribution: 0 for the Gaussian, tau for the EMG. For a table
 * it is integrated cell by cell in closed form on every call.
 */
double plidar_irf_mean(const plidar_irf *irf);
/**
 * Variance: sigma^2 for the Gaussian, sigma^2 + tau^2 for the EMG. For a
 * table it is integrated like the mean, about the mean.
 */
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
 *
 * For a table it is the largest node value, which a piecewise linear density
 * reaches exactly.
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
