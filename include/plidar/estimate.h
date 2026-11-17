/**
 * @file estimate.h
 * @brief Three ways to read a distance out of an arrival time histogram, the
 * third under two models of the counts.
 *
 * They differ in how much of the forward model they use, and the whole point
 * of putting them side by side is that the difference is measurable.
 *
 * **Centroid.** Find the fullest bin, take a window around it, subtract the
 * background estimated from everything outside that window, and compute a
 * weighted mean of the bin centres. It costs nothing and a cheap
 * rangefinder does roughly this. It is biased late, because the diffusion
 * tail of the IRF puts mass on one side only and a mean does not care that
 * the shape is asymmetric.
 *
 * **Matched filter.** Cross correlate the histogram with the binned IRF and
 * take the peak, interpolated to sub bin resolution by fitting a parabola to
 * the three correlation values around it. This uses the shape, so the tail
 * no longer biases it. It is the optimal linear estimator when the noise is
 * additive and Gaussian. Here the noise is neither.
 *
 * **Maximum likelihood.** In plidar_est_mle counts are Poisson with mean
 * A*g_i(t0) + B, so maximise sum_i [ N_i log(A g_i + B) - (A g_i + B) ] over
 * the time of flight and over the signal and background levels at the same
 * time. Search in three stages: a matched filter for a starting point, then a
 * coarse scan of the whole period with the log matched statistic that the
 * likelihood reduces to at fixed amplitudes, then expectation maximisation on
 * the two amplitudes alternating with a golden section search on t0.
 * plidar_est_mle_first_photon fits the first photon model instead, in which
 * each cycle records at most one photon and the counts are multinomial over
 * the bins and the cycles that recorded nothing. It estimates the same three
 * parameters in the same three stages, its coarse scan using the weights of
 * that model, and refits the amplitudes by Newton steps at every t0 the
 * golden section tries.
 *
 * The coarse scan covers every candidate the template fits behind, rather
 * than a neighbourhood of the matched filter peak. Restricting it would be
 * faster and would hide the behaviour this repository exists to measure:
 * below a certain photon budget the likelihood surface grows a competing
 * maximum on a background fluctuation, the estimator occasionally takes it,
 * and the error jumps from millimetres to metres. An estimator that is not
 * allowed to make that mistake cannot be shown making it.
 *
 * "Every candidate the template fits behind" is not quite the whole period:
 * the candidates stop a template half width short of either end. The half
 * width is 10 standard deviations of a parametric response: 31 bins of
 * 50 ps, 1.55 ns or 23 cm of range, for the response of the simulated
 * sweeps. For a table it is the longer of the two sides of the support,
 * measured from the origin: 40 bins of 750 ps for the one fitted to the
 * recorded frame, whose whole support spans 46. Only the matched filter is
 * held to the candidates. It refuses a return nearer an end than that,
 * since the edge of its search is all it could report. Both likelihood fits
 * can reach such a return: their golden section moves t0 past the first or
 * the last candidate, with the window held at that end of the record. A
 * return far inside that zone can still leave them on a wrong t0.
 *
 * ## Pile up
 *
 * The centroid, the matched filter and plidar_est_mle assume the histogram
 * is undistorted. If the data were taken with one record per cycle and the
 * detection rate was not negligible, run plidar_coates_invert first and pass
 * the recovered rates multiplied by the cycle count. Feeding a distorted
 * histogram straight in produces a distance that is short by a predictable
 * amount, which is itself one of the figures in this repository.
 *
 * Once most cycles record, that pair still weighs the bins wrongly: the late
 * bins are seen only by the few cycles still listening, and the Poisson fit
 * weighs them as if all of them were. plidar_est_mle_first_photon fits the
 * first photon model to the counts as recorded instead, so it needs no
 * correction in front and weighs each bin by the cycles that could still
 * record in it.
 *
 * ## Memory
 *
 * Nothing is allocated. Every routine that needs working space takes a
 * caller owned buffer, sized in its own comment.
 */
#ifndef PLIDAR_ESTIMATE_H
#define PLIDAR_ESTIMATE_H

#include <stddef.h>
#include <stdint.h>

#include "plidar/sim.h"
#include "plidar/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Half width of the IRF template, in units of its standard deviation. */
#define PLIDAR_TEMPLATE_SIGMAS 10.0

/** Golden section stops when the bracket is this fraction of a bin. */
#define PLIDAR_MLE_TOL_BINS 1e-4

typedef struct {
    double   t0;         /**< s, estimated time of flight */
    double   range;      /**< m, the same thing as a distance */
    double   signal;     /**< photons per cycle, maximum likelihood only */
    double   background; /**< photons per second, maximum likelihood only */
    double   loglik;     /**< maximum likelihood only */
    uint32_t iterations; /**< amplitude updates actually taken */
} plidar_estimate;

/**
 * Centre of mass of the return.
 *
 * @param tmpl        supplies the time grid and the IRF; its t0, signal and
 *                    background fields are not read
 * @param counts      histogram, nbins entries, counts or rates alike
 * @param half_window s, half width of the averaging window. Roughly three
 *                    standard deviations of the IRF is a sensible choice:
 *                    wider lets in background, narrower clips the tail and
 *                    trades one bias for another.
 */
plidar_status_t plidar_est_centroid(const plidar_scene *tmpl, const double *counts,
                                    size_t nbins, double half_window,
                                    plidar_estimate *out);

/**
 * Peak of the cross correlation with the IRF, interpolated by parabola.
 *
 * A peak on the first or the last candidate gives PLIDAR_ERR_DOMAIN and no
 * estimate: the return lies at or beyond the reach of the template.
 *
 * @param scratch caller owned, nbins doubles
 */
plidar_status_t plidar_est_matched(const plidar_scene *tmpl, const double *counts,
                                   size_t nbins, double *scratch,
                                   plidar_estimate *out);

/**
 * Joint maximum likelihood over time of flight, signal and background.
 *
 * @param cycles  laser pulses the histogram was accumulated over, needed to
 *                turn the fitted amplitudes back into physical units
 * @param scratch caller owned, nbins doubles
 */
plidar_status_t plidar_est_mle(const plidar_scene *tmpl, const double *counts,
                               double cycles, size_t nbins, double *scratch,
                               plidar_estimate *out);

/**
 * Joint maximum likelihood of the first photon model on the counts as
 * recorded, with no pile up correction in front.
 *
 * loglik is the multinomial log likelihood, its log q terms included. Counts
 * that are negative or not finite, or that add up to more than cycles, give
 * PLIDAR_ERR_DOMAIN; counts that add up to zero give PLIDAR_ERR_NO_CONVERGE.
 *
 * @param cycles  laser pulses of the whole acquisition, M
 * @param keep    probability a recorded count was kept, in (0, 1]: 1 for a
 *                histogram as recorded, 1/K for one of K groups of a
 *                multinomial split
 * @param scratch caller owned, 4*nbins doubles
 */
plidar_status_t plidar_est_mle_first_photon(const plidar_scene *tmpl, const double *counts,
                                            double cycles, double keep, size_t nbins,
                                            double *scratch, plidar_estimate *out);

#ifdef __cplusplus
}
#endif

#endif /* PLIDAR_ESTIMATE_H */
