/**
 * @file estimate.h
 * @brief Three ways to read a distance out of an arrival time histogram.
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
 * **Maximum likelihood.** Counts are Poisson with mean A*g_i(t0) + B, so
 * maximise sum_i [ N_i log(A g_i + B) - (A g_i + B) ] over the time of
 * flight and over the signal and background levels at the same time. Search
 * in three stages: a matched filter for a starting point, then a coarse scan
 * of the whole period with the log matched statistic that the likelihood
 * reduces to at fixed amplitudes, then expectation maximisation on the two
 * amplitudes alternating with a golden section search on t0.
 *
 * The coarse scan covers the full period on purpose. Restricting it to a
 * neighbourhood of the matched filter peak would be faster and would hide
 * the behaviour this repository exists to measure: below a certain photon
 * budget the likelihood surface grows a competing maximum on a background
 * fluctuation, the estimator occasionally takes it, and the error jumps from
 * millimetres to metres. An estimator that is not allowed to make that
 * mistake cannot be shown making it.
 *
 * ## Pile up
 *
 * All three assume the histogram is undistorted. If the data were taken with
 * one record per cycle and the detection rate was not negligible, run
 * plidar_coates_invert first and pass the recovered rates multiplied by the
 * cycle count. Feeding a distorted histogram straight in produces a distance
 * that is short by a predictable amount, which is itself one of the figures
 * in this repository.
 *
 * ## Memory
 *
 * Nothing is allocated. Every routine that needs working space takes a
 * caller owned buffer of nbins doubles.
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

#ifdef __cplusplus
}
#endif

#endif /* PLIDAR_ESTIMATE_H */
