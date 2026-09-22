/**
 * @file crb.h
 * @brief How well any unbiased estimator could possibly do on this scene.
 *
 * Counts in bin i are Poisson with mean M*(A*g_i(t0) + B), where M is the
 * number of cycles, g_i the IRF mass in the bin, A the signal amplitude and
 * B the background level per bin. For a Poisson model the Fisher information
 * is a sum over bins of the squared parameter sensitivity divided by the
 * mean,
 *
 *     I_jk = M * sum_i  (d mu_i / d theta_j) (d mu_i / d theta_k) / mu_i
 *
 * and the Cramer Rao bound on the variance of any unbiased estimate of theta
 * is the corresponding diagonal entry of the inverse.
 *
 * Two versions are worth having and they are not the same number:
 *
 *   known  the signal and background levels are given, and only the time of
 *          flight is estimated. This is the bound usually quoted.
 *
 *   joint  all three are estimated from the same histogram, which is what
 *          plidar_est_mle actually does. Having to pay for the nuisance
 *          parameters can only widen the bound, never narrow it.
 *
 * The derivative with respect to t0 comes from plidar_irf_bin_dt0, which is
 * the difference of two density values by the fundamental theorem of
 * calculus. Nothing here is finite differenced, so the bound does not
 * inherit a step size.
 *
 * A bound is not a promise. Below a certain photon budget the likelihood
 * grows a second maximum on a background fluctuation, the estimator
 * occasionally takes it, and the measured error leaves the bound far behind.
 * Separating that regime from the one where the bound holds is the point of
 * comparing the two.
 */
#ifndef PLIDAR_CRB_H
#define PLIDAR_CRB_H

#include <stddef.h>

#include "plidar/sim.h"
#include "plidar/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double var_tof_known; /**< s^2, signal and background known */
    double var_tof_joint; /**< s^2, all three parameters estimated */
    double information;   /**< the (t0, t0) entry of the information matrix */
} plidar_crb_result;

/**
 * @param cycles  number of laser pulses
 * @param scratch caller owned, 2*nbins doubles
 */
plidar_status_t plidar_crb(const plidar_scene *sc, double cycles,
                           double *scratch, size_t nbins,
                           plidar_crb_result *out);

/** Standard deviation in metres from a variance in seconds squared. */
double plidar_crb_range_sigma(double var_tof);

#ifdef __cplusplus
}
#endif

#endif /* PLIDAR_CRB_H */
