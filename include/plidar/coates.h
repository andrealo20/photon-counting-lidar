/**
 * @file coates.h
 * @brief Undo the distortion that recording one photon per cycle imposes.
 *
 * When the timing electronics keep only the first photon of each cycle, a
 * photon in an early bin prevents any later bin from being filled in that
 * cycle. The histogram leans towards earlier times and the estimated
 * distance comes out short. The effect grows with the detection rate, which
 * is why these systems are run at a few percent detections per pulse and no
 * more.
 *
 * The distortion has an exact inverse, published by Coates in 1968. Writing
 * M for the number of cycles, N_i for the recorded counts and R_i for the
 * number of cycles still undetected when bin i begins,
 *
 *     R_0 = M,  R_{i+1} = R_i - N_i,  mu_i = log( R_i / R_{i+1} )
 *
 * recovers the expected photons per cycle per bin that the detector would
 * have seen with no first photon rule at all. It is the maximum likelihood
 * estimator of mu_i under that rule, not a fitted correction.
 *
 * It fails in one way, and the failure is real rather than numerical: if
 * every cycle produced a detection at or before some bin, nothing was ever
 * observed past it and no finite rate is consistent with the data. The
 * function reports that instead of returning an infinity.
 */
#ifndef PLIDAR_COATES_H
#define PLIDAR_COATES_H

#include <stddef.h>
#include <stdint.h>

#include "plidar/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @param hist   recorded counts, one record per cycle at most
 * @param cycles number of laser pulses, M above
 * @param mu     out, expected detected photons per cycle per bin
 * @return PLIDAR_ERR_SATURATED if the cycles run out inside the histogram
 */
plidar_status_t plidar_coates_invert(const uint64_t *hist, uint64_t cycles,
                                     double *mu, size_t nbins);

/** Same, for an already averaged histogram held as doubles. */
plidar_status_t plidar_coates_invert_d(const double *hist, double cycles,
                                       double *mu, size_t nbins);

/**
 * Forward direction, for tests and for anyone who wants the distorted shape
 * without running the simulator: expected recorded counts given the true
 * per cycle rates.
 */
plidar_status_t plidar_coates_forward(const double *mu, double cycles,
                                      double *hist, size_t nbins);

#ifdef __cplusplus
}
#endif

#endif /* PLIDAR_COATES_H */
