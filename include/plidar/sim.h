/**
 * @file sim.h
 * @brief Forward model of a photon counting depth measurement, twice over.
 *
 * A pulsed laser fires at a fixed repetition rate. Within each cycle the
 * detected photons form an inhomogeneous Poisson process of rate
 *
 *     lambda(t) = eta * ( S * irf(t - t0) + B )
 *
 * where t0 = 2d/c is the time of flight, S the mean number of signal photons
 * a cycle would deliver to a perfect detector, B the flat rate of ambient
 * light and dark counts, and eta the detection efficiency. Arrival times are
 * accumulated into a histogram over millions of cycles. That histogram, and
 * nothing else, is what a depth estimator gets to see.
 *
 * Two independent paths produce that histogram, and the parity test between
 * them is the main correctness argument of this repository:
 *
 *   plidar_sim_analytic  works from the closed form. Bin masses come from
 *                        differences of the IRF distribution function, and
 *                        pile up is applied with the exact first photon
 *                        expression below.
 *
 *   plidar_sim_mc        works event by event. Signal and background are two
 *                        independent Poisson processes, so they are drawn
 *                        separately and merged. Background times are uniform
 *                        over the cycle. Signal times come from the way the
 *                        EMG is defined, a normal variate plus an
 *                        exponential one, added to t0. The merged stream
 *                        then goes through a detector model with dead time
 *                        and afterpulsing.
 *
 * The two paths share the IRF parameters and nothing else. The analytic one
 * evaluates the distribution function; the Monte Carlo one never calls it,
 * nor the density, because it samples from the construction the
 * distribution came from. So the parity test between them is a test of the
 * closed forms in irf.c against the definition of the distribution, not two
 * spellings of the same code. An error in either shows up as a chi squared
 * mismatch.
 *
 * ## Pile up
 *
 * Classic time correlated single photon counting electronics record the
 * first photon of a cycle and ignore the rest. Early photons therefore mask
 * late ones, the histogram leans towards earlier bins, and the estimated
 * distance comes out short. Writing mu_i for the expected number of detected
 * photons in bin i per cycle, the probability that the recorded photon of a
 * cycle lands in bin i is exactly
 *
 *     P_i = exp( - sum_{j<i} mu_j ) * ( 1 - exp( -mu_i ) )
 *
 * This is not an approximation. It is the probability that no photon arrived
 * before bin i and at least one arrived in it. plidar_coates_invert undoes
 * it.
 *
 * ## What the closed form does not cover
 *
 * The analytic path assumes the detector is armed at the start of every
 * cycle, so it describes the first photon mechanism alone. Dead time that
 * reaches across a cycle boundary, paralyzable quenching and afterpulsing
 * exist only in the Monte Carlo path, which is the point of having it.
 *
 * Neither path wraps returns that fall past the end of the repetition
 * period: mass beyond the window is dropped rather than aliased back. Both
 * drop it the same way, so parity is unaffected, but a scene with t0 close
 * to the period is not physically meaningful here.
 */
#ifndef PLIDAR_SIM_H
#define PLIDAR_SIM_H

#include <stddef.h>
#include <stdint.h>

#include "plidar/irf.h"
#include "plidar/rng.h"
#include "plidar/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Speed of light in vacuum, m/s, exact by definition of the metre. */
#define PLIDAR_C 299792458.0

typedef struct {
    double     period;      /**< s, one over the laser repetition rate */
    double     bin_width;   /**< s, timing resolution of the TDC */
    size_t     nbins;       /**< bins covering [0, nbins*bin_width) */
    double     t0;          /**< s, true time of flight, 2d/c */
    double     signal;      /**< photons per cycle in the return, before eta */
    double     background;  /**< photons per second, ambient plus dark counts */
    double     efficiency;  /**< eta, detection probability of an arriving photon */
    plidar_irf irf;
} plidar_scene;

/** Photons a single cycle may deliver before the simulator gives up. */
#define PLIDAR_MC_MAX_PER_CYCLE 256u

typedef struct {
    double dead_time;       /**< s, blind interval after a detection, 0 disables */
    int    paralyzable;     /**< non zero: photons arriving during the blind interval extend it */
    int    first_photon_only; /**< non zero: the TDC records one photon per cycle */
    double afterpulse_prob; /**< probability that a detection is followed by a spurious one */
    double afterpulse_tau;  /**< s, mean delay of that spurious detection */
} plidar_detector;

typedef enum {
    PLIDAR_SIM_IDEAL       = 0, /**< every detected photon is recorded */
    PLIDAR_SIM_FIRST_PHOTON = 1 /**< one record per cycle, pile up applied */
} plidar_sim_mode;

/** Round trip time of flight for a distance in metres, and its inverse. */
double plidar_range_to_tof(double range_m);
double plidar_tof_to_range(double tof_s);

/** Largest distance that does not fold back into the histogram. */
double plidar_unambiguous_range(const plidar_scene *sc);

/** A scene is usable if every field is in range and the grid covers the period. */
plidar_status_t plidar_scene_check(const plidar_scene *sc);

/**
 * Expected number of detected photons per cycle in each bin, for a
 * hypothesised time of flight. This is mu_i above, the quantity every
 * estimator and the Fisher information are written in terms of.
 */
plidar_status_t plidar_scene_rate(const plidar_scene *sc, double t0,
                                  double *mu, size_t nbins);

/**
 * Expected histogram after a given number of cycles, in closed form.
 * @param out expected counts, nbins entries, not necessarily integers
 */
plidar_status_t plidar_sim_analytic(const plidar_scene *sc, plidar_sim_mode mode,
                                    double cycles, double *out, size_t nbins);

/**
 * Sampled histogram, one cycle at a time, through the detector model.
 *
 * @param cycles    number of laser pulses
 * @param recorded  optional, receives the number of cycles that produced at
 *                  least one recorded photon. Divided by cycles this is the
 *                  detection probability per pulse, the number that has to
 *                  stay low for pile up to stay negligible.
 */
plidar_status_t plidar_sim_mc(const plidar_scene *sc, const plidar_detector *det,
                              uint64_t cycles, plidar_rng *rng,
                              uint64_t *hist, size_t nbins, uint64_t *recorded);

#ifdef __cplusplus
}
#endif

#endif /* PLIDAR_SIM_H */
