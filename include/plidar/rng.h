/**
 * @file rng.h
 * @brief Reproducible pseudo-random source for the photon simulator.
 *
 * Every figure in this repository has to be regenerable from a seed, so the
 * generator is part of the library rather than something borrowed from the
 * platform. PCG32 is used because its state is 128 bits, a different increment gives a
 * different stream from the same seed, and its output does not depend on the
 * C library version, which `rand` does.
 *
 * Streams matter here: the Monte Carlo simulator draws arrival times and
 * afterpulse delays from separate streams, so turning afterpulsing on does
 * not shift the arrival times that would have been drawn anyway. Without
 * that separation a comparison between two detector settings would confound
 * the physics with a different walk through the same number sequence.
 */
#ifndef PLIDAR_RNG_H
#define PLIDAR_RNG_H

#include <stdint.h>

#include "plidar/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Largest mean accepted by plidar_rng_poisson. See the note on that function. */
#define PLIDAR_POISSON_MU_MAX 100.0

typedef struct {
    uint64_t state;
    uint64_t inc; /**< stream selector, always odd */
} plidar_rng;

/**
 * Seed a generator.
 * @param seed   initial state, any value
 * @param stream stream selector; two generators with the same seed and
 *               different streams produce independent sequences
 */
void plidar_rng_seed(plidar_rng *r, uint64_t seed, uint64_t stream);

/** Uniform 32-bit word. */
uint32_t plidar_rng_u32(plidar_rng *r);

/** Uniform double in [0, 1), built from 53 random bits. */
double plidar_rng_uniform(plidar_rng *r);

/**
 * Standard normal variate by Box and Muller, one value per call.
 *
 * The transform produces a pair and this discards the second rather than
 * caching it. Caching would put a parity bit of history inside the
 * generator, so the sequence a caller sees would depend on how many normals
 * some earlier routine happened to draw. Two uniforms are cheap; a
 * reproducibility surprise is not.
 */
double plidar_rng_normal(plidar_rng *r);

/**
 * Exponential variate with the given rate, in the same time unit as 1/rate.
 * Computed through log1p so that the common case of a small uniform keeps
 * its precision.
 */
double plidar_rng_exponential(plidar_rng *r, double rate);

/**
 * Poisson variate by Knuth's product method.
 *
 * The method costs one uniform per unit of mean and loses its exponential
 * factor to underflow for large means, so the domain is capped at
 * PLIDAR_POISSON_MU_MAX. That is not a limitation in practice: the
 * simulator itself never draws a Poisson variate, because it generates
 * arrivals by thinning, and the callers that do want one are tests and
 * tooling working in the photon starved regime this library is about.
 *
 * @return PLIDAR_ERR_DOMAIN if mu is negative or above the cap.
 */
plidar_status_t plidar_rng_poisson(plidar_rng *r, double mu, uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* PLIDAR_RNG_H */
