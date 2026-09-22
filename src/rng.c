#include <math.h>
#include <stddef.h>

#include "plidar/rng.h"

/* PCG32, the minimal variant. The multiplier is the one from the reference
 * implementation; the increment carries the stream and must stay odd for the
 * underlying LCG to have full period. */
#define PLIDAR_PCG_MULT 6364136223846793005ULL

void plidar_rng_seed(plidar_rng *r, uint64_t seed, uint64_t stream)
{
    if (r == NULL) {
        return;
    }
    r->state = 0u;
    r->inc   = (stream << 1u) | 1u;
    (void)plidar_rng_u32(r);
    r->state += seed;
    (void)plidar_rng_u32(r);
}

uint32_t plidar_rng_u32(plidar_rng *r)
{
    uint64_t old = r->state;
    r->state = old * PLIDAR_PCG_MULT + r->inc;

    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t)(old >> 59u);

    /* (32 - rot) & 31 rather than (-rot) & 31: same rotation, no unary minus
     * on an unsigned value for -Wconversion to complain about. */
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

double plidar_rng_uniform(plidar_rng *r)
{
    /* 53 significant bits, so the spacing of the result matches the spacing
     * of doubles in [0, 1) instead of leaving 24-bit gaps. */
    uint64_t hi = (uint64_t)(plidar_rng_u32(r) >> 5); /* 27 bits */
    uint64_t lo = (uint64_t)(plidar_rng_u32(r) >> 6); /* 26 bits */
    return (double)((hi << 26) | lo) * (1.0 / 9007199254740992.0);
}

double plidar_rng_exponential(plidar_rng *r, double rate)
{
    if (!(rate > 0.0)) {
        return 0.0;
    }
    /* uniform() is in [0, 1), so -log1p(-u) is in [0, inf) and never sees
     * log(0). log1p keeps the precision when u is small, which is the case
     * that dominates when the rate is high. */
    return -log1p(-plidar_rng_uniform(r)) / rate;
}

plidar_status_t plidar_rng_poisson(plidar_rng *r, double mu, uint32_t *out)
{
    if (r == NULL || out == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (!(mu >= 0.0) || mu > PLIDAR_POISSON_MU_MAX) {
        return PLIDAR_ERR_DOMAIN;
    }

    const double limit = exp(-mu);
    double p = 1.0;
    uint32_t k = 0u;

    for (;;) {
        p *= plidar_rng_uniform(r);
        if (p <= limit) {
            break;
        }
        k++;
    }
    *out = k;
    return PLIDAR_OK;
}

double plidar_rng_normal(plidar_rng *r)
{
    /* 1 - u rather than u so the logarithm never sees zero. */
    const double u1 = plidar_rng_uniform(r);
    const double u2 = plidar_rng_uniform(r);
    const double radius = sqrt(-2.0 * log1p(-u1));
    return radius * cos(6.28318530717958647692 * u2);
}
