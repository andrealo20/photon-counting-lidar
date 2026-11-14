/*
 * A tabulated response shaped like a measured return, for the tests that need
 * a table rather than one of the two models.
 *
 * With x = t/unit, the density is
 *
 *     s(x) = r(x) * [ f(x) + 0.04 * (1 - f(x)) * exp(-max(x - 9, 0)/8) ]
 *     r(x) = 1/(1 + exp(-x/1.138)),  f(x) = 1/(1 + exp(x - 9))
 *
 * r is the rising edge, from 10 to 90 percent in 5 units with its half height
 * point at x = 0, f ends a flat top about 9 units wide at half height, and the
 * last term is a long tail at about 4 percent of the peak that decays over 8
 * units. Nothing about it is symmetric or close to a Gaussian, which is the
 * reason to have it.
 *
 * The nodes sit every eighth of a unit over [-6, 40], with the two end nodes
 * set to zero as plidar_irf_init_table requires. The function is static, so
 * include this only where it is called, or -Wunused-function warns.
 */
#ifndef PLIDAR_FIXTURE_TABLE_H
#define PLIDAR_FIXTURE_TABLE_H

#include <math.h>

#include "plidar/irf.h"

enum { FIXTURE_NODES = 369 }; /* support [-6, 40] units, 8 nodes per unit */

static plidar_status_t fixture_flat_top(plidar_irf *irf, double unit,
                                        double *work /* 3*FIXTURE_NODES */)
{
    double s[FIXTURE_NODES];

    for (int k = 0; k < FIXTURE_NODES; k++) {
        const double x = -6.0 + (double)k / 8.0;
        const double r = 1.0 / (1.0 + exp(-x / 1.138));
        const double f = 1.0 / (1.0 + exp(x - 9.0));
        s[k] = r * (f + 0.04 * (1.0 - f) * exp(-fmax(x - 9.0, 0.0) / 8.0));
    }
    s[0] = 0.0;
    s[FIXTURE_NODES - 1] = 0.0;
    return plidar_irf_init_table(irf, s, FIXTURE_NODES, -6.0 * unit, unit / 8.0,
                                 work);
}

#endif /* PLIDAR_FIXTURE_TABLE_H */
