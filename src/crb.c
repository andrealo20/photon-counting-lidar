#include <math.h>

#include "plidar/crb.h"

double plidar_crb_range_sigma(double var_tof)
{
    if (!(var_tof >= 0.0)) {
        return 0.0;
    }
    return 0.5 * PLIDAR_C * sqrt(var_tof);
}

plidar_status_t plidar_crb(const plidar_scene *sc, double cycles,
                           double *scratch, size_t nbins,
                           plidar_crb_result *out)
{
    plidar_status_t st;
    double *g, *dg;
    double amp_a, amp_b;
    double i00 = 0.0, i01 = 0.0, i02 = 0.0, i11 = 0.0, i12 = 0.0, i22 = 0.0;

    if (sc == NULL || scratch == NULL || out == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (!(cycles > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    st = plidar_scene_check(sc);
    if (st != PLIDAR_OK) {
        return st;
    }
    if (nbins != sc->nbins) {
        return PLIDAR_ERR_ARG;
    }

    g = scratch;
    dg = scratch + nbins;

    st = plidar_irf_bin(&sc->irf, sc->t0, 0.0, sc->bin_width, nbins, g);
    if (st != PLIDAR_OK) {
        return st;
    }
    st = plidar_irf_bin_dt0(&sc->irf, sc->t0, 0.0, sc->bin_width, nbins, dg);
    if (st != PLIDAR_OK) {
        return st;
    }

    amp_a = sc->efficiency * sc->signal;
    amp_b = sc->efficiency * sc->background * sc->bin_width;

    for (size_t i = 0; i < nbins; i++) {
        const double mu = amp_a * g[i] + amp_b;
        double d0, d1, d2;

        /* A bin that no photon of either kind can reach carries no
         * information and would divide by zero. With any background at all
         * this never triggers, which is why the noiseless case has to be
         * asked for deliberately. */
        if (!(mu > 0.0)) {
            continue;
        }
        d0 = amp_a * dg[i]; /* d mu / d t0 */
        d1 = g[i];          /* d mu / d A  */
        d2 = 1.0;           /* d mu / d B  */

        i00 += d0 * d0 / mu;
        i01 += d0 * d1 / mu;
        i02 += d0 * d2 / mu;
        i11 += d1 * d1 / mu;
        i12 += d1 * d2 / mu;
        i22 += d2 * d2 / mu;
    }

    i00 *= cycles;
    i01 *= cycles;
    i02 *= cycles;
    i11 *= cycles;
    i12 *= cycles;
    i22 *= cycles;

    if (!(i00 > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }

    out->information = i00;
    out->var_tof_known = 1.0 / i00;

    /* Top left entry of the inverse of a symmetric 3 by 3 matrix, written as
     * its cofactor over the determinant. */
    {
        const double c00 = i11 * i22 - i12 * i12;
        const double c01 = i02 * i12 - i01 * i22;
        const double c02 = i01 * i12 - i02 * i11;
        const double det = i00 * c00 + i01 * c01 + i02 * c02;

        if (isfinite(det) && isfinite(c00) && det > 0.0 && c00 > 0.0) {
            out->var_tof_joint = c00 / det;
            return PLIDAR_OK;
        }
    }

    /*
     * The three parameter system degenerates when the background goes to
     * zero. Every bin the return does not reach then has an expected count
     * of zero, and observing nothing there pins the background level down
     * exactly, so the information about it runs away and the matrix stops
     * being invertible in floating point.
     *
     * That is a real feature of the model rather than a numerical accident,
     * and it has a limit: a background known exactly leaves two parameters
     * to estimate, not three. Dropping the background from the system is
     * therefore the right thing to do, not a fallback. The remaining pair is
     * close to orthogonal, because the derivative of the response integrates
     * to zero, so the answer lands near the bound with everything known.
     */
    {
        const double det2 = i00 * i11 - i01 * i01;

        if (!(det2 > 0.0)) {
            return PLIDAR_ERR_DOMAIN;
        }
        out->var_tof_joint = i11 / det2;
    }
    return PLIDAR_OK;
}
