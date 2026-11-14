#include <math.h>

#include "plidar/crb.h"

double plidar_crb_range_sigma(double var_tof)
{
    if (!(var_tof >= 0.0)) {
        return 0.0;
    }
    return 0.5 * PLIDAR_C * sqrt(var_tof);
}

/*
 * The bounds from the information matrix of (t0, A, B), given as its upper
 * triangle in the order i00, i01, i02, i11, i12, i22.
 *
 * background_known drops B from the joint bound, and both bounds ask for it
 * when there is no background. B = 0 is then pinned down exactly by the bins
 * the return cannot reach, so its information is infinite, but those bins
 * have a rate of zero, which is what the callers skip so as not to divide by
 * it. With a Gaussian or an EMG the far tail leaves rates next to them small
 * enough that the information overflows and the cofactor below fails, or
 * pins B so well that the cofactor agrees with the pair to rounding, so
 * going straight to the pair moves their bounds by rounding at most. A table
 * is exactly zero outside its support: nothing overflows, the skipped bins
 * take their information with them, and the cofactor would return a finite
 * three parameter bound that is too pessimistic.
 */
static plidar_status_t invert_information(const double info[6], int background_known,
                                          plidar_crb_result *out)
{
    const double i00 = info[0], i01 = info[1], i02 = info[2];
    const double i11 = info[3], i12 = info[4], i22 = info[5];

    if (!(i00 > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }

    out->information = i00;
    out->var_tof_known = 1.0 / i00;

    /* Top left entry of the inverse of a symmetric 3 by 3 matrix, written as
     * its cofactor over the determinant. */
    if (!background_known) {
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

    {
        const double info[6] = {i00, i01, i02, i11, i12, i22};

        return invert_information(info, !(sc->background > 0.0), out);
    }
}

plidar_status_t plidar_crb_first_photon(const plidar_scene *sc, double cycles, double keep,
                                        double *scratch, size_t nbins,
                                        plidar_crb_result *out)
{
    plidar_status_t st;
    double *g, *dg;
    double amp_a, amp_b, p_none, w;
    double i00 = 0.0, i01 = 0.0, i02 = 0.0, i11 = 0.0, i12 = 0.0, i22 = 0.0;
    double s = 0.0, ds0 = 0.0, ds1 = 0.0, ds2 = 0.0;

    if (sc == NULL || scratch == NULL || out == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (!(cycles > 0.0) || !(keep > 0.0) || !(keep <= 1.0)) {
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

    /*
     * With S_i the rate summed over the bins before i, a cycle records in bin
     * i with probability P_i = exp(-S_i) (1 - exp(-mu_i)), and
     *
     *     dP_i = exp(-S_i) (exp(-mu_i) dmu_i - (1 - exp(-mu_i)) dS_i)
     *
     * so S_i and dS_i, kept as running sums, give every bin in one pass. Each
     * bin adds dP_i dP_i^T / P_i, written as exp(-S_i) v v^T / (1 - exp(-mu_i))
     * with v the bracket. exp(-S_i) cancels once instead of being squared and
     * divided back out, so a bin whose survival underflows in a very bright
     * record adds zero rather than 0/0.
     */
    for (size_t i = 0; i < nbins; i++) {
        const double mu = amp_a * g[i] + amp_b;
        const double d0 = amp_a * dg[i]; /* d mu / d t0 */
        const double d1 = g[i];          /* d mu / d A  */
        const double d2 = 1.0;           /* d mu / d B  */

        /* A bin no photon can reach is skipped as in plidar_crb. It still
         * enters the sums: raising B would put photons there, so it moves
         * dS for every bin after it. */
        if (mu > 0.0) {
            const double surv = exp(-s);
            const double hit = -expm1(-mu);
            const double miss = exp(-mu);
            const double v0 = miss * d0 - hit * ds0;
            const double v1 = miss * d1 - hit * ds1;
            const double v2 = miss * d2 - hit * ds2;

            i00 += surv * v0 * v0 / hit;
            i01 += surv * v0 * v1 / hit;
            i02 += surv * v0 * v2 / hit;
            i11 += surv * v1 * v1 / hit;
            i12 += surv * v1 * v2 / hit;
            i22 += surv * v2 * v2 / hit;
        }
        s += mu;
        ds0 += d0;
        ds1 += d1;
        ds2 += d2;
    }

    /*
     * Cycles with no kept record have probability 1 - q + q P_0, with
     * P_0 = exp(-S_n) and dP_0 = -P_0 dS_n, which adds
     * q^2 P_0^2 dS_n dS_n^T / (1 - q + q P_0). The denominator is at least
     * 1 - q, and at q = 1 it is P_0 itself, so the guard is for a P_0 that
     * underflows.
     */
    p_none = exp(-s);
    w = 0.0;
    if (p_none > 0.0) {
        w = keep * p_none * (keep * p_none / ((1.0 - keep) + keep * p_none));
    }

    {
        const double info[6] = {
            cycles * (keep * i00 + w * ds0 * ds0),
            cycles * (keep * i01 + w * ds0 * ds1),
            cycles * (keep * i02 + w * ds0 * ds2),
            cycles * (keep * i11 + w * ds1 * ds1),
            cycles * (keep * i12 + w * ds1 * ds2),
            cycles * (keep * i22 + w * ds2 * ds2),
        };

        return invert_information(info, !(sc->background > 0.0), out);
    }
}
