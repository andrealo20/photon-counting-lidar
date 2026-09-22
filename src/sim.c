#include <math.h>
#include <string.h>

#include "plidar/sim.h"

double plidar_range_to_tof(double range_m)
{
    return 2.0 * range_m / PLIDAR_C;
}

double plidar_tof_to_range(double tof_s)
{
    return 0.5 * PLIDAR_C * tof_s;
}

double plidar_unambiguous_range(const plidar_scene *sc)
{
    if (sc == NULL) {
        return 0.0;
    }
    return plidar_tof_to_range(sc->period);
}

plidar_status_t plidar_scene_check(const plidar_scene *sc)
{
    if (sc == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (sc->nbins == 0u) {
        return PLIDAR_ERR_ARG;
    }
    if (!(sc->period > 0.0) || !(sc->bin_width > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    /* The histogram has to tile the repetition period. A grid that stops
     * short would silently discard late returns, and one that runs past the
     * end would hold bins no photon can ever reach. */
    {
        const double span = (double)sc->nbins * sc->bin_width;
        if (fabs(span - sc->period) > 1e-9 * sc->period) {
            return PLIDAR_ERR_DOMAIN;
        }
    }
    if (!(sc->t0 >= 0.0) || !(sc->t0 < sc->period)) {
        return PLIDAR_ERR_DOMAIN;
    }
    if (!(sc->signal >= 0.0) || !(sc->background >= 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    if (!(sc->efficiency > 0.0) || !(sc->efficiency <= 1.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    if (!(sc->irf.sigma > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    if (sc->irf.kind == PLIDAR_IRF_EMG && !(sc->irf.tau > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    return PLIDAR_OK;
}

plidar_status_t plidar_scene_rate(const plidar_scene *sc, double t0,
                                  double *mu, size_t nbins)
{
    plidar_status_t st;

    if (sc == NULL || mu == NULL) {
        return PLIDAR_ERR_ARG;
    }
    if (nbins != sc->nbins) {
        return PLIDAR_ERR_ARG;
    }
    st = plidar_scene_check(sc);
    if (st != PLIDAR_OK) {
        return st;
    }

    /* The IRF mass lands in mu first and is scaled in place, so no scratch
     * buffer is needed and the function stays allocation free like the rest
     * of the library. */
    st = plidar_irf_bin(&sc->irf, t0, 0.0, sc->bin_width, nbins, mu);
    if (st != PLIDAR_OK) {
        return st;
    }

    {
        const double a = sc->efficiency * sc->signal;
        const double b = sc->efficiency * sc->background * sc->bin_width;
        for (size_t i = 0; i < nbins; i++) {
            mu[i] = a * mu[i] + b;
        }
    }
    return PLIDAR_OK;
}

plidar_status_t plidar_sim_analytic(const plidar_scene *sc, plidar_sim_mode mode,
                                    double cycles, double *out, size_t nbins)
{
    plidar_status_t st;

    if (sc == NULL || out == NULL || !(cycles > 0.0)) {
        return PLIDAR_ERR_ARG;
    }
    st = plidar_scene_rate(sc, sc->t0, out, nbins);
    if (st != PLIDAR_OK) {
        return st;
    }

    if (mode == PLIDAR_SIM_IDEAL) {
        for (size_t i = 0; i < nbins; i++) {
            out[i] *= cycles;
        }
        return PLIDAR_OK;
    }

    /* P_i = exp(-sum_{j<i} mu_j) * (1 - exp(-mu_i)). The second factor is
     * written with expm1 because mu_i is routinely around 1e-6 here, where
     * 1 - exp(-mu) loses most of its significant digits. */
    {
        double cum = 0.0;
        for (size_t i = 0; i < nbins; i++) {
            const double mu = out[i];
            out[i] = cycles * exp(-cum) * (-expm1(-mu));
            cum += mu;
        }
    }
    return PLIDAR_OK;
}

/* Keep the cycle's arrival times ordered as they are generated. The counts
 * involved are single digits in the regime this library is about, so an
 * insertion sort is both the fastest and the smallest thing to do. */
static void insert_sorted(double *t, size_t n, double value)
{
    size_t i = n;
    while (i > 0u && t[i - 1u] > value) {
        t[i] = t[i - 1u];
        i--;
    }
    t[i] = value;
}

plidar_status_t plidar_sim_mc(const plidar_scene *sc, const plidar_detector *det,
                              uint64_t cycles, plidar_rng *rng,
                              uint64_t *hist, size_t nbins, uint64_t *recorded)
{
    plidar_status_t st;
    plidar_rng arrivals;
    plidar_rng afterpulses;
    double times[PLIDAR_MC_MAX_PER_CYCLE];
    double t_ready = -1.0; /* absolute time the detector is armed again */
    uint64_t hits = 0u;

    if (det == NULL || hist == NULL || rng == NULL) {
        return PLIDAR_ERR_ARG;
    }
    st = plidar_scene_check(sc);
    if (st != PLIDAR_OK) {
        return st;
    }
    if (nbins != sc->nbins) {
        return PLIDAR_ERR_ARG;
    }
    if (!(det->dead_time >= 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    if (det->afterpulse_prob > 0.0 && !(det->afterpulse_tau > 0.0)) {
        return PLIDAR_ERR_DOMAIN;
    }
    if (sc->efficiency * sc->signal > PLIDAR_POISSON_MU_MAX ||
        sc->efficiency * sc->background * sc->period > PLIDAR_POISSON_MU_MAX) {
        return PLIDAR_ERR_DOMAIN;
    }

    /* Two streams off one caller seed. Arrival times and afterpulse draws
     * then advance independently, so switching afterpulsing on does not
     * change which photons arrive. The two words are drawn in separate
     * statements because the order of evaluation inside one expression is
     * not the compiler's to promise. */
    {
        uint64_t seed = (uint64_t)plidar_rng_u32(rng);
        seed = (seed << 32) | (uint64_t)plidar_rng_u32(rng);
        plidar_rng_seed(&arrivals, seed, 1u);
        plidar_rng_seed(&afterpulses, seed, 2u);
    }

    memset(hist, 0, nbins * sizeof(hist[0]));

    for (uint64_t cycle = 0u; cycle < cycles; cycle++) {
        const double base = (double)cycle * sc->period;
        size_t n = 0u;
        size_t next = 0u;
        uint32_t k = 0u;
        int recorded_here = 0;
        double pending_ap = -1.0;

        /* Background: a homogeneous process over the cycle. */
        st = plidar_rng_poisson(&arrivals,
                                sc->efficiency * sc->background * sc->period, &k);
        if (st != PLIDAR_OK) {
            return st;
        }
        for (uint32_t j = 0u; j < k; j++) {
            if (n >= PLIDAR_MC_MAX_PER_CYCLE) {
                return PLIDAR_ERR_DOMAIN;
            }
            insert_sorted(times, n++, plidar_rng_uniform(&arrivals) * sc->period);
        }

        /* Signal: the photon count is Poisson in the mean the return
         * carries, and each time is drawn from the construction of the IRF
         * itself, a normal jitter plus a diffusion delay. Photons falling
         * outside the cycle are dropped, which is the marking property of a
         * Poisson process and needs no correction. */
        st = plidar_rng_poisson(&arrivals, sc->efficiency * sc->signal, &k);
        if (st != PLIDAR_OK) {
            return st;
        }
        for (uint32_t j = 0u; j < k; j++) {
            double t = sc->t0 + sc->irf.sigma * plidar_rng_normal(&arrivals);
            if (sc->irf.kind == PLIDAR_IRF_EMG) {
                t += plidar_rng_exponential(&arrivals, 1.0 / sc->irf.tau);
            }
            if (t < 0.0 || t >= sc->period) {
                continue;
            }
            if (n >= PLIDAR_MC_MAX_PER_CYCLE) {
                return PLIDAR_ERR_DOMAIN;
            }
            insert_sorted(times, n++, t);
        }

        for (;;) {
            double event;

            /* Take whichever comes first, the next real photon or an
             * afterpulse left over from a detection earlier in the cycle. */
            if (pending_ap >= 0.0 && (next >= n || pending_ap <= times[next])) {
                event = pending_ap;
                pending_ap = -1.0;
            } else if (next < n) {
                event = times[next++];
            } else {
                break;
            }

            {
                const double abs_t = base + event;

                if (det->dead_time > 0.0 && abs_t < t_ready) {
                    /* A paralyzable detector is retriggered by photons it
                     * cannot report, so the blind interval keeps extending
                     * and a bright enough scene shuts it down completely. */
                    if (det->paralyzable) {
                        t_ready = abs_t + det->dead_time;
                    }
                    continue;
                }
                t_ready = abs_t + det->dead_time;
            }

            if (det->first_photon_only == 0 || recorded_here == 0) {
                const size_t bin = (size_t)(event / sc->bin_width);
                if (bin < nbins) {
                    hist[bin]++;
                }
                if (recorded_here == 0) {
                    hits++;
                    recorded_here = 1;
                }
            }

            if (det->afterpulse_prob > 0.0 &&
                plidar_rng_uniform(&afterpulses) < det->afterpulse_prob) {
                const double delay =
                    plidar_rng_exponential(&afterpulses, 1.0 / det->afterpulse_tau);
                const double ap = event + delay;
                if (ap < sc->period) {
                    pending_ap = ap;
                }
            }
        }
    }

    if (recorded != NULL) {
        *recorded = hits;
    }
    return PLIDAR_OK;
}
