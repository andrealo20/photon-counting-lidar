/*
 * Cost report. Prints a markdown table so a continuous integration run can
 * paste it straight into the job summary, which is where the numbers in the
 * README come from.
 */
#include <stdio.h>
#include <time.h>

#include "plidar/coates.h"
#include "plidar/crb.h"
#include "plidar/estimate.h"

enum { NB = 2000 };

static double hist_d[NB];
static double scratch[2 * NB];
static uint64_t hist_u[NB];

/* clock() rather than clock_gettime(). It is plain C99, so the bench needs
 * no feature test macro and no per platform reasoning about which one. Every
 * figure below is a loop long enough that the coarser resolution does not
 * show, and processor time is the more honest number for a single threaded
 * benchmark anyway. */
static double seconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static plidar_scene scene(void)
{
    plidar_scene sc;
    sc.period = 100e-9;
    sc.bin_width = 50e-12;
    sc.nbins = NB;
    sc.t0 = 37.3e-9;
    sc.signal = 0.02;
    sc.background = 1e6;
    sc.efficiency = 0.3;
    (void)plidar_irf_init_emg(&sc.irf, 60e-12, 140e-12);
    return sc;
}

int main(void)
{
    plidar_scene sc = scene();
    plidar_detector det;
    plidar_rng rng;
    plidar_estimate est;
    plidar_crb_result crb;
    const uint64_t cycles = 2000000u;
    double t;

    det.dead_time = 50e-9;
    det.paralyzable = 0;
    det.first_photon_only = 1;
    det.afterpulse_prob = 0.02;
    det.afterpulse_tau = 3e-9;

    printf("| step | work | time (ms) | rate |\n");
    printf("|---|---|---|---|\n");

    plidar_rng_seed(&rng, 1u, 0u);
    t = seconds();
    (void)plidar_sim_mc(&sc, &det, cycles, &rng, hist_u, NB, NULL);
    t = seconds() - t;
    printf("| simulate | %llu cycles | %.1f | %.2f Mcycle/s |\n",
           (unsigned long long)cycles, t * 1e3, (double)cycles / t / 1e6);

    t = seconds();
    for (int i = 0; i < 100; i++) {
        (void)plidar_coates_invert(hist_u, cycles, hist_d, NB);
    }
    t = (seconds() - t) / 100.0;
    printf("| correct pile up | %d bins | %.4f | %.1f Mbin/s |\n", NB, t * 1e3,
           (double)NB / t / 1e6);

    for (size_t i = 0; i < NB; i++) {
        hist_d[i] *= (double)cycles;
    }

    t = seconds();
    for (int i = 0; i < 100; i++) {
        (void)plidar_est_matched(&sc, hist_d, NB, scratch, &est);
    }
    t = (seconds() - t) / 100.0;
    printf("| matched filter | %d bins | %.3f | %.0f estimates/s |\n", NB, t * 1e3,
           1.0 / t);

    t = seconds();
    for (int i = 0; i < 20; i++) {
        (void)plidar_est_mle(&sc, hist_d, (double)cycles, NB, scratch, &est);
    }
    t = (seconds() - t) / 20.0;
    printf("| maximum likelihood | %d bins, %u updates | %.3f | %.0f estimates/s |\n",
           NB, est.iterations, t * 1e3, 1.0 / t);

    t = seconds();
    for (int i = 0; i < 100; i++) {
        (void)plidar_crb(&sc, (double)cycles, scratch, NB, &crb);
    }
    t = (seconds() - t) / 100.0;
    printf("| bound | %d bins | %.3f | %.0f evaluations/s |\n", NB, t * 1e3, 1.0 / t);

    printf("\nestimate %.4f ns, bound %.1f mm, %u amplitude updates\n",
           est.t0 * 1e9, plidar_crb_range_sigma(crb.var_tof_joint) * 1e3,
           est.iterations);
    return 0;
}
