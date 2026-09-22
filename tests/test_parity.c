/*
 * The load bearing test of this repository.
 *
 * Two paths produce the same histogram and share almost nothing. One
 * evaluates the IRF distribution function and applies the first photon
 * expression in closed form. The other draws a Poisson number of photons per
 * cycle, places each one as a normal jitter plus a diffusion delay, and lets
 * a detector model decide what gets recorded. If the closed forms in irf.c
 * or the pile up expression in sim.c were wrong, the two would disagree and
 * the chi squared statistic below would say so.
 *
 * The statistic is compared against a fixed threshold rather than a p value:
 * the seeds are fixed, so the test either passes every time or fails every
 * time, and a flaky statistical test in continuous integration is worse than
 * no test.
 */
#include <math.h>

#include "plidar/sim.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

enum { NB = 1000 };

static double expected[NB];
static uint64_t observed[NB];

static plidar_scene make_scene(int emg)
{
    plidar_scene sc;

    sc.period = 100e-9;      /* 10 MHz repetition, 15 m unambiguous */
    sc.bin_width = 100e-12;  /* coarse on purpose: the test is about counts */
    sc.nbins = NB;
    sc.t0 = 37.3e-9;
    sc.signal = 0.02;
    sc.background = 2e6;
    sc.efficiency = 0.3;
    if (emg) {
        (void)plidar_irf_init_emg(&sc.irf, 200e-12, 350e-12);
    } else {
        (void)plidar_irf_init_gaussian(&sc.irf, 200e-12);
    }
    return sc;
}

static plidar_detector ideal_detector(int first_photon)
{
    plidar_detector d;
    d.dead_time = 0.0;
    d.paralyzable = 0;
    d.first_photon_only = first_photon;
    d.afterpulse_prob = 0.0;
    d.afterpulse_tau = 0.0;
    return d;
}

/*
 * Pearson statistic over the bins the asymptotic distribution actually
 * applies to. Bins with fewer than five expected counts are dropped rather
 * than pooled: with a thousand bins there are enough left, and pooling would
 * make the degrees of freedom depend on the scene.
 */
static double chi_squared(size_t *dof)
{
    double acc = 0.0;
    size_t used = 0u;

    for (size_t i = 0; i < NB; i++) {
        if (expected[i] < 5.0) {
            continue;
        }
        {
            const double d = (double)observed[i] - expected[i];
            acc += d * d / expected[i];
            used++;
        }
    }
    *dof = used;
    return acc;
}

/* A chi squared variable with k degrees of freedom has mean k and variance
 * 2k, so this is how many standard deviations the statistic sits from where
 * it belongs. Anything inside five is agreement. */
static double sigmas_off(double chi2, size_t dof)
{
    return (chi2 - (double)dof) / sqrt(2.0 * (double)dof);
}

static void run_parity(int emg, int first_photon, uint64_t cycles, uint64_t seed)
{
    plidar_scene sc = make_scene(emg);
    plidar_detector det = ideal_detector(first_photon);
    plidar_rng rng;
    size_t dof = 0u;
    double chi2, off;

    plidar_rng_seed(&rng, seed, 0u);

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc,
                                              first_photon ? PLIDAR_SIM_FIRST_PHOTON
                                                           : PLIDAR_SIM_IDEAL,
                                              (double)cycles, expected, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_mc(&sc, &det, cycles, &rng, observed, NB, NULL));

    chi2 = chi_squared(&dof);
    off = sigmas_off(chi2, dof);
    TEST_ASSERT_TRUE(dof > 900u);
    TEST_ASSERT_TRUE(fabs(off) < 5.0);
}

static void test_gaussian_ideal(void)
{
    run_parity(0, 0, 2000000u, 1u);
}

static void test_gaussian_first_photon(void)
{
    run_parity(0, 1, 2000000u, 2u);
}

static void test_emg_ideal(void)
{
    run_parity(1, 0, 2000000u, 3u);
}

static void test_emg_first_photon(void)
{
    run_parity(1, 1, 2000000u, 4u);
}

/*
 * Does the parity test have the power to catch the thing it is there to
 * catch?
 *
 * It is a fair question, and at the detection rate of the cases above the
 * answer is no. Running the same observed histogram against the model with
 * the pile up term removed entirely, at six percent of cycles recording,
 * moves the statistic by 4.3 standard deviations, which is inside the five
 * the gate allows. A missing pile up term would pass.
 *
 * The rate is what fixes that, not the number of cycles. The scene below
 * records a photon in 36 percent of cycles, where the same control moves the
 * statistic by 987 standard deviations. So the test is run twice: once as a
 * parity check, which has to agree, and once against the model with pile up
 * removed, which has to disagree by a margin that leaves no room for doubt.
 */
static void test_parity_can_tell_pile_up_apart(void)
{
    plidar_scene sc = make_scene(1);
    plidar_detector det = ideal_detector(1);
    plidar_rng rng;
    const uint64_t cycles = 2000000u;
    size_t dof = 0u;
    double chi2, matching, mismatching;

    sc.signal = 0.5;
    sc.background = 1e7; /* together, 36 percent of cycles record something */

    plidar_rng_seed(&rng, 7u, 0u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_mc(&sc, &det, cycles, &rng, observed, NB, NULL));

    /* chi_squared fills dof, so its result is taken into a variable before
     * sigmas_off is called. Passing both as arguments of one call would
     * leave the compiler free to read dof first, and it does. */
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON,
                                              (double)cycles, expected, NB));
    chi2 = chi_squared(&dof);
    matching = sigmas_off(chi2, dof);

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_IDEAL,
                                              (double)cycles, expected, NB));
    chi2 = chi_squared(&dof);
    mismatching = sigmas_off(chi2, dof);

    TEST_ASSERT_TRUE(fabs(matching) < 5.0);
    TEST_ASSERT_TRUE(mismatching > 100.0);
}

static void test_detection_probability_matches_the_closed_form(void)
{
    plidar_scene sc = make_scene(1);
    plidar_detector det = ideal_detector(1);
    plidar_rng rng;
    const uint64_t cycles = 2000000u;
    uint64_t recorded = 0u;
    double predicted = 0.0;

    plidar_rng_seed(&rng, 5u, 0u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON,
                                              (double)cycles, expected, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_mc(&sc, &det, cycles, &rng, observed, NB,
                                        &recorded));

    for (size_t i = 0; i < NB; i++) {
        predicted += expected[i];
    }
    /* The number of cycles that produced a record is a binomial count, so
     * its standard deviation is under 500 here. Five of those is the gate. */
    TEST_ASSERT_DOUBLE_WITHIN(5.0 * sqrt(predicted), predicted, (double)recorded);
}

static void test_dead_time_only_removes_counts(void)
{
    plidar_scene sc = make_scene(1);
    plidar_detector open_det = ideal_detector(0);
    plidar_detector dead_det = ideal_detector(0);
    plidar_rng rng;
    const uint64_t cycles = 500000u;
    uint64_t with_dead = 0u, without = 0u;

    dead_det.dead_time = 50e-9;

    plidar_rng_seed(&rng, 11u, 0u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_mc(&sc, &open_det, cycles, &rng, observed, NB,
                                        NULL));
    for (size_t i = 0; i < NB; i++) {
        without += observed[i];
    }

    plidar_rng_seed(&rng, 11u, 0u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_mc(&sc, &dead_det, cycles, &rng, observed, NB,
                                        NULL));
    for (size_t i = 0; i < NB; i++) {
        with_dead += observed[i];
    }

    /* Same seed, same arrivals. A blind interval can only ever throw
     * detections away, so this is a strict inequality and not a statistical
     * statement. */
    TEST_ASSERT_TRUE(with_dead < without);
}

static void test_afterpulsing_only_adds_counts(void)
{
    plidar_scene sc = make_scene(1);
    plidar_detector plain = ideal_detector(0);
    plidar_detector noisy = ideal_detector(0);
    plidar_rng rng;
    const uint64_t cycles = 200000u;
    static uint64_t baseline[NB];
    uint64_t base_total = 0u, noisy_total = 0u;

    noisy.afterpulse_prob = 0.05;
    noisy.afterpulse_tau = 3e-9;

    plidar_rng_seed(&rng, 21u, 0u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_mc(&sc, &plain, cycles, &rng, baseline, NB,
                                        NULL));

    plidar_rng_seed(&rng, 21u, 0u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_mc(&sc, &noisy, cycles, &rng, observed, NB,
                                        NULL));

    /* Afterpulse draws come from their own stream, so the photons that arrive
     * are the same ones, and with every detection recorded and no dead time
     * every one of them still lands in the bin it landed in before. Switching
     * afterpulsing on can therefore only add counts, never move or remove
     * them, and checking that bin by bin is a stronger statement about the
     * two streams being separate than comparing totals would be. */
    for (size_t i = 0; i < NB; i++) {
        TEST_ASSERT_TRUE(observed[i] >= baseline[i]);
        base_total += baseline[i];
        noisy_total += observed[i];
    }
    TEST_ASSERT_TRUE(noisy_total > base_total);

    /* And it adds about as many as it should: one afterpulse per twenty
     * detections, give or take the ones whose delay lands them outside the
     * record. */
    {
        const double added = (double)(noisy_total - base_total);
        const double predicted = 0.05 * (double)base_total;
        TEST_ASSERT_DOUBLE_WITHIN(0.2 * predicted, predicted, added);
    }
}

static void test_scene_validation(void)
{
    plidar_scene sc = make_scene(0);
    plidar_detector det = ideal_detector(0);
    plidar_rng rng;

    plidar_rng_seed(&rng, 1u, 0u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_scene_check(&sc));

    sc.t0 = sc.period;
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, plidar_scene_check(&sc));
    sc = make_scene(0);

    sc.bin_width *= 2.0; /* grid no longer tiles the period */
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, plidar_scene_check(&sc));
    sc = make_scene(0);

    sc.efficiency = 1.5;
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_sim_mc(&sc, &det, 10u, &rng, observed, NB, NULL));
}

static void test_range_and_time_of_flight_are_inverses(void)
{
    plidar_scene sc = make_scene(0);

    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 5.0,
                              plidar_tof_to_range(plidar_range_to_tof(5.0)));
    /* 10 MHz gives just under 15 m before a return folds back. */
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, 14.9896229, plidar_unambiguous_range(&sc));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_gaussian_ideal);
    RUN_TEST(test_gaussian_first_photon);
    RUN_TEST(test_emg_ideal);
    RUN_TEST(test_emg_first_photon);
    RUN_TEST(test_parity_can_tell_pile_up_apart);
    RUN_TEST(test_detection_probability_matches_the_closed_form);
    RUN_TEST(test_dead_time_only_removes_counts);
    RUN_TEST(test_afterpulsing_only_adds_counts);
    RUN_TEST(test_scene_validation);
    RUN_TEST(test_range_and_time_of_flight_are_inverses);
    return UNITY_END();
}
