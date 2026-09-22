#include <math.h>

#include "plidar/crb.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

enum { NB = 5000 };

static double scratch[2 * NB];
static double rate_true[NB];
static double rate_shift[NB];

#define CYCLES 1.0e6

static plidar_scene base_scene(void)
{
    plidar_scene sc;

    sc.period = 100e-9;
    sc.bin_width = 20e-12;
    sc.nbins = NB;
    sc.t0 = 40.0e-9;
    sc.signal = 0.02;
    sc.background = 1e6;
    sc.efficiency = 0.3;
    (void)plidar_irf_init_gaussian(&sc.irf, 200e-12);
    return sc;
}

/*
 * With no background the Fisher information for a Gaussian response collapses
 * to N/sigma^2, where N is the mean number of detected signal photons. The
 * timing precision is then the width of the response divided by the square
 * root of the photon count, which is the headline claim of the whole
 * approach: resolution well below the width of the instrument response, paid
 * for in photons.
 */
static void test_background_free_limit(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;
    double expected_var, photons;

    sc.background = 0.0;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb(&sc, CYCLES, scratch, NB, &crb));

    photons = CYCLES * sc.efficiency * sc.signal;
    expected_var = sc.irf.sigma * sc.irf.sigma / photons;

    /* The bins are a tenth of the response width, so the sum sits within a
     * percent of the integral it approximates. */
    TEST_ASSERT_DOUBLE_WITHIN(0.02 * expected_var, expected_var,
                              crb.var_tof_known);
}

static void test_precision_beats_the_response_width(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;
    double sigma_range, photons;

    sc.background = 0.0;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb(&sc, CYCLES, scratch, NB, &crb));

    photons = CYCLES * sc.efficiency * sc.signal;
    sigma_range = plidar_crb_range_sigma(crb.var_tof_known);

    /* Six thousand photons against a 200 ps response: the bound is under a
     * millimetre, while a single photon would place the return to about
     * three centimetres. */
    TEST_ASSERT_TRUE(photons > 1000.0);
    TEST_ASSERT_TRUE(sigma_range < 1e-3);
    TEST_ASSERT_TRUE(sigma_range > 0.0);
}

static void test_variance_scales_inversely_with_cycles(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result one, two;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &one));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb(&sc, 2.0 * CYCLES, scratch, NB, &two));

    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * one.var_tof_known, one.var_tof_known,
                              2.0 * two.var_tof_known);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * one.var_tof_joint, one.var_tof_joint,
                              2.0 * two.var_tof_joint);
}

static void test_nuisance_parameters_cost_something(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &crb));

    /* Estimating the signal and background levels from the same histogram
     * cannot make the time of flight easier to pin down. */
    TEST_ASSERT_TRUE(crb.var_tof_joint >= crb.var_tof_known);
    /* With a thousand bins of background to fix the level from, it is not
     * much harder either. */
    TEST_ASSERT_TRUE(crb.var_tof_joint < 1.05 * crb.var_tof_known);
}

static void test_more_background_is_never_better(void)
{
    plidar_scene dark = base_scene();
    plidar_scene bright = base_scene();
    plidar_crb_result a, b;

    bright.background = 20e6;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&dark, CYCLES, scratch, NB, &a));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&bright, CYCLES, scratch, NB, &b));

    TEST_ASSERT_TRUE(b.var_tof_known > a.var_tof_known);
}

/*
 * Independent route to the same number. The expected log likelihood as a
 * function of a hypothesised time of flight has its maximum at the true one,
 * and the curvature there is the Fisher information. Taking that curvature
 * with a finite difference uses none of the analytic derivatives in irf.c,
 * so agreeing with the closed form is evidence about both.
 */
static double expected_loglik(const plidar_scene *sc, double hypothesis)
{
    double acc = 0.0;

    (void)plidar_scene_rate(sc, sc->t0, rate_true, NB);
    (void)plidar_scene_rate(sc, hypothesis, rate_shift, NB);

    for (size_t i = 0; i < NB; i++) {
        const double mean = CYCLES * rate_true[i];
        const double model = CYCLES * rate_shift[i];
        if (model > 0.0) {
            acc += mean * log(model) - model;
        }
    }
    return acc;
}

static void test_information_matches_the_curvature_of_the_likelihood(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;
    const double step = sc.bin_width;
    double up, mid, down, curvature;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &crb));

    up = expected_loglik(&sc, sc.t0 + step);
    mid = expected_loglik(&sc, sc.t0);
    down = expected_loglik(&sc, sc.t0 - step);
    curvature = -(up - 2.0 * mid + down) / (step * step);

    TEST_ASSERT_DOUBLE_WITHIN(0.01 * crb.information, crb.information, curvature);
}

static void test_rejects_bad_arguments(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_crb(NULL, CYCLES, scratch, NB, &crb));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, plidar_crb(&sc, 0.0, scratch, NB, &crb));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_crb(&sc, CYCLES, scratch, 7u, &crb));
    TEST_ASSERT_DOUBLE_WITHIN(1e-30, 0.0, plidar_crb_range_sigma(-1.0));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_background_free_limit);
    RUN_TEST(test_precision_beats_the_response_width);
    RUN_TEST(test_variance_scales_inversely_with_cycles);
    RUN_TEST(test_nuisance_parameters_cost_something);
    RUN_TEST(test_more_background_is_never_better);
    RUN_TEST(test_information_matches_the_curvature_of_the_likelihood);
    RUN_TEST(test_rejects_bad_arguments);
    return UNITY_END();
}
