#include <math.h>

#include "plidar/coates.h"
#include "plidar/estimate.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

enum { NB = 2000 };

static double noiseless[NB];
static double corrected[NB];
static double scratch[NB];

#define CYCLES 1.0e6
#define T0     37.3e-9

static plidar_scene base_scene(void)
{
    plidar_scene sc;

    sc.period = 100e-9;
    sc.bin_width = 50e-12;
    sc.nbins = NB;
    sc.t0 = T0;
    sc.signal = 0.02;
    sc.background = 1e6;
    sc.efficiency = 0.3;
    (void)plidar_irf_init_gaussian(&sc.irf, 150e-12);
    return sc;
}

/*
 * The expected histogram is the model itself, so an estimator that inverts
 * the model correctly has to return the parameters that produced it. No
 * noise is involved and no tolerance is statistical: what is left is the
 * search accuracy and the template truncation.
 */
static void test_maximum_likelihood_inverts_the_model(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate est;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_IDEAL, CYCLES,
                                              noiseless, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle(&sc, noiseless, CYCLES, NB, scratch, &est));

    TEST_ASSERT_DOUBLE_WITHIN(0.01 * sc.bin_width, sc.t0, est.t0);
    TEST_ASSERT_DOUBLE_WITHIN(0.01 * sc.signal, sc.signal, est.signal);
    TEST_ASSERT_DOUBLE_WITHIN(0.01 * sc.background, sc.background, est.background);
    TEST_ASSERT_DOUBLE_WITHIN(1e-9, plidar_tof_to_range(est.t0), est.range);
}

static void test_matched_filter_finds_the_peak(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate est;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_IDEAL, CYCLES,
                                              noiseless, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_matched(&sc, noiseless, NB, scratch, &est));

    /* The parabola is an approximation to a shape that is not one, so half a
     * percent of a bin is as close as this gets and is still a tenth of a
     * millimetre in distance. */
    TEST_ASSERT_DOUBLE_WITHIN(0.05 * sc.bin_width, sc.t0, est.t0);
}

static void test_centroid_is_unbiased_on_a_symmetric_response(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate est;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_IDEAL, CYCLES,
                                              noiseless, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_centroid(&sc, noiseless, NB,
                                              3.0 * sc.irf.sigma, &est));

    TEST_ASSERT_DOUBLE_WITHIN(0.05 * sc.bin_width, sc.t0, est.t0);
}

/*
 * The result this repository exists to show, in its smallest form: on a
 * response with a diffusion tail the centroid reports a distance that is too
 * far away, and the matched filter does not, from the same histogram.
 */
static void test_the_tail_biases_the_centroid_and_not_the_matched_filter(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate centroid, matched;
    double bias_centroid, bias_matched;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&sc.irf, 60e-12, 140e-12));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_IDEAL, CYCLES,
                                              noiseless, NB));

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_centroid(&sc, noiseless, NB,
                                              3.0 * sqrt(plidar_irf_variance(&sc.irf)),
                                              &centroid));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_matched(&sc, noiseless, NB, scratch, &matched));

    bias_centroid = centroid.t0 - sc.t0;
    bias_matched = matched.t0 - sc.t0;

    TEST_ASSERT_TRUE(bias_centroid > 0.2 * sc.irf.tau);
    TEST_ASSERT_TRUE(fabs(bias_matched) < 0.1 * fabs(bias_centroid));
}

/*
 * Pile up and its undoing, end to end. The histogram is built with one
 * record per cycle at a detection rate no careful operator would use, which
 * is what makes the effect large enough to see without noise.
 */
static void test_pile_up_shortens_the_range_and_coates_restores_it(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate raw, fixed;
    const double cycles = 1.0e7;
    double bias_raw_mm, bias_fixed_mm;

    /* Forty two percent of cycles produce a record, which is an order of
     * magnitude past the few percent a careful operator would allow. The
     * distortion is then large enough to state as a distance rather than as
     * a fraction of a bin. */
    sc.signal = 1.0;
    sc.background = 8e6;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON, cycles,
                                              noiseless, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle(&sc, noiseless, cycles, NB, scratch, &raw));

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_coates_invert_d(noiseless, cycles, corrected, NB));
    for (size_t i = 0; i < NB; i++) {
        corrected[i] *= cycles;
    }
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle(&sc, corrected, cycles, NB, scratch,
                                         &fixed));

    bias_raw_mm = plidar_tof_to_range(raw.t0 - sc.t0) * 1e3;
    bias_fixed_mm = plidar_tof_to_range(fixed.t0 - sc.t0) * 1e3;

    /* Short, as the mechanism requires, by around two millimetres. */
    TEST_ASSERT_TRUE(bias_raw_mm < -1.0);
    /* And gone to under a micrometre once the distortion is inverted, which
     * is the search tolerance rather than anything left of the physics. */
    TEST_ASSERT_TRUE(fabs(bias_fixed_mm) < 1e-3);
    /* The amplitudes come back too, not only the distance. */
    TEST_ASSERT_DOUBLE_WITHIN(0.01 * sc.signal, sc.signal, fixed.signal);
    TEST_ASSERT_DOUBLE_WITHIN(0.01 * sc.background, sc.background, fixed.background);
}

static void test_estimators_agree_with_each_other_on_clean_data(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate matched, mle;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_IDEAL, CYCLES,
                                              noiseless, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_matched(&sc, noiseless, NB, scratch, &matched));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle(&sc, noiseless, CYCLES, NB, scratch, &mle));

    TEST_ASSERT_DOUBLE_WITHIN(0.1 * sc.bin_width, matched.t0, mle.t0);
}

static void test_rejects_bad_arguments(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate est;

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG,
                          plidar_est_mle(&sc, NULL, CYCLES, NB, scratch, &est));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_est_mle(&sc, noiseless, 0.0, NB, scratch, &est));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG,
                          plidar_est_matched(&sc, noiseless, 4u, scratch, &est));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_est_centroid(&sc, noiseless, NB, 0.0, &est));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_maximum_likelihood_inverts_the_model);
    RUN_TEST(test_matched_filter_finds_the_peak);
    RUN_TEST(test_centroid_is_unbiased_on_a_symmetric_response);
    RUN_TEST(test_the_tail_biases_the_centroid_and_not_the_matched_filter);
    RUN_TEST(test_pile_up_shortens_the_range_and_coates_restores_it);
    RUN_TEST(test_estimators_agree_with_each_other_on_clean_data);
    RUN_TEST(test_rejects_bad_arguments);
    return UNITY_END();
}
