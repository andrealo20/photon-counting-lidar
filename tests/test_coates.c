#include <math.h>

#include "plidar/coates.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

enum { NB = 512 };

static double truth[NB];
static double recorded[NB];
static double recovered[NB];

static void fill_truth(double peak, double base)
{
    for (size_t i = 0; i < NB; i++) {
        const double d = ((double)i - 200.0) / 12.0;
        truth[i] = base + peak * exp(-0.5 * d * d);
    }
}

static void test_round_trip_is_exact(void)
{
    const double cycles = 4e6;

    /* A detection probability near a quarter per cycle is far past what a
     * real instrument would be run at, and that is the point: the inverse is
     * exact rather than a small signal approximation, so it has to hold
     * where the distortion is severe. */
    fill_truth(2e-3, 3e-4);

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_coates_forward(truth, cycles, recorded, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_coates_invert_d(recorded, cycles, recovered, NB));

    for (size_t i = 0; i < NB; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-12 * truth[i] + 1e-18, truth[i],
                                  recovered[i]);
    }
}

static void test_distortion_is_towards_earlier_bins(void)
{
    const double cycles = 1e6;
    double ideal_mean = 0.0, ideal_mass = 0.0;
    double seen_mean = 0.0, seen_mass = 0.0;

    fill_truth(2e-3, 3e-4);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_coates_forward(truth, cycles, recorded, NB));

    for (size_t i = 0; i < NB; i++) {
        ideal_mass += truth[i];
        ideal_mean += truth[i] * (double)i;
        seen_mass += recorded[i];
        seen_mean += recorded[i] * (double)i;
    }
    ideal_mean /= ideal_mass;
    seen_mean /= seen_mass;

    /* Photons that arrive first hide the ones behind them, so the recorded
     * histogram has to sit earlier than the true one. A system that reported
     * this as a distance would report it short. */
    TEST_ASSERT_TRUE(seen_mean < ideal_mean);
}

static void test_low_rate_limit_is_the_plain_histogram(void)
{
    const double cycles = 1e7;

    /* At a detection probability of a few parts in ten thousand the
     * correction should do almost nothing, which is the regime these
     * instruments are deliberately operated in. */
    fill_truth(2e-6, 3e-7);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_coates_forward(truth, cycles, recorded, NB));

    for (size_t i = 0; i < NB; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(2e-3 * truth[i] * cycles,
                                  truth[i] * cycles, recorded[i]);
    }
}

static void test_more_counts_than_cycles_is_rejected(void)
{
    const double cycles = 1000.0;
    double bad[4] = {400.0, 700.0, 0.0, 0.0};
    double out[4];

    /* One record per cycle is the rule the inverse is built on, so a
     * histogram holding more records than there were cycles is not distorted
     * data, it is impossible data. */
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_coates_invert_d(bad, cycles, out, 4u));
}

static void test_saturation_is_reported(void)
{
    const double cycles = 1000.0;
    double sat[8] = {400.0, 400.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double out[8];

    /* Every cycle has produced a detection by the end of the third bin, so
     * no finite rate explains that bin and nothing at all is known about the
     * ones after it. */
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_SATURATED,
                          plidar_coates_invert_d(sat, cycles, out, 8u));
}

static void test_integer_and_double_paths_agree(void)
{
    const uint64_t cycles = 100000u;
    uint64_t hist[16];
    double as_double[16];
    double a[16], b[16];

    for (size_t i = 0; i < 16u; i++) {
        hist[i] = (uint64_t)(100u + 37u * i);
        as_double[i] = (double)hist[i];
    }
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_coates_invert(hist, cycles, a, 16u));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_coates_invert_d(as_double, (double)cycles, b, 16u));
    for (size_t i = 0; i < 16u; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, a[i], b[i]);
    }
}

static void test_rejects_bad_arguments(void)
{
    double one = 1.0;
    uint64_t h = 1u;

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG,
                          plidar_coates_invert_d(NULL, 10.0, &one, 1u));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_coates_invert_d(&one, 0.0, &one, 1u));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_coates_invert(&h, 0u, &one, 1u));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG,
                          plidar_coates_forward(&one, 10.0, NULL, 1u));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_round_trip_is_exact);
    RUN_TEST(test_distortion_is_towards_earlier_bins);
    RUN_TEST(test_low_rate_limit_is_the_plain_histogram);
    RUN_TEST(test_saturation_is_reported);
    RUN_TEST(test_more_counts_than_cycles_is_rejected);
    RUN_TEST(test_integer_and_double_paths_agree);
    RUN_TEST(test_rejects_bad_arguments);
    return UNITY_END();
}
