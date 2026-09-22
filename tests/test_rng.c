#include <math.h>

#include "plidar/rng.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

#define N_DRAWS 200000

static void test_same_seed_same_sequence(void)
{
    plidar_rng a, b;
    plidar_rng_seed(&a, 12345u, 1u);
    plidar_rng_seed(&b, 12345u, 1u);

    for (int i = 0; i < 1000; i++) {
        TEST_ASSERT_EQUAL_UINT32(plidar_rng_u32(&a), plidar_rng_u32(&b));
    }
}

static void test_streams_diverge(void)
{
    plidar_rng a, b;
    int same = 0;

    plidar_rng_seed(&a, 12345u, 1u);
    plidar_rng_seed(&b, 12345u, 2u);

    for (int i = 0; i < 1000; i++) {
        if (plidar_rng_u32(&a) == plidar_rng_u32(&b)) {
            same++;
        }
    }
    /* Two independent 32 bit streams agree by chance about once in four
     * billion draws, so a thousand of them agreeing even once would be a
     * sign the stream selector is not reaching the generator. */
    TEST_ASSERT_EQUAL_INT(0, same);
}

static void test_uniform_range_and_moments(void)
{
    plidar_rng r;
    double sum = 0.0, sumsq = 0.0;

    plidar_rng_seed(&r, 7u, 1u);
    for (int i = 0; i < N_DRAWS; i++) {
        const double u = plidar_rng_uniform(&r);
        TEST_ASSERT_TRUE(u >= 0.0 && u < 1.0);
        sum += u;
        sumsq += u * u;
    }
    {
        const double mean = sum / N_DRAWS;
        const double var = sumsq / N_DRAWS - mean * mean;
        /* Standard error of the mean is 1/sqrt(12*N), around 6.5e-4 here. */
        TEST_ASSERT_DOUBLE_WITHIN(4e-3, 0.5, mean);
        TEST_ASSERT_DOUBLE_WITHIN(4e-3, 1.0 / 12.0, var);
    }
}

static void test_exponential_moments(void)
{
    plidar_rng r;
    const double rate = 4.0;
    double sum = 0.0, sumsq = 0.0;

    plidar_rng_seed(&r, 99u, 3u);
    for (int i = 0; i < N_DRAWS; i++) {
        const double x = plidar_rng_exponential(&r, rate);
        TEST_ASSERT_TRUE(x >= 0.0);
        sum += x;
        sumsq += x * x;
    }
    {
        const double mean = sum / N_DRAWS;
        const double var = sumsq / N_DRAWS - mean * mean;
        TEST_ASSERT_DOUBLE_WITHIN(5e-3, 1.0 / rate, mean);
        TEST_ASSERT_DOUBLE_WITHIN(1e-2, 1.0 / (rate * rate), var);
    }
}

static void test_normal_moments(void)
{
    plidar_rng r;
    double sum = 0.0, sumsq = 0.0, sum4 = 0.0;

    plidar_rng_seed(&r, 4242u, 5u);
    for (int i = 0; i < N_DRAWS; i++) {
        const double z = plidar_rng_normal(&r);
        sum += z;
        sumsq += z * z;
        sum4 += z * z * z * z;
    }
    {
        const double mean = sum / N_DRAWS;
        const double var = sumsq / N_DRAWS - mean * mean;
        /* The fourth moment is checked as well: a transform that got the
         * radius wrong can still produce a symmetric unit variance shape. */
        const double m4 = sum4 / N_DRAWS;
        TEST_ASSERT_DOUBLE_WITHIN(1e-2, 0.0, mean);
        TEST_ASSERT_DOUBLE_WITHIN(2e-2, 1.0, var);
        TEST_ASSERT_DOUBLE_WITHIN(1e-1, 3.0, m4);
    }
}

static void test_poisson_moments(void)
{
    plidar_rng r;
    const double mu = 0.35;
    double sum = 0.0, sumsq = 0.0;

    plidar_rng_seed(&r, 31337u, 7u);
    for (int i = 0; i < N_DRAWS; i++) {
        uint32_t k = 0u;
        TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_rng_poisson(&r, mu, &k));
        sum += (double)k;
        sumsq += (double)k * (double)k;
    }
    {
        const double mean = sum / N_DRAWS;
        const double var = sumsq / N_DRAWS - mean * mean;
        /* Mean and variance are equal for a Poisson variate, which is the
         * property a broken implementation is most likely to lose. */
        TEST_ASSERT_DOUBLE_WITHIN(1e-2, mu, mean);
        TEST_ASSERT_DOUBLE_WITHIN(1e-2, mu, var);
    }
}

static void test_poisson_zero_and_domain(void)
{
    plidar_rng r;
    uint32_t k = 99u;

    plidar_rng_seed(&r, 1u, 1u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_rng_poisson(&r, 0.0, &k));
    TEST_ASSERT_EQUAL_UINT32(0u, k);

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, plidar_rng_poisson(&r, -1.0, &k));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_rng_poisson(&r, PLIDAR_POISSON_MU_MAX + 1.0, &k));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_same_seed_same_sequence);
    RUN_TEST(test_streams_diverge);
    RUN_TEST(test_uniform_range_and_moments);
    RUN_TEST(test_exponential_moments);
    RUN_TEST(test_normal_moments);
    RUN_TEST(test_poisson_moments);
    RUN_TEST(test_poisson_zero_and_domain);
    return UNITY_END();
}
