#include <math.h>

#include "plidar/irf.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

#define SIGMA 42e-12
#define TAU   100e-12

static const double INV_SQRT_2PI = 0.39894228040143267794;

/* Central difference of the distribution function.
 *
 * The tolerances below are set by this, not by the library. A central
 * difference carries a truncation error of order (step/scale)^2 times the
 * ratio of the third derivative to the first, and in the tails of the
 * response that ratio is an order of magnitude above one over the width
 * squared. A step of sigma/200 then lands around one part in ten thousand,
 * which is where the assertions sit. Shrinking the step further trades that
 * error for a cancellation error and does not help. */
static double numeric_pdf(const plidar_irf *irf, double t, double step)
{
    return (plidar_irf_cdf(irf, t + step) - plidar_irf_cdf(irf, t - step)) /
           (2.0 * step);
}

static void test_gaussian_closed_forms(void)
{
    plidar_irf g;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_gaussian(&g, SIGMA));

    TEST_ASSERT_DOUBLE_WITHIN(1e-6 * INV_SQRT_2PI / SIGMA,
                              INV_SQRT_2PI / SIGMA, plidar_irf_pdf(&g, 0.0));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.5, plidar_irf_cdf(&g, 0.0));
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 1.0, plidar_irf_cdf(&g, 40.0 * SIGMA));
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.0, plidar_irf_cdf(&g, -40.0 * SIGMA));
    TEST_ASSERT_DOUBLE_WITHIN(1e-30, 0.0, plidar_irf_mean(&g));
    TEST_ASSERT_DOUBLE_WITHIN(1e-30, SIGMA * SIGMA, plidar_irf_variance(&g));
}

static void test_fwhm_conversion(void)
{
    /* A 100 ps full width at half maximum is the number a datasheet quotes;
     * everything inside works in standard deviations. */
    const double sigma = plidar_sigma_from_fwhm(100e-12);
    plidar_irf g;
    double half;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_gaussian(&g, sigma));
    half = plidar_irf_pdf(&g, 50e-12) / plidar_irf_pdf(&g, 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.5, half);
}

static void test_emg_density_is_the_derivative_of_its_distribution(void)
{
    plidar_irf e;
    const double step = SIGMA / 200.0;
    const double offsets[] = {-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0, 12.0};

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&e, SIGMA, TAU));

    /* The distribution function is written as Phi(t/sigma) minus tau times
     * the density, so this checks that the algebra behind that shortcut is
     * right, not that two copies of one formula agree. */
    for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        const double t = offsets[i] * SIGMA;
        const double analytic = plidar_irf_pdf(&e, t);
        const double numeric = numeric_pdf(&e, t, step);
        TEST_ASSERT_DOUBLE_WITHIN(1e-3 * (analytic + 1e-30) + 1e-6, analytic,
                                  numeric);
    }
}

static void test_emg_moments_from_the_binned_shape(void)
{
    plidar_irf e;
    enum { NB = 4000 };
    static double bins[NB];
    const double width = 8.0 * (SIGMA + TAU) / (double)NB;
    const double start = -10.0 * SIGMA;
    double mass = 0.0, mean = 0.0, second = 0.0;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&e, SIGMA, TAU));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_irf_bin(&e, 0.0, start, width, NB, bins));

    for (size_t i = 0; i < NB; i++) {
        const double centre = start + ((double)i + 0.5) * width;
        mass += bins[i];
        mean += bins[i] * centre;
        second += bins[i] * centre * centre;
    }
    mean /= mass;
    second = second / mass - mean * mean;

    /* The grid covers the shape to better than a part in a thousand, and
     * what is left out is the far tail, so the moments are recovered rather
     * than merely approached. */
    TEST_ASSERT_DOUBLE_WITHIN(1e-3, 1.0, mass);
    TEST_ASSERT_DOUBLE_WITHIN(0.02 * TAU, plidar_irf_mean(&e), mean);
    TEST_ASSERT_DOUBLE_WITHIN(0.05 * plidar_irf_variance(&e),
                              plidar_irf_variance(&e), second);
}

static void test_emg_becomes_gaussian_as_the_tail_vanishes(void)
{
    plidar_irf g, e;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_gaussian(&g, SIGMA));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&e, SIGMA, SIGMA / 1e4));

    for (double k = -4.0; k <= 4.0; k += 0.5) {
        const double t = k * SIGMA;
        TEST_ASSERT_DOUBLE_WITHIN(1e-3 * plidar_irf_pdf(&g, 0.0),
                                  plidar_irf_pdf(&g, t), plidar_irf_pdf(&e, t));
        TEST_ASSERT_DOUBLE_WITHIN(1e-3, plidar_irf_cdf(&g, t),
                                  plidar_irf_cdf(&e, t));
    }
}

/*
 * The Gaussian limit again, but pushed far enough to break an implementation
 * that forms the two halves of the exponent separately. Writing v for
 * sigma/tau, both halves are close to v^2/2, so adding and subtracting them
 * throws away every digit they share: at v = 1e8 that leaves the answer
 * wrong by tens of percent, and past v = 1e154 it overflows to a not a
 * number. Neither ratio is physical, but a sweep that drives tau towards
 * zero to watch the response become Gaussian passes through all of them.
 *
 * The tolerance is proportional to 1/v because that is the real distance
 * between the two distributions at this ratio, the mean of the EMG being tau
 * rather than zero.
 */
static void test_extreme_ratio_stays_accurate(void)
{
    static const double ratios[] = {1e4, 1e6, 1e8, 1e12};
    plidar_irf g;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_gaussian(&g, SIGMA));

    for (size_t r = 0; r < sizeof(ratios) / sizeof(ratios[0]); r++) {
        plidar_irf e;
        const double v = ratios[r];
        TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&e, SIGMA, SIGMA / v));

        for (double k = -3.0; k <= 3.0; k += 1.0) {
            const double t = k * SIGMA;
            const double want = plidar_irf_pdf(&g, t);
            const double got = plidar_irf_pdf(&e, t);
            TEST_ASSERT_DOUBLE_WITHIN((100.0 / v) * want, want, got);
            TEST_ASSERT_DOUBLE_WITHIN(100.0 / v, plidar_irf_cdf(&g, t),
                                      plidar_irf_cdf(&e, t));
        }
    }

    /* And at a ratio past anything the arithmetic can hold, the answers are
     * still numbers. */
    {
        plidar_irf e;
        TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                              plidar_irf_init_emg(&e, SIGMA, SIGMA / 1e150));
        for (double k = -3.0; k <= 3.0; k += 1.0) {
            const double f = plidar_irf_pdf(&e, k * SIGMA);
            const double F = plidar_irf_cdf(&e, k * SIGMA);
            TEST_ASSERT_TRUE(isfinite(f) && f >= 0.0);
            TEST_ASSERT_TRUE(isfinite(F) && F >= 0.0 && F <= 1.0);
        }
    }
}

static void test_far_tails_stay_finite(void)
{
    plidar_irf e;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&e, SIGMA, TAU));

    /* The two factors of the density diverge and vanish separately, so the
     * point of assembling them in the exponent is that neither of these
     * comes back as an infinity or a not a number. */
    for (double k = -200.0; k <= 200.0; k += 10.0) {
        const double t = k * SIGMA;
        const double f = plidar_irf_pdf(&e, t);
        const double F = plidar_irf_cdf(&e, t);
        TEST_ASSERT_TRUE(isfinite(f) && f >= 0.0);
        TEST_ASSERT_TRUE(isfinite(F) && F >= 0.0 && F <= 1.0);
    }
}

static void test_peak_bound_is_a_bound(void)
{
    plidar_irf e, g;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_gaussian(&g, SIGMA));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&e, SIGMA, TAU));

    for (double k = -20.0; k <= 60.0; k += 0.25) {
        TEST_ASSERT_TRUE(plidar_irf_pdf(&g, k * SIGMA) <=
                         plidar_irf_peak_bound(&g) * (1.0 + 1e-12));
        TEST_ASSERT_TRUE(plidar_irf_pdf(&e, k * SIGMA) <=
                         plidar_irf_peak_bound(&e) * (1.0 + 1e-12));
    }
}

static void test_bin_derivative_matches_a_finite_difference(void)
{
    plidar_irf e;
    enum { NB = 64 };
    static double up[NB], down[NB], deriv[NB];
    const double width = 16e-12;
    const double start = 0.0;
    const double t0 = 32.0 * width;
    const double step = width / 100.0;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&e, SIGMA, TAU));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_irf_bin(&e, t0 + step, start, width, NB, up));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_irf_bin(&e, t0 - step, start, width, NB, down));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_irf_bin_dt0(&e, t0, start, width, NB, deriv));

    for (size_t i = 0; i < NB; i++) {
        const double numeric = (up[i] - down[i]) / (2.0 * step);
        TEST_ASSERT_DOUBLE_WITHIN(1e-3 * fabs(deriv[i]) + 1e-9, deriv[i], numeric);
    }
}

static void test_rejects_bad_parameters(void)
{
    plidar_irf x;
    double one;

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_irf_init_gaussian(NULL, SIGMA));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, plidar_irf_init_gaussian(&x, 0.0));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, plidar_irf_init_emg(&x, SIGMA, 0.0));

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_gaussian(&x, SIGMA));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG,
                          plidar_irf_bin(&x, 0.0, 0.0, 1e-12, 0u, &one));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_irf_bin(&x, 0.0, 0.0, 0.0, 1u, &one));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_gaussian_closed_forms);
    RUN_TEST(test_fwhm_conversion);
    RUN_TEST(test_emg_density_is_the_derivative_of_its_distribution);
    RUN_TEST(test_emg_moments_from_the_binned_shape);
    RUN_TEST(test_emg_becomes_gaussian_as_the_tail_vanishes);
    RUN_TEST(test_extreme_ratio_stays_accurate);
    RUN_TEST(test_far_tails_stay_finite);
    RUN_TEST(test_peak_bound_is_a_bound);
    RUN_TEST(test_bin_derivative_matches_a_finite_difference);
    RUN_TEST(test_rejects_bad_parameters);
    return UNITY_END();
}
