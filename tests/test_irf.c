#include <math.h>

#include "fixture_table.h"
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

static void test_table_triangle_closed_forms(void)
{
    /* -1, 0, 1 ns holding 0, 5, 0: area 5 ns, normalised peak 1 per ns */
    static const double s[3] = {0.0, 5.0, 0.0};
    static double work[9];
    const double var = 1e-18 / 6.0;
    plidar_irf irf;
    double m[1];

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_table(&irf, s, 3u, -1e-9, 1e-9, work));
    TEST_ASSERT_DOUBLE_WITHIN(1e-3, 1e9, plidar_irf_pdf(&irf, 0.0));
    TEST_ASSERT_DOUBLE_WITHIN(1e-3, 0.5e9, plidar_irf_pdf(&irf, 0.5e-9));
    TEST_ASSERT_DOUBLE_WITHIN(0.0, 0.0, plidar_irf_pdf(&irf, 1.5e-9));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.5, plidar_irf_cdf(&irf, 0.0));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.125, plidar_irf_cdf(&irf, -0.5e-9));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_bin(&irf, 0.0, 0.5e-9, 0.5e-9, 1u, m));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.125, m[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-24, 0.0, plidar_irf_mean(&irf));
    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * var, var, plidar_irf_variance(&irf));
    TEST_ASSERT_DOUBLE_WITHIN(1e-3, 1e9, plidar_irf_peak_bound(&irf));
}

/* Asymmetric, with a different slope in every cell, so that a sign or an
 * index slip in the cell formulas cannot cancel against a symmetry. Shared by
 * the next two tests. */
static const double LUMPY[6] = {0.0, 1.0, 3.0, 2.0, 0.5, 0.0};
#define LUMPY_FIRST (-0.5e-9)
#define LUMPY_DT    0.25e-9

static void test_table_density_is_the_derivative_of_its_distribution(void)
{
    /* In units of dt from the first node: none within h of a node, so t - h
     * and t + h always fall in the same cell. */
    static const double at[7] = {0.3, 0.7, 1.5, 2.2, 2.9, 3.4, 4.6};
    static double work[18];
    const double h = 1e-4 * LUMPY_DT;
    plidar_irf irf;
    double peak;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_table(&irf, LUMPY, 6u, LUMPY_FIRST,
                                                           LUMPY_DT, work));
    peak = plidar_irf_peak_bound(&irf);

    /* Inside a cell the distribution function is quadratic, so the central
     * difference has no truncation error at all and what is left is
     * rounding. A density and a distribution function that disagree about
     * the shape would show up here as a difference of order the peak. */
    for (size_t i = 0; i < sizeof(at) / sizeof(at[0]); i++) {
        const double t = LUMPY_FIRST + at[i] * LUMPY_DT;
        const double numeric =
            (plidar_irf_cdf(&irf, t + h) - plidar_irf_cdf(&irf, t - h)) / (2.0 * h);
        TEST_ASSERT_DOUBLE_WITHIN(1e-6 * peak, plidar_irf_pdf(&irf, t), numeric);
    }
}

static void test_table_moments_match_a_direct_integration(void)
{
    enum { STEPS = 100000 };
    static double work[18];
    const double step = 5.0 * LUMPY_DT / (double)STEPS;
    plidar_irf irf;
    double first = 0.0, second = 0.0, mean, var;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_table(&irf, LUMPY, 6u, LUMPY_FIRST,
                                                           LUMPY_DT, work));

    /* The steps divide every cell evenly, so the kinks of the density sit on
     * step edges and the midpoint rule only meets polynomials. Its error is
     * then exactly zero on the first moment and step^2/12 on the second,
     * about 2.5e-10 of the variance. */
    for (int j = 0; j < STEPS; j++) {
        const double t = LUMPY_FIRST + ((double)j + 0.5) * step;
        const double p = plidar_irf_pdf(&irf, t) * step;
        first += t * p;
        second += t * t * p;
    }
    mean = plidar_irf_mean(&irf);
    var = plidar_irf_variance(&irf);
    TEST_ASSERT_DOUBLE_WITHIN(1e-8 * fabs(mean), mean, first);
    TEST_ASSERT_DOUBLE_WITHIN(1e-8 * var, var, second - first * first);
}

static void test_table_far_tail_bins_keep_their_precision(void)
{
    enum { N = 200 };
    static double s[N], work[3 * N];
    const double dt = 10e-12;
    plidar_irf irf;
    double area = 0.0, want, m;

    for (int k = 1; k < N - 1; k++) {
        s[k] = exp(-(double)k / 4.0);
        area += s[k];
    }
    area *= dt; /* the trapezoidal rule, with both ends at zero */

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_table(&irf, s, N, 0.0, dt, work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_bin(&irf, 0.0, 190.0 * dt, dt, 1u, &m));

    /* The mass is about 6e-22, far below the spacing of doubles near one, so
     * two values of the distribution function would round to the same number
     * and difference to exactly zero. */
    want = dt * (s[190] + s[191]) / 2.0 / area;
    TEST_ASSERT_TRUE(m > 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(1e-9 * want, want, m);
}

static void test_table_bin_masses_add_up_on_node_aligned_edges(void)
{
    enum { NB = 220 };
    static double work[3 * FIXTURE_NODES], on[NB], off[NB];
    /* Nodes every 12.5 ps from -600 ps to 4 ns. Bins of two node spacings
     * from -1 ns, which is on the node grid extended to the left, so with
     * t0 = 0 every edge inside the support lands on a node. */
    const double width = 25e-12;
    const double start = -1e-9;
    const double shift = 1e-15;
    plidar_irf irf;
    double sum_on = 0.0, sum_off = 0.0, peak;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, fixture_flat_top(&irf, 100e-12, work));
    peak = plidar_irf_peak_bound(&irf);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_bin(&irf, 0.0, start, width, NB, on));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_bin(&irf, shift, start, width, NB, off));

    /* Moving the return moves each mass at the rate pdf(lo) - pdf(hi), which
     * the peak bounds, so a femtosecond can only move a mass by the peak
     * times a femtosecond. A distribution function that jumped at the nodes
     * would move the masses by the size of the jump instead, however small
     * the shift. */
    for (size_t i = 0; i < NB; i++) {
        sum_on += on[i];
        sum_off += off[i];
        TEST_ASSERT_DOUBLE_WITHIN(2.0 * peak * shift, on[i], off[i]);
    }
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 1.0, sum_on);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 1.0, sum_off);
}

static void test_parametric_init_clears_the_table_fields(void)
{
    static const double s[3] = {0.0, 5.0, 0.0};
    static double work[9];
    plidar_irf irf;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_table(&irf, s, 3u, -1e-9, 1e-9, work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_IRF_TABLE, irf.kind);
    TEST_ASSERT_TRUE(irf.table == work && irf.n == 3u);
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_gaussian(&irf, SIGMA));
    TEST_ASSERT_NULL(irf.table);
    TEST_ASSERT_TRUE(irf.n == 0u);

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_table(&irf, s, 3u, -1e-9, 1e-9, work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&irf, SIGMA, TAU));
    TEST_ASSERT_NULL(irf.table);
    TEST_ASSERT_TRUE(irf.n == 0u);
}

static void test_table_rejects_bad_input(void)
{
    static const double good[4] = {0.0, 1.0, 2.0, 0.0};
    static double work[12];
    /* One defect each. HUGE_VAL and nan() rather than INFINITY and NAN,
     * which are floats in C99. */
    const double nan_value = nan("");
    const double bad[5][4] = {
        {0.0, -1.0, 2.0, 0.0},       /* negative */
        {0.0, nan_value, 2.0, 0.0},  /* not a number */
        {1.0, 1.0, 2.0, 0.0},        /* first sample not zero */
        {0.0, 1.0, 2.0, 1.0},        /* last sample not zero */
        {0.0, 0.0, 0.0, 0.0},        /* no area */
    };
    plidar_irf irf;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_gaussian(&irf, SIGMA));

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_irf_init_table(&irf, good, 1u, 0.0, 1e-12, work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_irf_init_table(NULL, good, 4u, 0.0, 1e-12, work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_irf_init_table(&irf, NULL, 4u, 0.0, 1e-12, work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_irf_init_table(&irf, good, 4u, 0.0, 1e-12, NULL));

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, plidar_irf_init_table(&irf, good, 4u, 0.0, 0.0, work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_irf_init_table(&irf, good, 4u, 0.0, nan_value, work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_irf_init_table(&irf, good, 4u, HUGE_VAL, 1e-12, work));
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                              plidar_irf_init_table(&irf, bad[i], 4u, 0.0, 1e-12, work));
    }

    /* A refused table leaves the response it was meant to replace alone. */
    TEST_ASSERT_EQUAL_INT(PLIDAR_IRF_GAUSSIAN, irf.kind);
    TEST_ASSERT_DOUBLE_WITHIN(0.0, SIGMA, irf.sigma);
    TEST_ASSERT_NULL(irf.table);
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
    RUN_TEST(test_table_triangle_closed_forms);
    RUN_TEST(test_table_density_is_the_derivative_of_its_distribution);
    RUN_TEST(test_table_moments_match_a_direct_integration);
    RUN_TEST(test_table_far_tail_bins_keep_their_precision);
    RUN_TEST(test_table_bin_masses_add_up_on_node_aligned_edges);
    RUN_TEST(test_parametric_init_clears_the_table_fields);
    RUN_TEST(test_table_rejects_bad_input);
    return UNITY_END();
}
