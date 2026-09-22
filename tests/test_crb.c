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

/*
 * Estimating the signal and background levels from the same histogram cannot
 * make the time of flight easier to pin down, so the joint bound can never
 * fall below the one that takes them as known. What is worth measuring is
 * how much it costs, and the answer is nothing that shows.
 *
 * The reason is a symmetry. The derivative of a symmetric response is odd
 * about the peak while the response itself is even, so the (t0, A) entry of
 * the information matrix is a sum of an odd function against an even weight
 * and vanishes. The (t0, B) entry is a sum of the derivative alone, which
 * telescopes to nothing over a grid that covers the response. The time of
 * flight is orthogonal to both amplitudes, and an orthogonal parameter is
 * free.
 *
 * So the assertion is equality to the last few bits, in both directions. A
 * one sided test would be checking which way the final rounding went.
 */
static void test_nuisance_parameters_are_free(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &crb));
    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * crb.var_tof_known, crb.var_tof_known,
                              crb.var_tof_joint);

    /* The tailed response breaks the symmetry argument, and the answer is
     * the same: over a record long enough for the derivative to telescope,
     * the two are still indistinguishable. */
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&sc.irf, 60e-12, 140e-12));
    sc.background = 20e6;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &crb));
    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * crb.var_tof_known, crb.var_tof_known,
                              crb.var_tof_joint);
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

static double curvature_at(const plidar_scene *sc, double step)
{
    const double up = expected_loglik(sc, sc->t0 + step);
    const double mid = expected_loglik(sc, sc->t0);
    const double down = expected_loglik(sc, sc->t0 - step);

    return -(up - 2.0 * mid + down) / (step * step);
}

static void test_information_matches_the_curvature_of_the_likelihood(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;
    double one, two, err_one, err_two;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &crb));

    one = curvature_at(&sc, sc.bin_width);
    two = curvature_at(&sc, 2.0 * sc.bin_width);

    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * crb.information, crb.information, one);

    /*
     * Agreeing to a tolerance only says the two are close. What identifies
     * the remaining difference as the truncation error of the difference,
     * rather than a disagreement about the information itself, is how it
     * behaves when the step changes: a central second difference carries an
     * error proportional to the square of the step, so doubling the step has
     * to quadruple it. It does, to better than a percent.
     */
    err_one = crb.information - one;
    err_two = crb.information - two;
    TEST_ASSERT_TRUE(err_one != 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(0.01 * 4.0, 4.0, err_two / err_one);
}

/*
 * The curvature check above moves only the time of flight, so it tests the
 * (t0, t0) entry of the information matrix and the bound that treats the two
 * amplitudes as known. The bound the sweeps publish is the other one, where
 * all three are estimated, and that one goes through a three by three
 * cofactor and determinant that nothing else here touches.
 *
 * So it is rebuilt from scratch and inverted a different way: each parameter
 * is rescaled so the diagonal of the matrix is one, which is the usual guard
 * against a badly scaled symmetric system losing its cofactor to
 * cancellation, and the answer is scaled back afterwards. Agreement to
 * machine precision says the cofactor in crb.c is right and, just as
 * usefully, that the tiny gap between the two bounds is real orthogonality
 * rather than lost digits.
 */
static double joint_by_rescaled_inverse(const plidar_scene *sc, double cycles)
{
    static double g[NB], dg[NB];
    double m[3][3];
    double s[3];
    const double amp_a = sc->efficiency * sc->signal;
    const double amp_b = sc->efficiency * sc->background * sc->bin_width;

    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            m[j][k] = 0.0;
        }
    }
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_irf_bin(&sc->irf, sc->t0, 0.0, sc->bin_width,
                                         NB, g));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_irf_bin_dt0(&sc->irf, sc->t0, 0.0, sc->bin_width,
                                             NB, dg));

    for (size_t i = 0; i < NB; i++) {
        const double mu = amp_a * g[i] + amp_b;
        double d[3];

        if (!(mu > 0.0)) {
            continue;
        }
        d[0] = amp_a * dg[i];
        d[1] = g[i];
        d[2] = 1.0;
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                m[j][k] += cycles * d[j] * d[k] / mu;
            }
        }
    }

    for (int j = 0; j < 3; j++) {
        s[j] = sqrt(m[j][j]);
    }
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            m[j][k] /= s[j] * s[k];
        }
    }
    {
        const double c00 = m[1][1] * m[2][2] - m[1][2] * m[1][2];
        const double c01 = m[0][2] * m[1][2] - m[0][1] * m[2][2];
        const double c02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        const double det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;

        return (c00 / det) / (s[0] * s[0]);
    }
}

static void test_joint_bound_survives_a_different_inverse(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;
    double other;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &crb));
    other = joint_by_rescaled_inverse(&sc, CYCLES);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * crb.var_tof_joint, crb.var_tof_joint, other);

    /* And again on a tailed response with a brighter background, where the
     * three parameters are less nearly orthogonal. */
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_irf_init_emg(&sc.irf, 60e-12, 140e-12));
    sc.background = 20e6;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &crb));
    other = joint_by_rescaled_inverse(&sc, CYCLES);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * crb.var_tof_joint, crb.var_tof_joint, other);
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
    RUN_TEST(test_nuisance_parameters_are_free);
    RUN_TEST(test_more_background_is_never_better);
    RUN_TEST(test_information_matches_the_curvature_of_the_likelihood);
    RUN_TEST(test_joint_bound_survives_a_different_inverse);
    RUN_TEST(test_rejects_bad_arguments);
    return UNITY_END();
}
