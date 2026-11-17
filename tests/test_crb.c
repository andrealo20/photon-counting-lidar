#include <math.h>

#include "fixture_table.h"
#include "plidar/crb.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

enum { NB = 5000 };

static double scratch[2 * NB];
static double rate_true[NB];
static double rate_shift[NB];
static double table_work[3 * FIXTURE_NODES];

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
static double rescaled_inverse_t0(double m[3][3])
{
    double s[3];

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

static double joint_by_rescaled_inverse(const plidar_scene *sc, double cycles)
{
    static double g[NB], dg[NB];
    double m[3][3];
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
    return rescaled_inverse_t0(m);
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

/*
 * With photons this rare the first photon model is the Poisson one. A cycle
 * here detects about 4e-4 photons, and the masking of late bins by early
 * ones, which is all that separates the two models, is a correction of that
 * order to every entry of the information.
 */
static void test_first_photon_bound_matches_poisson_at_low_flux(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result poisson, first;

    sc.signal = 2e-4;
    sc.background = 1e4;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&sc, CYCLES, scratch, NB, &poisson));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb_first_photon(&sc, CYCLES, 1.0, scratch, NB, &first));
    TEST_ASSERT_DOUBLE_WITHIN(2e-3 * poisson.var_tof_known, poisson.var_tof_known,
                              first.var_tof_known);
    TEST_ASSERT_DOUBLE_WITHIN(2e-3 * poisson.var_tof_joint, poisson.var_tof_joint,
                              first.var_tof_joint);
}

/*
 * Thinning keeps each recorded count with probability q, which scales every
 * bin's share of the information by q exactly. The residual category, cycles
 * with no kept record, does not scale, but it depends on the time of flight
 * only through dP_0/dt0 = -P_0 * A * sum_i dg_i, and with the whole response
 * inside the window that sum telescopes to zero. So the timing information
 * is q times the unthinned one at any flux, pile up included, and the
 * tolerance is rounding.
 */
static void test_thinning_scales_the_timing_information_exactly(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result all, quarter;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, fixture_flat_top(&sc.irf, 100e-12, table_work));
    sc.signal = 2.0;
    sc.background = 1e6;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb_first_photon(&sc, CYCLES, 1.0, scratch, NB, &all));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb_first_photon(&sc, CYCLES, 0.25, scratch, NB, &quarter));
    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * 0.25 * all.information, 0.25 * all.information,
                              quarter.information);
}

/*
 * The curvature check again, on the multinomial the first photon model is.
 * A cycle either records in bin i and the count is kept, with probability
 * q*P_i, or leaves no kept record, with probability 1 - q + q*P_0. P_i comes
 * from the simulator and P_0 from the rates, so none of the running sums in
 * crb.c are involved. The signal is set so that A = 2: pile up then hides
 * most of the trailing half of the return, and the model is far from the
 * Poisson one.
 */
static void first_photon_probabilities(const plidar_scene *sc, double *p, double *p_none)
{
    double total = 0.0;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(sc, PLIDAR_SIM_FIRST_PHOTON, 1.0, p, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_scene_rate(sc, sc->t0, rate_shift, NB));
    for (size_t i = 0; i < NB; i++) {
        total += rate_shift[i];
    }
    *p_none = exp(-total);
}

static double first_photon_loglik(const plidar_scene *sc, double keep, double hypothesis)
{
    static double p_true[NB], p_model[NB];
    plidar_scene shifted = *sc;
    double none_true, none_model, acc;

    shifted.t0 = hypothesis;
    first_photon_probabilities(sc, p_true, &none_true);
    first_photon_probabilities(&shifted, p_model, &none_model);

    acc = ((1.0 - keep) + keep * none_true) * log((1.0 - keep) + keep * none_model);
    for (size_t i = 0; i < NB; i++) {
        if (p_model[i] > 0.0) {
            acc += keep * p_true[i] * log(keep * p_model[i]);
        }
    }
    return CYCLES * acc;
}

static double first_photon_curvature_at(const plidar_scene *sc, double keep, double step)
{
    const double up = first_photon_loglik(sc, keep, sc->t0 + step);
    const double mid = first_photon_loglik(sc, keep, sc->t0);
    const double down = first_photon_loglik(sc, keep, sc->t0 - step);

    return -(up - 2.0 * mid + down) / (step * step);
}

static void test_first_photon_information_matches_the_curvature(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;
    double one, two, err_one, err_two;

    sc.signal = 2.0 / 0.3;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb_first_photon(&sc, CYCLES, 0.5, scratch, NB, &crb));

    one = first_photon_curvature_at(&sc, 0.5, sc.bin_width);
    two = first_photon_curvature_at(&sc, 0.5, 2.0 * sc.bin_width);
    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * crb.information, crb.information, one);

    /* As for the Poisson bound, what is left over has to be the truncation
     * error of the second difference, so it quadruples when the step
     * doubles. */
    err_one = crb.information - one;
    err_two = crb.information - two;
    TEST_ASSERT_TRUE(err_one != 0.0);
    TEST_ASSERT_DOUBLE_WITHIN(0.02 * 4.0, 4.0, err_two / err_one);
}

/*
 * The joint bound through the definition rather than through the algebra in
 * crb.c. Every category probability is differenced numerically in all three
 * parameters, the information is summed as M * sum_c dpi_c dpi_c^T / pi_c,
 * and the matrix goes through the rescaled inverse above. The steps keep the
 * truncation error of the differences far below the tolerance, which is set
 * by their rounding. It runs at q = 1, where the residual category is P_0
 * alone, and at q = 1/4, where q and 1 - q differ.
 */
static double first_photon_joint_by_rescaled_inverse(const plidar_scene *sc, double keep)
{
    static double p[NB], up[NB], down[NB], dp[3][NB];
    const double step[3] = {1e-3 * sc->bin_width, 1e-5 * sc->signal,
                            1e-5 * sc->background};
    double p_none, dnone[3], m[3][3];

    for (int j = 0; j < 3; j++) {
        plidar_scene hi = *sc, lo = *sc;
        double none_hi, none_lo;

        if (j == 0) {
            hi.t0 += step[0];
            lo.t0 -= step[0];
        } else if (j == 1) {
            hi.signal += step[1];
            lo.signal -= step[1];
        } else {
            hi.background += step[2];
            lo.background -= step[2];
        }
        first_photon_probabilities(&hi, up, &none_hi);
        first_photon_probabilities(&lo, down, &none_lo);
        for (size_t i = 0; i < NB; i++) {
            dp[j][i] = (up[i] - down[i]) / (2.0 * step[j]);
        }
        dnone[j] = (none_hi - none_lo) / (2.0 * step[j]);
    }

    first_photon_probabilities(sc, p, &p_none);
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            m[j][k] = CYCLES * (keep * dnone[j]) * (keep * dnone[k]) /
                      ((1.0 - keep) + keep * p_none);
        }
    }
    for (size_t i = 0; i < NB; i++) {
        if (!(p[i] > 0.0)) {
            continue;
        }
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                m[j][k] += CYCLES * (keep * dp[j][i]) * (keep * dp[k][i]) / (keep * p[i]);
            }
        }
    }
    return rescaled_inverse_t0(m);
}

static void test_first_photon_joint_bound_survives_a_different_inverse(void)
{
    const double keeps[2] = {1.0, 0.25};
    plidar_scene sc = base_scene();
    plidar_crb_result crb;

    sc.signal = 2.0 / 0.3;
    for (int k = 0; k < 2; k++) {
        TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb_first_photon(&sc, CYCLES, keeps[k],
                                                                 scratch, NB, &crb));
        TEST_ASSERT_DOUBLE_WITHIN(1e-6 * crb.var_tof_joint, crb.var_tof_joint,
                                  first_photon_joint_by_rescaled_inverse(&sc, keeps[k]));
        TEST_ASSERT_TRUE(crb.var_tof_joint >= crb.var_tof_known * (1.0 - 1e-12));
    }
}

/*
 * No background at all has to be the limit of a very dim one, for a table
 * as much as for the two models. A table is exactly zero outside its
 * support, so with no background the bins past either end have a rate of
 * exactly zero and are skipped, and what they say about the background is
 * lost with them.
 *
 * t0 sits 1 ps early, so the first bin of the support holds only its first
 * picosecond. The density rises linearly from zero there, which gives that
 * bin a fixed share of the timing information however thin it is, and it is
 * also the kept bin that says most about the background: estimate the
 * background from the kept bins and the share goes, 3 to 6 percent here. An
 * end node exactly on a bin edge would not do. Rounding can leave a sliver
 * with a mass near 1e-30 there, a share that 1e-15 background photons per
 * bin already drown, so the dim case would not be near the limit.
 */
static void test_zero_background_bound_is_continuous(void)
{
    plidar_scene dark = base_scene();
    plidar_scene dim;
    plidar_crb_result zero, tiny;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, fixture_flat_top(&dark.irf, 100e-12, table_work));
    dark.t0 -= 1e-12;
    dark.signal = 1.0;
    dark.background = 0.0;
    dim = dark;
    dim.background = 1e-3; /* about 1e-15 photons per bin and cycle */

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&dark, CYCLES, scratch, NB, &zero));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, plidar_crb(&dim, CYCLES, scratch, NB, &tiny));
    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * tiny.var_tof_joint, tiny.var_tof_joint,
                              zero.var_tof_joint);

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb_first_photon(&dark, CYCLES, 1.0, scratch, NB, &zero));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb_first_photon(&dim, CYCLES, 1.0, scratch, NB, &tiny));
    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * tiny.var_tof_joint, tiny.var_tof_joint,
                              zero.var_tof_joint);
}

static void test_first_photon_bound_rejects_bad_arguments(void)
{
    plidar_scene sc = base_scene();
    plidar_crb_result crb;

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG,
                          plidar_crb_first_photon(NULL, CYCLES, 1.0, scratch, NB, &crb));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_crb_first_photon(&sc, 0.0, 1.0, scratch, NB, &crb));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_crb_first_photon(&sc, CYCLES, 0.0, scratch, NB, &crb));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_crb_first_photon(&sc, CYCLES, 1.5, scratch, NB, &crb));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                          plidar_crb_first_photon(&sc, CYCLES, nan(""), scratch, NB, &crb));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG,
                          plidar_crb_first_photon(&sc, CYCLES, 1.0, scratch, 7u, &crb));
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
    RUN_TEST(test_first_photon_bound_matches_poisson_at_low_flux);
    RUN_TEST(test_thinning_scales_the_timing_information_exactly);
    RUN_TEST(test_first_photon_information_matches_the_curvature);
    RUN_TEST(test_first_photon_joint_bound_survives_a_different_inverse);
    RUN_TEST(test_zero_background_bound_is_continuous);
    RUN_TEST(test_first_photon_bound_rejects_bad_arguments);
    return UNITY_END();
}
