#include <math.h>
#include <string.h>

#include "fixture_table.h"
#include "plidar/coates.h"
#include "plidar/crb.h"
#include "plidar/estimate.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

enum { NB = 2000 };

static double noiseless[NB];
static double corrected[NB];
static double scratch[NB];
static double scratch_fp[4 * NB];
static double probability[NB];
static double table_work[3 * FIXTURE_NODES];

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

/*
 * The fixture is not a Gaussian and not an EMG, so the template, the search
 * and the closed form fit all have to follow the table. With a 100 ps unit
 * and 50 ps bins the template half width is 80 bins, from the 40 unit end of
 * the support.
 */
static void test_maximum_likelihood_inverts_a_tabulated_model(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate est;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK, fixture_flat_top(&sc.irf, 100e-12, table_work));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_IDEAL, CYCLES,
                                              noiseless, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle(&sc, noiseless, CYCLES, NB, scratch, &est));
    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * sc.bin_width, sc.t0, est.t0);
    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * sc.signal, sc.signal, est.signal);
    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * sc.background, sc.background, est.background);

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_matched(&sc, noiseless, NB, scratch, &est));
    TEST_ASSERT_DOUBLE_WITHIN(0.25 * sc.bin_width, sc.t0, est.t0);
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

/* The first photon log likelihood from its definition, at the parameters of
 * sc: each kept count is a cycle that recorded in its bin, q P_i, and every
 * other cycle left no kept record, 1 - q (1 - P_none). */
static double first_photon_loglik(const plidar_scene *sc, const double *counts,
                                  double cycles, double keep)
{
    double acc = 0.0, recorded = 0.0, total = 0.0;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(sc, PLIDAR_SIM_FIRST_PHOTON, 1.0, probability,
                                              NB));
    for (size_t i = 0; i < NB; i++) {
        recorded += probability[i];
        total += counts[i];
        if (counts[i] > 0.0) {
            acc += counts[i] * log(keep * probability[i]);
        }
    }
    return acc + (cycles - total) * log(1.0 - keep * recorded);
}

/*
 * The first photon fit inverts its own model, like the Poisson fit above: the
 * expected histogram of a cycle that records its first photon only, and a
 * quarter of it for a group of a four way split. The three responses, a
 * background and none, and both values of keep give twelve cases. With no
 * background the bins outside the response hold nothing, and the background
 * comes back below a photon per second.
 *
 * Except for the EMG. Its template stops at ten standard deviations, like the
 * Poisson fit's, and 1.2e-5 of this response lies past that. The fit has only
 * the background to put those counts in, which comes to 150 photons per
 * second, so that case is held to twice that.
 *
 * The table's template holds all of its response, so for the table the fit
 * and the simulator describe the same model, and two more things are exact.
 * With a background, loglik is the definition at the parameters the fit
 * returns, log q terms included. Without one it is not compared: the ends of
 * the support fall on bin edges here, and whether the bin past an end holds
 * a sliver of mass or none is decided by rounding, which the fit and the
 * simulator do differently: a count in a bin that one of them leaves empty
 * makes its likelihood minus infinity. The background is then zero itself:
 * the fit reaches b = 0 rather than stopping at the floor of its coarse scan.
 */
static void test_first_photon_mle_inverts_the_model(void)
{
    const double backgrounds[2] = {8e6, 0.0};
    const double keeps[2] = {1.0, 0.25};

    for (int shape = 0; shape < 3; shape++) {
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < 2; j++) {
                plidar_scene sc = base_scene();
                plidar_estimate est;

                if (shape == 1) {
                    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                                          plidar_irf_init_emg(&sc.irf, 60e-12, 140e-12));
                } else if (shape == 2) {
                    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                                          fixture_flat_top(&sc.irf, 100e-12, table_work));
                }
                sc.signal = 3.0 / 0.3;
                sc.background = backgrounds[k];
                TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                                      plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON,
                                                          CYCLES, noiseless, NB));
                for (size_t i = 0; i < NB; i++) {
                    noiseless[i] *= keeps[j];
                }
                TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                                      plidar_est_mle_first_photon(&sc, noiseless, CYCLES,
                                                                  keeps[j], NB, scratch_fp,
                                                                  &est));

                TEST_ASSERT_DOUBLE_WITHIN(1e-3 * sc.bin_width, sc.t0, est.t0);
                TEST_ASSERT_DOUBLE_WITHIN(1e-4 * sc.signal, sc.signal, est.signal);
                if (sc.background > 0.0) {
                    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * sc.background, sc.background,
                                              est.background);
                } else if (shape == 1) {
                    TEST_ASSERT_LESS_THAN_DOUBLE(300.0, est.background);
                } else {
                    TEST_ASSERT_LESS_THAN_DOUBLE(1.0, est.background);
                }
                if (shape == 2 && sc.background > 0.0) {
                    plidar_scene fit = sc;
                    double expect;

                    fit.t0 = est.t0;
                    fit.signal = est.signal;
                    fit.background = est.background;
                    expect = first_photon_loglik(&fit, noiseless, CYCLES, keeps[j]);
                    TEST_ASSERT_DOUBLE_WITHIN(1e-9 * fabs(expect), expect, est.loglik);
                } else if (shape == 2) {
                    TEST_ASSERT_EQUAL_DOUBLE(0.0, est.background);
                }
            }
        }
    }
}

/*
 * At four photons in ten thousand cycles the first photon model is the
 * Poisson one: what separates them is the masking of late photons by early
 * ones, a correction of that order. So on the same histogram the two fits
 * have to agree.
 */
static void test_first_photon_mle_matches_poisson_at_low_flux(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate first, poisson;
    const double cycles = 1e8;

    sc.signal = 1e-4 / 0.3;
    sc.background = 1e4;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON, cycles,
                                              noiseless, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle_first_photon(&sc, noiseless, cycles, 1.0, NB,
                                                      scratch_fp, &first));
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle(&sc, noiseless, cycles, NB, scratch, &poisson));

    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * sc.bin_width, poisson.t0, first.t0);
    TEST_ASSERT_DOUBLE_WITHIN(2e-3 * poisson.signal, poisson.signal, first.signal);
}

/*
 * A scene shaped like the real data: 750 ps bins, the tabulated response at
 * the scale of the measured pulse, rising over five bins and nine bins wide
 * at half height, no background, and a return bright enough that 97 percent
 * of the 7000 cycles record a photon, as in the brightest class of pixels.
 */
enum { NP = 400, MC_REPS = 300, MC_GROUP_REPS = 150 };

#define MC_CYCLES 7000u

static uint64_t mc_hist[NP];
static double mc_counts[NP];
static double mc_rates[NP];
static double mc_groups[4][NP];
static double mc_scratch[4 * NP];
static double mc_first[MC_REPS];
static double mc_coates[MC_REPS];

static plidar_scene mc_scene(void)
{
    plidar_scene sc;

    sc.period = 300e-9;
    sc.bin_width = 750e-12;
    sc.nbins = NP;
    sc.t0 = 120.3e-9;
    sc.signal = 3.5;
    sc.background = 0.0;
    sc.efficiency = 1.0;
    (void)fixture_flat_top(&sc.irf, 750e-12, table_work);
    return sc;
}

static plidar_detector first_photon_detector(void)
{
    plidar_detector d;

    d.dead_time = 0.0;
    d.paralyzable = 0;
    d.first_photon_only = 1;
    d.afterpulse_prob = 0.0;
    d.afterpulse_tau = 0.0;
    return d;
}

/* Two passes, so the spread is not the difference of two large sums. */
static void mean_sd(const double *x, size_t n, double *mean, double *sd)
{
    double m = 0.0, v = 0.0;

    for (size_t i = 0; i < n; i++) {
        m += x[i];
    }
    m /= (double)n;
    for (size_t i = 0; i < n; i++) {
        v += (x[i] - m) * (x[i] - m);
    }
    *mean = m;
    *sd = sqrt(v / (double)(n - 1u));
}

/*
 * The case the estimator exists for. Most cycles record on the rising edge,
 * and the late part of the return is seen only by the few cycles still
 * listening. The fit on the counts as recorded weighs each bin by those
 * cycles and reaches the first photon bound. Coates followed by the Poisson
 * fit gets the mean right but weighs every bin as if every cycle could still
 * record in it, and stays measurably above the bound on the same histograms.
 *
 * The bound is the joint one with the true parameters. With no background it
 * treats the background as known, and the fit agrees: every empty bin the
 * cycles were listening in pins the background at zero, where it stays.
 */
static void test_first_photon_mle_reaches_its_bound_under_pile_up(void)
{
    plidar_scene sc = mc_scene();
    const plidar_detector det = first_photon_detector();
    plidar_crb_result crb;
    plidar_rng rng;
    double bound, mean, sd, coates_mean, coates_sd;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb_first_photon(&sc, (double)MC_CYCLES, 1.0, mc_scratch,
                                                  NP, &crb));
    bound = sqrt(crb.var_tof_joint);

    plidar_rng_seed(&rng, 41u, 0u);
    for (size_t r = 0; r < MC_REPS; r++) {
        plidar_estimate first, coates;

        TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                              plidar_sim_mc(&sc, &det, MC_CYCLES, &rng, mc_hist, NP, NULL));
        for (size_t i = 0; i < NP; i++) {
            mc_counts[i] = (double)mc_hist[i];
        }
        TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                              plidar_est_mle_first_photon(&sc, mc_counts, (double)MC_CYCLES,
                                                          1.0, NP, mc_scratch, &first));

        TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                              plidar_coates_invert(mc_hist, MC_CYCLES, mc_rates, NP));
        for (size_t i = 0; i < NP; i++) {
            mc_rates[i] *= (double)MC_CYCLES;
        }
        TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                              plidar_est_mle(&sc, mc_rates, (double)MC_CYCLES, NP, mc_scratch,
                                             &coates));

        mc_first[r] = first.t0 - sc.t0;
        mc_coates[r] = coates.t0 - sc.t0;
    }

    mean_sd(mc_first, MC_REPS, &mean, &sd);
    TEST_ASSERT_GREATER_OR_EQUAL_DOUBLE(0.85, sd / bound);
    TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(1.20, sd / bound);
    TEST_ASSERT_LESS_THAN_DOUBLE(4.0 * sd / sqrt((double)MC_REPS), fabs(mean));

    mean_sd(mc_coates, MC_REPS, &coates_mean, &coates_sd);
    TEST_ASSERT_GREATER_THAN_DOUBLE(1.5, coates_sd / bound);
}

/*
 * The real data have one acquisition per pixel, so their precision is read
 * from groups: each count goes to one of four groups at random, and a group
 * is the histogram thinned with q = 1/4. Fit and bound both take q. The
 * groups of one histogram share its cycles, so the spread is taken within
 * each histogram, around the mean of its four groups, and the standard error
 * of the overall mean comes from those per histogram means, which are
 * independent of one another.
 */
static void test_first_photon_mle_on_thinned_groups(void)
{
    plidar_scene sc = mc_scene();
    const plidar_detector det = first_photon_detector();
    plidar_crb_result crb;
    plidar_rng rng, split;
    double within = 0.0, mean, sd;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_crb_first_photon(&sc, (double)MC_CYCLES, 0.25, mc_scratch,
                                                  NP, &crb));

    plidar_rng_seed(&rng, 43u, 0u);
    plidar_rng_seed(&split, 44u, 0u);
    for (size_t r = 0; r < MC_GROUP_REPS; r++) {
        double t[4], centre = 0.0;

        TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                              plidar_sim_mc(&sc, &det, MC_CYCLES, &rng, mc_hist, NP, NULL));
        memset(mc_groups, 0, sizeof mc_groups);
        for (size_t i = 0; i < NP; i++) {
            for (uint64_t c = 0u; c < mc_hist[i]; c++) {
                size_t k = (size_t)(4.0 * plidar_rng_uniform(&split));
                if (k > 3u) {
                    k = 3u;
                }
                mc_groups[k][i] += 1.0;
            }
        }
        for (size_t k = 0; k < 4u; k++) {
            plidar_estimate est;

            TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                                  plidar_est_mle_first_photon(&sc, mc_groups[k],
                                                              (double)MC_CYCLES, 0.25, NP,
                                                              mc_scratch, &est));
            t[k] = est.t0 - sc.t0;
            centre += 0.25 * t[k];
        }
        for (size_t k = 0; k < 4u; k++) {
            within += (t[k] - centre) * (t[k] - centre);
        }
        mc_first[r] = centre;
    }
    within /= 3.0 * (double)MC_GROUP_REPS;

    TEST_ASSERT_GREATER_OR_EQUAL_DOUBLE(0.85, sqrt(within / crb.var_tof_joint));
    TEST_ASSERT_LESS_OR_EQUAL_DOUBLE(1.20, sqrt(within / crb.var_tof_joint));

    mean_sd(mc_first, MC_GROUP_REPS, &mean, &sd);
    TEST_ASSERT_LESS_THAN_DOUBLE(4.0 * sd / sqrt((double)MC_GROUP_REPS), fabs(mean));
}

/*
 * Two histograms on which a careless fit of this model returns a not a
 * number. In the first every cycle recorded, so there is no category of
 * cycles without a record and nothing bounds the total rate from above
 * except the shape. In the second a single count sits a hundred bins past
 * the return, where the response is exactly zero and there is no background
 * to explain it, so the background has to come out positive.
 */
static void test_first_photon_mle_survives_saturation(void)
{
    plidar_scene sc = mc_scene();
    plidar_estimate est;
    double sum = 0.0;
    size_t last = 0u;

    /* Rescaled so the counts add up to M exactly. The estimator sums them in
     * index order, and the last bin with counts takes up the rounding: the
     * sum before it is within a factor of two of M, so M minus that sum is
     * exact, and so is adding it back. */
    sc.signal = 12.0;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON, (double)MC_CYCLES,
                                              mc_counts, NP));
    for (size_t i = 0; i < NP; i++) {
        sum += mc_counts[i];
        if (mc_counts[i] > 0.0) {
            last = i;
        }
    }
    for (size_t i = 0; i < NP; i++) {
        mc_counts[i] *= (double)MC_CYCLES / sum;
    }
    sum = 0.0;
    for (size_t i = 0; i < last; i++) {
        sum += mc_counts[i];
    }
    mc_counts[last] = (double)MC_CYCLES - sum;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle_first_photon(&sc, mc_counts, (double)MC_CYCLES, 1.0,
                                                      NP, mc_scratch, &est));
    TEST_ASSERT_DOUBLE_WITHIN(0.01 * sc.bin_width, sc.t0, est.t0);
    TEST_ASSERT_GREATER_OR_EQUAL_DOUBLE(0.0, est.background);

    sc.signal = 3.5;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON, (double)MC_CYCLES,
                                              mc_counts, NP));
    mc_counts[(size_t)(sc.t0 / sc.bin_width) + 100u] += 1.0;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle_first_photon(&sc, mc_counts, (double)MC_CYCLES, 1.0,
                                                      NP, mc_scratch, &est));
    TEST_ASSERT_TRUE(isfinite(est.t0));
    TEST_ASSERT_TRUE(isfinite(est.loglik));
    TEST_ASSERT_GREATER_THAN_DOUBLE(0.0, est.background);
}

/*
 * The coarse scan compares candidates at fixed amplitudes, and what it
 * compares does not shift with them: a count before the return was not
 * masked by it, a count after it was. Two identical clusters a hundred bins
 * apart leave that as the only difference between their candidates. A
 * return at the later one leaves the earlier to the background, unmasked,
 * while a return at the earlier one would have to mask the later. So the fit
 * has to land on the later cluster. It does, half a bin late: a model with
 * one return can only call the earlier cluster background, spread over the
 * whole record, and that misfit moves the return a little. Which cluster is
 * the question here, so the tolerance is the two bins of the section.
 */
static void test_first_photon_mle_prefers_the_later_of_two_clusters(void)
{
    plidar_scene sc = mc_scene();
    plidar_estimate est;
    const size_t shift = 100u;

    sc.signal = 0.5;
    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON, (double)MC_CYCLES,
                                              mc_counts, NP));
    /* Downwards, so that every bin is read before it is added to. */
    for (size_t i = NP - 1u; i >= shift; i--) {
        mc_counts[i] += mc_counts[i - shift];
    }

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_est_mle_first_photon(&sc, mc_counts, (double)MC_CYCLES, 1.0,
                                                      NP, mc_scratch, &est));
    TEST_ASSERT_DOUBLE_WITHIN(2.0 * sc.bin_width, sc.t0 + (double)shift * sc.bin_width,
                              est.t0);
}

/*
 * The matched filter tries only the candidates whose window fits inside the
 * record, from h = 40 bins after the start to h bins before the end for this
 * response. A return nearer an end than that pulls the correlation to the
 * first or the last candidate, and the estimate used to be the edge of the
 * search wherever the return was. It is refused instead. A return whose peak
 * lands one candidate further in has a neighbour on each side and is found to
 * a hundredth of a bin, as in the middle of the record. Rising edges in bins,
 * refused then found.
 */
static void test_matched_filter_refuses_a_return_out_of_its_reach(void)
{
    plidar_scene sc = mc_scene();
    const double edges[4] = {30.3, 369.7, 41.3, 358.3};

    for (int k = 0; k < 4; k++) {
        plidar_estimate est;

        sc.t0 = edges[k] * sc.bin_width;
        TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                              plidar_sim_analytic(&sc, PLIDAR_SIM_IDEAL, (double)MC_CYCLES,
                                                  mc_counts, NP));
        est.t0 = -1.0;
        if (k < 2) {
            TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN,
                                  plidar_est_matched(&sc, mc_counts, NP, mc_scratch, &est));
            TEST_ASSERT_EQUAL_DOUBLE(-1.0, est.t0);
        } else {
            TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                                  plidar_est_matched(&sc, mc_counts, NP, mc_scratch, &est));
            TEST_ASSERT_DOUBLE_WITHIN(0.01 * sc.bin_width, sc.t0, est.t0);
        }
    }
}

/* The first photon fit with its buffers filled in, so each rejection below
 * fits on one line. */
static plidar_status_t first_photon(const plidar_scene *sc, const double *counts,
                                    double cycles, double keep, size_t nbins)
{
    plidar_estimate est;

    return plidar_est_mle_first_photon(sc, counts, cycles, keep, nbins, scratch_fp, &est);
}

static void test_first_photon_mle_rejects_bad_arguments(void)
{
    plidar_scene sc = base_scene();
    plidar_estimate est;
    double saved;

    TEST_ASSERT_EQUAL_INT(PLIDAR_OK,
                          plidar_sim_analytic(&sc, PLIDAR_SIM_FIRST_PHOTON, CYCLES,
                                              noiseless, NB));
    saved = noiseless[5];

    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, first_photon(NULL, noiseless, CYCLES, 1.0, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, first_photon(&sc, NULL, CYCLES, 1.0, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_est_mle_first_photon(&sc, noiseless, CYCLES,
                                                                      1.0, NB, NULL, &est));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, plidar_est_mle_first_photon(&sc, noiseless, CYCLES,
                                                                      1.0, NB, scratch_fp, NULL));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, 0.0, 1.0, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, HUGE_VAL, 1.0, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, CYCLES, 0.0, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, CYCLES, 1.5, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, CYCLES, nan(""), NB));
    noiseless[5] = -1.0;
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, CYCLES, 1.0, NB));
    noiseless[5] = HUGE_VAL;
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, CYCLES, 1.0, NB));
    noiseless[5] = nan("");
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, CYCLES, 1.0, NB));
    noiseless[5] = saved;
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_DOMAIN, first_photon(&sc, noiseless, 1e3, 1.0, NB));
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_ARG, first_photon(&sc, noiseless, CYCLES, 1.0, 4u));
    memset(noiseless, 0, sizeof noiseless);
    TEST_ASSERT_EQUAL_INT(PLIDAR_ERR_NO_CONVERGE, first_photon(&sc, noiseless, CYCLES, 1.0, NB));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_maximum_likelihood_inverts_the_model);
    RUN_TEST(test_maximum_likelihood_inverts_a_tabulated_model);
    RUN_TEST(test_matched_filter_finds_the_peak);
    RUN_TEST(test_centroid_is_unbiased_on_a_symmetric_response);
    RUN_TEST(test_the_tail_biases_the_centroid_and_not_the_matched_filter);
    RUN_TEST(test_pile_up_shortens_the_range_and_coates_restores_it);
    RUN_TEST(test_estimators_agree_with_each_other_on_clean_data);
    RUN_TEST(test_rejects_bad_arguments);
    RUN_TEST(test_first_photon_mle_inverts_the_model);
    RUN_TEST(test_first_photon_mle_matches_poisson_at_low_flux);
    RUN_TEST(test_first_photon_mle_reaches_its_bound_under_pile_up);
    RUN_TEST(test_first_photon_mle_on_thinned_groups);
    RUN_TEST(test_first_photon_mle_survives_saturation);
    RUN_TEST(test_first_photon_mle_prefers_the_later_of_two_clusters);
    RUN_TEST(test_matched_filter_refuses_a_return_out_of_its_reach);
    RUN_TEST(test_first_photon_mle_rejects_bad_arguments);
    return UNITY_END();
}
