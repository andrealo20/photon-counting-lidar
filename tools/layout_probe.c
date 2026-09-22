/*
 * Prints the size of every struct the Python binding mirrors, and the offset
 * of every field in it.
 *
 * tools/check_bindings.py runs this and compares the output against what
 * ctypes computes from the declarations in tools/plidar.py. A field added in
 * the wrong place, a type that is not the width it looks, or padding that
 * differs between the two would otherwise corrupt every number the sweeps
 * produce, silently and plausibly, because ctypes would happily read a
 * double out of the middle of two other fields.
 */
#include <stddef.h>
#include <stdio.h>

#include "plidar/crb.h"
#include "plidar/estimate.h"
#include "plidar/rng.h"

#define SIZE_OF(type)        printf("%s %zu\n", #type, sizeof(type))
#define OFFSET_OF(type, fld) printf("%s.%s %zu\n", #type, #fld, offsetof(type, fld))

int main(void)
{
    SIZE_OF(plidar_irf);
    OFFSET_OF(plidar_irf, kind);
    OFFSET_OF(plidar_irf, sigma);
    OFFSET_OF(plidar_irf, tau);

    SIZE_OF(plidar_scene);
    OFFSET_OF(plidar_scene, period);
    OFFSET_OF(plidar_scene, bin_width);
    OFFSET_OF(plidar_scene, nbins);
    OFFSET_OF(plidar_scene, t0);
    OFFSET_OF(plidar_scene, signal);
    OFFSET_OF(plidar_scene, background);
    OFFSET_OF(plidar_scene, efficiency);
    OFFSET_OF(plidar_scene, irf);

    SIZE_OF(plidar_detector);
    OFFSET_OF(plidar_detector, dead_time);
    OFFSET_OF(plidar_detector, paralyzable);
    OFFSET_OF(plidar_detector, first_photon_only);
    OFFSET_OF(plidar_detector, afterpulse_prob);
    OFFSET_OF(plidar_detector, afterpulse_tau);

    SIZE_OF(plidar_estimate);
    OFFSET_OF(plidar_estimate, t0);
    OFFSET_OF(plidar_estimate, range);
    OFFSET_OF(plidar_estimate, signal);
    OFFSET_OF(plidar_estimate, background);
    OFFSET_OF(plidar_estimate, loglik);
    OFFSET_OF(plidar_estimate, iterations);

    SIZE_OF(plidar_crb_result);
    OFFSET_OF(plidar_crb_result, var_tof_known);
    OFFSET_OF(plidar_crb_result, var_tof_joint);
    OFFSET_OF(plidar_crb_result, information);

    SIZE_OF(plidar_rng);
    OFFSET_OF(plidar_rng, state);
    OFFSET_OF(plidar_rng, inc);
    return 0;
}
