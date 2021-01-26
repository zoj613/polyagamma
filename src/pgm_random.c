/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "../include/pgm_random.h"
#include "pgm_devroye.h"
#include "pgm_alternate.h"
#include "pgm_saddle.h"

/*
 * Sample from a PG(h. z) using a Normal Approximation. For sufficiently large
 * h, the density of a Polya-Gamma resembles that of a Normal distribution.
 * (See this plot for h=12: https://user-images.githubusercontent.com/44142765/105776797-e736bf00-5f71-11eb-9b45-1a366562df4d.png).
 *
 * - For z > 0, the mean and variance can be directly calculated using the
 *   distribution's moment generating function (MGF).
 * - For z = 0, we use calculate the limit of the derivative of the MGF as t
 *   approaches 0. The formular can be easily generated using any online math
 *   equation calculator.
 */
static NPY_INLINE double
random_polyagamma_normal_approx(bitgen_t* bitgen_state, double h, double z)
{
    double x, mean, variance;

    if (z == 0) {
        mean = 0.25 * h;
        variance = 0.041666688 * h;
    }
    else {
        x = tanh(0.5 * z);
        mean = 0.5 * h * x / z;
        variance = 0.25 * h * (sinh(z) - z) * (1 - x * x) / (z * z * z);
    }
    return mean + random_standard_normal(bitgen_state) * sqrt(variance);
}

/*
 * The hybrid sampler. The most efficient sampling method is picked according
 * to the value of the shape parameter.
 *
 * - If h is less than 1 (where DEVROYE and ALTERNATE do not apply), then
 *   the SADDLE method is used since it i much faster than the GAMMA method.
 * - if h is a whole number that is less than 20, DEVROYE is the most efficient
 *   method of choice. For values greater than 20, SADDLE method seems the fastest.
 * - If h is non-integer (where the DEVROYE method does not apply) less than
 *   or equal to 10, ALTERNATE method is the quickest. For values greater than
 *   10, SADDLE method is.
 * - If h > 50, we use a Normal approximation the Polya-Gamma distribution.
 *   Tests have shown this approximation is very accurate even for smaller
 *   values of h.
 */
static NPY_INLINE double
random_polyagamma_hybrid(bitgen_t* bitgen_state, double h, double z)
{
    double fract_part, int_part;

    if (h < 1) {
        return random_polyagamma_saddle(bitgen_state, h, z);
    }

    if (h > 50) {
        return random_polyagamma_normal_approx(bitgen_state, h, z);
    }

    fract_part = modf(h, &int_part);
    // for integer values of h
    if (fract_part == 0 && h < 20) {
        return random_polyagamma_devroye(bitgen_state, (uint64_t)int_part, z);
    }
    else if (fract_part == 0) {
        return random_polyagamma_saddle(bitgen_state, h, z);
    }

    // for non-integer values of h
    if (h <= 10) {
        return random_polyagamma_alternate(bitgen_state, h, z);
    }
    else {
        return random_polyagamma_saddle(bitgen_state, h, z);
    }
}


double
pgm_random_polyagamma(bitgen_t* bitgen_state, double h, double z, sampler_t method)
{
    switch(method) {
        case GAMMA:
            return random_polyagamma_gamma_conv(bitgen_state, h, z);
        case DEVROYE:
            return random_polyagamma_devroye(bitgen_state, (uint64_t)h, z);
        case ALTERNATE:
            return random_polyagamma_alternate(bitgen_state, h, z);
        case SADDLE:
            return random_polyagamma_saddle(bitgen_state, h, z);
        default:
            return random_polyagamma_hybrid(bitgen_state, h, z);
    }
}


NPY_INLINE void
pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                           sampler_t method, size_t n, double* out)
{
    for (size_t i = n; i--; )
        out[i] = pgm_random_polyagamma(bitgen_state, h, z, method);
}


NPY_INLINE void
pgm_random_polyagamma_fill2(bitgen_t* bitgen_state, const double* h,
                            const double* z, sampler_t method, size_t n,
                            double* restrict out)
{
    for (size_t i = n; i--; )
        out[i] = pgm_random_polyagamma(bitgen_state, h[i], z[i], method);
}
