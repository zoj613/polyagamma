/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <numpy/random/distributions.h>
#include "pgm_devroye.h"
#include "pgm_alternate.h"
#include "pgm_saddle.h"
#include "../include/pgm_random.h"


extern NPY_INLINE void
pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                           sampler_t method, size_t n, double* out);

extern NPY_INLINE void
pgm_random_polyagamma_fill2(bitgen_t* bitgen_state, const double* h,
                            const double* z, sampler_t method, size_t n,
                            double* restrict out);

/*
 * Sample from a PG(h. z) using a Normal Approximation. For sufficiently large
 * h, the density of a Polya-Gamma resembles that of a Normal distribution.
 *
 * - For z > 0, the mean and variance can be directly calculated using the
 *   distribution's moment generating function (MGF).
 * - For z = 0, we calculate the limit of the derivative of the MGF as t
 *   approaches 0. The formula can be easily generated using any online math
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

#ifndef PGM_GAMMA_LIMIT
#define PGM_GAMMA_LIMIT 200
#endif
/*
 * Sample from PG(h, z) using the Gamma convolution approximation method.
 *
 * The infinite sum is truncated to 200 terms.
 */
static NPY_INLINE double
random_polyagamma_gamma_conv(bitgen_t* bitgen_state, double h, double z)
{
    static const double pi2 = 9.869604401089358;
    const double z2 = z * z;
    double c, out = 0;

    for (size_t n = PGM_GAMMA_LIMIT; n--; ) {
        c = n + 0.5;
        out += random_standard_gamma(bitgen_state, h) / (pi2 * c * c + z2);
    }
    return 0.5 * out;
}

/*
 * The hybrid sampler. The most efficient sampling method is picked according
 * to the value of the shape parameter.
 *
 * - If h is less than 1 (where DEVROYE and ALTERNATE do not apply), then
 *   the SADDLE method is used since it is much faster than the GAMMA method.
 * - if h is a whole number that is less than 20 and z < 2, DEVROYE is the most
 *   efficient method of choice. Otherwise, the SADDLE method seems the fastest.
 * - If h is non-integer (and/or z > 2) less than or equal to 20, ALTERNATE
 *   method is the quickest on average. For values greater than 20, the SADDLE
 *   method is.
 * - If h > 50, we use a Normal approximation to the Polya-Gamma distribution.
 *   Tests have shown this approximation is very accurate even for smaller
 *   values of h greater than 30.
 */
static NPY_INLINE double
random_polyagamma_hybrid(bitgen_t* bitgen_state, double h, double z)
{
    bool is_integer;

    if (h < 1) {
        return random_polyagamma_saddle(bitgen_state, h, z);
    }

    if (h > 50) {
        return random_polyagamma_normal_approx(bitgen_state, h, 2 * z);
    }

    is_integer = h == (size_t)h;
    if (h > 15) {
        return random_polyagamma_saddle(bitgen_state, h, z);
    }
    else if (is_integer && z <= 1) {
        return random_polyagamma_devroye(bitgen_state, h, z);
    }
    else {
        return random_polyagamma_alternate(bitgen_state, h, z);
    }
}


NPY_INLINE double
pgm_random_polyagamma(bitgen_t* bitgen_state, double h, double z, sampler_t method)
{
    z = z == 0 ? 0 : 0.5 * (z < 0 ? -z : z);
    switch(method) {
        case GAMMA:
            return random_polyagamma_gamma_conv(bitgen_state, h, z);
        case DEVROYE:
            return random_polyagamma_devroye(bitgen_state, h, z);
        case ALTERNATE:
            return random_polyagamma_alternate(bitgen_state, h, z);
        case SADDLE:
            return random_polyagamma_saddle(bitgen_state, h, z);
        default:
            return random_polyagamma_hybrid(bitgen_state, h, z);
    }
}
