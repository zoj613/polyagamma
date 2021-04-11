/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <numpy/random/distributions.h>
#include <math.h>
#include "pgm_devroye.h"
#include "pgm_alternate.h"
#include "pgm_saddle.h"
#include "../include/pgm_random.h"


DECLDIR NPY_INLINE void
pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                           sampler_t method, size_t n, double* out);

DECLDIR NPY_INLINE void
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
    double out = 0, n = 0.5, z2 = z * z;

    do {
        out += random_standard_gamma(bitgen_state, h) / (pi2 * n * n + z2);
    } while (++n < PGM_GAMMA_LIMIT);
    return 0.5 * out;
}

/*
 * The hybrid sampler. The most efficient sampling method is picked according
 * to the values of the parameters.
 *
 * Refer to README.md file for more details.
 */
static NPY_INLINE double
random_polyagamma_hybrid(bitgen_t* bitgen_state, double h, double z)
{
    if (h > 50) {
        return random_polyagamma_normal_approx(bitgen_state, h, 2 * z);
    }
    else if (h >= 25 || (((h > 12 && h == (size_t)h) || h >= 8) && z < 1)) {
        return random_polyagamma_saddle(bitgen_state, h, z);
    }
    else if (h == 1 || (h == (size_t)h && z < 1)) {
        return random_polyagamma_devroye(bitgen_state, h, z);
    }
    else {
       return random_polyagamma_alternate(bitgen_state, h, z);
    }
}


typedef double (*pgm_func_t)(bitgen_t* bitgen_state, double h, double z);

const pgm_func_t sampling_method_table[] = {
    [ALTERNATE] = random_polyagamma_alternate,
    [DEVROYE] = random_polyagamma_devroye,
    [SADDLE] = random_polyagamma_saddle,
    [HYBRID] = random_polyagamma_hybrid,
    [GAMMA] = random_polyagamma_gamma_conv,
};


NPY_INLINE double
pgm_random_polyagamma(bitgen_t* bitgen_state, double h, double z, sampler_t method)
{
    return sampling_method_table[method](bitgen_state, h, 0.5 * fabs(z));
}
