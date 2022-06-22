/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <math.h>

#include "../include/pgm_random.h"

#if defined(_MSC_VER)
    #define PGM_INLINE __inline
    #define PGM_FORCEINLINE static __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define PGM_INLINE inline
    #define PGM_FORCEINLINE static PGM_INLINE __attribute__((always_inline))
#else
    #define PGM_INLINE
    #define PGM_FORCEINLINE static
#endif

/* numpy c-api declarations */
double
random_standard_normal(bitgen_t* bitgen_state);
double
random_standard_gamma(bitgen_t* bitgen_state, double shape);

/* forward declarations of supported sampling methods */
void
random_polyagamma_devroye(bitgen_t* bitgen_state, double h, double z,
                          size_t n, double* out);
void
random_polyagamma_alternate(bitgen_t* bitgen_state, double h, double z,
                            size_t n, double* out);
void
random_polyagamma_saddle(bitgen_t* bitgen_state, double h, double z,
                         size_t n, double* out);


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
static PGM_INLINE void
random_polyagamma_normal_approx(bitgen_t* bitgen_state, double h, double z,
                                size_t n, double* out)
{
    double x, mean, stdev;

    if (z == 0.) {
        mean = 0.25 * h;
        stdev = sqrt(h / 24);
    }
    else {
        x = tanh(0.5 * z);
        mean = 0.5 * h * x / z;
        stdev = sqrt(0.25 * h * (sinh(z) - z) * (1. - x * x) / (z * z * z));
    }

    while (n--) {
        out[n] = mean + random_standard_normal(bitgen_state) * stdev;
    }
}

#ifndef PGM_GAMMA_LIMIT
#define PGM_GAMMA_LIMIT 200
#endif

void*
memset(void* s, int c, size_t n);
/*
 * Sample from PG(h, z) using the Gamma convolution approximation method.
 *
 * The infinite sum is truncated to 200 terms.
 */
static PGM_INLINE void
random_polyagamma_gamma_conv(bitgen_t* bitgen_state, double h, double z,
                             size_t n, double* out)
{
    z = 0.5 * fabs(z);
    static const double pi2 = 9.869604401089358;
    double z2 = z * z;

    memset(out, 0, n * sizeof(*out));

    while(n--) {
        double m = 0.5;
        do {
            out[n] += random_standard_gamma(bitgen_state, h) / (pi2 * m * m + z2);
        } while (++m < PGM_GAMMA_LIMIT);

        out[n] *= 0.5;
    }
}

/*
 * The hybrid sampler. The most efficient sampling method is picked according
 * to the values of the parameters.
 *
 * Refer to README.md file for more details.
 */
static PGM_INLINE void
random_polyagamma_hybrid(bitgen_t* bitgen_state, double h, double z,
                         size_t n, double* out)
{
    if (h > 50.) {
        random_polyagamma_normal_approx(bitgen_state, h, z, n, out);
    }
    else if (h >= 8. || (h > 4. &&  z <= 4.)) {
        random_polyagamma_saddle(bitgen_state, h, z, n, out);
    }
    else if (h == 1. || (h == (size_t)h && z <= 1.)) {
        random_polyagamma_devroye(bitgen_state, h, z, n, out);
    }
    else {
       random_polyagamma_alternate(bitgen_state, h, z, n, out);
    }
}


typedef void
(*pgm_func_t)(bitgen_t* bitgen_state, double h, double z, size_t n, double* out);

static const pgm_func_t sampling_method_table[] = {
    [ALTERNATE] = random_polyagamma_alternate,
    [DEVROYE] = random_polyagamma_devroye,
    [SADDLE] = random_polyagamma_saddle,
    [HYBRID] = random_polyagamma_hybrid,
    [GAMMA] = random_polyagamma_gamma_conv,
};


/* Return the appropriate value for invalid shape parameter */
PGM_FORCEINLINE double
nan_or_inf(double h)
{
    return islessequal(h, 0.) | isnan(h) ? nan("") : INFINITY;
}


double
pgm_random_polyagamma(bitgen_t* bitgen_state, double h, double z, sampler_t method)
{
    double out; 
    
    if (isfinite(h) & isgreater(h, 0.)) {
        sampling_method_table[method](bitgen_state, h, z, 1., &out);
        return out;
    }

    return nan_or_inf(h);
}


void
pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                           sampler_t method, size_t n, double* out)
{
    if (islessequal(h, 0.) | isnan(h)) {
        double retval = nan_or_inf(h);
        while (n--) {
            out[n] = retval;
        }
        return;
    }
    sampling_method_table[method](bitgen_state, h, z, n, out);
}


void
pgm_random_polyagamma_fill2(bitgen_t* bitgen_state, const double* h, const double* z,
                            sampler_t method, size_t n, double* PGM_RESTRICT out)
{
    while (n--) {
        out[n] = pgm_random_polyagamma(bitgen_state, h[n], z[n], method);
    }
}
