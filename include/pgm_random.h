#ifndef PGM_RANDOM_H
#define PGM_RANDOM_H

#include <numpy/npy_common.h>
#include <numpy/random/bitgen.h>


typedef enum {GAMMA, DEVROYE, ALTERNATE, SADDLE, HYBRID} sampler_t;

/*
 * generate a sample from a Polya-Gamma distribution PG(h, z)
 *
 * Samples are draw from a Polya-Gamma distribution with specified para-
 *  meters `h` and `z`.
 *
 *  Parameters
 *  ----------
 *  h : double 
 *      The `h` parameter as described in [1]. The value(s) must be
 *      positive.
 *  z : double
 *      The exponential tilting parameter as described in [1].
 *  method : sampler_t
 *      The type of method to use when sampling. Must be one of {GAMMA,
 *      DEVROYE, ALTERNATE, SADDLE, HYBRID}. The HYBRID sampler automatically
 *      chooses the appropriate method using the parameter values. The DEVROYE
 *      metthod can only be used with positive integer values of h. If h is not
 *      a positive whole number, then it will be truncated to an integer before
 *      sampling.
 *
 *  References
 *  ----------
 * [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
 *     "Bayesian inference for logistic models using Pólya–Gamma latent
 *     variables." Journal of the American statistical Association
 *     108.504 (2013): 1339-1349.
 * [2] Windle, Jesse, Nicholas G. Polson, and James G. Scott.
 *     "Sampling Polya-Gamma random variates: alternate and approximate
 *     techniques." arXiv preprint arXiv:1405.0506 (2014)
 *
 */
double pgm_random_polyagamma(bitgen_t* bitgen_state, double h, double z, sampler_t method);

/*
 * Generate n samples from a PG(h, z) distribution.
 *
 * Parameters
 * ----------
 *  n : size_t
 *      The number of samples to generate.
 *  out: array of type double
 *      The array to place the generated samples. Only the first n elements
 *      will be populated.
 */
NPY_INLINE void
pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                           sampler_t method, size_t n, double* out)
{
    while (n--) {
        out[n] = pgm_random_polyagamma(bitgen_state, h, z, method);
    }
}

/*
 * Generate n samples from a PG(h[i], z[i]) distribution, where h and z are
 * arrays.
 *
 * h, z and out must be at least `n` in length. Only the first n elements of
 * `out` will be filled.
 */
NPY_INLINE void
pgm_random_polyagamma_fill2(bitgen_t* bitgen_state, const double* h,
                            const double* z, sampler_t method, size_t n,
                            double* restrict out)
{
    while (n--) {
        out[n] = pgm_random_polyagamma(bitgen_state, h[n], z[n], method);
    }
}

#endif
