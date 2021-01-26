/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_igammaq.h"
#include "pgm_alternate.h"
#include "pgm_alternate_trunc_points.h"

#define PGM_LOG2 0.6931471805599453  // log(2)
#define PGM_LS2PI 0.9189385332046727  // log(sqrt(2 * pi))

/* 
 * Return the optimal truncation point for a given value of h in the range
 * [1, 4]. Values are retrieved from a lookup table using binary search, then
 * the final value is calculated using linear interpolation.
 */
static NPY_INLINE double
get_truncation_point(double h)
{
    if (h == 1)
        return pgm_f[0];

    if (h == 4)
        return pgm_f[pgm_table_size - 1];

    // start binary search
    size_t index, offset = 0, len = pgm_table_size;

    while (len > 0) {
        index = offset + len / 2;
        if (pgm_h[index] < h) {
            len = len - (index + 1 - offset);
            offset = index + 1;
            continue;
        }
        else if (offset < index && pgm_h[index] > h) {
            len = index - offset;
            continue;
        }
        break;
    }
    double x0, x1, f0, f1;
    x0 = pgm_h[index - 1];
    f0 = pgm_f[index - 1];
    x1 = pgm_h[index + 1];
    f1 = pgm_f[index + 1];
    return f0 + (f1 - f0) * (pgm_h[index] - x0) / (x1 - x0);
}

/*
 * Compute a^L(x|h), the coefficient for the alternating sum S^L(x|h)
 */
static NPY_INLINE double
piecewise_coef(uint64_t n, double x, double h)
{
    double h_plus_2n = 2 * n + h;

    return exp(h * PGM_LOG2 + lgamma(n + h) + log(h_plus_2n) - lgamma(h) -
        lgamma(n + 1) - PGM_LS2PI - 1.5 * log(x) - 0.5 * h_plus_2n * h_plus_2n / x);
}


// compute: k(x|h)
static NPY_INLINE double
bounding_kernel(double x, double h, double t)
{
    static const double lsp_2 = 0.22579135264472733;  // log(sqrt(pi / 2))

    if (x > t) {
        return exp(h * lsp_2 + (h - 1) * log(x) - PGM_PI2_8 * x - lgamma(h));
    }
    else if (x > 0) {
        return exp(h * PGM_LOG2 + log(h) - PGM_LS2PI - 0.5 * h * h / x - 1.5 * log(x));
    }
    return 0;
}

/* 
 * Generate from J*(h, z) where {h | 1 <= h <= 4} using the alternate method.
 */
static NPY_INLINE double
random_jacobi_alternate_bounded(bitgen_t* bitgen_state, double h, double z)
{
    uint64_t n;
    double t, p, q, u, x, s, old_s, ratio, one_t;
    double h2 = h * h, half_h2 = 0.5 * h2, h_z = h / z;
    double lambda_z = PGM_PI2 / 8 + 0.5 * z * z;

    t = get_truncation_point(h);
    one_t = 1 / t;

    if (z > 0) {
        p = exp(h * (PGM_LOG2 - z)) * inverse_gaussian_cdf(t, h_z, h2);
    }
    else {
        /* UpperIncompleteGamma(0.5, x) == sqrt(pi) * erfc(sqrt(x)), the
         * regularized version of the function, which is what we want, can be
         * written as erfc(sqrt(x)) since the denominator of the regularized
         * version cancels with the sqrt(pi).*/
        p = exp(h * PGM_LOG2) * erfc(h / sqrt(2 * t));
    }
    q = exp(h * (PGM_LOGPI_2 - log(lambda_z))) * kf_gammaq(h, lambda_z * t);
    ratio = p / (p + q);

    for (;;) {
        if (next_double(bitgen_state) > ratio) {
            x = random_left_bounded_gamma(bitgen_state, h, lambda_z, t);
        }
        else if (z > 0) {
            do {
                x = random_wald(bitgen_state, h_z, h2);
            } while (x > t);
        }
        else {
            /* To sample from an inverse-gamma we can use the relation:
             * InvGamma(a, b) == 1 / Gamma(a, rate=b). To make sure our samples
             * remain less than t, we sample from a Gamma distribution left-
             * truncated at 1/t (i.e X > 1/t). Then 1/X < t is an Inverse-
             * Gamma right truncated at t. Which is what we want. */
            x = 1 / random_left_bounded_gamma(bitgen_state, 0.5, half_h2, one_t);
        }
        u = next_double(bitgen_state) * bounding_kernel(x, h, t);
        s = piecewise_coef(0, x, h);
        for (n = 1;; ++n) {
            old_s = s;
            if (n & 1) {
                s -= piecewise_coef(n, x, h);
                if ((old_s >= s) && (u < s))
                    return x;
            }
            else {
                s += piecewise_coef(n, x, h);
                if ((old_s >= s) && (u > s)) {
                    break;
                }
            }
        }
    }
}

/*
 * Sample from PG(h, z) using the alternate method, for h >= 1.
 *
 * For values of h >= 4, we sample J*(h, z/2) = sum(J*(b_i, z/2)) samples such
 * that sum(b_i) = h. Then use the relation PG(h, z) = J(h,z/2) / 4, to get a
 * sample from the Polya-Gamma distribution.
 *
 * We do this by sampling in chunks of size ``pgm_h_range / 2``, which is
 * half the difference between the largest and smallest optimal h value that
 * satisfies the alternating sum criterion for the sampler in the range We
 * chose ([1, 4]). We sample and decrement the `h` param until the its value
 * is less than or equal to 4. Then sample one more time if the remaining value
 * is not zero. 
 *
 * See: Section 4.3 of Windle et al. (2014)
 */
double
random_polyagamma_alternate(bitgen_t *bitgen_state, double h, double z)
{
    double out = 0, chunk_size = 0.5 * pgm_h_range;
    z = z == 0 ? 0 : 0.5 * (z < 0 ? -z : z);

    while (h >= 4) {
        out += random_jacobi_alternate_bounded(bitgen_state, chunk_size, z);
        h -= chunk_size;
    }
    out += random_jacobi_alternate_bounded(bitgen_state, h, z);
    return 0.25 * out;
}
