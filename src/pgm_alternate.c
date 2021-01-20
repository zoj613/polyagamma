/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_igammaq.h"
#include "pgm_alternate.h"
#include "pgm_alternate_trunc_points.h"

#define PGM_LOG2 0.6931471805599453  // log(2)
#define PGM_LS2PI 0.9189385332046727  // log(sqrt(2 * pi))

/* 
 * Return the smallest optimal truncation point greater than the input.
 * Values are retrieved from a lookup table for `h` in the range [1, 4].
 *
 * This function uses binary search for the lookup.
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
        else if (offset < index && pgm_h[index - 1] >= h) {
            len = index - offset;
            continue;
        }
        return pgm_f[index];
    }
    // Getting  means something went wrong, but it should never happen.
    return -1;
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
 * Compute an apporximation of the inverse of the error function.
 *
 * Reference
 * ----------
 * https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
 */
static NPY_INLINE double
erfinv(double x)
{
    static const double a = 0.147;
    static const double two_pia = 4.330746750799873;
    // sign(x) is never zero since the function is used to evaluate (1 - u)
    // where u is uniform value from an interval
    double sgn_x = x > 0 ? 1 : -1;
    double logx_2 = log(1 - x * x);
    
    x = two_pia + 0.5 * logx_2;
    return sgn_x * sqrt(sqrt(x * x - logx_2 / a) - x);
}

/*
 * Sample X ~ Levy(0, c) where X < t.
 *
 * A Levy(0, c) distribution is equivalent to an Inverse-Gamma(0.5, 0.5 * c)
 * distribution. Thus sampling from an Inverse-Gamma(0.5, h^2 / 2) can be done
 * by sampling from a Levy(0, h^2). Sampling from the Levy distribution is
 * done using the Inverse-Transform method. For this problem, it is superior
 * over standard rejection sampling because the truncated values (X < t) can
 * be generated directly without throwing away samples. We can use the fact
 * that to sample from an interval (a, b], we can generate a uniform variable
 * from the unterval (F(a), F[b)], where F(x) is the CDF of the distribution.
 * Then use inverse-transform to sample X such that a <= X <= b.
 *
 * Thus, to get X from an Inverse-Gamma(0.5, h^2 / 2) right truncated at t, we
 * 1) Generate a uniform V from the interval (0, F(t)], where F is the cdf of a
 *    Levy(0, h^2).
 * 2) Generate X using inverse transform X = Finv(V)
 * 3) return X
 */
static NPY_INLINE double
random_bounded_levy(bitgen_t* bitgen_state, double c, double t)
{
    double x, u, cdf_t;

    cdf_t = erfc(sqrt(0.5 * c / t));  // F(t)
    u = next_double(bitgen_state) * cdf_t;
    x = erfinv(1 - u);
    x = c / (2 * x * x); 
    return x;
}

/* 
 * Generate from J*(h, z) where {h | 1 <= h <= 4} using the alternate method.
 */
static NPY_INLINE double
random_jacobi_alternate_bounded(bitgen_t* bitgen_state, double h, double z)
{
    static const double logpi_2 = 0.4515827052894548;  // log(pi / 2)
    uint64_t n;
    double t, p, q, u, x, h_z, s, old_s, lambda_z, ratio;
    double h2 = h * h;

    lambda_z = PGM_PI2 / 8 + 0.5 * z * z;

    t = get_truncation_point(h);

    if (z > 0) {
        h_z = h / z;
        p = exp(h * (PGM_LOG2 - z)) * inverse_gaussian_cdf(t, h_z, h2);
    }
    else {
        /* UpperIncompleteGamma(0.5, x) == sqrt(pi) * erfc(sqrt(x)), the
         * regularized version of the function, which is what we want, can be
         * written as erfc(sqrt(x)) since the denominator of the regularized
         * version cancels with the sqrt(pi).*/
        p = exp(h * PGM_LOG2) * erfc(h / sqrt(2 * t));
    }
    q = exp(h * (logpi_2 - log(lambda_z))) * kf_gammaq(h, lambda_z * t);
    ratio = p / (p + q);

    for (;;) {
        if (next_double(bitgen_state) > ratio) {
            x = random_left_bounded_gamma(bitgen_state, h, lambda_z, t);
        }
        else if (z > 0) {
            h_z = h / z;
            do {
                x = random_wald(bitgen_state, h_z, h2);
            } while (x > t);
        }
        else {
            x = random_bounded_levy(bitgen_state, h2, t);
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
 * For values of h > 4, we sample J*(h, z/2) = sum(J*(b_i, z/2)) samples such
 * that sum(b_i) = h. Then use the relation PG(h, z) = J(h,z/2) / 4, to get a
 * sample from the Polya-Gamma distribution.
 *
 * We do this by sampling in chunks of size ``pgm_h_range``, which is
 * the difference between the largest and smallest optimal h value that
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
    double out = 0;
    z = 0.5 * (z < 0 ? fabs(z) : z);

    while (h >= 4) {
        out += random_jacobi_alternate_bounded(bitgen_state, pgm_h_range, z);
        h -= pgm_h_range;
    }
    out += random_jacobi_alternate_bounded(bitgen_state, h, z);
    return 0.25 * out;
}
