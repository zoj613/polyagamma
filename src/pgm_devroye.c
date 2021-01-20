/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_devroye.h"

/* 
 * Compute a_n(x|t,z), the nth term of the alternating sum S_n(x|t,z)
 */
static NPY_INLINE double
piecewise_coef(size_t n, double x, double t, double z, double coshz)
{
    static const double l2_pi = -0.4515827052894548;  // log(2 / pi) 
    double n_plus_half = n + 0.5;
    double n_plus_half2 = n_plus_half * n_plus_half;
    double n_plus_halfpi = NPY_PI * n_plus_half;
    double z2 = z * z;

    if (x > t) {
        return exp(-0.5 * x * (z2 + n_plus_halfpi * n_plus_halfpi)) *
            coshz * n_plus_halfpi;
    }
    else if (x > 0) {
        return coshz * n_plus_halfpi * exp(1.5 * (l2_pi - log(x)) -
            0.5 * z2 * x - 2 * n_plus_half2 / x);
    }
    return 0;
}

/*
 * Sample from an Inverse-Gaussian(mu, 1) truncated on the set {x | x < t}.
 *
 * We sample using two algorithms depending on whether mu > t or mu < t.
 */
static NPY_INLINE double
random_right_bounded_inverse_gaussian(bitgen_t* bitgen_state, double mu)
{
    static const double t = NPY_2_PI;  // 0.64
    double e1, e2, x, half_lambda_mu2;

    if (mu > t) {
        half_lambda_mu2 = -0.5 / (mu * mu);
        for (;;) {
            /* Below is an algorithm to sample from the tail of a normal
             * distribution such that the value is greater than 1/sqrt(t).
             * Once we obtain the sample, we square and invert it to
             * obtain a sample from an Inverse-Chi-Square distribution(df=1)
             * that is less than t, as shown in Devroye (1986) [page 382] &
             * Devroye (2009) [page 7]. This sample becomes our proposal.
             * We accept the sample only if we sample a uniform less than the
             * acceptance porbability. The probability is exp(-0.5 * lamda * x / mu^2).
             * Refer to Appendix S1 of Polson et al. (2013). */
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while ((e1 * e1) > (NPY_PI * e2));
            x = (1 + t * e1);
            x = t / (x * x);
            if (log(next_double(bitgen_state)) < half_lambda_mu2 * x)
                return x;
        }
    }
    /* If the truncation point t is greater than the mean (mu), the use
     * rejection sampling by sampling until x < t. */
    do {
        x = random_wald(bitgen_state, mu, 1);
    } while(x > t);
    return x;
}

/*
 * Generate a random sample from J*(1, 0) using algorithm described in
 * Devroye(2009), page 7.
 */
static NPY_INLINE double
random_jacobi_0(bitgen_t* bitgen_state)
{
    static const double t = NPY_2_PI;  // 0.64
    static const double p = 0.422599094; 
    static const double q = 0.57810262346829443;
    static const double ratio = p / (p + q);
    double x, e1, e2, s, u;
    size_t n;

    for (;;) {
        if (next_double(bitgen_state) < ratio) {
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while ((e1 * e1) > (NPY_PI * e2));  // 2 / t = pi
            x = (1 + t * e1);
            x = t / (x * x);
        }
        else {
            x = t + 8 * random_standard_exponential(bitgen_state) / PGM_PI2;
        }
        s = piecewise_coef(0, x, t, 0, 1);
        u = next_double(bitgen_state) * s;
        for (n = 1;; ++n) {
            if (n & 1) {
                s -= piecewise_coef(n, x, t, 0, 1);
                if (u < s)
                    return x;
            }
            else {
                s += piecewise_coef(n, x, t, 0, 1);
                if (u > s)
                    break;
            }
        }
    }
}

/*
 * Generate a random sample J*(1, z) using method described in Polson et al
 * (2013). Note that for z = 0, we use the one described in Devroye (2009).
 */
static NPY_INLINE double
random_jacobi(bitgen_t* bitgen_state, double z)
{
    if (z == 0)
        return random_jacobi_0(bitgen_state);

    static const double t = NPY_2_PI;  // 0.64
    double x, s, u;
    size_t n;

    double mu = 1 / z;
    double coshz = cosh(z);
    double k = PGM_PI2_8 + 0.5 * z * z;
    double q = coshz * (NPY_PI_2 / k) * exp(-k * t);
    double p = (1 + exp(-2 * z)) * inverse_gaussian_cdf(t, mu, 1);
    double ratio = p / (p + q);

    for (;;) {
        if (next_double(bitgen_state) < ratio) {
            x = random_right_bounded_inverse_gaussian(bitgen_state, mu);
        }
        else {
            x = t + random_standard_exponential(bitgen_state) / k;
        }
        /* Here we use S_n(x|t) instead of S_n(x|z,t) as explained in page 13 of
         * Polson et al.(2013) and page 14 of Windle et al. (2014). This
         * convenience avoids issues with S_n blowing up when z is very large.*/
        s = piecewise_coef(0, x, t, 0, 1);
        u = next_double(bitgen_state) * s;
        for (n = 1;; ++n) {
            if (n & 1) {
                s -= piecewise_coef(n, x, t, 0, 1);
                if (u < s)
                    return x;
            }
            else {
                s += piecewise_coef(n, x, t, 0, 1);
                if (u > s)
                    break;
            }
        }
    }
}

/*
 * sample from J*(n, z), where n is a positive integer greater than 1
 */
static NPY_INLINE double
random_jacobi_n(bitgen_t *bitgen_state, uint64_t n, double z)
{
    double out = 0;
    for (size_t i = n; i--; )
        out += random_jacobi(bitgen_state, z);
    return out;
}


#ifndef PGM_GAMMA_LIMIT
#define PGM_GAMMA_LIMIT 200
#endif
/*
 * Sample from J*(b, z) using a convolution of Gamma(b, 1) variates.
 */
static NPY_INLINE double
gamma_convolution_approx(bitgen_t* bitgen_state, double b, double z)
{
    const double z2 = z * z;
    double n_plus_half, out = 0;

    for (size_t n = PGM_GAMMA_LIMIT; n--; ) {
        n_plus_half = n + 0.5;
        out += random_standard_gamma(bitgen_state, b) /
            (PGM_PI2 * n_plus_half * n_plus_half + z2);
    }
    return 2 * out;
}

/*
 * Sample from PG(h, z) using the Gamma convolution approximation method
 */
double
random_polyagamma_gamma_conv(bitgen_t* bitgen_state, double h, double z)
{
    z = 0.5 * (z < 0 ? fabs(z) : z);
    return 0.25 * gamma_convolution_approx(bitgen_state, h, z);
}

/*
 * Sample from Polya-Gamma PG(n, z) using the Devroye method, where n is a
 * positive integer.
 */
double
random_polyagamma_devroye(bitgen_t *bitgen_state, uint64_t n, double z)
{
    z = 0.5 * (z < 0 ? fabs(z) : z);
    if (n > 1)
        return 0.25 * random_jacobi_n(bitgen_state, n, z);
    return 0.25 * random_jacobi(bitgen_state, z);
}
