/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"
#include "pgm_devroye.h"

// the truncation point
#define T 0.64

/* a struct to store frequently used values. This avoids unnecessary
 * recalculation of these values during a single call to the sampler.
 */
typedef struct {
    double proposal_probability;
    double logx;
    double z2;
    double z;
    double k;
    double x;
} parameter_t;

/* 
 * Compute a_n(x|t), the nth term of the alternating sum S_n(x|t)
 */
static NPY_INLINE float
piecewise_coef(int n, parameter_t const* pr)
{
    if (pr->x > T) {
        double b = NPY_PI * (n + 0.5);
        return (float)b * expf(-0.5 * pr->x * b * b);
    }
    else if (pr->x > 0) {
        double a = n + 0.5;
        return expf(-1.5 * (PGM_LOGPI_2 + pr->logx) - 2. * a * a / pr->x) *
               (float)(NPY_PI * a);
    }
    return 0;
}

/*
 * Initialize constants used during sampling. The values for z = 0 are obtained
 * from the J*(1, 0) sampler  described in Devroye(2009), page 7.
 *
 * Setting mu=Inf if z=0 ensures that sampling from a truncated inverse-gaussian
 * uses most efficient sampling algorithm in `random_right_bounded_inverse_gaussian`.
 *
 * NOTE
 * ----
 * We do not need to calculate the cdf of the inverse gaussian in order to get
 * the value of `p`. Since the value of the cdf is always evaluated at the
 * trancation point T, its value is only dependent on `z`. Thus we simplify the
 * expression for p and arrive at:
 *      p = erfc(a - b) / exp(z) + exp(z) * erfc(a + b),
 * where a = 1 / sqrt(2 * T) and b = z * sqrt(T/2).
 */
static NPY_INLINE void
set_sampling_parameters(parameter_t* const pr, double z)
{
    if (z > 0) {
        static const float a = 0.8838834764831844f;  // 1 / sqrt(2 * T)
        static const float t_two = 0.565685424949238f;  // sqrt(T/2)
        double p, q;
        float b = z * t_two;
        float ez = expf(z);

        p = pgm_erfc(a - b) / ez + pgm_erfc(a + b) * ez;
        pr->z2 = z * z;
        pr->k = PGM_PI2_8 + 0.5 * pr->z2;
        q = NPY_PI_2 * expf(-pr->k * T) / pr->k;
        pr->proposal_probability = p / (p + q);
    }
    else {
        pr->proposal_probability = 0.4223027567786595;
        pr->k = PGM_PI2_8;
        pr->z2 = 0.;
    }
    pr->logx = 0.;
    pr->z = z;
}

/*
 * Sample from an Inverse-Gaussian(1/z, 1) truncated on the set {x | x < 0.64}.
 *
 * We sample using two algorithms depending on whether 1/z > 0.64 or z < 1.5625.
 *
 * When 1/z < 0.64, We use a known sampling algorithm from Devroye
 * (1986), page 149. We sample until the generated variate is less than 0.64.
 *
 * When mu > 0.64, we use a Inverse-Chi-square distribution as a proposal,
 * as explained in [1], page 134. To generate a sample from this proposal, we
 * sample from the tail of a standard normal distribution such that the value
 * is greater than 1/sqrt(0.64). Once we obtain the sample, we square and invert
 * it to obtain a sample from a Inverse-Chi-Square(df=1) that is less than t.
 * An efficient algorithm to sample from the tail of a normal distribution
 * using a pair of exponential variates is shown in Devroye (1986) [page 382]
 * & Devroye (2009) [page 7]. This sample becomes our proposal. We accept the
 * sample only if we sample a uniform less than the acceptance porbability.
 * The probability is exp(-0.5 * z2 * x) (Refer to Appendix 1 of [1] for a
 * derivation of this probablity).
 *
 * References
 * ----------
 *  [1] Windle, J. (2013). Forecasting high-dimensional, time-varying
 *      variance-covariance matrices with high-frequency data and sampling
 *      Pólya-Gamma random variates for posterior distributions derived from
 *      logistic likelihoods.(PhD thesis). Retrieved from
 *      http://hdl.handle.net/2152/21842
 */
static NPY_INLINE double
random_right_bounded_invgauss(bitgen_t* bitgen_state, parameter_t* const pr)
{
    double x;
    // 1 / T = 1.5625
    if (pr->z < 1.5625) {
        do {
            double e1, e2;
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while (e1 * e1 > 3.125 * e2);  // 2 / T = 3.125
            x = (1 + T * e1);
            x = T / (x * x);
        } while (pr->z > 0 && log1pf(-next_float(bitgen_state)) >= -0.5 * pr->z2 * x);
        return x;
    }
    do {
        double y = random_standard_normal(bitgen_state);
        double w = (pr->z + 0.5 * y * y) / pr->z2;
        x = w - sqrt(w * w - 1 / pr->z2);
        if (next_double(bitgen_state) * (1. + x * pr->z) > 1.) {
            x = 1. / (x * pr->z2);
        }
    } while (x >= T);
    return x;
}

/*
 * Generate a random sample J*(1, z) using method described in Polson et al (2013)
 *
 * Here we use S_n(x|t) instead of S_n(x|z,t) as explained in page 13 of
 * Polson et al.(2013) and page 14 of Windle et al. (2014).
 */
static NPY_INLINE double
random_jacobi_star(bitgen_t* bitgen_state, parameter_t* const pr)
{
    for (;;) {
        if (next_double(bitgen_state) < pr->proposal_probability) {
            pr->x = random_right_bounded_invgauss(bitgen_state, pr);
            pr->logx = logf(pr->x);
        }
        else {
            pr->x = T + random_standard_exponential(bitgen_state) / pr->k;
        }
        float s = piecewise_coef(0, pr);
        float u = next_float(bitgen_state) * s;
        for (int i = 1;; ++i) {
            if (i & 1) {
                s -= piecewise_coef(i, pr);
                if (u <= s)
                    return pr->x;
            }
            else {
                s += piecewise_coef(i, pr);
                if (u > s)
                    break;
            }
        }
    }
}

/*
 * Sample from Polya-Gamma PG(h, z) using the Devroye method, where h is a
 * positive whole number / integer.
 */
double
random_polyagamma_devroye(bitgen_t* bitgen_state, double h, double z)
{
    parameter_t pr;
    int n = h;
    double out = 0;

    set_sampling_parameters(&pr, z);
    while (n--) {
        out += random_jacobi_star(bitgen_state, &pr);
    }
    return 0.25 * out; 
}

#undef T
