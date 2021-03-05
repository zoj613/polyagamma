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
struct config {
    double z;
    double z2;
    double k;
    double ratio;
    double x;
    double logx;
};

/* 
 * Compute a_n(x|t), the nth term of the alternating sum S_n(x|t)
 */
static NPY_INLINE double
piecewise_coef(size_t n, struct config* cfg)
{
    if (cfg->x > T) {
        double b = NPY_PI * (n + 0.5);
        return b * exp(-0.5 * cfg->x * b * b);
    }
    else if (cfg->x > 0) {
        double a = n + 0.5;
        return NPY_PI * a * exp(-1.5 * (PGM_LOGPI_2 + cfg->logx) - 2 * a * a / cfg->x);
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
 * Note that we do not need to calculate the cdf of the inverse gaussian in
 * order to get the value of `p`. Since the value of the cdf is always
 * evaluated at the trancation point T, its value is only dependent on `z`.
 * Thus we simplify the expression for p and arrive at:
 *      p = erfc(a - b) / exp(z) + exp(z) * erfc(a + b),
 * where a = 1 / sqrt(2 * T) and b = z * sqrt(T/2).
 */
static NPY_INLINE void
initialize_config(struct config* cfg, double z)
{
    if (z > 0) {
        static const double a = 0.8838834764831844;  // 1 / sqrt(2 * T)
        static const double t_two = 0.565685424949238;  // sqrt(T/2)
        double p, q;
        double b = z * t_two;
        double ez = exp(z);

        p = pgm_erfc(a - b) / ez + pgm_erfc(a + b) * ez;
        cfg->z2 = z * z;
        cfg->k = PGM_PI2_8 + 0.5 * cfg->z2;
        q = NPY_PI_2 * exp(-cfg->k * T) / cfg->k;
        cfg->ratio = p / (p + q);
    }
    else {
        cfg->k = PGM_PI2_8;
        cfg->ratio = 0.4223027567786595;
        cfg->z2 = 0;
    }
    cfg->z = z;
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
random_right_bounded_invgauss(bitgen_t* bitgen_state, struct config* cfg)
{
    static const double t_inv = 1.5625;  // 1 / T
    static const double two_t = 3.125;  // 2 / T
    double x;

    if (cfg->z < t_inv) {
        do {
            double e1, e2;
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while ((e1 * e1) > (two_t * e2));
            x = (1 + T * e1);
            x = T / (x * x);
        } while (cfg->z > 0 && log1p(-next_double(bitgen_state)) >= -0.5 * cfg->z2 * x);
        return x;
    }
    do {
        double y = random_standard_normal(bitgen_state);
        double w = (cfg->z + 0.5 * y * y) / cfg->z2;
        x = w - sqrt(w * w - 1 / cfg->z2);
        if (next_double(bitgen_state) * (1 + x * cfg->z) > 1) {
            x = 1 / (x * cfg->z2);
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
random_jacobi(bitgen_t* bitgen_state, struct config* cfg)
{
    for (;;) {
        if (next_double(bitgen_state) < cfg->ratio) {
            cfg->x = random_right_bounded_invgauss(bitgen_state, cfg);
        }
        else {
            cfg->x = T + random_standard_exponential(bitgen_state) / cfg->k;
        }
        cfg->logx = log(cfg->x);
        double s = piecewise_coef(0, cfg);
        double u = next_double(bitgen_state) * s;
        for (size_t i = 1;; ++i) {
            if (i & 1) {
                s -= piecewise_coef(i, cfg);
                if (u <= s)
                    return cfg->x;
            }
            else {
                s += piecewise_coef(i, cfg);
                if (u > s)
                    break;
            }
        }
    }
}

/*
 * Sample from Polya-Gamma PG(n, z) using the Devroye method, where n is a
 * positive integer.
 */
NPY_INLINE double
random_polyagamma_devroye(bitgen_t *bitgen_state, uint64_t n, double z)
{
    struct config cfg;
    double out = 0;

    initialize_config(&cfg, z);
    while (n--) {
        out += random_jacobi(bitgen_state, &cfg);
    }

    return 0.25 * out; 
}

#undef T
