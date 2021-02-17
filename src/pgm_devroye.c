/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_devroye.h"


#ifndef PGM_GAMMA_LIMIT
#define PGM_GAMMA_LIMIT 200
#endif
/*
 * Sample from PG(h, z) using the Gamma convolution approximation method.
 *
 * The infinite sum is truncated to 200 terms.
 */
NPY_INLINE double
random_polyagamma_gamma_conv(bitgen_t* bitgen_state, double h, double z)
{
    const double z2 = z * z;
    double c, out = 0;

    for (size_t n = PGM_GAMMA_LIMIT; n--; ) {
        c = n + 0.5;
        out += random_standard_gamma(bitgen_state, h) / (PGM_PI2 * c * c + z2);
    }
    return 0.5 * out;
}

// the truncation point
#define T 0.64

/* a struct to store frequently used values. This avoids unnecessary
 * recalculation of these values during a single call to the sampler.
 */
struct config {
    double mu;
    double k;
    double half_mu2;
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
    double a = n + 0.5;
    double b = NPY_PI * a;

    if (cfg->x > T) {
        return b * exp(-0.5 * cfg->x * b * b);
    }
    else if (cfg->x > 0) {
        return b * exp(-1.5 * (PGM_LOGPI_2 + cfg->logx) - 2 * a * a / cfg->x);
    }
    return 0;
}

/*
 * Initialize constants used during sampling. The values for z = 0 are obtained
 * from the J*(1, 0) sampler  described in Devroye(2009), page 7.
 *
 * Setting mu=Inf if z=0 ensures that sampling from a truncated inverse-gaussian
 * uses most efficient sampling algorithm in `random_right_bounded_inverse_gaussian`.
 */
static NPY_INLINE void
initialize_config(struct config* cfg, double z)
{
    double p, q;

    if (z > 0) {
        cfg->mu = 1 / z;
        cfg->k = PGM_PI2_8 + 0.5 * z * z;
        q = NPY_PI_2 * exp(-cfg->k * T) / cfg->k;
        p = 2 * exp(-z) * inverse_gaussian_cdf(T, cfg->mu, 1);
        cfg->ratio = p / (p + q);
    }
    else {
        cfg->mu = NPY_INFINITY;
        cfg->k = PGM_PI2_8;
        cfg->ratio = 0.4223027567786595;
    }
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
    double s, u;
    size_t i;

    for (;;) {
        if (next_double(bitgen_state) < cfg->ratio) {
            cfg->x = random_right_bounded_inverse_gaussian(bitgen_state, cfg->mu, 1, T);
        }
        else {
            cfg->x = T + random_standard_exponential(bitgen_state) / cfg->k;
        }
        cfg->logx = log(cfg->x);
        s = piecewise_coef(0, cfg);
        u = next_double(bitgen_state) * s;
        for (i = 1;; ++i) {
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
