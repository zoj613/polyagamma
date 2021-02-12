/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_devroye.h"


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
    return 0.25 * gamma_convolution_approx(bitgen_state, h, z);
}

// 0.64, the truncation point
#define T NPY_2_PI

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
    double n_plus_half = n + 0.5;
    double n_plus_half2 = n_plus_half * n_plus_half;
    double n_plus_halfpi = NPY_PI * n_plus_half;
    double x = cfg->x;

    if (x > T) {
        return n_plus_halfpi * exp(-0.5 * x * n_plus_halfpi * n_plus_halfpi); 
    }
    else if (x > 0) {
        return n_plus_halfpi * exp(-1.5 * (PGM_LOGPI_2 + cfg->logx) - 2 * n_plus_half2 / x);
    }
    return 0;
}

/*
 * Sample from an Inverse-Gaussian(mu, 1) truncated on the set {x | x < t}.
 *
 * We sample using two algorithms depending on whether mu > t or mu < t.
 */
static NPY_INLINE double
random_right_bounded_inverse_gaussian(bitgen_t* bitgen_state, struct config* cfg)
{
    double e1, e2, x; 

    if (cfg->mu > T) {
        for (;;) {
            /* Below is an algorithm to sample from the tail of a normal
             * distribution such that the value is greater than 1/sqrt(t).
             * Once we obtain the sample, we square and invert it to
             * obtain a sample from an Inverse-Chi-Square distribution(df=1)
             * that is less than t, as shown in Devroye (1986) [page 382] &
             * Devroye (2009) [page 7]. This sample becomes our proposal.
             * We accept the sample only if we sample a uniform less than the
             * acceptance porbability. The probability is exp(-0.5 * x / mu^2).
             * Refer to Appendix S1 of Polson et al. (2013). */
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while ((e1 * e1) > (NPY_PI * e2));
            x = (1 + T * e1);
            x = T / (x * x);
            if (log(next_double(bitgen_state)) < cfg->half_mu2 * x)
                return x;
        }
    }
    /* If the truncation point t is greater than the mean (mu), the use
     * rejection sampling by sampling until x < t. */
    do {
        x = random_wald(bitgen_state, cfg->mu, 1);
    } while(x > T);
    return x;
}

/*
 * Generate a random sample from J*(1, 0) using algorithm described in
 * Devroye(2009), page 7.
 */
static NPY_INLINE double
random_jacobi_0(bitgen_t* bitgen_state, struct config* cfg)
{
    static const double p = 0.422599094; 
    static const double q = 0.57810262346829443;
    static const double ratio = p / (p + q);
    double e1, e2, s, u;
    size_t i;

    for (;;) {
        if (next_double(bitgen_state) < ratio) {
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while ((e1 * e1) > (NPY_PI * e2));  // 2 / t = pi
            cfg->x = (1 + T * e1);
            cfg->x = T / (cfg->x * cfg->x);
        }
        else {
            cfg->x = T + 8 * random_standard_exponential(bitgen_state) / PGM_PI2;
        }
        cfg->logx = log(cfg->x);
        s = piecewise_coef(0, cfg);
        u = next_double(bitgen_state) * s;
        for (i = 1;; ++i) {
            if (i & 0x1) {
                s -= piecewise_coef(i, cfg);
                if (u < s)
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
 * Initialize the constant values that are used frequently and in more than one
 * place per call to `random_polyagamma_devroye`
 */
static NPY_INLINE void
initialize_config(struct config* cfg, double z)
{
    cfg->mu = 1 / z;
    cfg->k = PGM_PI2_8 + 0.5 * z * z;
    cfg->half_mu2 = -0.5 / (cfg->mu * cfg->mu);
}

/*
 * Generate a random sample J*(1, z) using method described in Polson et al (2013)
 */
static NPY_INLINE double
random_jacobi(bitgen_t* bitgen_state, struct config* cfg)
{
    double s, u;
    size_t i;

    for (;;) {
        if (next_double(bitgen_state) < cfg->ratio) {
            cfg->x = random_right_bounded_inverse_gaussian(bitgen_state, cfg);
        }
        else {
            cfg->x = T + random_standard_exponential(bitgen_state) / cfg->k;
        }
        /* Here we use S_n(x|t) instead of S_n(x|z,t) as explained in page 13 of
         * Polson et al.(2013) and page 14 of Windle et al. (2014). This
         * convenience avoids issues with S_n blowing up when z is very large.*/
        cfg->logx = log(cfg->x);
        s = piecewise_coef(0, cfg);
        u = next_double(bitgen_state) * s;
        for (i = 1;; ++i) {
            if (i & 0x1) {
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
    double q, p;
    double out = 0;

    if (z == 0) {
        do {
            out += random_jacobi_0(bitgen_state, &cfg);
            n--;
        } while (n);
        return 0.25 * out; 
    }

    initialize_config(&cfg, z);

    q = NPY_PI_2 * exp(-cfg.k * T) / cfg.k;
    p = 2 * exp(-z) * inverse_gaussian_cdf(T, cfg.mu, 1);
    cfg.ratio = p / (p + q);
    
    do {
        out += random_jacobi(bitgen_state, &cfg);    
        n--;
    } while (n);
    return 0.25 * out; 
}

#undef T
