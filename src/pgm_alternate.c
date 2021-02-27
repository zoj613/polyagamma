/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"
#include "pgm_alternate.h"
#include "pgm_alternate_trunc_points.h"

#define PGM_LOG2 0.6931471805599453  // log(2)

/* a struct to store frequently used values. This avoids unnecessary
 * recalculation of these values during a single call to the sampler.
 */
struct config {
    double h;
    double t;
    double z;
    double x;
    double h2;
    double half_h2;
    double logx;  
    double logh;
    double ratio;
    double h_z;  // h / z
    double lgammah;  // loggamma(h)
    double one_t;  // 1 / t
    double lambda_z;  // pi^2 / 8 + 0.5 * z * z
    double hlog2;  // h * log(2)
    double log_lambda_z;
};

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
    double x0 = pgm_h[index - 1];
    double f0 = pgm_f[index - 1];
    double x1 = pgm_h[index + 1];
    double f1 = pgm_f[index + 1];
    return f0 + (f1 - f0) * (pgm_h[index] - x0) / (x1 - x0);
}

/*
 * Compute a^L(x|h), the n'th coefficient for the alternating sum S^L(x|h)
 */
static NPY_INLINE double
piecewise_coef(size_t n, struct config* cfg)
{
    double h_plus_2n = 2 * n + cfg->h;
    double lgamh_plus_n = n ? pgm_lgamma(n + cfg->h) : cfg->lgammah;

    return exp(cfg->hlog2 + lgamh_plus_n + log(h_plus_2n) - cfg->lgammah -
               pgm_lgamma(n + 1) - PGM_LS2PI - 1.5 * cfg->logx - 
               0.5 * (h_plus_2n * h_plus_2n) / cfg->x);
}


// compute: k(x|h)
static NPY_INLINE double
bounding_kernel(struct config* cfg)
{
    if (cfg->x > cfg->t) {
        // log(sqrt(pi / 2))
        static const double lsp_2 = 0.22579135264472733;
        return exp(cfg->h * lsp_2 + (cfg->h - 1) * cfg->logx -
                   PGM_PI2_8 * cfg->x - cfg->lgammah);
    }
    else if (cfg->x > 0) {
        return exp(cfg->hlog2 + cfg->logh - PGM_LS2PI -
                   0.5 * cfg->h2 / cfg->x - 1.5 * cfg->logx);
    }
    return 0;
}

/*
 * Calculate the probability of sampling on either side of the truncation point
 *
 * UpperIncompleteGamma(0.5, x) == sqrt(pi) * erfc(sqrt(x)), the
 * regularized version of the function, which is what we want, can be
 * written as erfc(sqrt(x)) since the denominator of the regularized
 * version cancels with the sqrt(pi). This simplifies the calculation of `p`.
 */
static NPY_INLINE void
calculate_ratio(struct config* cfg)
{
    double p, q;

    if (cfg->z > 0) {
        p = exp(cfg->hlog2 - cfg->h * cfg->z) *
            inverse_gaussian_cdf(cfg->t, cfg->h_z, cfg->h2, false);
    }
    else {
        p = exp(cfg->hlog2) * pgm_erfc(cfg->h / sqrt(2 * cfg->t));
    }
    q = exp(cfg->h * (PGM_LOGPI_2 - cfg->log_lambda_z)) *
            pgm_gammaq(cfg->h, cfg->lambda_z * cfg->t, true);

    cfg->ratio = p / (p + q);
}

/*
 * Initialize the values used frequently during sampling and store them in
 * the config struct
 */
static NPY_INLINE void
initialize_config(struct config* cfg, double h, double z)
{
    cfg->z = z;
    cfg->lambda_z = z > 0 ? PGM_PI2_8 + 0.5 * (z * z) : PGM_PI2_8;
    cfg->log_lambda_z = log(cfg->lambda_z);

    cfg->h = h;
    cfg->t = get_truncation_point(h);
    cfg->one_t = 1.0 / cfg->t;
    cfg->h2 = h * h;
    cfg->logh = log(h);
    cfg->lgammah = pgm_lgamma(h);
    cfg->half_h2 = 0.5 * cfg->h2;
    cfg->hlog2 = h * PGM_LOG2;
    cfg->h_z = z > 0 ? h / z : 0;

    calculate_ratio(cfg);
}


static NPY_INLINE void
update_config(struct config* cfg, double h)
{
    cfg->h = h;
    cfg->t = get_truncation_point(h);
    cfg->one_t = 1.0 / cfg->t;
    cfg->h2 = h * h;
    cfg->logh = log(h);
    cfg->lgammah = pgm_lgamma(h);
    cfg->half_h2 = 0.5 * cfg->h2;
    cfg->hlog2 = h * PGM_LOG2;
    if (cfg->z > 0) cfg->h_z = h / cfg->z;

    calculate_ratio(cfg);
}

/* 
 * Generate from J*(h, z) where {h | 1 <= h <= 4} using the alternate method.
 *
 * To sample from an inverse-gamma we can use the relation:
 * InvGamma(a, b) == 1 / Gamma(a, rate=b). To make sure our samples
 * remain less than t, we sample from a Gamma distribution left-
 * truncated at 1/t (i.e X > 1/t). Then 1/X < t is an Inverse-
 * Gamma right truncated at t. Which is what we want.
 */
static NPY_INLINE double
random_jacobi_alternate_bounded(bitgen_t* bitgen_state, struct config* cfg)
{
    for (;;) {
        if (next_double(bitgen_state) > cfg->ratio) {
            cfg->x = random_left_bounded_gamma(bitgen_state, cfg->h,
                                               cfg->lambda_z, cfg->t);
        }
        else if (cfg->z > 0) {
            cfg->x = random_right_bounded_inverse_gaussian(bitgen_state, cfg->h_z,
                                                           cfg->h2, cfg->t);
        }
        else {
            cfg->x = 1 / random_left_bounded_gamma(bitgen_state, 0.5,
                                                   cfg->half_h2, cfg->one_t);
        }

        cfg->logx = log(cfg->x);
        double u = next_double(bitgen_state) * bounding_kernel(cfg);
        double s = piecewise_coef(0, cfg);

        for (size_t n = 1;; ++n) {
            double old_s = s;
            if (n & 1) {
                s -= piecewise_coef(n, cfg);
                if ((old_s >= s) && (u <= s))
                    return cfg->x;
            }
            else {
                s += piecewise_coef(n, cfg);
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
 * that sum(b_i) = h. Then use the relation PG(h, z) = J*(h, z/2) / 4, to get a
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
 *
 * We pre-calculate all values dependant on h once and avoid
 * unnecessary recalculation as long as h remains larger than 4. This is
 * because the value of ``chunk_size`` is constant and thus caching the
 * constants during repeated calls avoids unecessary overhead.
 */
NPY_INLINE double
random_polyagamma_alternate(bitgen_t *bitgen_state, double h, double z)
{
    struct config cfg;
    double out = 0;

    if (h > 4) {
        initialize_config(&cfg, pgm_h_range, z);
        while (h > 4) {
            out += random_jacobi_alternate_bounded(bitgen_state, &cfg);
            h -= pgm_h_range;
        }
        update_config(&cfg, h);
        out += random_jacobi_alternate_bounded(bitgen_state, &cfg);
        return 0.25 * out;
    }

    initialize_config(&cfg, h, z);
    return 0.25 * random_jacobi_alternate_bounded(bitgen_state, &cfg);
}
