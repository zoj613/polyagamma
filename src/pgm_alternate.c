/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"
#include "pgm_alternate.h"
#include "pgm_alternate_trunc_points.h"

/* a struct to store frequently used values. This avoids unnecessary
 * recalculation of these values during a single call to the sampler.
 */
typedef struct {
    // q / (p + q)
    float proposal_probability;
    double log_lambda_z;
    // pi^2 / 8 + 0.5 * z * z
    double lambda_z;
    double half_h2;
    // loggamma(h)
    double lgammah;
    double hlog2;
    // 1 / t;
    double t_inv;
    double logx;
    // (h / z) ** 2
    double h_z2;
    double h_z;
    double z2;
    double h;
    double z;
    double x;
    double t;
} parameter_t;

/* 
 * Return the optimal truncation point for a given value of h in the range
 * [1, 4]. Values are retrieved from a lookup table using binary search, then
 * the final value is calculated using linear interpolation.
 */
static NPY_INLINE double
get_truncation_point(double h)
{
    if (h <= 1) {
        return pgm_f[0];
    }
    else if (h == pgm_maxh) {
        return pgm_f[pgm_table_size - 1];
    }
    else {
        // start binary search
        size_t index, offset = 0, len = pgm_table_size - 1;

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
}

/*
 * Compute a^L(x|h), the n'th coefficient for the alternating sum S^L(x|h)
 */
static NPY_INLINE float
piecewise_coef(unsigned int n, parameter_t const* pr)
{
    double a = 2 * n + pr->h;
    double b = n ? pgm_lgamma(n + pr->h) - pr->lgammah : 0;

    return expf(pr->hlog2 + b - pgm_lgamma(n + 1) - PGM_LS2PI -
                1.5 * pr->logx - 0.5 * a * a / pr->x) * (float)a;
}

// compute: k(x|h)
static NPY_INLINE float
bounding_kernel(parameter_t const* pr)
{
    if (pr->x > pr->t) {
        static const double a = 0.22579135264472733;  // log(sqrt(pi / 2))
        return expf(pr->h * a + (pr->h - 1.) * pr->logx -
                    PGM_PI2_8 * pr->x - pr->lgammah);
    }
    else if (pr->x > 0) {
        return expf(pr->hlog2 - pr->half_h2 / pr->x -
                    1.5 * pr->logx - PGM_LS2PI) * (float)pr->h;
    }
    return 0.f;
}

/*
 * Compute the cdf of the inverse-gaussian distribution.
 */
static NPY_INLINE float
invgauss_cdf(parameter_t const* pr)
{
    static const double sqrt2_inv = 0.7071067811865475f;
    double st = sqrt(pr->t);
    double a = sqrt2_inv * pr->h / st;
    double b = pr->z * st * sqrt2_inv;
    float ez = expf(pr->h * pr->z);

    return 0.5f * (pgm_erfc(a - b) + ez * pgm_erfc(b + a) * ez);
}

/*
 * Initialize the values used frequently during sampling and store them in
 * the config struct
 *
 * Parameters
 * ----------
 *  pr : parameter_t*
 *      Pointer to a `config` struct that strores sampling parameters.
 *  h : double
 *      shape parameter of the distribution.
 *  update_params : bool
 *      Whether to update a previously initialized set of parameters with a
 *      new `h` value.
 *
 * Notes
 * -----
 * To calculate the probability of sampling on either side of the truncation,
 * point we note that:
 * - UpperIncompleteGamma(0.5, x) == sqrt(pi) * erfc(sqrt(x)), the regularized
 *   version of the function, can be written as erfc(sqrt(x)) since the
 *   denominator of the regularized version cancels with the sqrt(pi).
 *   This simplifies the calculation of `p` in the ratio = p / (p + q).
 */
static NPY_INLINE void
set_sampling_parameters(parameter_t* const pr, double h, bool update_params)
{
    float p, q;

    pr->h = h;
    pr->t = get_truncation_point(h);
    pr->t_inv = 1. / pr->t;
    pr->half_h2 = 0.5 * h * h;
    pr->lgammah = pgm_lgamma(h);
    pr->hlog2 = h * PGM_LOG2;

    if (!update_params && pr->z > 0) {
        pr->h_z = h / pr->z;
        pr->z2 = pr->z * pr->z;
        pr->h_z2 = pr->h_z * pr->h_z;
        pr->lambda_z = PGM_PI2_8 + 0.5 * pr->z2;
        pr->log_lambda_z = logf(pr->lambda_z);
        p = expf(pr->hlog2 - h * pr->z) * invgauss_cdf(pr);
    }
    else if (pr->z > 0) {
        pr->h_z = h / pr->z;
        pr->h_z2 = pr->h_z * pr->h_z;
        p = expf(pr->hlog2 - h * pr->z) * invgauss_cdf(pr);
    }
    else if (!update_params) {
        pr->lambda_z = PGM_PI2_8;
        pr->log_lambda_z = logf(pr->lambda_z);
        p = expf(pr->hlog2) * pgm_erfc(h / sqrt(2. * pr->t));
    }
    else {
        p = expf(pr->hlog2) * pgm_erfc(h / sqrt(2. * pr->t));
    }
    q = expf(h * (PGM_LOGPI_2 - pr->log_lambda_z)) *
        pgm_gammaq(h, pr->lambda_z * pr->t, true);

    pr->proposal_probability = q / (p + q);
}

/*
 * Sample from an Inverse-Gaussian(mu, lambda) truncated on the set {x | x < t}.
 *
 * We sample using two algorithms depending on whether mu > t or mu < t.
 *
 * When mu < t, We use a known sampling algorithm from Devroye
 * (1986), page 149. We sample until the generated variate is less than t.
 *
 * When mu > t, we use a Scaled-Inverse-Chi-square distribution as a proposal,
 * as explained in [1], page 134. This is equivalent to an Inverse-Gamma with
 * shape=0.5 and scale=lambda/2. We accept the sample only if we sample a
 * uniform less than the acceptance porbability. The probability is
 * exp(-0.5 * z^2 * z). (Refer to Appendix 1 of [1] for a derivation of this probablity).
 *
 * References
 * ----------
 *  [1] Windle, J. (2013). Forecasting high-dimensional, time-varying
 *      variance-covariance matrices with high-frequency data and sampling
 *      Pólya-Gamma random variates for posterior distributions derived from
 *      logistic likelihoods.(PhD thesis). Retrieved from
 *      http://hdl.handle.net/2152/21842
 */
static NPY_INLINE void
random_right_bounded_invgauss(bitgen_t* bitgen_state, parameter_t* const pr)
{
    if (pr->t < pr->h_z) {
        do {
            pr->x = 1. / random_left_bounded_gamma(bitgen_state, 0.5,
                                                   pr->half_h2, pr->t_inv);
        } while (log1pf(-next_float(bitgen_state)) >= -0.5 * pr->z2 * pr->x);
        return;
    }
    do {
        double y = random_standard_normal(bitgen_state);
        double w = pr->h_z + 0.5 * y * y / pr->z2;
        pr->x = w - sqrt(w * w - pr->h_z2);
        if (next_double(bitgen_state) * (pr->h_z + pr->x) > pr->h_z) {
            pr->x = pr->h_z2 / pr->x;
        }
    } while (pr->x >= pr->t);
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
random_jacobi_star(bitgen_t* bitgen_state, parameter_t* const pr)
{
    for (;;) {
        if (next_float(bitgen_state) <= pr->proposal_probability) {
            pr->x = random_left_bounded_gamma(bitgen_state, pr->h,
                                              pr->lambda_z, pr->t);
        }
        else if (pr->z > 0) {
            random_right_bounded_invgauss(bitgen_state, pr);
        }
        else {
            pr->x = 1. / random_left_bounded_gamma(bitgen_state, 0.5,
                                                   pr->half_h2, pr->t_inv);
        }

        pr->logx = logf(pr->x);
        float u = next_float(bitgen_state) * bounding_kernel(pr);
        float s = piecewise_coef(0, pr);

        for (unsigned int n = 1;; ++n) {
            float old_s = s;
            if (n & 1) {
                s -= piecewise_coef(n, pr);
                if (isgreaterequal(old_s, s) && islessequal(u, s))
                    return pr->x;
            }
            else {
                s += piecewise_coef(n, pr);
                if (isgreaterequal(old_s, s) && isgreater(u, s))
                    break;
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
 * See: Section 4.3 of Windle et al. (2014)
 *
 * We pre-calculate all values dependant on h only once and avoid
 * unnecessary recalculation as long as h remains larger than 4.
 */
double
random_polyagamma_alternate(bitgen_t *bitgen_state, double h, double z)
{
    parameter_t pr = {.z = z};

    if (h > pgm_maxh) {
        double out = 0;
        size_t chunk = h >= (pgm_maxh + 1) ? pgm_maxh : pgm_maxh - 1;

        set_sampling_parameters(&pr, chunk, false);
        while (h > pgm_maxh) {
            out += random_jacobi_star(bitgen_state, &pr);
            h -= chunk;
        }

        set_sampling_parameters(&pr, h, true);
        out += random_jacobi_star(bitgen_state, &pr);
        return 0.25 * out;
    }
    set_sampling_parameters(&pr, h, false);
    return 0.25 * random_jacobi_star(bitgen_state, &pr);
}
