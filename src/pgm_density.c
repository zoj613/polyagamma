/* Copyright (c) 2021, Zolisa Bleki
 *
 * This file is part of the polyagamma python package.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "pgm_common.h"
#include "../include/pgm_density.h"

#define PGM_2PI 6.283185307179586  // 2 * PI
// Maximum number of series terms to use when approximating the infinite sum
// representation of the PG(h, z) distribution
#ifndef PGM_MAX_SERIES_TERMS
#define PGM_MAX_SERIES_TERMS 200
#endif

/*
 * Compute the density function of PG(h, z). PG(h, z) is written as an
 * infinite alternating-sign sum of Inverse-Gaussian densities when z > 0, or
 * a sum of Inverse-Gamma densities when z = 0. Refer to pages 5 & 6 of [1].
 *
 * References
 * ----------
 * [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
 *     "Bayesian inference for logistic models using Pólya–Gamma latent
 *     variables." Journal of the American statistical Association
 *     108.504 (2013): 1339-1349.
 */
double
pgm_polyagamma_pdf(double x, double h, double z)
{
    if (islessequal(x, 0) || isinf(x)) {
        return 0;
    }

    double a = (fabs(z) > 0 ? h * log(cosh(0.5 * z)) - 0.5 * z * z * x : 0) +
               (h - 1) * PGM_LOG2;
    double sum = exp(a - 0.125 * h * h / x) * h;
    int sign = -1;

    a -= pgm_lgamma(h);
    for (unsigned int n = 1; n < PGM_MAX_SERIES_TERMS; n++, sign *= -1) {
        double twonh = 2 * n + h;
        double term = exp(a + pgm_lgamma(n + h) - 0.125 * twonh * twonh / x -
                          pgm_lgamma(n + 1)) * twonh;
        double prev_sum = sum;
        sum += sign * term;

        if (PGM_ISCLOSE(sum, prev_sum, 0, DBL_EPSILON)) {
            break;
        }
    }
    return sum / sqrt(PGM_2PI * x * x * x);
}

/*
 * Approximate the logarithm of the density function of PG(h, z).
 *
 * logsumexp is applied as an attempt to prevent underflow. The sum of terms
 * is truncated at `PGM_MAX_SERIES_TERMS` terms.
 */
double
pgm_polyagamma_logpdf(double x, double h, double z)
{
    if (islessequal(x, 0) || isinf(x)) {
        return -INFINITY;
    }

    double lg = pgm_lgamma(h);
    double a = (fabs(z) > 0 ? h * log(cosh(0.5 * z)) - 0.5 * z * z * x : 0) +
               (h - 1) * PGM_LOG2 - PGM_LS2PI - 1.5 * log(x) - lg;
    double first = lg - 0.125 * h * h / x;
    double sum = 1;
    int sign = -1;

    for (unsigned int n = 1; n < PGM_MAX_SERIES_TERMS; n++, sign *= -1) {
        double t = 2 * n + h;
        double curr = pgm_lgamma(n + h) - 0.125 * t * t / x - pgm_lgamma(n + 1);
        sum += sign * exp(curr - first) * t / h;
    }

    return a + (log(h) + first + log(sum));
}

/*
 * Struct to store arguments passed to the cdf functions.
 * `s2x` is sqrt(2x) if z == 0, else sqrt(x). `a` is (2n + h)
 */
struct cdf_args {
    double s2x;
    double a;
    double x;
    double z;
};

/*
 * CDF of the Inverse-Gamma(0.5, 0.125 * a^2) distribution, where `a` is the (2n + h).
 *
 * The CDF for these parameters simplifies to a simple expression involving a
 * call to `erfc`. For x >= 26.55 where erfc is guaranteed to underflow, we
 * use an approximation based on equation 7.1.28 of [1].
 *
 * References
 * ----------
 *  [1] Abramowitz, Milton; Stegun, Irene A. Handbook of mathematical functions
 *      with formulas, graphs, and mathematical tables. National Bureau of
 *      Standards Applied Mathematics Series, 55 For sale by the Superintendent of
 *      Documents, U.S. Government Printing Office, Washington, D.C. 1964 xiv+1046
 *      pp.
 */
static NPY_INLINE double
invgamma_logcdf(struct cdf_args* const arg)
{
    double x = 0.5 * arg->a / arg->s2x;

    if (isgreater(x, 26.55)) {
        static const double p0 = 0.3275911;
        static const double p1 = 0.254829592;
        static const double p2 = -0.284496736;
        static const double p3 = 1.421413741;
        static const double p4 = -1.453152027;
        static const double p5 = 1.061405429;
        double t = 1 / (1 + p0 * x);
        return log(t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))))) - x * x;
    }
    return log(erfc(x));
}

/*
 * Compute the logarithm of the CDF of the standard normal distribution.
 *
 * Care is taken to prevent underflow when input in a large negative number.
 *
 * For x < -37.5 where the erfc(|x| / sqrt(2)) is guaranteed to underflow, we
 * use an approximation based on [1] to avoid log(0). We use the relation
 * P[X < x] = P[X > |x|] for x < 0.
 *
 * References
 * ----------
 *  [1] Byrc, W. (2001).A uniform approximation to the right normal tail
 *     integral. Applied Mathematics and Computation,127, 365-374.
 */
static NPY_INLINE double
norm_logcdf(double x)
{
    if (isless(x, -37.5)) {
        static const double p0 = 12.77436324;
        static const double p1 = 5.575192695;
        static const double q0 = 25.54872648;
        static const double q1 = 31.53531977;
        static const double q2 = 14.38718147;
        static const double s2p = 2.5066282746310002;  // sqrt(2*pi)
        x = -x;
        double r = (p0 + x * (p1 + x)) / (x * x * x * s2p + q0 + x * (q1 + q2 * x));
        return log(r) - 0.5 * x * x;
    }

    double y = x / 1.4142135623730951;
    double z = fabs(y);

    if (isless(z, 1)) {
        return log(0.5 + 0.5 * erf(y));
    }

    double a = 0.5 * erfc(z);
    if (y > 0) {
        return log1p(-a);
    }

    return log(a);
}

/*
 * CDF of the Inverse-Gaussian(0.5 * a / z, a^2 / 4) distribution, where
 * a = (2n + h). We use the method of Goknur & Smyth (2016) to prevent
 * underflow/overflow when parameter values are either very small or large.
 */
static NPY_INLINE double
invgauss_logcdf(struct cdf_args* const arg)
{
    double qm = 2 * arg->x * arg->z / arg->a;
    double r = 2 * arg->s2x / arg->a;
    double a = norm_logcdf((qm - 1) / r);
    double b = arg->z * arg->a + norm_logcdf(-(qm + 1) / r);

    return a + log1p(exp(b - a));
}


typedef double (*logcdf_func_t)(struct cdf_args*);

/*
 * Approximate the distribution function of PG(h, z).
 *
 * Note: The first term of the sum is evaluated before the loop to avoid
 * redundancy.
 */
double
pgm_polyagamma_cdf(double x, double h, double z)
{
    if (islessequal(x, 0)) {
        return 0;
    }
    else if (isinf(x)) {
        return 1;
    }

    z = fabs(z);
    double c, zn;
    logcdf_func_t logcdf;
    struct cdf_args arg = {.a = h, .x = x, .z = z};

    if (z > 0) {
        logcdf = invgauss_logcdf;
        c = h * log1p(exp(-z));
        arg.s2x = sqrt(x);
        zn = z;
    }
    else {
        logcdf = invgamma_logcdf;
        c = h * PGM_LOG2;
        arg.s2x = sqrt(2 * x);
        zn = 0;
    }

    double sum = exp(c + logcdf(&arg));
    int sign = -1;

    c -= pgm_lgamma(h);
    for (unsigned int n = 1; n < PGM_MAX_SERIES_TERMS; n++, sign *= -1, zn = z * n) {
        arg.a = 2 * n + h;
        double term = exp(c + pgm_lgamma(n + h) + logcdf(&arg) - pgm_lgamma(n + 1) - zn);
        double prev_sum = sum;
        sum += sign * term;

        if (PGM_ISCLOSE(sum, prev_sum, 0, DBL_EPSILON)) {
            break;
        }
    }

    return sum;
}

/*
 * Approximate the logarithm of the distribution function of PG(h, z).
 *
 * logsumexp is applied as an attempt to prevent underflow. The sum of
 * terms is truncated at `PGM_MAX_SERIES_TERMS` terms.
 *
 * Note: The first term of the sum is evaluated before the loop to avoid
 * redundancy.
 */
double
pgm_polyagamma_logcdf(double x, double h, double z)
{
    if (islessequal(x, 0)) {
        return -INFINITY;
    }
    else if (isinf(x)) {
        return 0;
    }

    z = fabs(z);
    double c, zn;
    logcdf_func_t logcdf;
    double lg = pgm_lgamma(h);
    struct cdf_args arg = {.a = h, .x = x, .z = z};

    if (z > 0) {
        logcdf = invgauss_logcdf;
        c = h * log1p(exp(-z)) - lg;
        arg.s2x = sqrt(x);
        zn = z;
    }
    else {
        logcdf = invgamma_logcdf;
        c = h * PGM_LOG2 - lg;
        arg.s2x = sqrt(2 * x);
        zn = 0;
    }

    double first = lg + logcdf(&arg);
    double sum = 1;
    int sign = -1;

    for (unsigned int n = 1; n < PGM_MAX_SERIES_TERMS; n++, sign *= -1, zn = z * n) {
        arg.a = 2 * n + h;
        double curr = pgm_lgamma(n + h) + logcdf(&arg) - pgm_lgamma(n + 1) - zn;
        sum += sign * exp(curr - first);
    }

    return c + (first + log(sum));
}
