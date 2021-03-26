/* Copyright (c) 2021, Zolisa Bleki
 *
 * This file is part of the polyagamma python package.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "pgm_common.h"
#include "../include/pgm_density.h"

#define PGM_2PI 6.283185307179586  // 2 * PI
/* Maximum number of series terms to use when approximating the infinite sum
 * representation of the PG(h, z) distribution
 */
#ifndef PGM_MAX_SERIES_TERMS
#define PGM_MAX_SERIES_TERMS 200
#endif

DECLDIR bool
is_close(double a, double b, double atol, double rtol);

/*
 * Compute the density function of PG(h, z). PG(h, z) is written as an
 * infinite alternating-sign sum of Inverse-Gaussian densities when z > 0, or
 * a sum of Inverse-Gamma densities when z = 0.
 */
double
pgm_polyagamma_pdf(double x, double h, double z)
{
    if (islessequal(x, 0) || isinf(x)) {
        return 0;
    }

    double sum = 0;
    double a = (fabs(z) > 0 ? h * log(cosh(0.5 * z)) - 0.5 * z * z * x : 0) +
               (h - 1) * PGM_LOG2 - pgm_lgamma(h);

    for (size_t n = 0; n < PGM_MAX_SERIES_TERMS; n++) {
        double twonh = 2 * n + h;
        double term = exp(a + pgm_lgamma(n + h) - 0.125 * twonh * twonh / x -
                          pgm_lgamma(n + 1)) * twonh;
        double prev_sum = sum;
        sum += n & 1 ? -term: term;

        if (is_close(sum, prev_sum, 0, DBL_EPSILON)) {
            break;
        }
    }
    return sum / sqrt(PGM_2PI * x * x * x);
}

/*
 * Approximate the logarithm of the density function of PG(h, z).
 *
 * logsumexp is applied as an attempt to minimize numerical error. The sum of
 * terms is truncated at `PGM_MAX_SERIES_TERMS` terms.
 */
double
pgm_polyagamma_logpdf(double x, double h, double z)
{
    if (islessequal(x, 0) || isinf(x)) {
        return -INFINITY;
    }

    static double arr[PGM_MAX_SERIES_TERMS];
    double a = (fabs(z) > 0 ? h * log(cosh(0.5 * z)) - 0.5 * z * z * x : 0) +
               (h - 1) * PGM_LOG2 - pgm_lgamma(h);
    double b = -PGM_LS2PI - 1.5 * log(x);
    double logh = log(h);
    double sum = 0;

    for (size_t n = 0; n < PGM_MAX_SERIES_TERMS; n++) {
        double t = 2 * n + h;
        arr[n] = a + b + pgm_lgamma(n + h) - 0.125 * t * t / x - pgm_lgamma(n + 1);
        sum += (n & 1 ? -1 : 1) * exp(arr[n] - arr[0]) * t / h;
    }

    return (logh + arr[0]) + log(sum);
}

/*
 * Struct to store arguments passed to the cdf functions.
 * `s2x` is  sqrt(2x) if z == 0, else sqrt(x)
 * `a` is (2n + h)
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
invgamma_logcdf(struct cdf_args* arg)
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
 * Care is taken to minimize error when input in negative.
 *
 * For x < -38 where the erfc(|x| / sqrt(2)) is guaranteed to underflow, we use
 * an approximation based on [1] to avoid log(0). We use the relation
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
 * a = (2n + h). We use the method of Goknur & Smyth (2016) to minimize error.
 */
static NPY_INLINE double
invgauss_logcdf(struct cdf_args* arg)
{
    double qm = 2 * arg->x * arg->z / arg->a;
    double r = 2 * arg->s2x / arg->a;
    double a = norm_logcdf((qm - 1) / r);
    double b = arg->z * arg->a + norm_logcdf(-(qm + 1) / r);

    return a + log1p(exp(b - a));
}


typedef double (*logcdf_func)(struct cdf_args*);

/*
 * Approximate the distribution function of PG(h, z).
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
    double sum = 0;
    double c = (z > 0 ? h * log1p(exp(-z)) : h * PGM_LOG2) - pgm_lgamma(h);
    logcdf_func logcdf = z > 0 ? invgauss_logcdf : invgamma_logcdf;
    struct cdf_args arg = {
        .s2x = (z > 0 ? sqrt(x) : sqrt(2 * x)),
        .x = x,
        .z = z,
    };

    for (size_t n = 0; n < PGM_MAX_SERIES_TERMS; n++) {
        arg.a = 2 * n + h;
        double term = exp(c + pgm_lgamma(n + h) + logcdf(&arg) -
                          pgm_lgamma(n + 1) - (z > 0 ? z * n : 0));
        double prev_sum = sum;
        sum += n & 1 ? -term : term;
        if (is_close(sum, prev_sum, 0, DBL_EPSILON)) {
            break;
        }
    }

    return sum;
}

/*
 * Approximate the logarithm of the distribution function of PG(h, z).
 *
 * logsumexp is applied as an attempt to minimize numerical error. The sum of
 * terms is truncated at `PGM_MAX_SERIES_TERMS` terms. If an underflow is
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
    double sum = 0;
    static double arr[PGM_MAX_SERIES_TERMS];
    double c = (z > 0 ? h * log1p(exp(-z)) : h * PGM_LOG2) - pgm_lgamma(h);
    logcdf_func logcdf = z > 0 ? invgauss_logcdf : invgamma_logcdf;
    struct cdf_args arg = {
        .s2x = (z > 0 ? sqrt(x) : sqrt(2 * x)),
        .x = x,
        .z = z
    };

    for (size_t n = 0; n < PGM_MAX_SERIES_TERMS; n++) {
        arg.a = 2 * n + h;
        arr[n] = pgm_lgamma(n + h) + logcdf(&arg) -
                 pgm_lgamma(n + 1) - (z > 0 ? z * n : 0);
        sum += (n & 1 ? -1 : 1) * exp(arr[n] - arr[0]);
    }

    return (c + arr[0]) + log(sum);
}
