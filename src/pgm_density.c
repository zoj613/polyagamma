/* Copyright (c) 2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This file is part of the polyagamma python package.
 * See: https://github.com/zoj613/polyagamma
 *
 * NOTE
 * ----
 * This module provides functions to compute the Polya-Gamma distribution's
 * density function, its cumulative distribution function and their logarithms.
 *
 * Polson et. al (2013) [Page 5 & 6] shows an expression of the distribution's
 * density function as an infinite alternating sum. A bit of algebraic
 * manipulation shows that when z=0, the density function can be re-written as:
 *
 *  f(x|h) = 2^h / gamma(h) * \Sigma^{\infty}_{n=0}
 *          ((-1)^n * gamma(n + h) / gamma(n + 1) * g(x| 0.5, 0.125 * (2n + h)^2))
 *  where g() is the density function of an Inverse-Gamma distribution with
 *  shape parameter 0.5 and scale parameter (2n + h)^2 / 8.
 *
 * Similarly, for |z| > 0 the density can be written more conveniently as:
 *
 *  f(x|h, z) = (1 + exp(-|z|))^h / gamma(h) * \Sigma^{\infty}_{n=0}
 *              ((-1)^n * gamma(n + h) / gamma(n + 1) * exp(-|z| * n) *
 *              g(x| 0.5 * (2n + h) / |z|, 0.25 * (2n + h)^2))
 *  where g() is the density function of an Inverse-Gaussian distribution with
 *  mean = 0.5 * (2n + h) / |z| and shape parameter (2n + h)^2 / 4.
 *
 * The above expressions for f(x) are central to the implementation used to
 * approximate the CDF of the Polya-Gamma distribution.
 */
#include <stdlib.h>
#include "pgm_macros.h"
#include "../include/pgm_density.h"

// Maximum number of series terms to use when approximating the infinite sum
// representation of the PG(h, z) distribution
#ifndef PGM_MAX_SERIES_TERMS
#define PGM_MAX_SERIES_TERMS 200
#endif

#ifndef DBL_EPSILON
#define DBL_EPSILON 2.22045e-16
#endif

PGM_EXTERN double
pgm_lgamma(double z);

/* Parameters used as arguments to the log density functions.
 *
 * x: value at which the density is evaluated.
 * h: shape parameter of the PG distribution.
 * z: exponential tilting parameter of the PG distribution.
 * n: n'th iteration of the alternating infinite sum used to evaluate density.
 */
typedef struct {
    double x;
    double h;
    double z;
    double n;
} pdf_parameter_t;

/*
 * This function implements the log-density function of an Inverse-Gamma distribution
 * with shape parameter 0.5 and scale parameter 0.125 * (2n + h)^2. This is
 * equivalent to a scaled-Inverse-Chi-Squared distribution with degrees of
 * freedom 1 and scale parameter rho^2 = 0.25 * (2n + h)^2. This is also
 * equivalent to a Levy distribution with location parameter equal to 0 and a
 * scale parameter equal to 0.25 * (2n + h)^2.
 *
 * To summarize, the following are all equivalent:
 * - Inverse-Gamma(shape=0.5, scale=0.125 * (2n + h)^2)
 * - Scaled-Inv-Chi2(df=1, scale=0.25 * (2n + h)^2)
 * - Levy(loc=0, scale=0.25 * (2n + h)^2)
 *
 * We use the Levy distribution parametrization. We also scale the distribution
 * using the relation:
 *      f(x; loc, scale) = f(x/scale; loc, 1) / scale
 * leading to a log density of the form:
 *      logf = log(f(x/scale; loc, 1)) - log(scale).
 * This form is more stable and avoids overflow/underflow when scale parameter
 * is large.
 *
 * References
 * ----------
 *  [1] https://en.wikipedia.org/wiki/Inverse-gamma_distribution
 *  [2] https://en.wikipedia.org/wiki/Scaled-inverse-chi-squared_distribution
 *  [3] https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
 */
static PGM_INLINE double
invgamma_logpdf(const pdf_parameter_t* pr)
{
    double scale = pr->n * pr->n + pr->n * pr->h + (0.5 * pr->h) * (0.5 * pr->h);
    double y = pr->x / scale;

    return -PGM_LS2PI - 1.5 * log(y) - 0.5 / y - log(scale);
}

/*
 * This function implements the log-density function of an Inverse-Gaussian
 * distribution with mean = 0.5 * (2n + h) / |z| and shape 0.25 * (2n + h)^2.
 * As outlined in [1], the mean can act as a scale parameter of the
 * distribution. Therefore to avoid instability at extreme values, we re-scale
 * so that we end up with a mean of 1 and a shape of 0.5 * (2n + h) * |z|,
 * using:
 *      f(x; mean, shape) = f(x/mean; 1, shape/mean) / mean.
 *
 * References
 * ----------
 * [1] Giner, G., & Smyth, G. K. (2016). statmod: probability calculations for
 *     the inverse Gaussian distribution. In R Journal (Vol. 8, Issue 1, pp. 339-351).
 */
static PGM_INLINE double
invgauss_logpdf(const pdf_parameter_t* pr)
{
    double mean = pr->n / pr->z + 0.5 * pr->h / pr->z;
    double shape = pr->n * pr->z + 0.5 * pr->h * pr->z;
    double y = pr->x / mean;

    return -PGM_LS2PI - 1.5 * log(y) + 0.5 * log(shape) -
            (0.5 * shape) * (y - 2. + 1. / y) - log(mean);
}


typedef double (*logpdf_func_t)(const pdf_parameter_t*);


static PGM_INLINE double
logsumexp(size_t n, double array[const n], double max_val, double scale[const n])
{
    double out = 0;

    for (size_t i = 0; i < n; ++i) {
        out += scale[i] * exp(array[i] - max_val);
    }

    return max_val + log(fabs(out));
}


double
pgm_polyagamma_logpdf(double x, double h, double z)
{
    if (islessequal(x, 0.) || isinf(x) || isinf(h)) {
        return -INFINITY;
    }

    z = fabs(z);
    double c;
    logpdf_func_t logpdf;
    double lg = pgm_lgamma(h);
    pdf_parameter_t pr = {.x = x, .z = z, .h = h, .n = 0};
    double* sign = malloc(PGM_MAX_SERIES_TERMS * sizeof(*sign));
    double* elems = malloc(PGM_MAX_SERIES_TERMS * sizeof(*elems));

    if (z > 0.) {
        logpdf = invgauss_logpdf;
        c = h * log1p(exp(-z)) - lg;
    }
    else {
        logpdf = invgamma_logpdf;
        c = h * PGM_LOG2 - lg;
    }

    sign[0] = 1;
    elems[0] = lg + logpdf(&pr);
    double max_elem = elems[0];
    size_t n = 1;

    /* TODO: Compute the loggamma difference using a numerically stable way.
     * something like: log(pochhammer(n + 1, h - 1)) */
    do {
        pr.n = n;
        elems[n] = pgm_lgamma(n + h) - pgm_lgamma(n + 1) + logpdf(&pr) - z * n;
        sign[n] = -sign[n - 1];
        max_elem = elems[n] > max_elem ? elems[n] : max_elem;
    } while (++n < PGM_MAX_SERIES_TERMS);

    double out = c + logsumexp(n, elems, max_elem, sign);
    free(sign);
    free(elems);

    return out;
}


PGM_INLINE double
pgm_polyagamma_pdf(double x, double h, double z)
{
    return exp(pgm_polyagamma_logpdf(x, h, z));
}

/*
 * Struct to store arguments passed to the cdf functions.
 * `s2x` is sqrt(2x) if z == 0, else sqrt(x). `a` is (2n + h)
 */
typedef struct {
    double s2x;
    double a;
    double x;
    double z;
} cdf_parameter_t;

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
static PGM_INLINE double
invgamma_logcdf(const cdf_parameter_t* pr)
{
    double x = 0.5 * pr->a / pr->s2x;

    if (isgreater(x, 26.55)) {
        static const double p0 = 0.3275911;
        static const double p1 = 0.254829592;
        static const double p2 = -0.284496736;
        static const double p3 = 1.421413741;
        static const double p4 = -1.453152027;
        static const double p5 = 1.061405429;
        double t = 1. / (1. + p0 * x);
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
static PGM_INLINE double
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
    /* y = x / sqrt(2) */
    double y = x * 0.7071067811865475;
    double z = fabs(y);

    if (isless(z, 1.)) {
        return log(0.5 + 0.5 * erf(y));
    }

    double a = 0.5 * erfc(z);
    if (y > 0.) {
        return log1p(-a);
    }

    return log(a);
}

/*
 * CDF of the Inverse-Gaussian(0.5 * a / |z|, a^2 / 4) distribution, where
 * a = (2n + h). We use the method of Goknur & Smyth (2016) to prevent
 * underflow/overflow when parameter values are either very small or large.
 */
static PGM_INLINE double
invgauss_logcdf(const cdf_parameter_t* pr)
{
    double qm = 2. * pr->x * pr->z / pr->a;
    double r = 2. * pr->s2x / pr->a;
    double a = norm_logcdf((qm - 1.) / r);
    double b = pr->z * pr->a + norm_logcdf(-(qm + 1.) / r);

    return a + log1p(exp(b - a));
}


typedef double (*logcdf_func_t)(const cdf_parameter_t*);

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
    if (islessequal(x, 0.)) {
        return -INFINITY;
    }
    else if (isinf(x)) {
        return 0.;
    }

    z = fabs(z);
    double c;
    logcdf_func_t logcdf;
    double lg = pgm_lgamma(h);
    cdf_parameter_t pr = {.a = h, .x = x, .z = z};
    double* sign = malloc(PGM_MAX_SERIES_TERMS * sizeof(*sign));
    double* elems = malloc(PGM_MAX_SERIES_TERMS * sizeof(*elems));

    if (z > 0.) {
        logcdf = invgauss_logcdf;
        c = h * log1p(exp(-z)) - lg;
        pr.s2x = sqrt(x);
    }
    else {
        logcdf = invgamma_logcdf;
        c = h * PGM_LOG2 - lg;
        pr.s2x = sqrt(2. * x);
    }

    sign[0] = 1;
    elems[0] = lg + logcdf(&pr);
    double max_elem = elems[0];
    size_t n = 1;

    /* TODO: Compute the loggamma difference using a numerically stable way.
     * something like: log(pochhammer(n + 1, h - 1)) */
    do {
        pr.a = 2 * n + h;
        elems[n] = pgm_lgamma(n + h) - pgm_lgamma(n + 1) + logcdf(&pr) - z * n;
        sign[n] = -sign[n - 1];
        max_elem = elems[n] > max_elem ? elems[n] : max_elem;
    } while (++n < PGM_MAX_SERIES_TERMS);

    double out = c + logsumexp(n, elems, max_elem, sign);
    free(sign);
    free(elems);

    return out;
}


double
pgm_polyagamma_cdf(double x, double h, double z)
{
    return fmax(fmin(exp(pgm_polyagamma_logcdf(x, h, z)), 1.), 0.);
}
