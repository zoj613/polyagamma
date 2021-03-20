/* Copyright (c) 2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"
#include "../include/pgm_density.h"

#define PGM_LOG2 0.6931471805599453  // log(2)
#define PGM_2PI 6.283185307179586

DECLDIR bool
is_close(double a, double b, double atol, double rtol);

/*
 * Approximate the density function of PG(h, z).
 *
 * The value is calculated with accuracy of up `terms` terms. The calculate
 * will terminate early if successive terms are very small such that the
 * current series value is equal to the previous value with given tolerance.
 */
NPY_INLINE double
pgm_polyagamma_pdf(double x, double h, double z, size_t terms, double rtol)
{
    if (islessequal(x, 0) || isinf(x)) {
        return 0;
    }

    const double a = (fabs(z) > 0 ? h * log(cosh(0.5 * z)) - 0.5 * z * z * x : 0) +
                     (h - 1) * PGM_LOG2 - pgm_lgamma(h);
    double sum = 0;

    for (size_t n = 0; n < terms; n++) {
        double twonh = 2 * n + h;
        double term = twonh * exp(a + pgm_lgamma(n + h) - pgm_lgamma(n + 1) -
                                  0.125 * twonh * twonh / x);
        double prev_sum = sum;
        sum += n & 1 ? -term: term;
        if (is_close(sum, prev_sum, 0, rtol)) {
            break;
        }
    }
    return sum / sqrt(PGM_2PI * x * x * x);
}

/*
 * A struct for passing around arguments of the integrand for the Romberg method
 */
struct integrand_args {
    // (2 * n + h)^2
    double twonh2;
    // 0.5 * z^2
    double halfz2;
};

/*
 * The function to integrate.
 * f(x) = exp(-0.125 * (2n + h)^2 / x - 0.5 * z^2) / sqrt(2 * pi * x^3)
 */
static NPY_INLINE double
integrand(double x, struct integrand_args* arg)
{
    return exp(-0.125 * arg->twonh2 / x - arg->halfz2 * x) / sqrt(PGM_2PI * x * x * x);
}


static NPY_INLINE void
swap_pointers(double** a, double** b)
{
    static double* temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

#ifndef PGM_ROMBERG_ITERS
#define PGM_ROMBERG_ITERS 15
#endif
/*
 * Approximate the integral of the function `integrand` using Romberg's method.
 *
 *  The approximation is accurate up to the given tolerance.
 */
static NPY_INLINE double
romberg(double b, struct integrand_args* args, double rtol)
{
    static double R[PGM_ROMBERG_ITERS][PGM_ROMBERG_ITERS];
    static double* R1 = R[0];
    static double* R2 = R[1];
    static double a = 1e-03;
    double h = b - a;

    R1[0] = 0.5 * h * (integrand(a, args) + integrand(b, args));

    for (size_t i = 1; i < PGM_ROMBERG_ITERS; i++) {
        h *= 0.5;
        double sum = 0;
        for (size_t j = 1; j <= 1 << (i - 1); j++) {
            sum += integrand(a + (2 * j - 1) * h, args);
        }

        R2[0] = 0.5 * R1[0] + h * sum;
        for (size_t j = 1; j <= i; j++) {
            R2[j] = R2[j - 1] + (R2[j - 1] - R1[j - 1]) / ((1 << (2 * j)) - 1);
        }

        if (i > 1 && is_close(R2[i], R1[i - 1], 0, rtol)) {
            return R2[i - 1];
        }
        swap_pointers(&R1, &R2);
    }
    return R1[PGM_ROMBERG_ITERS - 1];
}

/*
 * Appromixate the comulative distribution function of PG(h, z) at x.
 *
 * Romberg's method is used to approximate the integral of
 *
 *      f(x) = exp(-0.125 * (2n + h)^2 / x - 0.5 * z^2) / sqrt(2 * pi * x^3)
 *
 * The infinite sum is truncated at `terms` terms. Convergence
 * of the series is tested after each term is calculated so that if the
 * successive terms are too small to be significant, the calculation is
 * terminated early.
 *
 * References
 * ----------
 * [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
 *     "Bayesian inference for logistic models using Pólya–Gamma latent
 *     variables." Journal of the American statistical Association
 *     108.504 (2013): 1339-1349.
 */
double
pgm_polyagamma_cdf(double x, double h, double z, size_t terms, double rtol)
{
    if (islessequal(x, 0)) {
        return 0;
    }
    else if (isinf(x)) {
        return 1;
    }

    const double a = (fabs(z) > 0 ? h * log(cosh(0.5 * z)) : 0) +
                     (h - 1) * PGM_LOG2 - pgm_lgamma(h);
    struct integrand_args args = {.halfz2 = 0.5 * z * z};
    double sum = 0;

    for (size_t n = 0; n < terms; n++) {
        double twonh = 2 * n + h;
        args.twonh2 = twonh * twonh;
        double term = twonh * exp(a + pgm_lgamma(n + h) - pgm_lgamma(n + 1)) *
                      romberg(x, &args, rtol);
        double prev_sum = sum;
        sum += n & 1 ? -term : term;
        if (is_close(sum, prev_sum, 0, rtol)) {
            break;
        }
    }
    return sum;
}
