/* Copyright (c) 2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"
#include "../include/pgm_density.h"

#define PGM_LOG2 0.6931471805599453  // log(2)

DECLDIR bool
is_close(double a, double b, double atol, double rtol);

/*
 * Approximate the density function of PG(h, z).
 *
 * The value is calculated with accuracy of up `terms` terms. The calculate
 * will terminate early if successive terms are very small such that the
 * current series value is equal to the previous value with given tolerance.
 */
double
pgm_polyagamma_pdf(double x, double h, double z, size_t terms, double atol, double rtol)
{
    if (x <= 0) {
        return 0;
    }

    double sum = 0;
    double clogx = 1.5 * log(x);
    double a = (fabs(z) > 0 ? h * log(cosh(0.5 * z)) - 0.5 * z * z * x : 0) +
               (h - 1) * PGM_LOG2 - pgm_lgamma(h);

    for (size_t n = 0; n < terms; n++) {
        double twonh = 2 * n + h;
        double term = exp(-PGM_LS2PI - clogx - 0.125 * twonh * twonh / x +
                          log(twonh) + pgm_lgamma(n + h) - pgm_lgamma(n + 1));
        double prev_sum = sum;
        sum += n & 1 ? -term: term;
        if (is_close(sum, prev_sum, atol, rtol)) {
            break;
        }
    }
    return exp(a) * sum;
}
