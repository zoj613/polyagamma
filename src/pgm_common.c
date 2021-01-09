/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"


NPY_INLINE double
inverse_gaussian_cdf(double x, double mu, double lambda)
{
    double w = sqrt(0.5 * lambda / x);
    double y = w * x / mu;

    return 0.5 * (1 + erf(y - x) + exp(2 * lambda / mu) * (1 - erf(y + x)));
}
