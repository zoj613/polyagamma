/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"


extern NPY_INLINE double
pgm_erfc(double x);

extern NPY_INLINE double
pgm_lgamma(double z);

extern NPY_INLINE double
inverse_gaussian_cdf(double x, double mu, double lambda, bool robust);

extern NPY_INLINE double
random_left_bounded_gamma(bitgen_t* bitgen_state, double a, double b, double t);

extern NPY_INLINE double
random_right_bounded_inverse_gaussian(bitgen_t* bitgen_state, double mu,
                                      double lambda, double t);
