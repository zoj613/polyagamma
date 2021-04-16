/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"


DECLDIR NPY_INLINE float
pgm_erfc(float x);

DECLDIR NPY_INLINE double
pgm_lgamma(double z);

DECLDIR NPY_INLINE double
random_left_bounded_gamma(bitgen_t* bitgen_state, double a, double b, double t);
