/* Copyright (c) 2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#pragma once
#ifndef PGM_MACROS_H
#define PGM_MACROS_H

#include <math.h>
#include <numpy/random/bitgen.h>

#define PGM_PI       3.141592653589793238462643383279503   // pi
#define PGM_PI_2     1.570796326794896619231321691639751   // pi / 2
#define PGM_PI2_8    1.233700550136169827354311374984519   // pi^2 / 8
#define PGM_LOGPI_2  0.4515827052894548647261952298948821  // log(pi / 2)
#define PGM_LS2PI    0.9189385332046727417803297364056177  // log(sqrt(2 * pi))
#define PGM_LOG2     0.6931471805599453094172321214581766  // log(2)

#if defined(_MSC_VER)
    #define PGM_INLINE __inline
    #define PGM_EXTERN extern
#elif defined(__GNUC__) || defined(__clang__)
    #define PGM_INLINE inline
    #define PGM_EXTERN extern
#else
    #define PGM_INLINE 
    #define PGM_EXTERN
#endif

#if defined(_MSC_VER)
    #define PGM_FORCEINLINE static __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define PGM_FORCEINLINE static PGM_INLINE __attribute__((always_inline))
#else
    #define PGM_FORCEINLINE static
#endif

#define PGM_MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
 * Test if two numbers equal within the given absolute and relative tolerences
 *
 * `rtol` is the relative tolerance – it is the maximum allowed difference
 * between a and b, relative to the larger absolute value of a or b.
 *
 * `atol` is the minimum absolute tolerance – useful for comparisons near zero.
 */
#define PGM_ISCLOSE(a, b, atol, rtol) \
    (fabs(((a) - (b))) <= PGM_MAX((rtol) * PGM_MAX(fabs((a)), fabs((b))), (atol)))

/*
 * Generate a random single precision float in the range [0, 1). This macros is
 * adapted from a private <numpy/random/distributions.h> function of a similar name
 */
#define next_float(rng) \
    (((rng)->next_uint32((rng)->state) >> 8) * (1.0f / 16777216.0f))

/*
 * Generate a random double precision float in the range [0, 1). This macros is
 * adapted from a private <numpy/random/distributions.h> function of a similar name
 */
#define next_double(rng) \
    ((rng)->next_double((rng)->state))

#endif
