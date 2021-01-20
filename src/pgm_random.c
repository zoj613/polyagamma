/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "../include/pgm_random.h"
#include "pgm_devroye.h"
#include "pgm_alternate.h"
#include "pgm_saddle.h"


static NPY_INLINE double
random_polyagamma_hybrid(bitgen_t* bitgen_state, double h, double z)
{
    if (h < 1) {
        return random_polyagamma_gamma_conv(bitgen_state, h, z);
    }
    else {
        double fract_part, int_part;
        fract_part = modf(h, &int_part);
        if (fract_part > 0)
            return random_polyagamma_alternate(bitgen_state, h, z);
        return random_polyagamma_devroye(bitgen_state, (uint64_t)int_part, z);
    }
}


double
pgm_random_polyagamma(bitgen_t* bitgen_state, double h, double z, sampler_t method)
{
    switch(method) {
        // TODO: Add saddle point apporoximation sampling method
        case GAMMA:
            return random_polyagamma_gamma_conv(bitgen_state, h, z);
        case DEVROYE:
            return random_polyagamma_devroye(bitgen_state, (uint64_t)h, z);
        case ALTERNATE:
            return random_polyagamma_alternate(bitgen_state, h, z);
        case SADDLE:
            return random_polyagamma_saddle(bitgen_state, h, z);
        default:
            return random_polyagamma_hybrid(bitgen_state, h, z);
    }
}


NPY_INLINE void
pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                           sampler_t method, size_t n, double* out)
{
    for (size_t i = n; i--; )
        out[i] = pgm_random_polyagamma(bitgen_state, h, z, method);
}


NPY_INLINE void
pgm_random_polyagamma_fill2(bitgen_t* bitgen_state, const double* h,
                            const double* z, sampler_t method, size_t n,
                            double* restrict out)
{
    for (size_t i = n; i--; )
        out[i] = pgm_random_polyagamma(bitgen_state, h[i], z[i], method);
}
