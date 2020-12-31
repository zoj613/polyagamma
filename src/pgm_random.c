#include "pgm_devroye.h"
#include "../include/pgm_random.h"


double
pgm_random_polyagamma(bitgen_t* bitgen_state, double h, double z, sampler_t type)
{
    switch(type) {
        //TODO: Add other samplers (alternative and saddle point apporoximation)
        default:
            return random_polyagamma_devroye(bitgen_state, h, z);
    }
}

NPY_INLINE void
pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                           sampler_t type, size_t n, double* out)
{
    for (size_t i = n; i--; )
        out[i] = pgm_random_polyagamma(bitgen_state, h, z, type);
}
