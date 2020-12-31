#ifndef PGM_RANDOM_H
#define PGM_RANDOM_H

#include "../src/pgm_common.h"

typedef enum {HYBRID, DEVROYE, ALTERNATE, SADDLE} sampler_t;

double pgm_random_polyagamma(bitgen_t* bitgen_state, double h, double z,
                             sampler_t type);
void pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                                sampler_t type, size_t n, double* out);
#endif
