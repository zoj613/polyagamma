#ifndef PGM_DEVROYE_H
#define PGM_DEVROYE_H
#include <numpy/random/bitgen.h>

double random_polyagamma_devroye(bitgen_t *bitgen_state, uint64_t n, double z);

#endif
