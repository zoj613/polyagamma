#ifndef PGM_DEVROYE_H
#define PGM_DEVROYE_H

#include "pgm_common.h"

double random_polyagamma_devroye(bitgen_t *bitgen_state, double h, double z);
double random_polyagamma_gamma_conv(bitgen_t* bitgen_state, double h, double z);

#endif
