#ifndef PGM_DEVROYE_H
#define PGM_DEVROYE_H
#include <stddef.h>

typedef struct bitgen bitgen_t;

void
random_polyagamma_devroye(bitgen_t* bitgen_state, double h, double z,
                          size_t n, double* out);

#endif
