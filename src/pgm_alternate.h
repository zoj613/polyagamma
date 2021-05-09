#ifndef PGM_ALTERNATE_H
#define PGM_ALTERNATE_H
#include <stddef.h>

typedef struct bitgen bitgen_t;

void
random_polyagamma_alternate(bitgen_t* bitgen_state, double h, double z,
                            size_t n, double* out);

#endif
