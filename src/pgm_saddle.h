#ifndef PGM_SADDLE_H
#define PGM_SADDLE_H
#include <stddef.h>

typedef struct bitgen bitgen_t;

void
random_polyagamma_saddle(bitgen_t* bitgen_state, double h, double z,
                         size_t n, double* out);

#endif
