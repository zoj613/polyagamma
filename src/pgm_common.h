#ifndef PGM_COMMON_H
#define PGM_COMMON_H
#include <numpy/random/distributions.h>

#define PGM_PI2 9.869604401089358  // pi^2
#define PGM_PI2_8 1.233700550136169  // pi^2 / 8


double inverse_gaussian_cdf(double x, double mu, double lambda);
double random_left_bounded_gamma(bitgen_t* bitgen_state, double a,
                                 double b, double t);

#endif
