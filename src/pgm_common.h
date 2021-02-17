#ifndef PGM_COMMON_H
#define PGM_COMMON_H
#include <numpy/random/distributions.h>

#define PGM_PI2 9.869604401089358  // pi^2
#define PGM_PI2_8 1.233700550136169  // pi^2 / 8
#define PGM_LOGPI_2 0.4515827052894548  // log(pi / 2)
#define PGM_LS2PI 0.9189385332046727  // log(sqrt(2 * pi))


double pgm_lgamma(double z);
double pgm_gammaq(double s, double x);
double inverse_gaussian_cdf(double x, double mu, double lambda);
double random_left_bounded_gamma(bitgen_t* bitgen_state, double a,
                                 double b, double t);
double random_right_bounded_inverse_gaussian(bitgen_t* bitgen_state, double mu,
                                             double lambda, double t);

#endif
