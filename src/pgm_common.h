#ifndef PGM_COMMON_H
#define PGM_COMMON_H

#include <numpy/random/distributions.h>

#define PGM_PI2 9.869604401089358  // pi^2
#define PGM_1_SQRT2PI 0.3989422804014327  // 1 / sqrt(2 * pi)
#define PGM_PI2_8 1.233700550136169  // pi^2 / 8


static NPY_INLINE double
standard_normal_cdf(double x)
{
    return 0.5 * (1 + erf(x * NPY_SQRT1_2));
}

/* Inverse-Gaussian cumulative density function */
static NPY_INLINE double
inverse_gaussian_cdf(double x, double mu, double lambda)
{
    double sqrt_lx = sqrt(lambda / x);
    double x_mu = x / mu;

    return standard_normal_cdf(sqrt_lx * (x_mu - 1)) +
        standard_normal_cdf(-sqrt_lx * (x_mu + 1)) * exp(2 * lambda / mu);
}

#endif
