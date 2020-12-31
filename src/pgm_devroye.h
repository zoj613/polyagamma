#ifndef PGM_DEVROYE_H
#define PGM_DEVROYE_H

#include "pgm_common.h"


// piecewise coefficient a_n(x|t)
static NPY_INLINE double
piecewise_coef(uint64_t n, double x, double t)
{
    double n_plus_half = n + 0.5;
    double n_plus_half_2 = n_plus_half * n_plus_half;
    double n_plus_half_pi = NPY_PI * n_plus_half;

    if (x < t) {
        double one_x = 1 / x;
        return n_plus_half_pi * pow(NPY_2_PI * one_x, 1.5) *
                    exp(-2 * n_plus_half_2 * one_x);
    }
    else {
        return n_plus_half_pi * exp(-0.5 * n_plus_half_pi * n_plus_half_pi * x);
    }
}

/* sample from an IG(mu, 1) bounded on the interval (0, bound) */
static NPY_INLINE double
random_bounded_inverse_gaussian(bitgen_t* bitgen_state, double mu, double bound)
{
    double e1, e2, x, y, z;

    if (mu > bound) {
        z = 1 / bound;
        for (;;) {
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while ((e1 * e1) <= 2 * e2 / bound);
            x = (1 + bound * e1);
            x = bound / (x * x); 
            if (next_double(bitgen_state) < exp(-0.5 * z * z * x))
                return x;
        }
    }
    else {
        double muy, mu2 = mu * mu;
        for (;;) {
            y = random_standard_normal(bitgen_state);
            y = y * y;
            muy = mu * y;
            x = mu + 0.5 * mu2 * y - 0.5 * mu * sqrt(4 * muy + muy * muy);
            if (next_double(bitgen_state) > (mu / (mu + x)))
                x = mu2 / x; 
            if (x < bound)
                return x;
        }
    }
}

/*
 * generate random variables from the polya-gamma distribution P(1, z) using
 * method inspired by Devroye(2009)
 */
static NPY_INLINE double
random_polyagamma_1z(bitgen_t* bitgen_state, double z)
{
    static const double t = NPY_2_PI;  // 0.64
    double k, p, q, x, mu, u, s, y;
    size_t n;

    z = fabs(z) * 0.5;
    mu = 1 / z;
    k = PGM_PI2_8 + 0.5 * z * z;
    p = (NPY_PI_2 / k) * exp(-k * t);
    q = 2 * exp(-z) * inverse_gaussian_cdf(t, mu, 1);

    for (;;) {
        u = next_double(bitgen_state);
        if (u < (p / (p + q))) {
            x = t + random_standard_exponential(bitgen_state) / k;
        }
        else {
            x = random_bounded_inverse_gaussian(bitgen_state, mu, t);
        }
        s = piecewise_coef(0, x, t);
        y = next_double(bitgen_state) * s;
        for (n = 0; ++n;) {
            if (n & 1) {
                s -= piecewise_coef(n, x, t);
                if (y < s)
                    return 0.25 * x;
            }
            else {
                s += piecewise_coef(n, x, t);
                if (y > s)
                    break;
            }
        }
    }
}


static NPY_INLINE double
gamma_convolution_approx(bitgen_t* bitgen_state, double b, double c, uint64_t n)
{
    const double c2_2 = 0.5 * c * c;
    double i_minus_half, out = 0;
    for (size_t i = n; i--; ) {
        i_minus_half = i - 0.5;
        out += random_standard_gamma(bitgen_state, b) /
            (2 * i_minus_half * i_minus_half * PGM_PI2 + c2_2);
    }
    return out;
}


static NPY_INLINE double
random_polyagamma_int(bitgen_t *bitgen_state, uint64_t h, double z)
{
    double out = 0;
    for (size_t i = h; i--; )
        out += random_polyagamma_1z(bitgen_state, z);
    return out;
}


static NPY_INLINE double
random_polyagamma_devroye(bitgen_t *bitgen_state, double h, double z)
{
    if (h < 1)
        return gamma_convolution_approx(bitgen_state, h, z, 200);

    double out, fract_part, int_part;
    // split h into its integer and fractional parts
    fract_part = modf(h, &int_part);
    out = random_polyagamma_int(bitgen_state, (uint64_t)int_part, z);
    if (fract_part > 0)
        out += gamma_convolution_approx(bitgen_state, fract_part, z, 200);
    return out;
}

#endif
