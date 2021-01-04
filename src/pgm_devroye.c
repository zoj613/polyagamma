#include "pgm_devroye.h"
#include "pgm_common.h"

/* 
 * Compute a_n(x|t,z), the nth term of the alternating sum S_n(x|t,z)
 */
static NPY_INLINE double
piecewise_coef(size_t n, double x, double t, double z, double coshz)
{
    static const double l2_pi = -0.4515827052894548;  // log(2 / pi) 
    double n_plus_half = n + 0.5;
    double n_plus_half2 = n_plus_half * n_plus_half;
    double n_plus_halfpi = NPY_PI * n_plus_half;
    double z2 = z * z;

    if (x > t) {
        return exp(-0.5 * x * (z2 + n_plus_halfpi * n_plus_halfpi)) *
            coshz * n_plus_halfpi;
    }
    else if (x > 0) {
        return coshz * n_plus_halfpi * exp(1.5 * (l2_pi - log(x)) -
            0.5 * z2 * x - 2 * n_plus_half2 / x);
    }
    return 0;
}

/*
 * Generate a random sample from J*(1, 0) using algorithm described in
 * Devroye(2009), page 7.
 */
static NPY_INLINE double
random_jacobi_0(bitgen_t* bitgen_state)
{
    static const double t = NPY_2_PI;  // 0.64
    static const double p = 0.422599094; 
    static const double q = 0.57810262346829443;
    static const double ratio = p / (p + q);
    double x, e1, e2, s, u;
    size_t n;

    for (;;) {
        if (next_double(bitgen_state) < ratio) {
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while ((e1 * e1) > (NPY_PI * e2));  // 2 / t = pi
            x = (1 + t * e1);
            x = t / (x * x);
        }
        else {
            x = t + 8 * random_standard_exponential(bitgen_state) / PGM_PI2;
        }
        s = piecewise_coef(0, x, t, 0, 1);
        u = next_double(bitgen_state) * s;
        for (n = 1;; ++n) {
            if (n & 1) {
                s -= piecewise_coef(n, x, t, 0, 1);
                if (u < s)
                    return x;
            }
            else {
                s += piecewise_coef(n, x, t, 0, 1);
                if (u > s)
                    break;
            }
        }
    }
}

/*
 * Generate a random sample J*(1, z) using method inspired by Devroye(2009),
 * where z > 0.
 */
static NPY_INLINE double
random_jacobi(bitgen_t* bitgen_state, double z)
{
    if (z == 0)
        return random_jacobi_0(bitgen_state);

    static const double t = NPY_2_PI;  // 0.64
    double x, s, u;
    size_t n;

    double mu = 1 / z;
    double coshz = cosh(z);
    double k = PGM_PI2_8 + 0.5 * z * z;
    double q = coshz * (NPY_PI_2 / k) * exp(-k * t);
    double p = (1 + exp(-2 * z)) * inverse_gaussian_cdf(t, mu, 1);
    double ratio = p / (p + q);

    for (;;) {
        if (next_double(bitgen_state) < ratio) {
            do {
                x = random_wald(bitgen_state, mu, 1);
            } while(x > t);
        }
        else {
            x = t + random_standard_exponential(bitgen_state) / k;
        }
        /* Here we use S_n(x|t) instead of S_n(x|z,t) as explained in page 13 of
         * Polson et al.(2013) and page 14 of Windle et al. (2014). This
         * convenience avoids issues with S_n blowing up when z is very large.*/
        s = piecewise_coef(0, x, t, 0, 1);
        u = next_double(bitgen_state) * s;
        for (n = 1;; ++n) {
            if (n & 1) {
                s -= piecewise_coef(n, x, t, 0, 1);
                if (u < s)
                    return x;
            }
            else {
                s += piecewise_coef(n, x, t, 0, 1);
                if (u > s)
                    break;
            }
        }
    }
}

/*
 * sample from J*(n, z), where n is a positive integer greater than 1
 */
static NPY_INLINE double
random_jacobi_n(bitgen_t *bitgen_state, size_t n, double z)
{
    double out = 0;
    for (size_t i = n; i--; )
        out += random_jacobi(bitgen_state, z);
    return out;
}


#ifndef PGM_GAMMA_LIMIT
#define PGM_GAMMA_LIMIT 200
#endif
/*
 * Sample from J*(b, z) using a convolution of Gamma(b, 1) variates.
 */
static NPY_INLINE double
gamma_convolution_approx(bitgen_t* bitgen_state, double b, double z)
{
    const double z2 = z * z;
    double n_plus_half, out = 0;

    for (size_t n = PGM_GAMMA_LIMIT; n--; ) {
        n_plus_half = n + 0.5;
        out += random_standard_gamma(bitgen_state, b) /
            (PGM_PI2 * n_plus_half * n_plus_half + z2);
    }
    return 2 * out;
}

/*
 * Sample from PG(h, z) using the Gamma convolution approximation method
 */
double
random_polyagamma_gamma_conv(bitgen_t* bitgen_state, double h, double z)
{
    z = 0.5 * (z < 0 ? fabs(z) : z);
    return 0.25 * gamma_convolution_approx(bitgen_state, h, z);
}

/*
 * Sample from Polya-Gamma PG(h, z) using the Devroye method
 */
double
random_polyagamma_devroye(bitgen_t *bitgen_state, double h, double z)
{
    z = 0.5 * (z < 0 ? fabs(z) : z);

    if (h < 1)
        return 0.25 * gamma_convolution_approx(bitgen_state, h, z);

    double out, fract_part, int_part;
    // split h into its integer and fractional parts
    fract_part = modf(h, &int_part);
    out = random_jacobi_n(bitgen_state, (size_t)int_part, z);
    if (fract_part > 0)
        out += gamma_convolution_approx(bitgen_state, fract_part, z);
    return 0.25 * out;
}
