/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_igammaq.h"
#include "pgm_saddle.h"
#include "pgm_saddle_tangent_points.h"


typedef enum {LEFT, RIGHT} SIDE_t;

struct config {
    // center point
    double xc;
    // inverse of center point
    double one_xc;
    // log of center point
    double logxc;
    // left tangent point
    double xl;
    // right tangent point
    double xr;
    // derivative of the line to xl
    double Lprime_l;
    // derivative of the line to xr
    double Lprime_r;
    // sqrt(1 / alpha_l) constant
    double sqrt_alpha_l;
    // sqrt(1 / alpha_r) constant
    double sqrt_alpha_r;
    // y intercept of tangent line to xl. ta_l = phi(t) - 0.5 * (1/xc - 1/xl)
    double intercept_l;
    // y intercept of tangent line to xr. eta_l = phi(t) - log(xr) + log(xc)
    double intercept_r;
    // the constant sqrt(h / (2 * pi))
    double sqrt_h_2pi;
    // config->sqrt_h_2pi * config->sqrt_alpha_l
    double coef_l;
    // config->sqrt_h_2pi * config->sqrt_alpha_r
    double coef_r;
};

/*
 * Compute K'(u), the derivatie of the Cumulant Generating Function of x.
 */
static NPY_INLINE double
cgf_prime(double u)
{
    double s;
    double tolerance = 5e-5;

    if (u == 0 || fabs(u) < tolerance)
        return 1;
    else if (u > 0) {
        s = 2 * u;
        s = sqrt(s);
        return tan(s) / s;
    }
    else {
        s = -2 * u;
        s = sqrt(s);
        return tanh(s) / s;
    }
}

/*
 * The function f(u|x) = K'(t) - x. We find solve for u in order to obtain t.
 */
static NPY_INLINE double
objective(double u, double x)
{
    return cgf_prime(u) - x;
}

/*
 * Select the solver bracket interval [lower, upper] for the solution `u`,
 * given a value of x.
 *
 * - When x = 1, then u = 0.
 * - When x < 1, then u < 0.
 * - When x > 1, then u > 0.
 *
 * Page 16 of Windle et al. (2014) shows that the upper bound of `u` is pi^2/8,
 * therefore we choose the bracketing interval accoding to the above, as
 * outlined in Page 19 of Windle et al. (2014).
 */
static NPY_INLINE void
select_bracket(double x, double* lower, double* upper)
{
    double tolerance = 5e-05;

    if (x == 1 || fabs(x) < tolerance) {
        *lower = -0.1;
        *upper = 0.1;
    }
    else if (x < 1) {
        *lower = -500;
        *upper = -0.001;
    }
    else {
        *lower = 0.01;
        *upper = 1.2;
    } 
}


#ifndef PGM_MAX_ITER
#define PGM_MAX_ITER 100
#endif
/* 
 * Apply the method of False Position to solve for the root f(u) = K'(u) - x.
 *
 * Adapted from: https://en.wikipedia.org/wiki/Regula_falsi#Example_code
 */
static NPY_INLINE double
regula_falsi(double arg)
{
    double r, s, t, fr;
    double e = 5e-5, m = PGM_MAX_ITER;
    int side = 0;

    /* starting values at endpoints of interval */
    select_bracket(arg, &s, &t);
    double fs = objective(s, arg);
    double ft = objective(t, arg);

    for (size_t n = 0; n < m; n++) {
        r = (fs * t - ft * s) / (fs - ft);
        if (fabs(t - s) < e * fabs(t + s)) {
            break;
        }
        
        fr = objective(r, arg);

        if (fr * ft > 0) {
            /* fr and ft have same sign, copy r to t */
            t = r; ft = fr;
            if (side==-1)
                fs /= 2;
            side = -1;
        }
        else if (fs * fr > 0) {
            /* fr and fs have same sign, copy r to s */
            s = r; fs = fr;
            if (side==+1)
                ft /= 2;
            side = +1;
        }
        else {
             break; /* fr * f_ very small (looks like zero) */
        }
    }
    return r;
}

/*
 * K(t), the cumulant generating function of X
 */
static NPY_INLINE double
cgf(double u, double z)
{
    double s, out;
    out = z == 0 ? 0 : log(cosh(z));
    if (u >= 0) {
        s = 2 * u;
        s = sqrt(s);
        out -= log(cos(s));
    }
    else {
        s = -2 * u;
        s = sqrt(s);
        out -= log(cosh(s));
    }
    return out; 
}

/*
 * Get pre-computed optimal values of xl, xc and xr from a lookup table,
 * for a given value of the tilting parameter z.
 */
static NPY_INLINE void
lookup_tangent_points(double z, double* xl, double* xc, double* xr)
{
    size_t index, offset = 0, len = pgm_saddle_tabsize;

    if (z == 0) {
        *xl = 1;
        *xc = 1.5;
        *xr = 1.65;
    }
    else if (z >= pgm_saddle_max_z) {
        *xl = tanh(z) / z;
        *xc = pgm_xc[len - 1];
        *xr = pgm_xr[len - 1];
    }
    else {
        // start binary search
        while (len > 0) {
            index = offset + len / 2;
            if (pgm_z[index] < z) {
                len = len - (index + 1 - offset);
                offset = index + 1;
                continue;
            }
            else if (offset < index && pgm_z[index - 1] >= z) {
                len = index - offset;
                continue;
            }
            break;
        }
        *xl = pgm_xl[index - 1];
        *xc = pgm_xc[index - 1];
        *xr = pgm_xr[index - 1];
    }
}

/*
 * Configure some constants and variables to be used during sampling.
 */
static NPY_INLINE void
initialize_config(struct config* cfg, double h, double z)
{
    const static double twopi = 6.283185307179586;
    double xl, xc, xr, ul, uc, ur, tr, alpha_l, alpha_r, one_xl, one_xc, half_z2;
    bool is_zero = z == 0 ? true : false;

    lookup_tangent_points(z, &xl, &xc, &xr);

    one_xl = 1 / xl;
    one_xc = 1 / xc;
    half_z2 = is_zero ? 0 : 0.5 * z * z;
    ul = is_zero ? 0 : -half_z2;
    uc = regula_falsi(xc);
    ur = regula_falsi(xr);
    tr = (ur + half_z2);

    // t = 0 at x = m, since K'(0) = m when t(x) = 0
    cfg->Lprime_l = -0.5 * one_xl * one_xl;
    cfg->Lprime_r = -tr - 1 / xr;

    cfg->xl = xl;
    cfg->xc = xc;
    cfg->xr = xr;
    cfg->one_xc = one_xc;
    cfg->logxc = log(xc);

    cfg->intercept_l = cgf(ul, z) - 0.5 * (one_xc - one_xl) - cfg->Lprime_l * xl;
    cfg->intercept_r = cgf(ur, z) - tr * xr - log(xr) + cfg->logxc - cfg->Lprime_r * xr; 

    alpha_r = 1 + 0.5 * one_xc * one_xc * (1 - xc) / uc;
    alpha_l = one_xc * alpha_r;
    cfg->sqrt_alpha_l = 1 / sqrt(alpha_l);
    cfg->sqrt_alpha_r = 1 / sqrt(alpha_r);

    cfg->sqrt_h_2pi = sqrt(h / twopi); 
    cfg->coef_l = cfg->sqrt_h_2pi * cfg->sqrt_alpha_l;
    cfg->coef_r = cfg->sqrt_h_2pi * cfg->sqrt_alpha_r;
}

/*
 * Compute L_i(x|z) or L_r(x|z) for a given x value.
 *
 * L_i(x|z) is the line touching the curve of eta(x) at xl.
 * L_r(x|z) is the line touching the curve of eta(x) at xr.
 */
static NPY_INLINE double
tangent_at_x(double x, struct config* cfg, SIDE_t side)
{
    switch(side) {
        case LEFT: return cfg->Lprime_l * x + cfg->intercept_l;
        case RIGHT: return cfg->Lprime_r * x + cfg->intercept_r;
    }
}

/*
 * Compute the saddle point estimate at x.
 */
static NPY_INLINE double
saddle_point(double x, double h, double z, double coef)
{
    double u = regula_falsi(x);
    double t = u + 0.5 * z * z; 
    double kprime2 = cgf_prime(u);

    kprime2 = kprime2 * kprime2 + 0.5 * (1 - kprime2) / u;
    return coef * exp(h * (cgf(u, z) - t * x)) / sqrt(kprime2);
}

/*
 * k(x|h,z): The bounding kernel of the saddle point approximation.
 */
static NPY_INLINE double
bounding_kernel(double x, double h, struct config* cfg)
{
    if (x > cfg->xc) {
        return cfg->coef_r * exp(h * (cfg->logxc + tangent_at_x(x, cfg, RIGHT)) +
                (h - 1) * log(x));
    }
    double half_h = 0.5 * h;
    return cfg->coef_l * exp(half_h * cfg->one_xc - 1.5 * log(x) -
            half_h / x + h * tangent_at_x(x, cfg, LEFT));
}

/*
 * Sample from PG(h, z) using the Saddle approximation method.
 *
 * NOTE: The accuracy and speed of this implementation seems to be heavily
 * dependant on the value of the tilting parameter ``z`` and the points xl, xc
 * and xr. The wrong choice of xl and xr for a given z can lead to a poor
 * saddle point approximation. To prevent this issue, we precompute all the
 * best values for xl, xc and xr for a given value of z. This ensures quality
 * samples and fast runtime since the envelope will be as close as possible to
 * the true density of the distribution.
 */
double
random_polyagamma_saddle(bitgen_t* bitgen_state, double h, double z)
{
    struct config cfg;
    double p, q, ratio, kappa_l, kappa_r, bl, br, sqrt_rho_l, one_srho_l, hrho_r, x, v;

    z = z == 0 ? 0 : 0.5 * (z < 0 ? -z : z);
    initialize_config(&cfg, h, z);

    bl = tangent_at_x(0, &cfg, LEFT);
    sqrt_rho_l = sqrt(-2 * cfg.Lprime_l);
    one_srho_l = 1 / sqrt_rho_l;
    kappa_l = cfg.sqrt_alpha_l * exp(h * (0.5 / cfg.xc + bl - sqrt_rho_l));
    p = kappa_l * inverse_gaussian_cdf(cfg.xc, one_srho_l, h);

    br = tangent_at_x(0, &cfg, RIGHT);
    hrho_r = -(h * cfg.Lprime_r);
    kappa_r = cfg.coef_r * exp(h * (br - log(hrho_r)) + lgamma(h));
    q = kappa_r * kf_gammaq(h, hrho_r * cfg.xc);
    ratio = p / (p + q);

    do {
        if (next_double(bitgen_state) < ratio) {
            do {
                x = random_wald(bitgen_state, one_srho_l, h);
            } while (x > cfg.xc);
        }
        else {
            x = random_left_bounded_gamma(bitgen_state, h, hrho_r, cfg.xc);
        }
        v = next_double(bitgen_state) * bounding_kernel(x, h, &cfg);
    } while(v > saddle_point(x, h, z, cfg.sqrt_h_2pi));

    return 0.25 * h * x;
}
