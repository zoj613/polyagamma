/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_igammaq.h"
#include "pgm_saddle.h"


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
    // y intercept of tangent line to xl.
    double intercept_l;
    // y intercept of tangent line to xr.
    double intercept_r;
    // the constant sqrt(h / (2 * pi))
    double sqrt_h_2pi;
    // config->sqrt_h_2pi * config->sqrt_alpha_l
    double coef_l;
    // config->sqrt_h_2pi * config->sqrt_alpha_r
    double coef_r;
    // h / 2
    double half_h;
    // 0.5 * h / xc
    double hh_xc;
};


/*
 * A tuned implementation of the hyperbolic tangent function based on a
 * continued fraction approximation.
 *
 * References
 * ---------
 *  - https://math.stackexchange.com/questions/107292/rapid-approximation-of-tanhx
 *  - https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
 */
static NPY_INLINE double
fast_tanh(double x) {
    double a, b, x2;

    if (x > 4.97)
        return 1;
    x2 = x * x;
    a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    b = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    return a / b;
}

/*
 * A struct to store a function's value and derivative at a point.
 */
struct func_return {
    double f;
    double fprime;
};

/*
 * Compute K'(u), the derivative of the Cumulant Generating Function (CGF) of x.
 *
 * For values in the neighborhood of zero, we use a faster Taylor series
 * expansion of the trigonometric and hyperbolic functions.
 */
static NPY_INLINE void
cgf_prime(double u, struct func_return* ret)
{
    double ss;
    double s = 2 * u;

    if (s == 0) {
        ret->f = 1;
    }
    else if (s < 0) {
        ss = sqrt(-s);
        ret->f = fast_tanh(ss) / ss;
    }
    else {
        ss = sqrt(s);
        ret->f = tan(ss) / ss;
    }
    ret->fprime = ret->f * ret->f + (1 - ret->f) / s;
}

/*
 * Select the starting guess for the solution `u` of Newton's method given a
 * value of x.
 *
 * - When x = 1, then u = 0.
 * - When x < 1, then u < 0.
 * - When x > 1, then u > 0.
 *
 * Page 16 of Windle et al. (2014) shows that the upper bound of `u` is pi^2/8.
 */
static NPY_INLINE double
select_starting_guess(double x)
{
    if (x >= 7.5)
        return 1.1;
    else if (x >= 5.5)
        return 1.05;
    else if (x >= 5)
        return 1.02;
    else if (x >= 4.5)
        return 1;
    else if (x >= 4)
        return 0.98;
    else if (x >= 3.5)
        return 0.95;
    else if (x >= 3)
        return 0.92;
    else if (x >= 2.5)
        return 0.81;
    else if (x >= 2)
        return 0.72;
    else if (x >= 1.5)
        return 0.58;
    else if (x >= 1)
        return 0.345;
    else if (x > 0.5)
        return -0.147;
    else if (x > 0.25)
        return -1.78;
    else return -2;
}


#ifndef PGM_MAX_ITER
#define PGM_MAX_ITER 25
#endif
/*
 * Solve for the root of f(u) = K'(t) - x using Newton's method.
 */
static NPY_INLINE double
newton_raphson(double arg, double x0, struct func_return* value)
{
    static const double tolerance = 1e-10;
    double x = 0;

    for (size_t i = 0; i < PGM_MAX_ITER; i++) {
        cgf_prime(x0, value);
        if (fabs(value->fprime) < tolerance) {
            break;
        }
        x = x0 - (value->f - arg) / value->fprime;
        if (fabs(x - x0) <= tolerance)
            return x;
        x0 = x;
    }
    return x;
}

/*
 * K(t), the cumulant generating function of X
 *
 * For values of u and z in the neighborhood of zero, we use a faster Taylor series
 * expansion of log(cosh(x)) and log(cos(x)).
 */
static NPY_INLINE double
cgf(double u, double z)
{
    double s, out;

    if (z == 0) {
        out = 0;
    }
    else {
        out = log(cosh(z));
    }

    if (u == 0) {
        return out;
    }
    else if (u > 0) {
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
 * Configure some constants and variables to be used during sampling.
 */
static NPY_INLINE void
initialize_config(struct config* cfg, double h, double z)
{
    const static double twopi = 6.283185307179586;
    struct func_return f;
    double xl, xc, xr, ul, uc, ur, tr, alpha_l, alpha_r, one_xl, one_xc, half_z2;
    bool is_zero = z == 0 ? true : false;

    xl = is_zero ? 1 : tanh(z) / z;
    xc = 2.75 * xl;
    xr = 3 * xl;

    one_xl = 1 / xl;
    one_xc = 1 / xc;
    half_z2 = is_zero ? 0 : 0.5 * z * z;
    ul = is_zero ? 0 : -half_z2;
    uc = newton_raphson(xc, select_starting_guess(xc), &f);
    ur = newton_raphson(xr, select_starting_guess(xr), &f);
    tr = (ur + half_z2);

    // t = 0 at x = m, since K'(0) = m when t(x) = 0
    cfg->Lprime_l = -0.5 * one_xl * one_xl;
    cfg->Lprime_r = -tr - 1 / xr;

    cfg->xl = xl;
    cfg->xc = xc;
    cfg->xr = xr;
    cfg->one_xc = one_xc;
    cfg->logxc = log(xc);

    cfg->intercept_l = cgf(ul, z) - 0.5 * one_xc + one_xl;
    cfg->intercept_r = cgf(ur, z) + 1 - log(xr) + cfg->logxc;

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
    if (side == LEFT)
        return cfg->Lprime_l * x + cfg->intercept_l;
    return cfg->Lprime_r * x + cfg->intercept_r;
}

/*
 * Compute the saddle point estimate at x.
 */
static NPY_INLINE double
saddle_point(double x, double h, double z, double coef)
{
    struct func_return f;
    double u = newton_raphson(x, select_starting_guess(x), &f);
    double t = u + 0.5 * z * z; 

    return coef * exp(h * (cgf(u, z) - t * x)) / sqrt(f.fprime);
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
    return cfg->coef_l * exp(cfg->hh_xc - 1.5 * log(x) - cfg->half_h / x +
                             h * tangent_at_x(x, cfg, LEFT));
}

/*
 * Sample from PG(h, z) using the Saddle approximation method.
 */
double
random_polyagamma_saddle(bitgen_t* bitgen_state, double h, double z)
{
    struct config cfg;
    double p, q, ratio, kappa_l, kappa_r, bl, br, sqrt_rho_l, one_srho_l, hrho_r, x, v;

    initialize_config(&cfg, h, z);
    cfg.half_h = 0.5 * h;
    cfg.hh_xc = cfg.half_h * cfg.one_xc;

    bl = tangent_at_x(0, &cfg, LEFT);
    sqrt_rho_l = sqrt(-2 * cfg.Lprime_l);
    one_srho_l = 1 / sqrt_rho_l;
    kappa_l = cfg.sqrt_alpha_l * exp(h * (0.5 * cfg.one_xc + bl - sqrt_rho_l));
    p = kappa_l * inverse_gaussian_cdf(cfg.xc, one_srho_l, h);

    br = tangent_at_x(0, &cfg, RIGHT);
    hrho_r = -(h * cfg.Lprime_r);
    kappa_r = cfg.coef_r * exp(h * (br - log(hrho_r)) + pgm_lgamma(h));
    q = kappa_r * pgm_gammaq(h, hrho_r * cfg.xc);

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
