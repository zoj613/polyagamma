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
 * Compute f(x) = tanh(x) / x in the range [0, infinity).
 *
 * This implementation is based off the analysis presented by Beebe [1].
 * For x <= 5, we use a rational polynomial approximation of Cody and Waite [2].
 * For x > 5, we use g(x) = 1 / x to approximate the function.
 *
 * Tests show that the absolute maximum relative error compared to output
 * produced by the standard library tanh(x) / x is 9.080398e-05.
 *
 * References
 * ---------
 * [1] Beebe, Nelson H. F.. (1993). Accurate hyperbolic tangent computation.
 *     Technical report, Center for Scientific Computing, Department of
 *     Mathematics, University ofUtah, Salt Lake City, UT 84112, USA, April 20
 *    1993.
 * [2] William J. Cody, Jr. and William Waite. Software Manual for the Elementary
 *     Functions. Prentice-Hall, Upper Saddle River, NJ 07458, USA, 1980.
 *     ISBN 0-13-822064-6. x + 269 pp. LCCN QA331 .C635 1980.
 */
static NPY_INLINE double
tanh_x(double x)
{
    static const double p0 = -0.16134119023996228053e+04;
    static const double p1 = -0.99225929672236083313e+02;
    static const double p2 = -0.96437492777225469787e+00;
    static const double q0 = 0.48402357071988688686e+04;
    static const double q1 = 0.22337720718962312926e+04;
    static const double q2 = 0.11274474380534949335e+03;
    double x2, r;

    if (x > 4.95) {
        return 1 / x;
    }
    x2 = x * x;
    r = x2 * ((p2 * x2 + p1) * x2 + p0) / (((x2 + q2) * x2 + q1) * x2 + q0);
    return 1 + r;
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
        ret->f = tanh_x(ss);
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
 * Test if two numbers equal within the given absolute and relative tolerences
 *
 * `rtol` is the relative tolerance – it is the maximum allowed difference
 * between a and b, relative to the larger absolute value of a or b.
 *
 * `atol` is the minimum absolute tolerance – useful for comparisons near zero.
 */
static NPY_INLINE bool
is_close(double a, double b, double atol, double rtol)
{
    return fabs(a - b) <= MAX(rtol * MAX(fabs(a), fabs(b)), atol);
}

/*
 * Solve for the root of f(u) = K'(t) - x using Newton's method.
 */
static NPY_INLINE double
newton_raphson(double arg, double x0, struct func_return* value)
{
    static const double atol = 1e-20, rtol = 1e-05;
    double fval, x = 0;

    for (size_t i = 0; i < PGM_MAX_ITER; i++) {
        cgf_prime(x0, value);
        fval = value->f - arg;
        if (fabs(fval) <= rtol) {
            return x0;
        }
        if (value->fprime <= atol) {
            break;
        }
        x = x0 - fval / value->fprime;
        if (is_close(x, x0, atol, rtol)) {
            return x;
        }
        x0 = x;
    }
    return x;
}

/*
 * K(t), the cumulant generating function of X
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
    double xl, xc, xr, ul, ur, tr, alpha_l, alpha_r, one_xl, one_xc, half_z2;
    bool is_zero = z == 0 ? true : false;

    xl = is_zero ? 1 : tanh(z) / z;
    xc = 2.75 * xl;
    xr = 3 * xl;

    one_xl = 1 / xl;
    one_xc = 1 / xc;
    half_z2 = is_zero ? 0 : 0.5 * z * z;
    ul = is_zero ? 0 : -half_z2;
    ur = newton_raphson(xr, select_starting_guess(xr), &f);
    newton_raphson(xc, select_starting_guess(xc), &f);
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

    alpha_r = f.fprime * one_xc * one_xc;  // K''(t(xc)) / xc^2
    alpha_l = one_xc * alpha_r;  // K''(t(xc)) / xc^3

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
NPY_INLINE double
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
            x = random_right_bounded_inverse_gaussian(bitgen_state, one_srho_l, h, cfg.xc);
        }
        else {
            x = random_left_bounded_gamma(bitgen_state, h, hrho_r, cfg.xc);
        }
        v = next_double(bitgen_state) * bounding_kernel(x, h, &cfg);
    } while(v > saddle_point(x, h, z, cfg.sqrt_h_2pi));

    return 0.25 * h * x;
}
