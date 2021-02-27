/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"
#include "pgm_saddle.h"


struct config {
    // shape parameter
    double h;
    // exponential tilting parameter
    double z;
    // a sample from the proposal distribution
    double x;
    // center point
    double xc;
    // log of center point
    double logxc;
    // derivative of the line to xl
    double Lprime_l;
    // derivative of the line to xr
    double Lprime_r;
    // sqrt(1 / alpha_l) constant
    double sqrt_alpha_l;
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
    // 0.5 * z * z
    double half_z2;
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
    if (x > 4.95) {
        return 1 / x;
    }
    static const double p0 = -0.16134119023996228053e+04;
    static const double p1 = -0.99225929672236083313e+02;
    static const double p2 = -0.96437492777225469787e+00;
    static const double q0 = 0.48402357071988688686e+04;
    static const double q1 = 0.22337720718962312926e+04;
    static const double q2 = 0.11274474380534949335e+03;
    double x2 = x * x;
    return 1 + x2 * ((p2 * x2 + p1) * x2 + p0) / (((x2 + q2) * x2 + q1) * x2 + q0);
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
    double s = 2 * u;

    if (s == 0) {
        ret->f = 1;
    }
    else if (s < 0) {
        ret->f = tanh_x(sqrt(-s));
    }
    else {
        double ss = sqrt(s);
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
    double x = 0;

    for (size_t i = 0; i < PGM_MAX_ITER; i++) {
        cgf_prime(x0, value);
        double fval = value->f - arg;
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
    double out = z > 0 ? log(cosh(z)) : 0;

    if (u == 0) {
        return out;
    }
    else if (u > 0) {
        out -= log(cos(sqrt(2 * u)));
    }
    else {
        out -= log(cosh(sqrt(-2 * u)));
    }
    return out; 
}

/*
 * Configure some constants and variables to be used during sampling.
 */
static NPY_INLINE void
initialize_config(struct config* cfg, double h, double z)
{
    bool is_zero = z > 0 ? false : true;
    double xl = is_zero ? 1 : tanh_x(z);
    cfg->xc = 2.75 * xl;
    double xr = 3 * xl;
    cfg->h = h;
    cfg->z = z;

    cfg->half_h = 0.5 * h;
    double one_xc = 1 / cfg->xc;
    cfg->hh_xc = cfg->half_h * one_xc;
    double one_xl = 1 / xl;
    cfg->half_z2 = is_zero ? 0 : 0.5 * (z * z);
    double ul = -cfg->half_z2;

    struct func_return f;
    double ur = newton_raphson(xr, select_starting_guess(xr), &f);
    newton_raphson(cfg->xc, select_starting_guess(cfg->xc), &f);
    double tr = ur + cfg->half_z2;

    // t = 0 at x = m, since K'(0) = m when t(x) = 0
    cfg->Lprime_l = -0.5 * (one_xl * one_xl);
    cfg->Lprime_r = -tr - 1 / xr;

    cfg->logxc = log(cfg->xc);

    cfg->intercept_l = cgf(ul, z) - 0.5 * one_xc + one_xl;
    cfg->intercept_r = cgf(ur, z) + 1 - log(xr) + cfg->logxc;

    double alpha_r = f.fprime * (one_xc * one_xc);  // K''(t(xc)) / xc^2
    double alpha_l = one_xc * alpha_r;  // K''(t(xc)) / xc^3

    cfg->sqrt_alpha_l = 1 / sqrt(alpha_l);

    cfg->sqrt_h_2pi = sqrt(h / 6.283185307179586);
    cfg->coef_l = cfg->sqrt_h_2pi * cfg->sqrt_alpha_l;
    cfg->coef_r = cfg->sqrt_h_2pi / sqrt(alpha_r);
}

/*
 * Compute the saddle point estimate at x.
 */
static NPY_INLINE double
saddle_point(struct config* cfg)//double x, double h, double z, double coef)
{
    struct func_return f;
    double u = newton_raphson(cfg->x, select_starting_guess(cfg->x), &f);
    double t = u + cfg->half_z2; 

    return cfg->sqrt_h_2pi * exp(cfg->h * (cgf(u, cfg->z) - t * cfg->x)) / sqrt(f.fprime);
}

/*
 * k(x|h,z): The bounding kernel of the saddle point approximation.
 */
static NPY_INLINE double
bounding_kernel(struct config* cfg)
{
    if (cfg->x > cfg->xc) {
        double tanline_at_x = cfg->Lprime_r * cfg->x + cfg->intercept_r;
        return cfg->coef_r * exp(cfg->h * (cfg->logxc + tanline_at_x) +
                                 (cfg->h - 1) * log(cfg->x));
    }
    double tanline_at_x = cfg->Lprime_l * cfg->x + cfg->intercept_l;
    return cfg->coef_l * exp(-cfg->half_h / cfg->x + cfg->h * tanline_at_x +
                             cfg->hh_xc - 1.5 * log(cfg->x));
}

/*
 * Sample from PG(h, z) using the Saddle approximation method.
 */
NPY_INLINE double
random_polyagamma_saddle(bitgen_t* bitgen_state, double h, double z)
{
    struct config cfg;
    double p, q, ratio, sqrt_rho_l, one_srho_l, hrho_r;

    initialize_config(&cfg, h, z);

    sqrt_rho_l = sqrt(-2 * cfg.Lprime_l);
    one_srho_l = 1 / sqrt_rho_l;
    p = cfg.sqrt_alpha_l * exp(h * (0.5 / cfg.xc + cfg.intercept_l - sqrt_rho_l)) *
        inverse_gaussian_cdf(cfg.xc, one_srho_l, h, true);

    hrho_r = -(h * cfg.Lprime_r);
    q = cfg.coef_r * exp(h * (cfg.intercept_r - log(hrho_r))) *
        pgm_gammaq(h, hrho_r * cfg.xc, false);

    ratio = p / (p + q);
    do {
        if (next_double(bitgen_state) < ratio) {
            cfg.x = random_right_bounded_inverse_gaussian(bitgen_state, one_srho_l, h, cfg.xc);
        }
        else {
            cfg.x = random_left_bounded_gamma(bitgen_state, h, hrho_r, cfg.xc);
        }
    } while (next_double(bitgen_state) * bounding_kernel(&cfg) > saddle_point(&cfg));

    return 0.25 * h * cfg.x;
}
