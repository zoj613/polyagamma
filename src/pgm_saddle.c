/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"
#include "pgm_saddle.h"


typedef struct {
    // y intercept of tangent line to xr.
    double right_tangent_intercept;
    // y intercept of tangent line to xl.
    double left_tangent_intercept;
    // derivative of the line to xr
    double right_tangent_slope;
    // derivative of the line to xl
    double left_tangent_slope;
    // config->sqrt_h2pi * config->sqrt_alpha_r
    float right_kernel_coef;
    // config->sqrt_h2pi * config->sqrt_alpha
    float left_kernel_coef;
    // sqrt(1 / alpha_l) constant
    float sqrt_alpha;
    // log(cosh(z))
    float log_cosh_z;
    // the constant sqrt(h / (2 * pi))
    float sqrt_h2pi;
    // 0.5 * z * z
    double half_z2;
    // log of center point
    double logxc;
    double xc;
    double h;
    double z;
    double x;
} parameter_t;

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
        return 1. / x;
    }
    static const double p0 = -0.16134119023996228053e+04;
    static const double p1 = -0.99225929672236083313e+02;
    static const double p2 = -0.96437492777225469787e+00;
    static const double q0 = 0.48402357071988688686e+04;
    static const double q1 = 0.22337720718962312926e+04;
    static const double q2 = 0.11274474380534949335e+03;
    double x2 = x * x;
    return 1. + x2 * ((p2 * x2 + p1) * x2 + p0) / (((x2 + q2) * x2 + q1) * x2 + q0);
}


static inline float
__attribute__((always_inline))
tan_x(float x)
{
    return tanf(x) / x;
}

/*
 * A struct to store a function's value and derivative at a point.
 */
struct func_return_value {double f, fprime;};

/*
 * compute K(t), the cumulant generating function of X
 */
#define cumulant(u, v)                                           \
    ((u) < 0 ? (v)->log_cosh_z - logf(coshf(sqrtf(-2. * (u)))) : \
     (u) > 0 ? (v)->log_cosh_z - logf(cosf(sqrtf(2. * (u)))) :   \
     (v)->log_cosh_z)                                            \


/*
 * Compute K'(t), the derivative of the Cumulant Generating Function (CGF) of X.
 */
static NPY_INLINE void
cumulant_prime(double u, struct func_return_value* rv)
{
    double s = 2. * u;

    rv->f = s < 0 ? tanh_x(sqrt(-s)) : s > 0 ? tan_x(sqrtf(s)) : 1.;
    rv->fprime = rv->f * rv->f + (1. - rv->f) / s;
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
#define select_starting_guess(x) \
    ((x) <= 0.25 ? -9 :          \
     (x) <= 0.5 ? -1.78 :        \
     (x) <= 1.0 ? -0.147 :       \
     (x) <= 1.5 ? 0.345 :        \
     (x) <= 2.5 ? 0.72 :         \
     (x) <= 4.0 ? 0.95 : 1.15)   \


#ifndef PGM_MAX_ITER
#define PGM_MAX_ITER 25
#endif
/*
 * Solve for the root of (f(u) = K'(t) - arg) using Newton's method.
 *
 * NOTE
 * ----
 * if the function value `fval` is equal to zero up to a specified absolute
 * tolerance or the derivative is too small, then we stop and return the
 * current value `x0` as the root's coarse estimate.
 */
static NPY_INLINE double
newton_raphson(double arg, double x0, struct func_return_value* value)
{
    static const double atol = 1e-05, rtol = 1e-05;
    double x = x0;
    unsigned int n = 0;

    do {
        x0 = x;
        cumulant_prime(x0, value);
        double fval = value->f - arg;
        if (fabs(fval) <= atol || value->fprime <= atol) {
            return x0;
        }
        x = x0 - fval / value->fprime;
    } while (!PGM_ISCLOSE(x, x0, atol, rtol) && ++n < PGM_MAX_ITER);

    return x;
}

/*
 * Configure some constants to be used during sampling.
 *
 * NOTE
 * ----
 * Note that unlike the recommendations of Windle et al (2014) to use
 * xc = 1.1 * xl and xr = 1.2 * xl, we found that using xc = 2.75 * xl and
 * xr = 3 * xl provides the best envelope for the target density function and
 * thus gives the best performance in terms of runtime due to the algorithm
 * rejecting much fewer proposals; while the former produces an envelope that
 * exceeds the target by too much in height and has a narrower variance, thus
 * leading to many rejected proposal samples. Tests show that the latter
 * selection results in the saddle approximation being over twice as fast as
 * when using the former. Also note that log(xr) and log(xc) need not be
 * computed directly. Since both xr and xc depend on xl, then their logs can
 * be written as log(xr) = log(3) + log(xl) and log(xc) = log(2.75) + log(xl).
 * Thus the log() function can be called once on xl and then the constants
 * be precalculated at compile time, making the calculation of the logs a
 * little more efficient.
 */
static NPY_INLINE void
set_sampling_parameters(parameter_t* pr, double h, double z)
{
    static const float log275 = 1.0116009116784799f;
    static const float log3 = 1.0986122886681098f;
    float logxl;
    double xl;

    if (z > 0) {
        xl = tanh_x(z);
        logxl = logf(xl);
        pr->half_z2 = 0.5 * (z * z);
        pr->log_cosh_z = logf(coshf(z));
    }
    else {
        xl = 1.;
        logxl = 0;
        pr->half_z2 = 0.;
        pr->log_cosh_z = 0.f;
    }

    pr->xc = 2.75 * xl;
    double xr = 3. * xl;
    pr->h = h;
    pr->z = z;

    double xc_inv = 1. / pr->xc;
    double xl_inv = 1. / xl;
    double ul = -pr->half_z2;

    struct func_return_value rv;
    double ur = newton_raphson(xr, select_starting_guess(xr), &rv);
    newton_raphson(pr->xc, select_starting_guess(pr->xc), &rv);
    double tr = ur + pr->half_z2;

    // t = 0 at x = m, since K'(0) = m when t(x) = 0
    pr->left_tangent_slope = -0.5 * (xl_inv * xl_inv);
    pr->left_tangent_intercept = cumulant(ul, pr) - 0.5 * xc_inv + xl_inv;
    pr->logxc = log275 + logxl;
    pr->right_tangent_slope = -tr - 1. / xr;
    pr->right_tangent_intercept = cumulant(ur, pr) + 1.0f - log3 - logxl + pr->logxc;

    double alpha_r = rv.fprime * (xc_inv * xc_inv);  // K''(t(xc)) / xc^2
    double alpha_l = xc_inv * alpha_r;  // K''(t(xc)) / xc^3

    pr->sqrt_alpha = 1.0f / sqrtf(alpha_l);

    pr->sqrt_h2pi = sqrtf((float)h / 6.283185307179586f);
    pr->left_kernel_coef = pr->sqrt_h2pi * pr->sqrt_alpha;
    pr->right_kernel_coef = pr->sqrt_h2pi / sqrtf(alpha_r);
}

/*
 * Compute the saddle point estimate at x.
 */
static NPY_INLINE float
saddle_point(parameter_t const* pr)
{
    struct func_return_value rv;
    double u = newton_raphson(pr->x, select_starting_guess(pr->x), &rv);
    double t = u + pr->half_z2; 

    return expf(pr->h * (cumulant(u, pr) - t * pr->x)) *
           pr->sqrt_h2pi / sqrt(rv.fprime);
}

/*
 * k(x|h,z): The bounding kernel of the saddle point approximation. See
 * Proposition 17 of Windle et al (2014).
 */
static NPY_INLINE float
bounding_kernel(parameter_t const* pr)
{
    double point;

    if (pr->x > pr->xc) {
        point = pr->right_tangent_slope * pr->x + pr->right_tangent_intercept;
        return expf(pr->h * (pr->logxc + point) + (pr->h - 1.) * logf(pr->x)) *
               pr->right_kernel_coef;
    }
    point = pr->left_tangent_slope * pr->x + pr->left_tangent_intercept;
    return expf(0.5 * pr->h * (1. / pr->xc - 1. / pr->x) +
                pr->h * point - 1.5 * logf(pr->x)) * pr->left_kernel_coef;
}

/*
 * Compute the logarithm of the standard normal distribution function (cdf).
 *
 * NOTE
 * ----
 *  The switch to using erf() when x is very close to zero is done implicitly
 *  inside `pgm_erfc`.
 */
#define log_norm_cdf(x) (log1pf(-0.5f * pgm_erfc((x) / 1.4142135623730951f)))

/*
 * Calculate the logarithm of the cumulative distribution function of an
 * Inverse-Gaussian.
 *
 * We use the computation method presented in [1] to avoid numerical issues
 * when the inputs have very large/small values.
 *
 * References
 * ----------
 *  [1] Giner, Goknur and G. Smyth. “statmod: Probability Calculations for the
 *      Inverse Gaussian Distribution.” R J. 8 (2016): 339.
 */
static NPY_INLINE double
invgauss_logcdf(double x, double mu, double lambda)
{
    double qm = x / mu;
    double tm =  mu / lambda;
    double r = sqrt(x / lambda);
    float a = log_norm_cdf((qm - 1.) / r);
    float b = 2. / tm + log_norm_cdf(-(qm + 1.) / r);

    return a + log1pf(expf(b - a));
}

/*
 * Sample from PG(h, z) using the Saddle approximation method.
 */
double
random_polyagamma_saddle(bitgen_t* bitgen_state, double h, double z)
{
    double sqrt_rho, sqrt_rho_inv, hrho;
    float proposal_probability, p, q;
    parameter_t pr;

    set_sampling_parameters(&pr, h, z);

    sqrt_rho = sqrt(-2. * pr.left_tangent_slope);
    sqrt_rho_inv = 1. / sqrt_rho;
    p = expf(h * (0.5 / pr.xc + pr.left_tangent_intercept - sqrt_rho) +
             invgauss_logcdf(pr.xc, sqrt_rho_inv, h)) * pr.sqrt_alpha;

    hrho = -h * pr.right_tangent_slope;
    q = pgm_gammaq(h, hrho * pr.xc, false) * pr.right_kernel_coef *
        expf(h * (pr.right_tangent_intercept - logf(hrho)));

    proposal_probability = p / (p + q);

    double mu2 = sqrt_rho_inv * sqrt_rho_inv;
    do {
        if (next_float(bitgen_state) < proposal_probability) {
            do {
                double y = random_standard_normal(bitgen_state);
                double w = sqrt_rho_inv + 0.5 * mu2 * y * y / h;
                pr.x = w - sqrt(w * w - mu2);
                if (next_double(bitgen_state) * (1. + pr.x * sqrt_rho) > 1.) {
                    pr.x = mu2 / pr.x;
                }
            } while (pr.x >= pr.xc);
        }
        else {
            pr.x = random_left_bounded_gamma(bitgen_state, h, hrho, pr.xc);
        }
    } while (next_float(bitgen_state) * bounding_kernel(&pr) > saddle_point(&pr));

    return 0.25 * h * pr.x;
}
