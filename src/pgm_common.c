/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"


extern NPY_INLINE double
pgm_erfc(double x, bool scaled);

extern NPY_INLINE double
pgm_lgamma(double z);

extern NPY_INLINE double
inverse_gaussian_cdf(double x, double mu, double lambda);

extern NPY_INLINE double
random_left_bounded_gamma(bitgen_t* bitgen_state, double a, double b, double t);

extern NPY_INLINE double
random_right_bounded_inverse_gaussian(bitgen_t* bitgen_state, double mu,
                                      double lambda, double t);


/*
 * Computes f(x) = log(1 + x) - x while avoiding cancellation issues.
 *
 * Algorithm is from [1]. For the range -0.7 <=x<=1.3 a rational approximation
 * is used.
 *
 * References
 * ----------
 * [1] Temme, N. (1994). A Set of Algorithms for the Incomplete Gamma Functions.
 *     Probability in the Engineering and Informational Sciences, 8(2),
 *     291-307. doi:10.1017/S0269964800003417.
 */
static NPY_INLINE double
log1pminx(double x)
{
    static const double p0 = -4.999999994526e-01;
    static const double p1 = -5.717084236157e-01;
    static const double p2 = -1.423751838241e-01;
    static const double p3 = -8.310525299547e-04;
    static const double p4 = 3.899341537646e-05;
    static const double q1 = 1.810083408290e+00;
    static const double q2 = 9.914744762863e-01;
    static const double q3 = 1.575899184525e-01;

    if (x <= -1) {
        return -DBL_MAX;
    }
    else if (x < -0.7 || x > 1.36) {
        return log(1 + x) - x;
    }
    else if (fabs(x) < DBL_EPSILON) {
        return -0.5 * sqrt(x);
    }
    else if (x > 0) {
        return sqrt(x) * ((((p4 * x + p3) * x + p2) * x + p1) * x + p0) /
                          (((q3 * x + q2) * x + q1) * x + 1);
    }
    else {
        double z = -x / (1 + x);
        if (z > 1.36) {
            return -(log(1 + z) - z) + x * z;
        }
        else {
            return -sqrt(z) * ((((p4 * z + p3) * z + p2) * z + p1) * z + p0) /
                                (((q3 * z + q2) * z + q1) * z + 1) + x * z;
        }
    }
}

/*
 * Computes the tempered gamma function for x > 0.
 *
 * The function is defined as Gamma(x) = * sqrt(2pi)*exp(-x)*x^(x-0.5)*gammastar(x).
 * where gammastar is the tempered gamma function, as described in [1].
 *
 * References
 * ----------
 * [1] Temme, N. (1994). A Set of Algorithms for the Incomplete Gamma Functions.
 *     Probability in the Engineering and Informational Sciences, 8(2),
 *     291-307. doi:10.1017/S0269964800003417.
 */
static NPY_INLINE double
gammastar(double x)
{
    if (x > 1e+10) {
        return x > (1 / (12 * DBL_EPSILON)) ? 1 : 1 + 1 / (12 * x);
    }
    else if (x >= 12) {
        static const double p0 = 1.000000000949e+00;
        static const double p1 = 9.781658613041e-01;
        static const double p2 = 7.806359425652e-02;
        static const double q1 = 8.948328926305e-01;
        double a = 1 / x;
        return ((p2 * a + p1) * a + p0) / (q1 * a + 1);
    }
    else if (x >= 1) {
        static const double p0 = 5.115471897484e-02;
        static const double p1 = 4.990196893575e-01;
        static const double p2 = 9.404953102900e-01;
        static const double p3 = 9.999999625957e-01;
        static const double q0 = 1.544892866413e-02;
        static const double q1 = 4.241288251916e-01;
        static const double q2 = 8.571609363101e-01;
        return (((p3 * x + p2) * x + p1) * x + p0) / (((x + q2) * x + q1) * x + q0);
    }
    else if (x > DBL_MIN) {
        double a = 1 + 1 / x;
        return gammastar(x + 1) * sqrt(a) * exp(x * log(a) - 1);
    }
    else {
        // 1 / (sqrt(2 * pi) * sqrt(DBL_MIN));
        return 2.6744707353778563e+153;
    }
}

/*
 * Computes g(x) in 1/Gamma(1 + x) = 1 + x * (x - 1) * g(x) for 0<=x<=1, as
 * described in [1].
 *
 * References
 * ----------
 * [1] Temme, N. (1994). A Set of Algorithms for the Incomplete Gamma Functions.
 *     Probability in the Engineering and Informational Sciences, 8(2),
 *     291-307. doi:10.1017/S0269964800003417.
 */
static NPY_INLINE double
auxgam(double x)
{
    static const double p0 = -5.772156647338e-01;
    static const double p1 = -1.087824060619e-01;
    static const double p2 = 4.369287357367e-02;
    static const double p3 = -6.127046810372e-03;
    static const double q1 = 3.247396119172e-01;
    static const double q2 = 1.776068284106e-01;
    static const double q3 = 2.322361333467e-02;
    static const double q4 = 8.148654046054e-03;

    if (x <= -1) {
        return -0.5;
    }
    else if (x < 0) {
        double z = x + 1;
        double r = ((((z + p3) * z + p2) * z + p1) * z + p0) /
                    ((((q4 * z + q3) * z + q2) * z + q1) * z + 1);
        return -(1 + sqrt(z) * r) / (1 - x);
    }
    else if (x <= 1) {
        return ((((x + p3) * x + p2) * x + p1) * x + p0) /
                ((((q4 * x + q3) * x + q2) * x + q1) * x + 1);
    }
    else if (x <= 2) {
        double z = x - 1;
        double r = ((((z + p3) * z + p2) * z + p1) * z + p0) /
                    ((((q4 * z + q3) * z + q2) * z + q1) * z + 1);
        return (r * (z - 1) - 1) / sqrt(x);
    }
    else {
        return (1 / tgamma(x + 1) - 1) / (x * (x - 1));
    }
}

/*
 * Computes the function D(a, x) = x^a * e^-x / Gamma(a + 1),.
 *
 * It is common occuring term in many representations of the incomplete gamma
 * function. Careful computaton of the term is required especially for large
 * arguments because it determines the accuracy of the result of the incomplete
 * gamma function.
 *
 * Here, mu = (x - a) / a
 *
 * References
 * ----------
 * [1] Temme, N. (1994). A Set of Algorithms for the Incomplete Gamma Functions.
 *     Probability in the Engineering and Informational Sciences, 8(2),
 *     291-307. doi:10.1017/S0269964800003417.
 */
static NPY_INLINE double
compute_dax(double a, double mu)
{
    static const double twopi = 6.283185307179586;
    static const double log_min = -708.3964185322641;  // log(DBL_MIN)

    double dp = a * log1pminx(mu) - 0.5 * log(twopi * a);
    return dp < log_min ? 0 : exp(dp) / gammastar(a);
}

/*
 * Compute Q(a, x) using Gautschi's algorithm for the taylor series expansion of
 * P(a, x) and the relation Q(a, x) = 1 - P(a, x).
 *
 * Gil et al [3] shows that no more than 30 terms are needed for a <= 10000 for
 * the series to converge if used when x <= 0.3 * a and epsilon = 1e-15.
 *
 * References
 * ----------
 * [1] Temme, N. (1994). A Set of Algorithms for the Incomplete Gamma Functions.
 *     Probability in the Engineering and Informational Sciences, 8(2),
 *     291-307. doi:10.1017/S0269964800003417.
 * [2] Gautschi, W. (1979). A computational procedure for incomplete gamma
 *     functions. ACM Trans-actions on Mathematical Software 5: 466-481.
 * [3] A. Gil, J. Segura, and N. M. Temme. 2012. Efficient and accurate
 *     algorithms for the computation and inversion of the incomplete gamma
 *     function ratios. SIAM Journal on Scientific Computing 34, 6 (2012),
 *     A2965--A2981.
 */
#define PGM_EPS 1e-15
static NPY_INLINE double
taylor_p(double a, double x)
{
    double dax = compute_dax(a, (x - a) / a);
    if (dax == 0) {
        return 1;
    }

    double sum = 1, r = 1;
	for (size_t n = 1; n < 30 && (r / sum) > PGM_EPS; n++) {
		sum += (r *= x / (a + n));
	}
    return 1 - dax * sum;
}

/*
 * Compute Q(a, x) using Gautschi's algorithm of a Taylor expansion when x<1.
 *
 * Gil et al [3] uses this algorithm when 0 < x <= 1.5 and a > threshold, where
 * the threshold depends on the value of x.
 *
 * References
 * ----------
 * [1] Temme, N. (1994). A Set of Algorithms for the Incomplete Gamma Functions.
 *     Probability in the Engineering and Informational Sciences, 8(2),
 *     291-307. doi:10.1017/S0269964800003417.
 * [2] Gautschi, W. (1979). A computational procedure for incomplete gamma
 *     functions. ACM Trans-actions on Mathematical Software 5: 466-481.
 * [3] A. Gil, J. Segura, and N. M. Temme. 2012. Efficient and accurate
 *     algorithms for the computation and inversion of the incomplete gamma
 *     function ratios. SIAM Journal on Scientific Computing 34, 6 (2012),
 *     A2965--A2981.
 */
static NPY_INLINE double
taylor_q(double a, double x, double logx)
{
    size_t i;
    double v, t;
    double s = -a * (a - 1) * auxgam(a);
    double u = s - expm1(a * logx) * (1 - s);
    double r = a + 3;
    double p = a * x;
    double q = a + 1;
    for (i = 0, v = t = 1; i < 100; r += 2, i++) {
        p += x;
        q += r;
        t *= -p / q;
        v += t;
        if (fabs(t / v) < PGM_EPS) {
            break;
        }
    }
    v = a * (1 - s) * exp((a + 1) * logx) * v / (a + 1);
    return u + v;
}

/*
 * Compute Q(a, x) using Gautschi's algorithm of a continued fraction.
 *
 * GIl et al [3] uses this expansion for when x > 2.35 * a and/or a < 12.
 *
 * References
 * ----------
 * [1] Temme, N. (1994). A Set of Algorithms for the Incomplete Gamma Functions.
 *     Probability in the Engineering and Informational Sciences, 8(2),
 *     291-307. doi:10.1017/S0269964800003417.
 * [2] Gautschi, W. (1979). A computational procedure for incomplete gamma
 *     functions. ACM Trans-actions on Mathematical Software 5: 466-481.
 * [3] A. Gil, J. Segura, and N. M. Temme. 2012. Efficient and accurate
 *     algorithms for the computation and inversion of the incomplete gamma
 *     function ratios. SIAM Journal on Scientific Computing 34, 6 (2012),
 *     A2965--A2981.
 */
static NPY_INLINE double
continued_frac(double a, double x)
{
    double dax = compute_dax(a, (x - a) / a);
    if (dax == 0) {
        return 1;
    }

    size_t i;
    double g, t, tau, ro, p;
    double c = x + 1 - a;
    double q = (x - 1 - a) * c;
    double r = 4 * c;
    double s = 1 - a;
    for (i = p = ro = 0, t = g = 1; i < 100; r += 8, s += 2, i++) {
        p += s;
        tau = p * (ro + 1);
        q += r;
        ro = tau / (q - tau);
        t *= ro;
        g += t;
        if (fabs(t / g) < PGM_EPS) {
            break;
        }
    }
    return (a / c) * g * dax;
}

/*
 * compute Q(a, x) using a uniformly asymptotic expansion explained in [1].
 *
 * Gil et al [4] uses this expansion when a >= 12 and |eta| <= 1, where eta
 * satisfies the expression 0.5 * eta^2 = (x/a) - 1 - log(x/a). They also use
 * 25 coefficients for the recursive infinite sum used to compute the
 * expansion. This is said to be sufficient for a = 12, and fewer coefficients
 * are needed for larger a. We use the representation in equation 5.1 of [4].
 * Cofficients 1-16 are obtained from [1] and coefficients 17-26 from the
 * cephes library, via Scipy's `scipy.special` module at:
 * https://github.com/scipy/scipy/blob/master/scipy/special/cephes/igam.h
 *
 * References
 * ----------
 * [1] Temme, N. (1994). A Set of Algorithms for the Incomplete Gamma Functions.
 *     Probability in the Engineering and Informational Sciences, 8(2),
 *     291-307. doi:10.1017/S0269964800003417.
 * [2] Temme, N.M. (1979). The asymptotic expansions of the incomplete gamma
 *     functions. SIAM Journal on Mathematical Analysis 10: 239-253. 8.
 * [3] Temme, N.M. (1987). On the computation of the incomplete gamma functions
 *     for large values of the parameters. In E.J.C. Mason & M.G. Cox (eds.),
 *     Algorithms for approximation. Pro-ceedings of the IMA-Conference,
 *     Shrivenham, July 15-19, 1985, Oxford, Clarendon, pp. 479-489.
 * [4] A. Gil, J. Segura, and N. M. Temme. 2012. Efficient and accurate
 *     algorithms for the computation and inversion of the incomplete gamma
 *     function ratios. SIAM Journal on Scientific Computing 34, 6 (2012),
 *     A2965--A2981.
 */
static NPY_INLINE double
q_asymp(double a, double x)
{
    static const double twopi = 6.283185307179586;
    static const double fm[] = {
        1.0000000000000000e+0, -3.3333333333333333e-1,
        8.3333333333333333e-2, -1.4814814814814815e-2,
        1.1574074074074074e-3, 3.527336860670194e-4,
        -1.7875514403292181e-4, 3.9192631785224378e-5,
        -2.1854485106799922e-6, -1.85406221071516e-6,
        8.296711340953086e-7, -1.7665952736826079e-7,
        6.7078535434014986e-9, 1.0261809784240308e-8,
        -4.3820360184533532e-9, 9.1476995822367902e-10,
        -2.551419399494625e-11, -5.8307721325504251e-11,
        2.4361948020667416e-11, -5.0276692801141756e-12,
        1.1004392031956135e-13, 3.3717632624009854e-13,
        -1.3923887224181621e-13, 2.8534893807047443e-14,
        -5.1391118342425726e-16, -1.9752288294349443e-15
    };

    double mu = (x - a) / a;
    double dax = compute_dax(a, mu);
    if (dax == 0) {
        return mu < 0 ? 1 : 0;
    }

    double y = -log1pminx(mu);
    double eta = sqrt(2 * y);
    double v = 0.5 * pgm_erfc(sqrt(a * y), true) * gammastar(a) * sqrt(twopi * a);
    size_t i;
    double s = mu < 0 ? -1 : 1;
    double bm[25] = {0};
    double u, t;
    eta *= s;
    for (i = 23, u = 0, bm[24] = fm[25], bm[23] = fm[24]; i > 0; i--) {
        t = fm[i] + (i + 1) * bm[i + 1] / a;
        u = eta * u + t;
        bm[i - 1] = t;
    }
    u *= s;
    return s == 1 ? (u + v) * dax : 1 - (u + v) * dax;
}

/*
 * Compute Q(a, x), the upper incomplete gamma ratio using Gil's algorithm [1].
 *
 * An appropriate expansion is used depending on the pair (a, x). Refer to [1]
 * for more details.
 *
 * For integer and half-integer values we use special cased expansion that are
 * more efficient than the ones outlined by Gil et al [2].
 *
 * References
 * ----------
 *  [1] A. Gil, J. Segura, and N. M. Temme. 2012. Efficient and accurate
 *      algorithms for the computation and inversion of the incomplete gamma
 *      function ratios. SIAM Journal on Scientific Computing 34, 6 (2012), A2965--A2981
 *  [2] https://www.boost.org/doc/libs/1_71_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
 */
NPY_INLINE double
pgm_gammaq(double a, double x)
{
    static const double loghalf = -0.6931471805599453;
    size_t aa = (size_t)a;

    if (a < 30 && a == aa) {
        size_t k;
        double sum, z;
        for (k = sum = z = 1; k < aa; k++) {
            sum += (z *= x / k);
        }
        return exp(-x) * sum;
    }
    else if (a < 30 && a == (aa + 0.5)) {
        static const double one_sqrtpi = 0.5641895835477563;
        size_t k;
        double sum, z;
        double sqrt_x = sqrt(x);
        for (k = z = 1, sum = 0; k < aa + 1; k++) {
            sum += (z *= x / (k - 0.5));
        }
        return pgm_erfc(sqrt_x, false) + exp(-x) * one_sqrtpi * sum / sqrt_x;
    }

    double logx = log(x);
    double alpha = x >= 0.5 ? x : loghalf / logx;

    if (a > alpha) {
        return (x < 0.3 * a) || a < 12 ? taylor_p(a, x) : q_asymp(a, x);
    }
    else if (a < -DBL_MIN / logx) {
        return 0;
    }
    else if (x <= 1.5) {
        return taylor_q(a, x, logx);
    }
    else {
        return (x > 2.35 * a) || a < 12 ? continued_frac(a, x) : q_asymp(a, x);
    }
}

#undef PGM_EPS
