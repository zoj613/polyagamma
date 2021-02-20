/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"
#include "pgm_igammaq.h"
#include <float.h>

#define SQRT_PI sqrt(NPY_PI)
NPY_INLINE double
ppgm_erfc(double z)
{
    size_t k;
    double a, b, c, b_c;
    double z2 = z * z;

    if (z > 37) {
        return z > 0 ? 0 : 2;
    }
    for (k = 1, a = 1, b = c =  1 / z2; k < 100; k++) {
        a = 1 / (1 + (0.5 * k * a) / z2);
        b *= (a - 1);
        c += b;
        if (fabs(b / c) < 1e-7) {
            break;
        }
    }
    a = exp(-z2) * z * c / SQRT_PI;
    return z > 0 ? a : 2 - a;
}
NPY_INLINE double
pgm_erfc(double z)
{
    double c, d, h, u, v, y, z2;
    size_t k;

    if (z > 37) {
        return z > 0 ? 0 : 2;
    }

    z2 = z * z;
    for (k = 1, d = 0, c = y = v = 2 * z2 + 1; k < 100; k += 2) {
        u = k * (k + 1);
        v += 4;
        c = v - u / c;
        d = 1 / (v - u * d);
        h = c * d;
        y *= h;
        if (fabs(h - 1) <= 1e-10) {
            break;
        }
    }
    c = exp(-z2) * 2 * z / y / SQRT_PI;
    return z > 0 ? c : 2 - c; 
}
#undef SQRT_PI
/*
 * Calculate logarithm of the gamma function of z.
 *
 * This implementation is based on an asymptotic expansion based on stirling's
 * approximation.
 *
 * - values less than 1 we use lgamma from <math.h> since it is more accurate.
 * - For interger values corresponding to logfactorial, we use a lookup table.
 */
NPY_INLINE double
pgm_lgamma(double z)
{
    /* lookup table for integer values of log-gamma function where x >= 3
     * courtesy of NumPy devs. */
    static const double logfactorial[126] = {
        0.000000000000000, 0.0000000000000000, 0.69314718055994529,
        1.791759469228055, 3.1780538303479458, 4.7874917427820458,
        6.5792512120101012, 8.5251613610654147, 10.604602902745251,
        12.801827480081469, 15.104412573075516, 17.502307845873887,
        19.987214495661885, 22.552163853123425, 25.19122118273868,
        27.89927138384089, 30.671860106080672, 33.505073450136891,
        36.395445208033053, 39.339884187199495, 42.335616460753485,
        45.380138898476908, 48.471181351835227, 51.606675567764377,
        54.784729398112319, 58.003605222980518, 61.261701761002001,
        64.557538627006338, 67.88974313718154, 71.257038967168015,
        74.658236348830158, 78.092223553315307, 81.557959456115043,
        85.054467017581516, 88.580827542197682, 92.136175603687093,
        95.719694542143202, 99.330612454787428, 102.96819861451381,
        106.63176026064346, 110.32063971475739, 114.03421178146171,
        117.77188139974507, 121.53308151543864, 125.3172711493569,
        129.12393363912722, 132.95257503561632, 136.80272263732635,
        140.67392364823425, 144.5657439463449, 148.47776695177302,
        152.40959258449735, 156.3608363030788, 160.3311282166309,
        164.32011226319517, 168.32744544842765, 172.35279713916279,
        176.39584840699735, 180.45629141754378, 184.53382886144948,
        188.6281734236716, 192.7390472878449, 196.86618167289001,
        201.00931639928152, 205.1681994826412, 209.34258675253685,
        213.53224149456327, 217.73693411395422, 221.95644181913033,
        226.1905483237276, 230.43904356577696, 234.70172344281826,
        238.97838956183432, 243.26884900298271, 247.57291409618688,
        251.89040220972319, 256.22113555000954, 260.56494097186322,
        264.92164979855278, 269.29109765101981, 273.67312428569369,
        278.06757344036612, 282.4742926876304, 286.89313329542699,
        291.32395009427029, 295.76660135076065, 300.22094864701415,
        304.68685676566872, 309.1641935801469, 313.65282994987905,
        318.1526396202093, 322.66349912672615, 327.1852877037752,
        331.71788719692847, 336.26118197919845, 340.81505887079902,
        345.37940706226686, 349.95411804077025, 354.53908551944079,
        359.1342053695754, 363.73937555556347, 368.35449607240474,
        372.97946888568902, 377.61419787391867, 382.25858877306001,
        386.91254912321756, 391.57598821732961, 396.24881705179155,
        400.93094827891576, 405.6222961611449, 410.32277652693733,
        415.03230672824964, 419.75080559954472, 424.47819341825709,
        429.21439186665157, 433.95932399501481, 438.71291418612117,
        443.47508812091894, 448.24577274538461, 453.02489623849613,
        457.81238798127816, 462.60817852687489, 467.4121995716082,
        472.22438392698058, 477.04466549258564, 481.87297922988796
    };
    // Coefficients from Pugh (2004)
    static const double d[11] = {
        2.48574089138753565546E-5, 1.05142378581721974210E+0,
        -3.45687097222016235469E+0, 4.51227709466894823700E+0,
        -2.98285225323576655721E+0, 1.05639711577126713077E+0,
        -1.95428773191645869583E-1, 1.70970543404441224307E-2,
        -5.71926117404305781283E-4, 4.63399473359905636708E-6,
        -2.71994908488607703910E-9
    };
    static const double a1 = 0.08333333333333333;  // 1 / 12
    static const double a2 = 0.002777777777777778;  // 1/360
    static const double a3 = 0.0007936507936507937;  // 1/1260
    static const double log2sepi = 0.6207822376352452;  // log(2 * sqrt(e/pi))

    double out, z2, sum, a, b;
    size_t zz, k;

    if (z < 1) {
        // use Pugh(2004)'s algorithm for small z.
         a = z - 1;
         for (k = 1, sum = d[0] + d[0] / a; k < 11; k++) {
             sum += d[k] / (a + k);
         }
         b = z - 0.5;
         return log(sum) + log2sepi - b + b * log(b + 10.900511);
    }

    zz = (size_t)z;
    if (z < 127 && z == zz) 
        return logfactorial[zz - 1];

    z2 = z * z;
    out = (z - 0.5) * log(z) - z + PGM_LS2PI;
    out += a1 / z - a2 / (z2 * z) + a3 / (z2 * z2 * z);
    return out;
}

/*
 * Calculate the Error function for x >= 0.
 *
 * This is an implementation based on Abramowitz and Stegun (1964), equation
 * 7.1.28, using a polynomial of order 4. It has a maximum error of 1.5×10−7.
 */
static NPY_INLINE double
pgm_erf(double x)
{
    static const double p = 0.3275911;
    static const double a1 = 0.254829592;
    static const double a2 = -0.284496736;
    static const double a3 = 1.421413741;
    static const double a4 = -1.453152027;
    static const double a5 = 1.061405429;
    double out;

    if (x == 0)
        return 0;

    double t = 1 / (1 + p * x), t2 = t * t, t3 = t2 * t;
    out = a1 * t + a2 * t2 + a3 * t3 + a4 * t2 * t2 + a5 * t3 * t2;
    return 1 - out * exp(-x * x);
}

/*
 * Calculate the cumulative distribution function of an Inverse-Gaussian.
 */
NPY_INLINE double
inverse_gaussian_cdf(double x, double mu, double lambda)
{
    double a = sqrt(0.5 * lambda / x);
    double b = a * (x / mu);
    double c = exp(lambda / mu);

    return 0.5 * (kf_erfc(a - b) + c * kf_erfc(b + a) * c);
}

/*
 * sample from X ~ Gamma(a, rate=b) truncated on the interval {x | x > t}.
 *
 * For a > 1 we use the algorithm described in Dagpunar (1978)
 * For a == 1, we truncate an Exponential of rate=b.
 * For a < 1, we use algorithm [A4] described in Philippe (1997)
 *
 * TODO: There is a more efficient algorithm for a > 1 in Philippe (1997), which
 * should replace this one in the future.
 */
NPY_INLINE double
random_left_bounded_gamma(bitgen_t* bitgen_state, double a, double b, double t)
{
    double x, log_rho, log_m, a_minus_1, b_minus_a, c0, one_minus_c0;

    if (a > 1) {
        b = t * b;
        a_minus_1 = a - 1;
        b_minus_a = b - a;
        c0 = 0.5 * (b_minus_a + sqrt(b_minus_a * b_minus_a + 4 * b)) / b;
        one_minus_c0 = 1 - c0;

        do {
            x = b + random_standard_exponential(bitgen_state) / c0;
            log_rho = a_minus_1 * log(x) - x * one_minus_c0;
            log_m = a_minus_1 * log(a_minus_1 / one_minus_c0) - a_minus_1;
        } while (log(next_double(bitgen_state)) > (log_rho - log_m));
        return t * (x / b);
    }
    else if (a == 1) {
        return t + random_standard_exponential(bitgen_state) / b;
    }
    else {
        do {
            x = 1 + random_standard_exponential(bitgen_state) / (t * b);
        } while (log(next_double(bitgen_state)) > (a - 1) * log(x));
        return t * x;
    }
}

/*
 * Sample from an Inverse-Gaussian(mu, lambda) truncated on the set {x | x < t}.
 *
 * We sample using two algorithms depending on whether mu > t or mu < t.
 *
 * When mu < t, We use a known sampling algorithm from Devroye
 * (1986), page 149. We sample until the generated variate is less than t.
 *
 * When mu > t, we use a Scaled-Inverse-Chi-square distribution as a proposal,
 * as explained in [1], page 134. To generate a sample from this proposal, we
 * sample from the tail of a standard normal distribution such that the value
 * is greater than 1/sqrt(t). Once we obtain the sample, we square and invert
 * it to obtain a sample from a Scaled-Inverse-Chi-Square(df=1, scale=lambda)
 * that is less than t. An efficient algorithm to sample from the tail of a
 * normal distribution using a pair of exponential variates is shown in
 * Devroye (1986) [page 382] & Devroye (2009) [page 7]. This sample becomes our
 * proposal. We accept the sample only if we sample a uniform less than the
 * acceptance porbability. The probability is exp(-0.5 * lambda * x / mu^2).
 * Refer to Appendix S1 of Polson et al. (2013).
 *
 * References
 * ----------
 *  [1] Windle, J. (2013). Forecasting high-dimensional, time-varying
 *      variance-covariance matrices with high-frequency data and sampling
 *      Pólya-Gamma random variates for posterior distributions derived from
 *      logistic likelihoods.(PhD thesis). Retrieved from
 *      http://hdl.handle.net/2152/21842
 */
NPY_INLINE double
random_right_bounded_inverse_gaussian(bitgen_t* bitgen_state, double mu,
                                      double lambda, double t)
{
    double x;

    if (t < mu) {
        double e1, e2;
        const double a = 1. / (mu * mu);
        const double half_lambda = -0.5 * lambda;
        do {
            do {
                e1 = random_standard_exponential(bitgen_state);
                e2 = random_standard_exponential(bitgen_state);
            } while ((e1 * e1) > (2 * e2 / t));
            x = (1 + t * e1);
            x = t / (x * x);
        } while (a > 0 && log(1 - next_double(bitgen_state)) >= half_lambda * a * x);
        return x;
    }
    do {
        x = random_wald(bitgen_state, mu, lambda);
    } while (x >= t);
    return x;
}


#define SQRT_MAX sqrt(DBL_MAX)
#define SQRT_MIN sqrt(DBL_MIN)
#define LOG_MAX log(DBL_MAX)
#define LOG_MIN log(DBL_MIN)
#define LOG_EPS log(DBL_EPSILON)
#define SQRT_MIN_LOG_EPS log(-LOG_EPS)
#define ONE_SQRT2EPS 1 / sqrt(2 * DBL_EPSILON)
#define EXP_LOW LOG_MIN
#define SQRT_MIN_EXPLOW sqrt(-EXPLOW)
#define EXP_HIGH LOG_MAX
#define SQRT_2PI sqrt(2 * NPY_PI)
#define LOG_SQRT_2PI log(SQRT_2PI)
#define SQRT_PI sqrt(NPY_PI)
#define ONE_SQRTPI 1 / SQRT_PI

static NPY_INLINE double
error_function(double x, bool erfc, bool expo)
{
    double y;
    if (erfc) {
        if (x < -SQRT_MIN_LOG_EPS) {
            return 2;
        }
        else if (x < -DBL_EPSILON) {
            return 2 - error_function(-x, true, false);
        }
        else if (x < DBL_EPSILON) {
           return 1; 
        }
        else if (x < 0.5) {
            return expo ? exp(x * x) : 1;
        }
        else if (x < 4) {
            double ak[] = {
                7.3738883116, 6.8650184849, 3.0317993362, 5.6316961891e-1, 4.3187787405e-5
            };
            double bk[] = {
                7.3739608908, 15.184908190, 12.795529509, 5.3542167949, 1.0000000000
            };
            return expo ? 1 : exp(-x * x) * ratfun(x, 4, 4, ak, bk);
        }
        else {
            double xl;
            if (expo) {
                xl = 1 / (DBL_MIN * SQRT_PI);
                if (x > xl) {
                    return 0;
                }
                else if (x > ONE_SQRT2EPS) {
                    return 1 / (x * SQRT_PI);
                }
                else {
                    double z = x * x;
                    double y = 1;
                }
            }
            else {
                if (x < SQRT_MIN_EXPLOW) {
                    double z = x * x;
                    double y = exp(-z);
                    if (x * DBL_MIN > y / SQRT_PI) {
                        return 0;
                    }
                }

            }
        }
    }

}
// function to perform a rational approximation
static NPY_INLINE double
ratfun(double x, size_t n, size_t m, const double* arr1, const double* arr2)
{
    double num, den;
    size_t i, k;

    for (i = n - 1, num = arr1[n]; -1 < i; i--) {
        num = num * x + arr1[i]; 
    }
    for (k = m - 1, den = arr2[m]; -1 < k; k--) {
        den = den * x + arr2[k]; 
    }
    return num / den;
}

// exp(x) - 1
static NPY_INLINE double
expmin1(double x)
{
    static double a[] = {
        9.999999998390e-01, 6.652950247674e-2,
        2.331217139081e-2, 1.107965764952e-3
    };
    static double b[] = {
        1.000000000000e+0, -4.334704979491e-1,
        7.338073943202e-2, -5.003986850699e-3
    };

    if (x < LOG_EPS) {
        return -1;
    }
    else if (x > EXP_HIGH) {
        return DBL_MAX;
    }
    else if (x < -0.69 || x > 0.41) {
        return exp(x) - 1;
    }
    else {
        return expm1(x);
    }
    //else return expm1(x);
    //else y = ratfun(x, 3, 3, a, b) * x;
}


// log(1 + x) - x
static NPY_INLINE double
log1pminx(double x)
{
    static double a[] = {
         -4.999999994526e-1, -5.717084236157e-1, -1.423751838241e-1,
         -8.310525299547e-4, 3.899341537646e-5
    };
    static double b[] = {
        1.000000000000e+0, 1.810083408290e+0, 
        9.914744762863e-1, 1.575899184525e-1
    };
    double z;

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
        return sqrt(x) * ratfun(x, 4, 3, a, b);
    }
    else {
        z = -x / (1 + x);
        if (z > 1.36) {
            return -(log(1 + z) - z) + x * z;
        }
        else {
            return -sqrt(z) * ratfun(z, 4, 3, a, b);
        }
    }
}

// Gamma*(x)
static NPY_INLINE double
gammastar(double x)
{
    double a;

    if (x > 1e10) {
        if (x > (1 / (12 * DBL_EPSILON))) {
            return 1;
        }
        else {
            return 1 + 1 / (12 * x);
        }
    }
    else if (x >= 12) {
        a = 1 / x;
        return (1.000000000949 + a * (9.781658613041e-1 + a * 7.806359425652e-2)) /
               (1 + a * 8.948328926305e-1);
    }
    else if (x >= 1) {
        static const double arr1[] = {
            5.115471897484e-2, 4.990196893575e-1,
            9.404953102900e-1, 9.999999625957e-1
        };
        static const double arr2[] = {
            1.544892866413e-2, 4.241288251916e-1,
            8.571609363101e-1, 1.000000000000e+0,
        };
        return ratfun(x, 3, 3, arr1, arr2);
    }
    else if (x > DBL_MIN) {
        a = 1 + 1 / x;
        return gammastar(x + 1) * sqrt(a) * exp(x * log(a) - 1);
    }
    else {
        return 1 / (SQRT_2PI * SQRT_MIN);
    }
}

//g(x) in 1/Gamma(1 + x) = 1 + x * (x - 1) * g(x) for 0<=x<=1
static NPY_INLINE double
auxgam(double x)
{
    static const double arr1[] = {
        -5.772156647338e-1, -1.087824060619e-1,
        4.369287357367e-2, -6.127046810372e-3,
    };
    static const double arr2[] = {
        1.000000000000e+0, 3.247396119172e-1, 1.776068284106e-1,
        2.322361333467e-2, 8.148654046054e-3,
    };

    if (x <= -1) {
        return -0.5;
    }
    else if (x < 0) {
        return -(1 + sqrt(x + 1) * ratfun(x + 1, 3, 4, arr1, arr2)) / (1 - x);
    }
    else if (x <= 1) {
        return ratfun(x, 3, 4, arr1, arr2);
    }
    else if (x <= 2) {
        return ((x - 2) * ratfun(x - 1, 3, 4, arr1, arr2) - 1) / sqrt(x);
    }
    else {
        return (1 / tgamma(x + 1) - 1) / (x * (x - 1));
    }
}

// D(a, x) = x^a * e^-x / Gamma(a + 1)
static NPY_INLINE double
compute_dax(double a, double x)
{
    static const double twopi = 6.283185307179586;
    double dp;

    dp = a * log1pminx((x - a) / a) - 0.5 * log(twopi * a);
    if (dp < EXP_LOW) {
        return 0;
    }
    else {
        return exp(dp) / gammastar(a);
    }
}

// compute Q(a, x) using the taylor series expansion for P(a, x);
#define PGM_EPS 1e-15
static NPY_INLINE double
q_taylorp(double a, double x)
{
	double sum, r, dax;
	size_t n;

    dax = compute_dax(a, x);
    if (dax == 0) {
        return 1; 
    }

	for (n = 1, sum = r = 1; n < 100; n++) {
		sum += (r *= x / (a + n));
		if (r / sum < PGM_EPS) {
            break;
        }
	}
    return 1 - dax * sum;
}


// compute Q(a, x) using a specialized Taylor expansion when x is small.
static NPY_INLINE double
q_taylorq(double a, double x, double logx)
{
    size_t i;
    double p, q, r, s, t, u, v;

    r = a * logx;
    q = expmin1(r);
    s = -a * (a - 1) * auxgam(a);
    u = s - q * (1 - s);

    for (i = 0, p = a * x, q = a + 1, r = a + 3, t = v = 1; i < 200; i++) {
        p += x;
        q += r;
        r += 2;
        t = -p * t / q;
        v += t;
        if (fabs(t / v) < PGM_EPS) {
            break;
        }
    }
    v = a * (1 - s) * exp((a + 1) * logx) * v / (a + 1);
    return u + v;
}

// compute Q(a, x) using a continued fraction expansion.
static NPY_INLINE double
q_continued_frac(double a, double x)
{
    size_t i;
    double dax, c, g, r, s, t, tau, ro, p, q; 

    dax = compute_dax(a, x);
    if (dax == 0) {
        return 0;
    }

    c = x + 1 - a;
    q = (x - 1 - a) * c;
    for (i = p = ro = 0, t = g = 1, r = 4 * c, s = 1 - a; i < 200; i++) {
        p += s;
        q += r;
        r += 8;
        s += 2;
        tau = p * (ro + 1);
        ro = tau / (q - tau);
        t *= ro;
        g += t;
        if (fabs(t / g) < PGM_EPS) {
            break;
        }
    }
    return (a / c) * g * dax;
}

// compute Q(a, x) using the asymptotic expansion
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
    double dax, mu, y, eta, v, s, u, t; 
    size_t i;

    dax = compute_dax(a, x);
    if (dax == 0) {
        return 1;
    }

    double bm[26] = {0};
    mu = (x - a) / a;
    y = -log1pminx(mu); 
    eta = sqrt(2 * y);
    v = 0.5 * kf_erfc(sqrt(a * y)) * gammastar(a) * sqrt(twopi * a);
    s = 1;
    if (mu < 0) {
        s = -1;
        eta *= s;
    }
    for (i = 23, u = 0, bm[24] = fm[25], bm[23] = fm[24]; i > 0; i--) {
        t = fm[i] + (i + 1) * bm[i + 1] / a;
        u = eta * u + t;
        bm[i - 1] = t;
    }
    u *= s;
    if (s == 1) {
        return (u + v) * dax;
    }
    return 1 - (u + v) * dax;
}


/*
 * Compute Q(a, x) using Gil's algorithm [1].
 *
 * References
 * ----------
 *  [1]
 */
NPY_INLINE double
pgm_gammaq(double a, double x)
{
    static const double loghalf = -0.6931471805599453;
    static const double one_sqrtpi = 0.5641895835477563;
    double alpha, sum, z, sqrt_x, logx;
    size_t aa, k;

    aa = (size_t)a;
    if (a < 30 && a == aa) {
        for (k = sum = z = 1; k < aa; k++) {
            sum += (z *= x / k);
        }
        return exp(-x) * sum;
    }
    else if (a < 30 && a == (aa + 0.5)) {
        sqrt_x = sqrt(x);
        for (k = z = 1, sum = 0; k < aa + 1; k++) {
            sum += (z *= x / (k - 0.5));
        }
        return kf_erfc(sqrt_x) + exp(-x) * one_sqrtpi * sum / sqrt_x;
    }
    else {
        logx = log(x);
        alpha = x >= 0.5 ? x : loghalf / logx; 
    }

    if (a > alpha) {
        return (x < 0.3 * a) || a < 12 ? q_taylorp(a, x) : q_asymp(a, x);
    }
    else if (a < -DBL_MIN / logx) {
        return 0;
    }
    else if (x < 1) {
        return q_taylorq(a, x, logx);
    }
    else {
        return (x > 2.35 * a) || a < 12 ? q_continued_frac(a, x) : q_asymp(a, x);
    }
}
