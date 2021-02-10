/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"

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
    static const double logfactorial[124] = {
        0.69314718055994529, 1.791759469228055, 3.1780538303479458,
        4.7874917427820458, 6.5792512120101012, 8.5251613610654147,
        10.604602902745251, 12.801827480081469, 15.104412573075516,
        17.502307845873887, 19.987214495661885, 22.552163853123425,
        25.19122118273868, 27.89927138384089, 30.671860106080672,
        33.505073450136891, 36.395445208033053, 39.339884187199495,
        42.335616460753485, 45.380138898476908, 48.471181351835227,
        51.606675567764377, 54.784729398112319, 58.003605222980518,
        61.261701761002001, 64.557538627006338, 67.88974313718154,
        71.257038967168015, 74.658236348830158, 78.092223553315307,
        81.557959456115043, 85.054467017581516, 88.580827542197682,
        92.136175603687093, 95.719694542143202, 99.330612454787428,
        102.96819861451381, 106.63176026064346, 110.32063971475739,
        114.03421178146171, 117.77188139974507, 121.53308151543864,
        125.3172711493569, 129.12393363912722, 132.95257503561632,
        136.80272263732635, 140.67392364823425, 144.5657439463449,
        148.47776695177302, 152.40959258449735, 156.3608363030788,
        160.3311282166309, 164.32011226319517, 168.32744544842765,
        172.35279713916279, 176.39584840699735, 180.45629141754378,
        184.53382886144948, 188.6281734236716, 192.7390472878449,
        196.86618167289001, 201.00931639928152, 205.1681994826412,
        209.34258675253685, 213.53224149456327, 217.73693411395422,
        221.95644181913033, 226.1905483237276, 230.43904356577696,
        234.70172344281826, 238.97838956183432, 243.26884900298271,
        247.57291409618688, 251.89040220972319, 256.22113555000954,
        260.56494097186322, 264.92164979855278, 269.29109765101981,
        273.67312428569369, 278.06757344036612, 282.4742926876304,
        286.89313329542699, 291.32395009427029, 295.76660135076065,
        300.22094864701415, 304.68685676566872, 309.1641935801469,
        313.65282994987905, 318.1526396202093, 322.66349912672615,
        327.1852877037752, 331.71788719692847, 336.26118197919845,
        340.81505887079902, 345.37940706226686, 349.95411804077025,
        354.53908551944079, 359.1342053695754, 363.73937555556347,
        368.35449607240474, 372.97946888568902, 377.61419787391867,
        382.25858877306001, 386.91254912321756, 391.57598821732961,
        396.24881705179155, 400.93094827891576, 405.6222961611449,
        410.32277652693733, 415.03230672824964, 419.75080559954472,
        424.47819341825709, 429.21439186665157, 433.95932399501481,
        438.71291418612117, 443.47508812091894, 448.24577274538461,
        453.02489623849613, 457.81238798127816, 462.60817852687489,
        467.4121995716082, 472.22438392698058, 477.04466549258564,
        481.87297922988796};

    static const double a1 = 0.08333333333333333;  // 1 / 12
    static const double a2 = 0.002777777777777778;  // 1/360
    static const double a3 = 0.0007936507936507937;  // 1/1260
    double out, z2;
    size_t zz;

    if (z < 1)
        return lgamma(z);

    if (z == 1 || z == 2)
        return 0;
    
    zz = (size_t)z;
    if (z < 127 && z == zz) 
        return logfactorial[zz - 3];

    z2 = z * z;
    out = (z - 0.5) * log(z) - z + PGM_LS2PI;
    out += a1 / z - a2 / (z2 * z) + a3 / (z2 * z2 * z);
    return out;
}

/*
 * Calculate the Error function for x >= 0.
 *
 * This is an implementation based on Abramowitz and Stegun (1964), equation
 * 7.1.25, using a polynomial of order 4. It has a maximum error of 5×10−4.
 */
static NPY_INLINE double
pgm_erf(double x)
{
    static const double a1 = 0.278393;
    static const double a2 = 0.230389;
    static const double a3 = 0.000972;
    static const double a4 = 0.078108;
    double out;
    double x2 = x * x;

    out = 1 + a1 * x + a2 * x2 + a3 * x2 * x + a4 * x2 * x2;
    out *= out;
    out *= out;

    return 1 - 1 / out;
}

/*
 * Calculate the cumulative distribution function of an Inverse-Gaussian.
 */
double
inverse_gaussian_cdf(double x, double mu, double lambda)
{
    double a = sqrt(0.5 * lambda / x);
    double b = a * (x / mu);
    double c = exp(lambda / mu);

    return 0.5 * (1 + pgm_erf(b - a) + c * (1 - pgm_erf(b + a)) * c);
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
        } while (log(random_standard_uniform(bitgen_state)) > (log_rho - log_m));
        return t * (x / b);
    }
    else if (a == 1) {
        return t + random_standard_exponential(bitgen_state) / b;
    }
    else {
        do {
            x = 1 + random_standard_exponential(bitgen_state) / (t * b);
        } while (log(random_standard_uniform(bitgen_state)) > (a - 1) * log(x));
        return t * x;
    }
}
