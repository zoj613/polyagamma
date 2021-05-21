/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef PGM_COMMON_H
#define PGM_COMMON_H

#include "pgm_macros.h"

/* numpy c-api declarations */
PGM_EXTERN double
random_standard_normal(bitgen_t* bitgen_state);
PGM_EXTERN double
random_standard_exponential(bitgen_t* bitgen_state);
PGM_EXTERN double
random_standard_gamma(bitgen_t* bitgen_state, double shape);

/* useful float constants */
#ifndef DBL_EPSILON
#define DBL_EPSILON 2.22045e-16
#endif
#ifndef DBL_MIN
#define DBL_MIN 2.22507e-308
#endif
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209e-07f
#endif
#ifndef FLT_MIN
#define FLT_MIN 1.17549e-38f
#endif
#define PGM_MAX_EXP 88.7228f    // maximum allowed expf() argument

/*
 * Calculate logarithm of the gamma function of z.
 *
 * NOTE
 * ----
 * - For integer values corresponding to logfactorial, we use a lookup table,
 *   otherwise we call the standard library implementation of lgamma.
 */
PGM_INLINE double
pgm_lgamma(double z)
{
    /* lookup table for integer values of log-gamma function where 1<=z<=200 */
    static const double logfactorial[200] = {
    0.00000000000000000000L, 0.00000000000000000000L, 0.69314718055994530943L,
    1.79175946922805500079L, 3.17805383034794561975L, 4.78749174278204599415L,
    6.57925121201010099526L, 8.52516136106541430086L, 10.60460290274525022719L,
    12.80182748008146961186L, 15.10441257307551529612L, 17.50230784587388584150L,
    19.98721449566188614923L, 22.55216385312342288610L, 25.19122118273868150135L,
    27.89927138384089156699L, 30.67186010608067280557L, 33.50507345013688888583L,
    36.39544520803305357320L, 39.33988418719949403668L, 42.33561646075348502624L,
    45.38013889847690802634L, 48.47118135183522388137L, 51.60667556776437357377L,
    54.78472939811231919027L, 58.00360522298051993775L, 61.26170176100200198341L,
    64.55753862700633105565L, 67.88974313718153497793L, 71.25703896716800901656L,
    74.65823634883016438751L, 78.09222355331531063849L, 81.55795945611503718065L,
    85.05446701758151741707L, 88.58082754219767880610L, 92.13617560368709247937L,
    95.71969454214320249114L, 99.33061245478742692927L, 102.96819861451381269979L,
    106.63176026064345913030L, 110.32063971475739543732L, 114.03421178146170323481L,
    117.77188139974507154195L, 121.53308151543863396132L, 125.31727114935689513381L,
    129.12393363912721488962L, 132.95257503561630989253L, 136.80272263732636846278L,
    140.67392364823425940368L, 144.56574394634488600619L, 148.47776695177303207807L,
    152.40959258449735784502L, 156.36083630307878519772L, 160.33112821663090702407L,
    164.32011226319518140682L, 168.32744544842765233028L, 172.35279713916280155961L,
    176.39584840699735171499L, 180.45629141754377104678L, 184.53382886144949050211L,
    188.62817342367159119398L, 192.73904728784490243687L, 196.86618167288999400877L,
    201.00931639928152667995L, 205.16819948264119853609L, 209.34258675253683563977L,
    213.53224149456326118324L, 217.73693411395422725452L, 221.95644181913033395059L,
    226.19054832372759332448L, 230.43904356577695233255L, 234.70172344281826774803L,
    238.97838956183432307379L, 243.26884900298271419139L, 247.57291409618688395045L,
    251.89040220972319437942L, 256.22113555000952545004L, 260.56494097186320932358L,
    264.92164979855280104726L, 269.29109765101982254532L, 273.67312428569370413856L,
    278.06757344036614290617L, 282.47429268763039605927L, 286.89313329542699396169L,
    291.32395009427030757587L, 295.76660135076062402293L, 300.22094864701413177710L,
    304.68685676566871547988L, 309.16419358014692195247L, 313.65282994987906178830L,
    318.15263962020932683727L, 322.66349912672617686327L, 327.18528770377521719404L,
    331.71788719692847316467L, 336.26118197919847702115L, 340.81505887079901787051L,
    345.37940706226685413927L, 349.95411804077023693038L, 354.53908551944080887464L,
    359.13420536957539877521L, 363.73937555556349016106L, 368.35449607240474959036L,
    372.97946888568902071293L, 377.61419787391865648951L, 382.25858877306002911456L,
    386.91254912321755249360L, 391.57598821732961960618L, 396.24881705179152582841L,
    400.93094827891574549739L, 405.62229616114488922607L, 410.32277652693730540800L,
    415.03230672824963956580L, 419.75080559954473413686L, 424.47819341825707464833L,
    429.21439186665157011769L, 433.95932399501482021331L, 438.71291418612118484521L,
    443.47508812091894095375L, 448.24577274538460572306L, 453.02489623849613509243L,
    457.81238798127818109829L, 462.60817852687492218733L, 467.41219957160817877195L,
    472.22438392698059622665L, 477.04466549258563309865L, 481.87297922988793424937L,
    486.70926113683941224841L, 491.55344822329800347216L, 496.40547848721762064228L,
    501.26529089157929280907L, 506.13282534203487522673L, 511.00802266523602676584L,
    515.89082458782239759554L, 520.78117371604415142272L, 525.67901351599506276635L,
    530.58428829443349222794L, 535.49694318016954425188L, 540.41692410599766910329L,
    545.34417779115487379116L, 550.27865172428556556072L, 555.22029414689486986889L,
    560.16905403727303813799L, 565.12488109487429888134L, 570.08772572513420617835L,
    575.05753902471020677645L, 580.03427276713078114545L, 585.01787938883911766030L,
    590.00831197561785385064L, 595.00552424938196893756L, 600.00947055532742813178L,
    605.02010584942368387473L, 610.03738568623860821782L, 615.06126620708488456080L,
    620.09170412847732001271L, 625.12865673089094925574L, 630.17208184781019580933L,
    635.22193785505973290251L, 640.27818366040804093364L, 645.34077869343500771793L,
    650.40968289565523929863L, 655.48485671088906617809L, 660.56626107587352919603L,
    665.65385741110591327763L, 670.74760761191267560699L, 675.84747403973687401857L,
    680.95341951363745458536L, 686.06540730199399785727L, 691.18340111441075296339L,
    696.30736509381401183605L, 701.43726380873708536878L, 706.57306224578734715758L,
    711.71472580229000698404L, 716.86222027910346005219L, 722.01551187360123895687L,
    727.17456717281576800138L, 732.33935314673928201890L, 737.50983714177743377771L,
    742.68598687435126293188L, 747.86777042464334813721L, 753.05515623048410311924L,
    758.24811308137431348220L, 763.44661011264013927846L, 768.65061679971693459068L,
    773.86010295255835550465L, 779.07503871016734109389L, 784.29539453524566594567L,
    789.52114120895886717477L, 794.75224982581345378740L, 799.98869178864340312440L,
    805.23043880370304542504L, 810.47746287586353153287L, 815.72973630391016147678L,
    820.98723167593794297625L, 826.24992186484282852277L, 831.51778002390615662787L,
    836.79077958246990348590L, 842.06889424170042068862L, 847.35209797043840918018L,
    852.64036500113294436698L, 857.93366982585743685252L};

    if (z < 201. && z == (size_t)z) {
        return logfactorial[(size_t)z - 1];
    }
    return lgamma(z);
}

/*
 * sample from X ~ Gamma(a, rate=b) truncated on the interval {x | x > t}.
 *
 * For a > 1 we use the algorithm described in Dagpunar (1978)
 * For a == 1, we truncate an Exponential of rate=b.
 * For a < 1, we use algorithm [A4] described in Philippe (1997)
 */
PGM_INLINE double
random_left_bounded_gamma(bitgen_t* bitgen_state, double a, double b, double t)
{
    double x;

    if (a > 1.) {
        b = t * b;
        float threshold;
        const float amin1 = a - 1.;
        const double bmina = b - a;
        const double c0 = 0.5 * (bmina + sqrt((bmina * bmina) + 4. * b)) / b;
        const float one_minus_c0 = 1. - c0;
        const float log_m = amin1 * (logf(amin1 / one_minus_c0) - 1.0f);

        do {
            x = b + random_standard_exponential(bitgen_state) / c0;
            threshold = amin1 * logf(x) - x * one_minus_c0 - log_m;
        } while (log1pf(-next_float(bitgen_state)) > threshold);
        return t * (x / b);
    }
    else if (a == 1.) {
        return t + random_standard_exponential(bitgen_state) / b;
    }
    else {
        const float amin1 = a - 1.;
        const double tb = t * b;
        do {
            x = 1. + random_standard_exponential(bitgen_state) / tb;
        } while (log1pf(-next_float(bitgen_state)) > amin1 * logf(x));
        return t * x;
    }
}

/*
 * Compute function G(p, x) (A confluent hypergeometric function ratio).
 * This function is defined in equation 14 of [1] and this implementation
 * uses a continued fraction (eq. 15) defined for x <= p. The continued
 * fraction is evaluated using the Modified Lentz method.
 *
 * G(p, x) = a_1/b_1+ a_2/b_2+ a_3/b_3+ ..., such that a_1 = 1 and for n >= 1:
 * a_2n = -(p - 1 + n)*x, a_(2n+1) = n*x, b_n = p - 1 + n.
 *
 * Note that b_n can be reduced to b_1 = p, b_n = b_(n-1) + 1 for n >= 2. Also
 * for odd n, the argument of a_n is "k=(n-1)/2" and for even n "k=n/2". This
 * means we can pre-compute constant terms s = 0.5 * x and r = -(p - 1) * x.
 * This simplifies a_n into: a_n = s * (n - 1) for odd n and a_n = r - s * n
 * for even n >= 2. The terms for the first iteration are pre-calculated as
 * explained in [1].
 *
 * References
 * ----------
 *  [1] Algorithm 1006: Fast and accurate evaluation of a generalized
 *      incomplete gamma function, Rémy Abergel and Lionel Moisan, ACM
 *      Transactions on Mathematical Software (TOMS), 2020. DOI: 10.1145/3365983
 */
static PGM_INLINE float
confluent_x_smaller(double p, double x)
{
    int n;
    float f, c, d, delta;
    float a = 1.0f, b = p;
    float r = -(p - 1.0f) * x;
    float s = 0.5f * x;
    for (n = 2, f = a / b, c = a / FLT_MIN, d = 1.0f / b; n < 100; ++n) {
        a =  n & 1 ? s * (n - 1) : r - s * n;
        b++;

        c = b + a / c;
        if (c < FLT_MIN) {
            c = FLT_MIN;
        }

        d = a * d + b;
        if (d < FLT_MIN) {
            d = FLT_MIN;
        }

        d = 1.0f / d;
        delta = c * d;
        f *= delta;
        if (fabsf(delta - 1.0f) < FLT_EPSILON) {
            break;
        }
    }
    return f;
}

/*
 * Compute function G(p, x) (A confluent hypergeometric function ratio).
 * This function is defined in equation 14 of [1] and this implementation
 * uses a continued fraction (eq. 16) defined for x > p. The continued
 * fraction is evaluated using the Modified Lentz method.
 *
 * G(p, x) = a_1/b_1+ a_2/b_2+ a_3/b_3+ ..., such that a_1 = 1 and for n > 1:
 * a_n = -(n - 1) * (n - p - 1), and for n >= 1: b_n = x + 2n - 1 - p.
 *
 * Note that b_n can be re-written as b_1 = x - p + 1 and
 * b_n = (((x - p + 1) + 2) + 2) + 2 ...) for n >= 2. Thus b_n = b_(n-1) + 2
 * for n >= 2. Also a_n can be re-written as a_n = (n - 1) * ((p - (n - 1)).
 * So if we can initialize the series with a_1 = 1 and instead of computing
 * (n - 1) at every iteration we can instead start the counter at n = 1 and
 * just compute a_(n+1) = n * (p - n). This doesnt affect b_n terms since all
 * we need is to keep incrementing b_n by 2 every iteration after initializing
 * the series with b_1 = x - p + 1.
 *
 * References
 * ----------
 *  [1] Algorithm 1006: Fast and accurate evaluation of a generalized
 *      incomplete gamma function, Rémy Abergel and Lionel Moisan, ACM
 *      Transactions on Mathematical Software (TOMS), 2020. DOI: 10.1145/3365983
 */
static PGM_INLINE float
confluent_p_smaller(double p, double x)
{
    int n;
    float f, c, d, delta;
    float a = 1.0f, b = x - p + 1.0f;
    for (n = 1, f = a / b, c = a / FLT_MIN, d = 1.0f / b; n < 100; ++n) {
        a = n * (p - n);
        b += 2.0f;

        c = b + a / c;
        if (c < FLT_MIN) {
            c = FLT_MIN;
        }

        d = a * d + b;
        if (d < FLT_MIN) {
            d = FLT_MIN;
        }

        d = 1.0f / d;
        delta = c * d;
        f *= delta;
        if (fabsf(delta - 1.0f) < FLT_EPSILON) {
            break;
        }
    }
    return f;
}

/*
 * Compute the (normalized) upper incomplete gamma function for the pair (p, x).
 *
 * We use the algorithm described in [1]. We use two continued fractions to
 * evaluate the function in the regions {0 < x <= p} and {0 <= p < x}
 * (algorithm 3 of [1]).
 *
 * We also use a terminating series to evaluate the normalized version for
 * integer and half-integer values of p <= 30 as described in [2]. This is
 * faster than the algorithm of [1] when p is small since not more than p terms
 * are required to evaluate the function.
 *
 * Parameters
 * ----------
 *  normalized : if true, the normalized upper incomplete gamma is returned,
 *      else the non-normalized version is returned for the arguments (p, x).
 *
 * References
 * ----------
 *  [1] Algorithm 1006: Fast and accurate evaluation of a generalized
 *      incomplete gamma function, Rémy Abergel and Lionel Moisan, ACM
 *      Transactions on Mathematical Software (TOMS), 2020. DOI: 10.1145/3365983
 *  [2] https://www.boost.org/doc/libs/1_71_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
 */
static PGM_INLINE float
upper_incomplete_gamma(float p, float x, bool normalized)
{
    if (normalized) {
        int p_int = (int)p;
        if (p == p_int && p < 30.f) {
            float sum, r;
            int k = 1;
            for (sum = r = 1.f; k < p_int; ++k) {
                sum += (r *= x / k);
            }
            return expf(-x) * sum;
        }
        else if (p == (p_int + 0.5f) && p < 30.f) {
            float sum, r;
            int k = 1;
            static const float one_sqrtpi = 0.5641895835477563f;
            float sqrt_x = sqrtf(x);
            for (r = 1.f, sum = 0.f; k < p_int + 1; ++k) {
                sum += (r *= x / (k - 0.5f));
            }
            return erfcf(sqrt_x) + expf(-x) * one_sqrtpi * sum / sqrt_x;
        }
    }

    bool x_smaller = p >= x;
    float f = x_smaller ? confluent_x_smaller(p, x) : confluent_p_smaller(p, x);

    if (normalized) {
        float out = f * expf(-x + p * logf(x) - pgm_lgamma(p));
        return x_smaller ? 1.0f - out : out;
    }
    else if (x_smaller) {
        float lgam = pgm_lgamma(p);
        float exp_lgam = lgam >= PGM_MAX_EXP ? expf(PGM_MAX_EXP) : expf(lgam);
        float arg = -x + p * logf(x) - lgam;

        if (arg >= PGM_MAX_EXP) {
            arg = PGM_MAX_EXP;
        }
        else if (arg <= -PGM_MAX_EXP) {
            arg = -PGM_MAX_EXP;
        }
        return (1.0f - f * expf(arg)) * exp_lgam;
    }
    else {
        float arg = -x + p * logf(x);
        return f * (arg >= PGM_MAX_EXP ? expf(PGM_MAX_EXP) : expf(arg));
    }
}

#endif
