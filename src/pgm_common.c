/* Copyright (c) 2020-2021, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include "pgm_common.h"

#include <stdint.h>

double random_left_bounded_gamma_philippe(bitgen_t* bitgen_state, double a, double b);

NPY_INLINE double
inverse_gaussian_cdf(double x, double mu, double lambda)
{
    double a = sqrt(0.5 * lambda / x);
    double b = a * (x / mu);
    double c = exp(1 * lambda / mu);

    return 0.5 * (1 + erf(b - a) + c * (1 + erf(-(b + a))) * c);
}

#ifndef PGM_MAX_TAB_SIZE
#define PGM_MAX_TAB_SIZE 150
#endif

struct mixture {
    double v;
    double w_bar;
};

static NPY_INLINE uint64_t
get_optimal_k(uint64_t a, double b, double u)
{
    static struct mixture weights[PGM_MAX_TAB_SIZE];
    static double w_a;
    static uint64_t old_a;
    static double old_b;
    if (a == old_a && b == old_b) {
        for (size_t i = 0; i < a; i++) {
           // printf("w[%ld] = %lf ", i, (1 - weights[i].w_bar));
        }
        //puts("");
        if (u < (1 - weights[0].w_bar)) {
            return a;
        }
        for (size_t i = 1; i < a; i++) {
            if (u < (1 - weights[i].w_bar))
                return a - i;
        }
        return a;
    }
    old_a = a;
    old_b = b;
    weights[0].v = 1;
    w_a = 1;
    for (size_t i = 1; i < a; i++) {
        weights[i].v = weights[i - 1].v * (a - (i + 1) + 1) / b;
        w_a += weights[i].v;
    } 
    weights[0].w_bar = weights[0].v / w_a;
    if (u < (1 - weights[0].w_bar)) {
        return a;
    }
    for (size_t i = 1; i < a; i++) {
        weights[i].w_bar = weights[i].v / w_a; 
        if (u < (1 - weights[i].w_bar)) {
            //printf("u=%lf, w=%lf, i=%ld\n", u, weights[i].w_bar, i + 1);
            return a - i;
        }
    }
    //printf("welp...u=%lf, w=%lf, i=%ld\n", u, weights[a - 1].w_bar, a);
    return a;
}

double
random_left_bounded_gamma_philippe(bitgen_t* bitgen_state, double a, double b)
{
    double u, bf, x, logrho, logm, af_a, a_minus_af;
    uint64_t af, k;

    af  = floor(a);
    af_a = af / a;
    a_minus_af = a - af;

    if (b <= a) {
        bf = b * af_a;
        do {
            u = random_standard_uniform(bitgen_state);
            k = get_optimal_k(af, bf, u); 
            x = 1 + random_standard_gamma(bitgen_state, k) / b;
            logrho = (a_minus_af) * log(x) - x * b * (1 - af_a);
        } while (logrho < -random_standard_uniform(bitgen_state) * a_minus_af);
        return x;
    }
    bf = b - a_minus_af;
    logm = a_minus_af * log(a / b) - a_minus_af;
    do {
        u = random_standard_uniform(bitgen_state);
        k = get_optimal_k(af, bf, u); 
        x = 1 + random_standard_gamma(bitgen_state, k) / b;
        logrho = (a_minus_af) * log(x) - x * a_minus_af;
    } while (logrho < random_standard_uniform(bitgen_state) * logm);
    return x;
}

/*
 * sample from X ~ Gamma(a, rate=b) truncated on the interval {x | x > t}.
 *
 * For a >= 1, we use algorithm [A5] described in Philippe (1997)
 * For a < 1, we use algorithm [A4] described in Philippe (1997)
 *
 * TODO: There is a more efficient algorithm for a > 1 in Philippe (1997), which
 * should replace this one in the future.
 */
double
random_left_bounded_gamma(bitgen_t* bitgen_state, double a, double b, double t)
{
    double x; 

    if (a >= 1) {
        b = t * b;
        x = random_left_bounded_gamma_philippe(bitgen_state, a, b);
        return x * t; 
    }
    else {
        do {
            x = 1 + random_standard_exponential(bitgen_state) / (t * b);
        } while (log(random_standard_uniform(bitgen_state)) > (a - 1) * log(x));
        return t * x;
    }
}
