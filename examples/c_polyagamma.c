/*
 * This module shows an examples of how to use polyagamma in a C program.
 * Here we use a custom bitgenerator called Xoroshiro128plus. Numpy's
 * bitgenerator struct requires defining function pointers for generating
 * integers and standard uniform numbers. We define these functions alongside the
 * bitgenerator.
 *
 * This example can be compiled with:
 *
 * gcc c_polyagamma.c src/*.c -I./include -I$(python -c "import numpy; print(numpy.get_include())")
 *  -I/usr/include/python3.9 -L$(python -c "import numpy; print(numpy.get_include())")/../../random/lib
 *  -lm -lnpyrandom -O3
 */
#include "../include/pgm_random.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {uint64_t s[2];} xrs128p_random_t;


static inline uint64_t
rotl(const uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}


static inline uint64_t
xrs128p_next_int(void* rng)
{
    xrs128p_random_t* xrs = rng;
	const uint64_t s0 = xrs->s[0];
	uint64_t s1 = xrs->s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	xrs->s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
	xrs->s[1] = rotl(s1, 37); // c

	return result;
}


static inline double
xrs128p_next_double(void* rng)
{
    xrs128p_random_t* xrs = rng;
    return (xrs128p_next_int(xrs) >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * Generate 100 samples from a PG(1, 1.5) distribution using the alternate
 * method.
 */
int
main(void)
{
    size_t n = 100;
    double* out = malloc(n * sizeof(*out));
    // setup the bitgen
    xrs128p_random_t xrs = {.s = {12132233, 1959324}};
    bitgen_t bitgen;
    bitgen.state = &xrs;
    bitgen.next_double = xrs128p_next_double;
    bitgen.next_uint64 = xrs128p_next_int;

    pgm_random_polyagamma_fill(&bitgen, 1, 1.5, ALTERNATE, n, out);
    puts("Samples: [ ");
    for (size_t i = 0; i < n; i++)
       printf("%lf ", out[i]);
    puts("]");
    free(out);
}
