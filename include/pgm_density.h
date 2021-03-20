#ifndef PGM_DENSITY_H
#define PGM_DENSITY_H
#include <stddef.h>

/*
 * Approximate the density function of PG(h, z).
 *
 * The value is calculated with accuracy of up `terms` terms. The calculate
 * will terminate early if successive terms are very small such that the
 * current series value is equal to the previous value with given tolerance.
 */
double
pgm_polyagamma_pdf(double x, double h, double z, size_t terms, double atol, double rtol);

#endif
