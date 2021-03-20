#ifndef PGM_DENSITY_H
#define PGM_DENSITY_H

double
pgm_polyagamma_pdf(double x, double h, double z);

double
pgm_polyagamma_cdf(double x, double h, double z);

double
pgm_polyagamma_logpdf(double x, double h, double z);

double
pgm_polyagamma_logcdf(double x, double h, double z);

#endif
