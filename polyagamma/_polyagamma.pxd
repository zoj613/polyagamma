# cython: language_level=3
from numpy.random cimport bitgen_t


cdef extern from "pgm_random.h" nogil:
    ctypedef enum sampler_t:
        GAMMA
        DEVROYE
        ALTERNATE
        SADDLE
        HYBRID

    double random_polyagamma "pgm_random_polyagamma" \
        (bitgen_t* bitgen_state, double h, double z, sampler_t method) noexcept

    void random_polyagamma_fill "pgm_random_polyagamma_fill" \
        (bitgen_t* bitgen_state, double h, double z,
         sampler_t method, size_t n, double* out) noexcept

    void random_polyagamma_fill2 "pgm_random_polyagamma_fill2" \
        (bitgen_t* bitgen_state, const double* h, const double* z,
         sampler_t method, size_t n, double* out) noexcept


cdef extern from "pgm_density.h" nogil:
    double polyagamma_pdf "pgm_polyagamma_pdf" (double x, double h, double z) noexcept
    double polyagamma_cdf "pgm_polyagamma_cdf" (double x, double h, double z) noexcept
    double polyagamma_logpdf "pgm_polyagamma_logpdf" \
        (double x, double h, double z) noexcept
    double polyagamma_logcdf "pgm_polyagamma_logcdf" \
        (double x, double h, double z) noexcept
