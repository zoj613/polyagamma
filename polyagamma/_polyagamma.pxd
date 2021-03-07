# cython: language_level=3
from numpy.random cimport bitgen_t

# this enum is duplicated from its C equivalent to avoid the need to link the
# corresponding C header when importing this file in cython modules.
ctypedef enum sampler_t:
    GAMMA
    DEVROYE
    ALTERNATE
    SADDLE
    HYBRID

# Cython-level declarations available to be cimported by other modules
cdef double random_polyagamma(bitgen_t* bitgen_state, double h, double z,
                              sampler_t method) nogil

cdef void random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                                 sampler_t method, size_t n, double* out) nogil

cdef void random_polyagamma_fill2(bitgen_t* bitgen_state, const double* h,
                                  const double* z, sampler_t method, size_t n,
                                  double* out) nogil
