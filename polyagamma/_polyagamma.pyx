# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
from cpython.pycapsule cimport PyCapsule_GetPointer
from cython.view cimport array as cvarray
from numpy.random.bit_generator cimport bitgen_t

import numpy as np


cdef extern from "../include/pgm_random.h":
    ctypedef enum sampler_t:
        HYBRID
        DEVROYE
        ALTERNATE
        SADDLE
    double pgm_random_polyagamma(bitgen_t* bg, double h, double z, sampler_t) nogil
    void pgm_random_polyagamma_fill(bitgen_t* bg, double h, double z,
                                    sampler_t, size_t n, double* out) nogil


cdef const char* NAME = "BitGenerator"


class Generator(np.random.Generator):
    def polyagamma(self, double h=1, double z=0, size=None, double[:] out=None):
        """
        polyagamma(h=1, z=0, size=None, out=None)

        Draw samples from a Polya-Gamma distribution.

        Samples are draw from a Polya-Gamma distribution with specified para-
        meters `h` and `z`.

        Parameters
        ----------
        h : scalar
            The `h` parameter as described in [1]_.
        z : scalar
            The exponential tilting parameter as described in [1]_.
        size : int, optional
            The number of elements to draw from the distribution. If size is
            ``None`` (default) then a single value is returned.
        out : numpy.ndarray, optional
            Output array in which to store samples. If give, then no value
            is returned.

        Returns
        -------
        out : numpy.ndarray or scalar
            Samples from a Polya-Gamma distribution with parameters `h` & `z`.

        References
        ----------
        .. [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
               "Bayesian inference for logistic models using Pólya–Gamma latent
               variables." Journal of the American statistical Association
               108.504 (2013): 1339-1349.
        .. [2] Windle, Jesse, Nicholas G. Polson, and James G. Scott.
               "Sampling Polya-Gamma random variates: alternate and approximate
               techniques." arXiv preprint arXiv:1405.0506 (2014)

        Examples
        --------
        >>> from polyagamma import default_rng
        >>> rng = default_rng()
        >>> out = rng.polyagamma(size=5)

        """
        cdef size_t n
        cdef sampler_t stype = DEVROYE
        cdef bint ret_value = True

        cdef bitgen_t* bitgen = <bitgen_t*>PyCapsule_GetPointer(
            self._bit_generator.capsule, NAME
        )

        if h <= 0:
            raise ValueError("`h` must positive")

        if out is not None:
            n = out.shape[0]
            ret_value = False
        elif size:
            n = size
            out = cvarray(shape=(n,), itemsize=sizeof(double), format="d")
        else:
            n = 1

        if n > 1:
            with self._bit_generator.lock, nogil:
                pgm_random_polyagamma_fill(bitgen, h, z, stype, n, &out[0])
            if ret_value:
                return np.asarray(out.base)
        else:
            return pgm_random_polyagamma(bitgen, h, z, stype)


def default_rng(seed=None):
    return Generator(np.random.PCG64(seed))
