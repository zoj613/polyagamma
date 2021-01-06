# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
from cpython.pycapsule cimport PyCapsule_GetPointer
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


cdef bint is_sequence(object x):
    cdef bint out
    try:
        iter(x)
        out = True
    except TypeError:
        out = False
    return out


class Generator(np.random.Generator):
    def polyagamma(self, h=1, z=0, size=None, double[:] out=None):
        """
        polyagamma(h=1, z=0, size=None, out=None)

        Draw samples from a Polya-Gamma distribution.

        Samples are draw from a Polya-Gamma distribution with specified para-
        meters `h` and `z`.

        Parameters
        ----------
        h : scalar or sequence, optional
            The `h` parameter as described in [1]_. The value(s) must be
            positive. Defaults to 1.
        z : scalar or sequence, optional
            The exponential tilting parameter as described in [1]_.
            Defaults to 0.
        size : int or tuple of ints, optional
            The number of elements to draw from the distribution. If size is
            ``None`` (default) then a single value is returned. If a tuple of
            integers is passed, the returned array will have the same shape.
            This parameter only applies if `h` and `z` are scalars.
        out : numpy.ndarray, optional
            1d output array in which to store samples. If given, then no value
            is returned. when `h` and/or `z` is a sequence, then `out` needs
            to have the same total size as the broadcasted result of the
            parameters.

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
        # outputs a 5 by 10 array of PG(1, 0) samples.
        >>> out = rng.polyagamma(size=(5, 10))
        # broadcasting to generate 5 values from PG(1, 5), PG(2, 5),...,PG(5, 5)
        >>> a = [1, 2, 3, 4, 5]
        >>> rng.polyagamma(a, 5)

        """
        # define an ``h`` value small enough to be regarded as a zero
        DEF zero = 1e-04

        cdef size_t n, idx
        cdef object bcast
        cdef double ch, cz
        cdef sampler_t stype = DEVROYE
        cdef object has_out = True if out is not None else False

        cdef bitgen_t* bitgen = <bitgen_t*>PyCapsule_GetPointer(
            self._bit_generator.capsule, "BitGenerator"
        )

        if is_sequence(h) or is_sequence(z):
            # TODO: Maybe use numpy's C-API for the broadcasting and iteration?
            # Would readability take a hit?
            bcast = np.broadcast(h, z)
            if has_out and out.size != bcast.size:
                raise ValueError(
                    "`out` must have the same total size as the broadcasted "
                    "result of `h` and `z`"
                )
            elif not has_out:
                out = np.empty(bcast.size)
            hvals, _ = bcast.iters
            if np.any([i <= zero for i in hvals]):
                raise ValueError("values of `h` must be positive")
            bcast.reset()
            for idx, iter_set in enumerate(bcast):
                ch, cz = iter_set
                with self._bit_generator.lock, nogil:
                    out[idx] = pgm_random_polyagamma(bitgen, ch, cz, stype);
            if not has_out:
                return out.base.reshape(bcast.shape)

        elif h <= zero:
            raise ValueError("`h` must positive")

        elif has_out:
            n = out.size
            ch, cz = h, z
            with self._bit_generator.lock, nogil:
                pgm_random_polyagamma_fill(bitgen, ch, cz, stype, n, &out[0])

        elif size:
            npy_arr = np.empty(size)
            out = npy_arr.ravel() if isinstance(size, tuple) else npy_arr
            n = out.size
            ch, cz = h, z
            with self._bit_generator.lock, nogil:
                pgm_random_polyagamma_fill(bitgen, ch, cz, stype, n, &out[0])
            return out.base.reshape(size)

        else:
            ch, cz = h, z
            with self._bit_generator.lock, nogil:
                cz = pgm_random_polyagamma(bitgen, ch, cz, stype)
            return cz


def default_rng(seed=None):
    return Generator(np.random.PCG64(seed))
