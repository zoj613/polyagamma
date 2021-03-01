# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""
Copyright (c) 2020-2021, Zolisa Bleki

SPDX-License-Identifier: BSD-3-Clause
"""
from cpython.float cimport PyFloat_Check
from cpython.int cimport PyInt_Check
from cpython.long cimport PyLong_Check
from cpython.number cimport PyNumber_Int
from cpython.object cimport PyObject_RichCompareBool, Py_LE, Py_LT, Py_NE
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.tuple cimport PyTuple_CheckExact, PyTuple_GET_SIZE, PyTuple_GET_ITEM
from numpy.random.bit_generator cimport BitGenerator, bitgen_t
cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "../include/pgm_random.h":
    ctypedef enum sampler_t:
        GAMMA
        DEVROYE
        ALTERNATE
        SADDLE
        HYBRID
    double pgm_random_polyagamma(bitgen_t* bg, double h, double z,
                                 sampler_t method) nogil
    void pgm_random_polyagamma_fill(bitgen_t* bg, double h, double z,
                                    sampler_t method, size_t n, double* out) nogil


cdef dict METHODS = {
    "gamma": GAMMA,
    "saddle": SADDLE,
    "devroye": DEVROYE,
    "alternate": ALTERNATE,
}


cdef inline bint params_are_numbers(object a, object b):
    return (PyFloat_Check(a) or PyLong_Check(a) or PyInt_Check(a)) and \
            (PyFloat_Check(b) or PyLong_Check(b) or PyInt_Check(b))


cdef inline bint is_a_number(object o):
    return PyFloat_Check(o) or PyLong_Check(o) or PyInt_Check(o)


cdef inline int check_method(object h, str method, bint disable_checks) except -1:
    cdef bint raise_error
    cdef object o

    if method not in METHODS:
        raise ValueError(f"`method` must be one of {set(METHODS)}")

    elif not disable_checks and method == "alternate":
        if is_a_number(h):
            raise_error = PyObject_RichCompareBool(h, 1, Py_LT)
        else:
            o = np.PyArray_FROM_O(h) < 1
            raise_error = np.PyArray_Any(o, np.NPY_MAXDIMS, <np.ndarray>NULL)
        if raise_error:
            raise ValueError("alternate method must have h >=1")

    elif not disable_checks and method == "devroye":
        if is_a_number(h):
            raise_error = PyObject_RichCompareBool(PyNumber_Int(h), h, Py_NE)
        else:
            o = np.PyArray_FROM_OT(h, np.NPY_INT) != np.PyArray_FROM_O(h)
            raise_error = np.PyArray_Any(o, np.NPY_MAXDIMS, <np.ndarray>NULL)
        if raise_error:
            raise ValueError("devroye method must have integer values for h")
    else:
        return METHODS[method]


def polyagamma(h=1, z=0, *, size=None, double[:] out=None, method=None,
               bint disable_checks=False, random_state=None):
    """
    polyagamma(h=1, z=0, *, size=None, out=None, method=None,
               disable_checks=False, random_state=None)

    Draw samples from a Polya-Gamma distribution.

    A sample is drawn from a Polya-Gamma distribution with specified shape
    (`h`) and tilting (`z`) parameters.

    Parameters
    ----------
    h : scalar or sequence, optional
        The shape parameter of the distribution as described in [1]_.
        The value(s) must be positive and finite. Defaults to 1.
    z : scalar or sequence, optional
        The exponential tilting parameter as described in [1]_. The value(s)
        must be finite. Defaults to 0.
    size : int or tuple of ints, optional
        The number of elements to draw from the distribution. If size is
        ``None`` (default) then a single value is returned. If a tuple of
        integers is passed, the returned array will have the same shape.
        This parameter only applies if `h` and `z` are scalars.
    out : array_like, optional
        1d array_like object in which to store samples. This object must
        implement the buffer protocol as described in [4]_. If given, then no
        value is returned. when `h` and/or `z` is a sequence, then `out` needs
        to have the same total size as the broadcasted result of the
        parameters.
    method : str or None, optional
        The method to use when sampling. If None (default) then a hybrid
        sampler is used that picks a method based on the value of `h`.
        A legal value must be one of {"gamma", "devroye", "alternate", "saddle"}.
        If the "alternate" method is used, then the value of `h` must be no
        less than 1. If the "devroye" method is used, the `h` must be a
        positive integer.
    disable_checks : bool, optional
        Whether to check that the `h` parameter contains only positive
        values(s). Disabling may give a performance gain, but may result
        in problems (crashes, non-termination, wrong return values)
        if the inputs do contain non-positive values.
    random_state : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A seed to initialize the random number generator. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
        Additionally, when passed a `BitGenerator`, it will be wrapped by
        `Generator`. If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    out : numpy.ndarray or scalar
        Samples from a Polya-Gamma distribution with parameters `h` & `z`.

    Notes
    -----
    To reduce overhead of creating a new generator instance every call to this
    function, it is recommended that the user pass an existing instance of
    ``numpy.random.Generator`` for the parameter `random_state`. This is
    especially important for reproducability of samples when calling this
    function repeatedly (e.g. as part of an MCMC algorithm).

    References
    ----------
    .. [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
           "Bayesian inference for logistic models using Pólya–Gamma latent
           variables." Journal of the American statistical Association
           108.504 (2013): 1339-1349.
    .. [2] Windle, Jesse, Nicholas G. Polson, and James G. Scott.
           "Sampling Polya-Gamma random variates: alternate and approximate
           techniques." arXiv preprint arXiv:1405.0506 (2014)
    .. [3] Luc Devroye. "On exact simulation algorithms for some distributions
           related to Jacobi theta functions." Statistics & Probability Letters,
           Volume 79, Issue 21, (2009): 2251-2259.
    .. [4] https://www.python.org/dev/peps/pep-3118/

    Examples
    --------
    >>> from polyagamma import polyagamma
    # outputs a 5 by 10 array of PG(1, 0) samples.
    >>> out = polyagamma(size=(5, 10))
    # broadcasting to generate 5 values from PG(1, 5), PG(2, 5),...,PG(5, 5)
    >>> a = [1, 2, 3, 4, 5]
    >>> polyagamma(a, 5)
    # using a specific method
    >>> out = polyagamma(method="devroye")
    # one can pass an existing instance of numpy.random.Generator as a parameter.
    >>> rng = np.random.default_rng(12345)
    >>> polyagamma(random_state=rng)
    # Output can be stored in an input array via the ``out`` parameter.
    >>> arr = np.empty(10)
    >>> polyagamma(size=10, out=arr)

    """
    # define an ``h`` value small enough to be regarded as a zero
    DEF zero = 1e-04

    cdef Py_ssize_t n, idx
    cdef np.broadcast bcast
    cdef double ch, cz
    cdef bint is_tuple
    cdef np.npy_intp dims
    cdef BitGenerator bitgenerator
    cdef bitgen_t* bitgen
    cdef sampler_t stype = HYBRID
    cdef bint has_out = True if out is not None else False

    bitgenerator = np.random.default_rng(random_state)._bit_generator
    bitgen = <bitgen_t*>PyCapsule_GetPointer(bitgenerator.capsule, "BitGenerator")

    if method is not None:
        stype = <sampler_t>check_method(h, method, disable_checks)

    if params_are_numbers(h, z):
        if not disable_checks and PyObject_RichCompareBool(h, zero, Py_LE):
            raise ValueError("`h` must be positive")
        elif has_out:
            n = out.shape[0]
            ch, cz = h, z
            with bitgenerator.lock, nogil:
                pgm_random_polyagamma_fill(bitgen, ch, cz, stype, n, &out[0])
            return
        elif size is not None:
            is_tuple = PyTuple_CheckExact(size)
            if is_tuple:
                total_size = 1
                for idx in range(PyTuple_GET_SIZE(size)):
                    total_size *= <object>PyTuple_GET_ITEM(size, idx)
                dims = <np.npy_intp>total_size
            else:
                dims = <np.npy_intp>size
            out = np.PyArray_EMPTY(1, &dims, np.NPY_DOUBLE, 0)
            n = <size_t>dims
            ch, cz = h, z
            with bitgenerator.lock, nogil:
                pgm_random_polyagamma_fill(bitgen, ch, cz, stype, n, &out[0])
            return np.PyArray_Reshape(out.base, size) if is_tuple else out.base
        else:
            ch, cz = h, z
            with bitgenerator.lock, nogil:
                cz = pgm_random_polyagamma(bitgen, ch, cz, stype)
            return cz

    else:
        h = np.PyArray_FROM_OT(h, np.NPY_DOUBLE)
        if not disable_checks and np.any(h <= zero):
            raise ValueError("values of `h` must be positive")

        z = np.PyArray_FROM_OT(z, np.NPY_DOUBLE)
        bcast = np.PyArray_MultiIterNew2(h, z)
        if has_out and out.shape[0] != bcast.size:
            raise ValueError(
                "`out` must have the same total size as the broadcasted "
                "result of `h` and `z`"
            )
        elif not has_out:
            dims = <np.npy_intp>(bcast.size)
            out = np.PyArray_EMPTY(1, &dims, np.NPY_DOUBLE, 0)

        n = out.shape[0]
        with bitgenerator.lock, nogil:
            for idx in range(n):
                ch = (<double*>np.PyArray_MultiIter_DATA(bcast, 0))[0]
                cz = (<double*>np.PyArray_MultiIter_DATA(bcast, 1))[0]
                out[idx] = pgm_random_polyagamma(bitgen, ch, cz, stype);
                np.PyArray_MultiIter_NEXT(bcast)

        if has_out:
            return
        else:
            return np.PyArray_Reshape(out.base, bcast.shape)
