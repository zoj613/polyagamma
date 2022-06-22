# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""
Copyright (c) 2020-2021, Zolisa Bleki

SPDX-License-Identifier: BSD-3-Clause
"""
from cpython.exc cimport PyErr_Clear
from cpython.float cimport PyFloat_Check
from cpython.long cimport PyLong_Check
from cpython.number cimport PyNumber_Long
from cpython.object cimport PyObject_RichCompareBool, Py_LE, Py_LT, Py_NE
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stdlib cimport free
from numpy.random.bit_generator cimport BitGenerator, bitgen_t
cimport numpy as np
from numpy.random import default_rng

np.import_array()

cdef extern from "pgm_random.h" nogil:
    double pgm_random_polyagamma(bitgen_t* bitgen_state, double h,
                                 double z, sampler_t method)
    void pgm_random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                                    sampler_t method, size_t n, double* out)
    void pgm_random_polyagamma_fill2(bitgen_t* bitgen_state, const double* h,
                                     const double* z, sampler_t method,
                                     size_t n, double* out)

# Cython-level function definitions to be shared with other cython modules
cdef inline double random_polyagamma(bitgen_t* bitgen_state, double h, double z,
                                     sampler_t method) nogil:
    return pgm_random_polyagamma(bitgen_state, h, z, method)


cdef inline void random_polyagamma_fill(bitgen_t* bitgen_state, double h, double z,
                                        sampler_t method, size_t n, double* out) nogil:
    pgm_random_polyagamma_fill(bitgen_state, h, z, method, n, out)


cdef inline void random_polyagamma_fill2(bitgen_t* bitgen_state, const double* h,
                                         const double* z, sampler_t method,
                                         size_t n, double* out) nogil:
    pgm_random_polyagamma_fill2(bitgen_state, h, z, method, n, out)


# python-level functions and helpers below

cdef dict METHODS = {
    "gamma": GAMMA,
    "saddle": SADDLE,
    "devroye": DEVROYE,
    "alternate": ALTERNATE,
}


cdef const char* BITGEN_NAME = "BitGenerator"


cdef inline bint is_a_number(object o):
    return PyFloat_Check(o) or PyLong_Check(o)


cdef inline bint params_are_scalars(object a, object b):
    return is_a_number(a) and is_a_number(b)


cdef inline int check_method(object h, str method, bint disable_checks) except -1:
    cdef bint raise_error
    cdef object o

    if method not in METHODS:
        raise ValueError(f"`method` must be one of {set(METHODS)}")

    if not disable_checks and method == "devroye":
        if is_a_number(h):
            raise_error = PyObject_RichCompareBool(PyNumber_Long(h), h, Py_NE)
        else:
            o = np.PyArray_FROM_OTF(h, np.NPY_LONG, np.NPY_ARRAY_FORCECAST) != np.PyArray_FROM_O(h)
            raise_error = any(np.PyArray_Ravel(o, np.NPY_CORDER))
        if raise_error:
            raise ValueError("devroye method must have integer values for h")
        return DEVROYE

    return METHODS[method]

# This has to be included seperately to function as expected.
# See: https://github.com/numpy/numpy/issues/19291
cdef extern from "numpy/ndarrayobject.h":
    int PyArray_IntpConverter(object size, np.PyArray_Dims* shape) except 0


cdef inline object _polyagamma_shape_broadcasted(bitgen_t* bitgen, object h, object z,
                                                 sampler_t stype, np.PyArray_Dims shape,
                                                 object lock):
    cdef np.flatiter h_iter, z_iter
    cdef double* arr_ptr
    cdef double ch, cz

    h_iter = np.PyArray_BroadcastToShape(h, shape.ptr, shape.len)
    z_iter = np.PyArray_BroadcastToShape(z, shape.ptr, shape.len)
    arr = np.PyArray_EMPTY(shape.len, shape.ptr, np.NPY_DOUBLE, 0)
    arr_ptr = <double*>np.PyArray_DATA(arr)
    free(shape.ptr)

    with lock, nogil:
        while np.PyArray_ITER_NOTDONE(h_iter):
            ch = (<double*>np.PyArray_ITER_DATA(h_iter))[0]
            cz = (<double*>np.PyArray_ITER_DATA(z_iter))[0]
            arr_ptr[0] = pgm_random_polyagamma(bitgen, ch, cz, stype)
            np.PyArray_ITER_NEXT(h_iter)
            np.PyArray_ITER_NEXT(z_iter)
            arr_ptr += 1
    return arr

# intentionally named `polyagamma` instead of `random_polyagamma` in this file
# to avoid name clashing with the cython function of the same name.
def polyagamma(h=1., z=0., *, size=None, double[:] out=None, method=None,
               bint disable_checks=False, random_state=None):
    """
    random_polyagamma(h=1., z=0., *, size=None, out=None, method=None,
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
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``h`` and ``z`` are both scalars.
        Otherwise, ``np.broadcast(h, z).size`` samples are drawn.

        .. versionchanged:: 1.3.2
           Can now handle a `size` argument when `h` and `z` are not both scalars.
           Also raises an exception when it contains non-integer values.

    out : array_like, optional
        1d array_like object in which to store samples. This object must
        implement the buffer protocol as described in [4]_ or the array
        protocol as described in [5]_. This object's elements must be of 64bit
        float type, C-contiguous and aligned. If given, then no value is
        returned. when `h` and/or `z` is a sequence, then `out` needs to have
        the same total size as the broadcasted result of the parameters. If
        both this and the `size` parameter are set when `h` and `z` are not
        both scalars, `size` is preferred while this parameter is ignored.

        .. versionchanged:: 1.3.2
           If set with `size` when `h` and `z` are not both scalars, this input
           is ignored in favor of `size`.

    method : str or None, optional
        The method to use when sampling. If None (default) then a hybrid
        sampler is used that picks the most efficient method based on the value
        of `h`. A legal value must be one of {"gamma", "devroye", "alternate", "saddle"}.
        - "gamma" method generates a sample using a convolution of gamma random
          variates.
        - "devroye" method generates a sample using an accept-rejection scheme
          introduced by [3]_.
        - "alternate" method is an accept-rejection scheme that addresses the
          inefficiencies of the "devroye" method when `h` is greater than 1.
        - "saddle" method uses a saddle point approximation of the target
          distribution's density as an envelope in order to speed up the
          accept-rejection scheme. It is mainly suitable for large values of `h`
          (e.g. h > 20).
        If the "devroye" method is used, the `h` must be a positive integer.
    disable_checks : bool, optional
        Whether to check that the `h` parameter contains only positive
        values(s). If ``h <= 0``, then return ``NaN`` for corresponding output
        values. Disabling _may_ give a performance gain if `h` is a very large array.

        .. versionchanged:: 1.3.4
           Now explicitly returns NaN if `disable_checks=True` and `h` is not
           positive.

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
    .. [5] https://numpy.org/doc/stable/reference/arrays.interface.html

    Examples
    --------
    >>> from polyagamma import random_polyagamma
    # outputs a 5 by 10 array of PG(1, 0) samples.
    >>> out = random_polyagamma(size=(5, 10))
    # broadcasting to generate 5 values from PG(1, 5), PG(2, 5),...,PG(5, 5)
    >>> a = [1, 2, 3, 4, 5]
    >>> random_polyagamma(a, 5)
    # using a specific method
    >>> out = random_polyagamma(method="devroye")
    # one can pass an existing instance of numpy.random.Generator as a parameter.
    >>> rng = np.random.default_rng(12345)
    >>> random_polyagamma(random_state=rng)
    # Output can be stored in an input array via the ``out`` parameter.
    >>> arr = np.empty(10)
    >>> random_polyagamma(size=10, out=arr)

    """
    # define an ``h`` value small enough to be regarded as a zero
    DEF zero = 1e-04

    cdef np.broadcast bcast
    cdef double ch, cz
    cdef double[:] ah, az
    cdef double* arr_ptr
    cdef np.PyArray_Dims shape
    cdef np.npy_intp arr_len
    cdef BitGenerator bitgenerator
    cdef bitgen_t* bitgen
    cdef sampler_t stype = HYBRID
    cdef bint has_out = True if out is not None else False

    bitgenerator = <BitGenerator>(default_rng(random_state)._bit_generator)
    bitgen = <bitgen_t*>PyCapsule_GetPointer(bitgenerator.capsule, BITGEN_NAME)

    if method is not None:
        stype = <sampler_t>check_method(h, method, disable_checks)

    if params_are_scalars(h, z):
        if not disable_checks and PyObject_RichCompareBool(h, zero, Py_LE):
            raise ValueError("`h` must be positive")
        ch, cz = h, z
        if not has_out and size is None:
            with bitgenerator.lock, nogil:
                cz = random_polyagamma(bitgen, ch, cz, stype)
            return cz
        elif has_out:
            with bitgenerator.lock, nogil:
                random_polyagamma_fill(bitgen, ch, cz, stype, out.shape[0], &out[0])
            return
        else:
            PyArray_IntpConverter(size, &shape)
            arr = np.PyArray_EMPTY(shape.len, shape.ptr, np.NPY_DOUBLE, 0)
            free(shape.ptr)
            arr_len = np.PyArray_SIZE(arr)
            arr_ptr = <double*>np.PyArray_DATA(arr)
            with bitgenerator.lock, nogil:
                pgm_random_polyagamma_fill(bitgen, ch, cz, stype, arr_len, arr_ptr)
            return arr

    h = np.PyArray_FROM_OT(h, np.NPY_DOUBLE)
    if not disable_checks and any(np.PyArray_Ravel(np.PyArray_FROM_O(h <= zero),
                                                   np.NPY_CORDER)):
        raise ValueError("values of `h` must be positive")
    z = np.PyArray_FROM_OT(z, np.NPY_DOUBLE)

    # handle cases where the user also passes a size argument value
    if size is not None:
        PyArray_IntpConverter(size, &shape)
        return _polyagamma_shape_broadcasted(bitgen, h, z, stype, shape, bitgenerator.lock)

    elif np.PyArray_NDIM(<np.ndarray>h) == np.PyArray_NDIM(<np.ndarray>z) == 1:
        ah, az = h, z
        if has_out and not (out.shape[0] == ah.shape[0] == az.shape[0]):
            raise IndexError("`out` must have the same length as parameters")
        elif not has_out:
            out = np.PyArray_EMPTY(1, <np.npy_intp*>ah.shape, np.NPY_DOUBLE, 0)
        with bitgenerator.lock, nogil:
            random_polyagamma_fill2(bitgen, &ah[0], &az[0], stype, out.shape[0], &out[0])
        if has_out:
            return
        else:
            return out.base

    else:
        bcast = np.PyArray_MultiIterNew2(h, z)
        if has_out and out.shape[0] != bcast.size:
            raise ValueError(
                "`out` must have the same total size as the broadcasted "
                "result of `h` and `z`"
            )
        elif has_out:
            arr_ptr = &out[0]
        else:
            arr = np.PyArray_EMPTY(bcast.nd, bcast.dimensions, np.NPY_DOUBLE, 0)
            arr_ptr = <double*>np.PyArray_DATA(arr)

        with bitgenerator.lock, nogil:
            while bcast.index < bcast.size:
                ch = (<double*>np.PyArray_MultiIter_DATA(bcast, 0))[0]
                cz = (<double*>np.PyArray_MultiIter_DATA(bcast, 1))[0]
                arr_ptr[bcast.index] = pgm_random_polyagamma(bitgen, ch, cz, stype)
                np.PyArray_MultiIter_NEXT(bcast)

        if has_out:
            return
        else:
            return arr


cdef extern from "pgm_density.h" nogil:
    double pgm_polyagamma_logpdf(double x, double h, double z)
    double pgm_polyagamma_logcdf(double x, double h, double z)
    double pgm_polyagamma_pdf(double x, double h, double z)
    double pgm_polyagamma_cdf(double x, double h, double z)


ctypedef double (*dist_func)(double x, double h, double z) nogil


cdef object dispatch(dist_func f, object x, object h, object z):
    cdef double cx, ch, cz

    if is_a_number(x) and is_a_number(h) and is_a_number(z):
        cx, ch, cz = x, h, z
        with nogil:
            cx = f(cx, ch, cz)
        return cx

    x = np.PyArray_FROM_OT(x, np.NPY_DOUBLE)
    h = np.PyArray_FROM_OT(h, np.NPY_DOUBLE)
    z = np.PyArray_FROM_OT(z, np.NPY_DOUBLE)
    cdef np.broadcast bcast = np.PyArray_MultiIterNew3(x, h, z)
    arr = np.PyArray_EMPTY(bcast.nd, bcast.dimensions, np.NPY_DOUBLE, 0)
    cdef double* arr_ptr = <double*>np.PyArray_DATA(arr)

    with nogil:
        while bcast.index < bcast.size:
            cx = (<double*>np.PyArray_MultiIter_DATA(bcast, 0))[0]
            ch = (<double*>np.PyArray_MultiIter_DATA(bcast, 1))[0]
            cz = (<double*>np.PyArray_MultiIter_DATA(bcast, 2))[0]
            arr_ptr[bcast.index] = f(cx, ch, cz)
            np.PyArray_MultiIter_NEXT(bcast)

    return arr


def polyagamma_pdf(x, h=1., z=0., bint return_log=False):
    """
    polyagamma_pdf(x, h=1., z=0., return_log=False)

    Calculate the density of PG(h, z) at `x`.

    Parameters
    ----------
    x : scalar or sequence
        The value(s) at which the function is evaluated.
    h : scalar or sequence, optional
        The shape parameter of the distribution as described in [1]_.
        The value(s) must be positive and finite. Defaults to 1.
    z : scalar or sequence, optional
        The exponential tilting parameter as described in [1]_. The value(s)
        must be finite. Defaults to 0.
    return_log : bool, optional
        Whether to return the logarithm of the density. This option uses a
        specialized function to accurately compute the log of the pdf and thus
        is better than explicitly calling ``log()`` on the result since this
        won't suffer from underflow if the result is nearly-zero. Defaults to
        False.

    Returns
    -------
    out : numpy.ndarray or scalar
        Value of the density function at `x` of PG(`h`, `z`).

    Notes
    -----

    .. versionadded:: 1.3.0

    This function implements the density function as shown in page 6 of [1]_.
    The infinite sum is truncated at a maximum of 200 terms. Convergence of
    the series is tested after each term is calculated so that if the
    successive terms are equal up to machine epsilon, the calculation is
    terminated early.

    References
    ----------
    .. [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
           "Bayesian inference for logistic models using Pólya–Gamma latent
           variables." Journal of the American statistical Association
           108.504 (2013): 1339-1349.

    Examples
    --------
    >>> from polyagamma import polyagamma_pdf
    >>> import numpy as np
    >>> polyagamma_pdf(0.2)
    2.339176537265802
    >>> polyagamma_pdf(3, h=6, z=1)
    0.012537773310487178
    >>> x = np.linspace(0.01, 0.5, 5)
    >>> polyagamma_pdf(x)
    array([1.48671951e-03, 3.21504108e+00, 1.78492273e+00, 9.75298427e-01,
           5.32845353e-01])
    >>> polyagamma_pdf(0.75, h=[4, 3, 1])
    array([1.08247188, 1.1022302 , 0.15517146])
    >>> polyagamma_pdf(0.75, h=[4, 3, 1], return_log=True)
    array([ 0.07924721,  0.09733558, -1.86322458])

    """
    cdef dist_func f = pgm_polyagamma_logpdf if return_log else pgm_polyagamma_pdf
    return dispatch(f, x, h, z)


def polyagamma_cdf(x, h=1., z=0., bint return_log=False):
    """
    polyagamma_cdf(x, h=1., z=0., return_log=False)

    Calculate the cumulative distribution function of PG(h, z) at `x`.

    Parameters
    ----------
    x : scalar or sequence
        The value(s) at which the function is evaluated.
    h : scalar or sequence, optional
        The shape parameter of the distribution as described in [1]_.
        The value(s) must be positive and finite. Defaults to 1.
    z : scalar or sequence, optional
        The exponential tilting parameter as described in [1]_. The value(s)
        must be finite. Defaults to 0.
    return_log : bool, optional
        Whether to return the logarithm of the CDF. This option uses a
        specialized function to accurately compute the log of the CDF and thus
        is better than explicitly calling ``log()`` on the result since this
        won't suffer from underflow if the result is nearly-zero. Defaults to
        False.

    Notes
    -----

    .. versionadded:: 1.3.0

    This function implements the distribution function as shown in page 6 of
    [1]_. The infinite sum is truncated at a maximum of 200 terms.

    References
    ----------
    .. [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
           "Bayesian inference for logistic models using Pólya–Gamma latent
           variables." Journal of the American statistical Association
           108.504 (2013): 1339-1349.

    Examples
    --------
    >>> from polyagamma import polyagamma_cdf
    >>> import numpy as np
    >>> polyagamma_cdf(0.2)
    0.525512539764972
    >>> polyagamma_cdf(3, h=6, z=1)
    0.9966435679024789
    >>> x = np.linspace(0.01, 1, 5)
    >>> polyagamma_cdf(x)
    array([1.14660629e-06, 6.42692997e-01, 8.94654581e-01, 9.68941224e-01,
           9.90843160e-01])
    >>> polyagamma_cdf(0.75, h=[4, 3, 1])
    array([0.30130807, 0.57523169, 0.96855568])
    >>> polyagamma_cdf(0.75, h=[4, 3, 1], return_log=True)
    array([-1.19962205, -0.55298237, -0.0319493 ])

    """
    cdef dist_func f = pgm_polyagamma_logcdf if return_log else pgm_polyagamma_cdf
    return dispatch(f, x, h, z)
