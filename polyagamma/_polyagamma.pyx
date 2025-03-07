# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Copyright (c) 2020-2023, Zolisa Bleki

SPDX-License-Identifier: BSD-3-Clause
"""
from cpython.float cimport PyFloat_Check
from cpython.long cimport PyLong_Check
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stdlib cimport free
from numpy.random.bit_generator cimport BitGenerator, bitgen_t
cimport numpy as np
from numpy.random import default_rng

np.import_array()


def rvs(h=1., z=0., *, size=None, out=None, method=None,
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
        An array_like object in which to store samples. This object must
        implement the buffer protocol as described in [4]_ or the array
        protocol as described in [5]_. This object's elements must be of
        floating type, contiguous and aligned. When `h` and/or `z` is a sequence,
        then `out` needs to have the same total size as the broadcasted result
        of the parameters. If both this and the `size` parameter are set, `size`
        is preferred while this parameter is ignored.

        .. versionchanged:: 1.3.2
           If set with `size` when `h` and `z` are not both scalars, this input
           is ignored in favor of `size`.

        .. versionchanged:: 1.3.7
           2D or more dimensional array objects are now accepted as values for
           this parameter.

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

    random_state : {int, array_like, SeedSequence, BitGenerator, Generator}, optional
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
    if method not in METHODS:
        raise ValueError(f"`method` must be one of {set(METHODS)}")

    cdef:
        void* data
        double ch, cz
        np.npy_intp n, ndim
        np.PyArray_Dims shape
        bint has_out = out is not None
        bint has_size = size is not None
        sampler_t sampler = <sampler_t>METHODS[method]
        BitGenerator bitgenerator = <BitGenerator>default_rng(random_state).bit_generator
        bitgen_t* bitgen = <bitgen_t*>PyCapsule_GetPointer(bitgenerator.capsule,
                                                           BITGEN_NAME)

    # We special-case when input params are scalars for maximum efficiency
    if is_number(h) and is_number(z):
        ch, cz = h, z
        if not disable_checks and (ch <= ZERO or (sampler == DEVROYE and ch != <int>ch)):
            raise ValueError(PARAM_CHECK_ERROR_MSG)

        if has_size:
            np.PyArray_IntpConverter(size, &shape)
            res = np.PyArray_EMPTY(shape.len, shape.ptr, np.NPY_DOUBLE, 0)
            free(shape.ptr)
            with bitgenerator.lock, nogil:
                ndim = shape.len
                data = np.PyArray_DATA(<np.ndarray[np.double_t, ndim=ndim]>res)
                n = np.PyArray_SIZE(<np.ndarray[np.double_t, ndim=ndim]>res)
                random_polyagamma_fill(bitgen, ch, cz, sampler, <size_t>n, <double*>data)
            return res
        elif has_out:
            # We use the WRITEBACKIFCOPY flag to wrap `out` incase the data is not
            # double precision float. Once we are done, this flag ensures the
            # sampled double precision data is copied back into the original data
            # of `out` and cast to its original type.
            res = np.PyArray_FROM_OTF(out, np.NPY_DOUBLE, np.NPY_ARRAY_WRITEBACKIFCOPY)
            with bitgenerator.lock, nogil:
                data = np.PyArray_DATA(<np.ndarray[np.double_t]>res)
                n = np.PyArray_SIZE(<np.ndarray[np.double_t]>res)
                random_polyagamma_fill(bitgen, ch, cz, sampler, <size_t>n, <double*>data)
            PyArray_ResolveWritebackIfCopy(<np.PyArrayObject*>res)
            return out
        else:
            with bitgenerator.lock:
                cz = random_polyagamma(bitgen, ch, cz, sampler)
            return cz

    harray = np.PyArray_FROM_OTF(h, np.NPY_DOUBLE, np.NPY_ARRAY_CARRAY_RO)

    if not disable_checks and any_nonpositive(<np.ndarray[np.double_t]>harray, sampler):
        raise ValueError(PARAM_CHECK_ERROR_MSG)

    zarray = np.PyArray_FROM_OT(z, np.NPY_DOUBLE)

    cdef:
        char** ptr
        np.NpyIter* it
        np.npy_intp* innersizeptr
        np.NpyIter_IterNextFunc iternext
        int** op_axes = [NULL, NULL, NULL]
        np.PyArray_Descr** op_dtypes = NULL
        np.PyArrayObject** ops = [<np.PyArrayObject*>harray,
                                  <np.PyArrayObject*>zarray, NULL]
        np.npy_uint32* op_flags = [np.NPY_ITER_READONLY | np.NPY_ITER_CONTIG,
                                   np.NPY_ITER_READONLY | np.NPY_ITER_CONTIG, 0]
        # We handle iteration of the innermost loop of the broadcasted contents
        # manually for efficiency using the EXTERNAL_LOOP flag, and thus get
        # an array of the loop contents instead of one-at-a-time. We use the
        # BUFFERED flag to satisfy data type, alignment, and byte-order requirements
        # as well as to get larger chunks of the inner loop when used with EXTERNAL_LOOP.
        # When buffering is enabled, the GROWINNER flag allows the size of the
        # inner loop to grow when buffering isn’t necessary, enabling a single
        # passthrough of all the data in some cases.
        np.npy_uint32 it_flags = (np.NPY_ITER_ZEROSIZE_OK | np.NPY_ITER_EXTERNAL_LOOP |
                                  np.NPY_ITER_BUFFERED | np.NPY_ITER_GROWINNER)

    if has_size:
        np.PyArray_IntpConverter(size, &shape)
        op_flags[2] = np.NPY_ITER_WRITEONLY | np.NPY_ITER_ALLOCATE | np.NPY_ITER_CONTIG
        it = np.NpyIter_AdvancedNew(3, ops, it_flags, np.NPY_KEEPORDER, np.NPY_NO_CASTING,
                                    op_flags, op_dtypes, shape.len, op_axes, shape.ptr, 0)
        free(shape.ptr)
    elif has_out:
        npy_arr = np.PyArray_FROM_OTF(out, np.NPY_DOUBLE, np.NPY_ARRAY_WRITEBACKIFCOPY)
        ops[2] = <np.PyArrayObject*>npy_arr
        op_flags[2] = (np.NPY_ITER_WRITEONLY | np.NPY_ITER_CONTIG |
                       np.NPY_ITER_NO_BROADCAST)
        it = np.NpyIter_MultiNew(3, ops, it_flags, np.NPY_KEEPORDER,
                                 np.NPY_NO_CASTING, op_flags, op_dtypes)
    else:
        op_flags[2] = np.NPY_ITER_WRITEONLY | np.NPY_ITER_ALLOCATE | np.NPY_ITER_CONTIG
        it = np.NpyIter_MultiNew(3, ops, it_flags, np.NPY_KEEPORDER,
                                 np.NPY_NO_CASTING, op_flags, op_dtypes)

    try:
        iternext = np.NpyIter_GetIterNext(it, NULL)
        with bitgenerator.lock, nogil:
            ptr = np.NpyIter_GetDataPtrArray(it)
            innersizeptr = np.NpyIter_GetInnerLoopSizePtr(it)
            while True:
                random_polyagamma_fill2(bitgen, <double*>ptr[0], <double*>ptr[1], sampler,
                                        <size_t>innersizeptr[0], <double*>ptr[2])
                if not iternext(it):
                    break
        if has_out:
            PyArray_ResolveWritebackIfCopy(ops[2])
        return <object>np.NpyIter_GetOperandArray(it)[2]
    finally:
        np.NpyIter_Deallocate(it)  # pragma: no cover


def pdf(x, h=1., z=0., bint return_log=False):
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
    cdef dist_func f = polyagamma_logpdf if return_log else polyagamma_pdf
    return dispatch(f, x, h, z)


def cdf(x, h=1., z=0., bint return_log=False):
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
    cdef dist_func f = polyagamma_logcdf if return_log else polyagamma_cdf
    return dispatch(f, x, h, z)


cdef extern from "numpy/ndarrayobject.h":
    # currently not available in numpy's cython declarations so we include manually
    int PyArray_ResolveWritebackIfCopy(np.PyArrayObject* obj)


cdef dict METHODS = {
    None: HYBRID,
    "gamma": GAMMA,
    "saddle": SADDLE,
    "devroye": DEVROYE,
    "alternate": ALTERNATE,
}
cdef double ZERO = 1e-04  # h value small enough to be regarded as 0
cdef const char* BITGEN_NAME = "BitGenerator"
cdef const char* PARAM_CHECK_ERROR_MSG = \
    "Values of `h` must be positive, and also integer if `method` is devroye."
ctypedef double (*dist_func)(double x, double h, double z) noexcept nogil


cdef inline bint is_number(object o) noexcept:
    """Check whether an object is a python float/int."""
    return PyFloat_Check(o) or PyLong_Check(o)


cdef inline bint any_nonpositive(np.ndarray h, sampler_t method) noexcept:
    """Verify if the shape paramemter is non-positive given a sampling method."""
    cdef np.npy_intp i, size
    cdef double* ch = <double*>np.PyArray_DATA(h)

    if np.PyArray_IsZeroDim(<object>h):
        return ch[0] <= ZERO or (method == DEVROYE and ch[0] != <size_t>ch[0])

    size = np.PyArray_SIZE(h)

    if method == DEVROYE:
        for i in range(size):
            if ch[i] <= ZERO or ch[i] != <size_t>ch[i]:
                return True
        return False

    for i in range(size):
        if ch[i] <= ZERO:
            return True
    return False


cdef inline object dispatch(dist_func fn, object x, object h, object z):
    """Apply any function f with signature f(x,h,z) to arguments `x`, `h` and `z`."""
    cdef double cx, ch, cz

    if is_number(x) and is_number(h) and is_number(z):
        cx, ch, cz = x, h, z
        with nogil:
            cx = fn(cx, ch, cz)
        return  cx

    xarray = np.PyArray_FROM_OT(x, np.NPY_DOUBLE)
    harray = np.PyArray_FROM_OT(h, np.NPY_DOUBLE)
    zarray = np.PyArray_FROM_OT(z, np.NPY_DOUBLE)

    cdef:
        char** ptr
        np.NpyIter_IterNextFunc iternext
        np.PyArray_Descr** op_dtypes = NULL
        np.npy_uint32 it_flags = np.NPY_ITER_ZEROSIZE_OK
        np.npy_uint32* op_flags = [np.NPY_ITER_READONLY,
                                   np.NPY_ITER_READONLY,
                                   np.NPY_ITER_READONLY,
                                   np.NPY_ITER_WRITEONLY | np.NPY_ITER_ALLOCATE]
        np.PyArrayObject** ops = [<np.PyArrayObject*>xarray, <np.PyArrayObject*>harray,
                                  <np.PyArrayObject*>zarray, NULL]
        np.NpyIter* it = np.NpyIter_MultiNew(4, ops, it_flags, np.NPY_KEEPORDER,
                                             np.NPY_NO_CASTING, op_flags, op_dtypes)

    try:
        iternext = np.NpyIter_GetIterNext(it, NULL)
        with nogil:
            ptr = np.NpyIter_GetDataPtrArray(it)
            while True:
                (<double*>ptr[3])[0] = fn((<double*>ptr[0])[0], (<double*>ptr[1])[0],
                                          (<double*>ptr[2])[0])
                if not iternext(it):
                    break
        return <object>np.NpyIter_GetOperandArray(it)[3]
    finally:
        np.NpyIter_Deallocate(it)  # pragma: no cover
