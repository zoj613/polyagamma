import functools

import numpy as np
import pytest

from polyagamma import (
    random_polyagamma,
    polyagamma_pdf,
    polyagamma_cdf,
)


def test_polyagamma():
    rng = np.random.default_rng(1)
    rng_polyagamma = functools.partial(random_polyagamma, random_state=rng)

    assert len(rng_polyagamma(size=5)) == 5
    assert rng_polyagamma(size=(5, 2)).shape == (5, 2)
    # test if non-integer elements of size get truncated to ints
    assert rng_polyagamma(size=(5.1, 2.9)).shape == (5, 2)
    assert rng_polyagamma(size=10.4).shape == (10,)

    h = [[[0.59103028], [0.15228518], [0.53494081], [0.85875483], [0.05796053]],
         [[0.63934113], [0.1341983 ], [0.73854957], [0.76351291], [0.38431413]],
         [[0.90747263], [0.19601777], [0.04178972], [0.08220703], [0.55739387]],
         [[0.83603371], [0.04006968], [0.36563885], [0.96736881], [0.61390783]]]
    z = [-8.2654694, 4.72289344, -1.07163445, 8.00492261]
    assert rng_polyagamma(h, z).shape == (4, 5, 4)
    # test if integer arrays dont crash things
    z = np.array([1, 2, 3, 4, 6])
    assert rng_polyagamma(h, z).shape == (4, 5, 5)
    assert rng_polyagamma(h, 0.12345).shape == (4, 5, 1)

    # should work on list/tuple input
    h = [[1000, 2, 3], [3, 3, 1]]
    out = rng_polyagamma(h)
    assert out.shape == (2, 3)
    assert not np.allclose(out, 0)
    z = (1000, 2, -10, 0)
    out = rng_polyagamma(z=z)
    assert out.shape == (4,)
    assert not np.allclose(out, 0)

    # test for when h and z are sequences of dimension 1 and equal length
    h = (1, 2, 3, 4, 5, 6)
    z = np.ones(6)
    out = rng_polyagamma(h, z)
    assert out.shape == (6,)
    out = np.zeros(6)
    rng_polyagamma(h, z, out=out)
    assert not np.allclose(out, 0)
    with pytest.raises(IndexError, match="must have the same length as parameters"):
        rng_polyagamma(h, z, out=out[1:])

    # tests for when one of the params is a sequence and the other is not
    out = np.array([0., 0., 0., 0., 0.])
    rng_polyagamma(out=out)
    assert not np.allclose(out, 0)
    out2 = np.array([0., 0., 0., 0., 0., 0.])
    rng_polyagamma(h, out=out2)
    assert not np.allclose(out2, 0)
    # test size of output array when a parameter is a sequence
    with pytest.raises(ValueError, match="`out` must have the same total"):
        rng_polyagamma(h, out=out)

    # raise an error when output array with dim > 1 is passed as an arg
    with pytest.raises(ValueError):
        rng_polyagamma(out=np.empty((2, 1)))

    # h must be greater than 0
    with pytest.raises(ValueError, match="`h` must be positive"):
        rng_polyagamma(0)
    with pytest.raises(ValueError, match="`h` must be positive"):
        rng_polyagamma(-1.3232)
    with pytest.raises(ValueError, match="`h` must be positive"):
        h = [1, 2, 3, 0.00001]
        rng_polyagamma(h)

    # should work for negative values of z
    rng_polyagamma(z=-10.5)

    # raise error on unknown method names
    with pytest.raises(ValueError, match="`method` must be one of"):
        rng_polyagamma(method="unknown method")
    # should work for all supported methods
    rng_polyagamma(method="gamma")
    rng_polyagamma(method="devroye")
    rng_polyagamma(method="alternate")
    rng_polyagamma(method="saddle")
    # test if sampling works for sequence input when devroye and alternate
    # methods are specified. See Issue #32
    h = (1, 2, 3)
    rng_polyagamma(h, method="devroye")
    rng_polyagamma(h, method="alternate")

    # raise an error when using devroye with non-integer values of h
    with pytest.raises(ValueError):
        rng_polyagamma(2.0000000001, method="devroye")
    # should work for whole numbers: 2.000000 == 2
    rng_polyagamma(2.0000000000, method="devroye")

    # raise error when passed a non-keyword arg after the first 2
    with pytest.raises(TypeError, match="takes at most 2 positional arguments"):
        rng_polyagamma(1, 0, 5)

    # don't raise error when passed non-integer h values if checks are disabled
    # when using the devroye method.
    rng_polyagamma(1.5, method="devroye", disable_checks=True)

    # test for reproducibility via random_state
    rng = np.random.default_rng(12345)
    expected = random_polyagamma(size=5, random_state=rng)
    rng2 = np.random.default_rng(12345)
    assert np.allclose(expected, random_polyagamma(size=5, random_state=rng2))
    assert not np.allclose(expected, random_polyagamma(size=5))

# "devroye" is not included because it does not play well with non-integer h
@pytest.mark.parametrize("method", ("alternate", "saddle", "gamma"))
@pytest.mark.parametrize("h", (0.5, 1, 4, 7, 15, 25))
@pytest.mark.parametrize("z", (0, 1, -4, 7, -15, 25))
def test_polyagamma_pdf_cdf(method, h, z):
    # test pdf calculation of points sampled using each method
    rng = np.random.default_rng(1)
    x = random_polyagamma(h, z, size=5000, method=method, random_state=rng)
    x.sort()
    d = polyagamma_pdf(x, h=h, z=z)
    area_under_curve = np.trapz(d, x)
    # relative tolerance is set so that 0.998 can pass the tests.
    assert np.isclose(1.0, area_under_curve, rtol=1e-2)

    xx = x.mean()
    mask = x <= xx
    # calculate the cdf of the distribution at the mean of the samples
    cdf = polyagamma_cdf(xx, h=h, z=z)
    # estimate empirical cdf from the sampled and corresponding density
    ecdf = np.trapz(d[mask], x[mask])
    # test if the empirical cdf is equal to the distribution's to 2 decimals
    assert np.allclose(ecdf, cdf, rtol=1e-2)

    # test value that is unlikely given distribution parameters
    assert 0 == polyagamma_pdf(0)
    assert 0 == polyagamma_pdf(np.inf)
    assert 0 == polyagamma_pdf(-1)
    assert -np.inf == polyagamma_pdf(0, return_log=True)
    assert -np.inf == polyagamma_pdf(np.inf, return_log=True)
    assert -np.inf == polyagamma_pdf(-1, return_log=True)
    assert 0 == polyagamma_cdf(0)
    assert 1 == polyagamma_cdf(np.inf)
    assert 0 == polyagamma_cdf(-1)
    assert -np.inf == polyagamma_cdf(0, return_log=True)
    assert 0 == polyagamma_cdf(np.inf, return_log=True)
    assert -np.inf == polyagamma_cdf(-1, return_log=True)

    # test if log versions agree with the unlogged distribution functions
    ld = np.exp(polyagamma_pdf(x, h=h, z=z, return_log=True))
    assert np.allclose(ld, d)
    lc = np.exp(polyagamma_cdf(xx, h=h, z=z, return_log=True))
    assert np.allclose(lc, cdf)


def test_log_extreme_value_behaviour():
    # test success of directly computing logcdf/logpdf instead of using log(*)
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
        np.log(polyagamma_cdf(0.01, h=10, z=3))
    assert np.isclose(polyagamma_cdf(0.01, h=10, z=3, return_log=True), -1238.6998500970105)

    with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
        np.log(polyagamma_pdf(0.01, h=10, z=3))
    # test extremely small values for logcdf and logpdf
    assert np.isclose(polyagamma_cdf(1e-3, h=10, z=3, return_log=True), -12489.8793293567)
    assert np.isclose(polyagamma_pdf(1e-3, h=10, z=3, return_log=True), -12473.46649418656)
    assert np.isclose(polyagamma_cdf(1e-16, return_log=True), -1250000000000017.2)
    assert np.isclose(polyagamma_pdf(1e-16, return_log=True), -1249999999999945.8)
