import functools

import numpy as np
import pytest

from polyagamma import polyagamma


def test_polyagamma():
    rng = np.random.default_rng(1)
    rng_polyagamma = functools.partial(polyagamma, random_state=rng)

    assert len(rng_polyagamma(size=5)) == 5
    assert rng_polyagamma(size=(5, 2)).shape == (5, 2)
    # raise an error if shape tuple element is not an integer
    with pytest.raises(TypeError):
        rng_polyagamma(size=(5.0, 2))

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

    out = np.array([0., 0., 0., 0., 0.])
    rng_polyagamma(out=out)
    assert not np.allclose(out, 0)
    # test size of output array when a parameter is a sequence
    with pytest.raises(ValueError):
        rng_polyagamma(h, out=out)

    # raise an error when output array with dim > 1 is passed as an arg
    with pytest.raises(ValueError):
        rng_polyagamma(out=np.empty((2, 1)))

    # h must be equal to or greater than 1
    with pytest.raises(ValueError):
        rng_polyagamma(0)
    with pytest.raises(ValueError):
        rng_polyagamma(-1.3232)
    with pytest.raises(ValueError):
        h = [1, 2, 3, 0.00001]
        rng_polyagamma(h)

    # should work for negative values of z
    rng_polyagamma(z=-10.5)

    # raise error on unknown method names
    with pytest.raises(ValueError):
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

    # raise error for values less than 1 with alternate method
    with pytest.raises(ValueError):
        rng_polyagamma(0.9, method="alternate")
    # raise an error when using devroye with non-integer values of h
    with pytest.raises(ValueError):
        rng_polyagamma(2.0000000001, method="devroye")
    # should work for whole numbers: 2.000000 == 2
    rng_polyagamma(2.0000000000, method="devroye")

    # raise error when passed a non-keyword arg after the first 2
    with pytest.raises(TypeError, match="takes at most 2 positional arguments"):
        rng_polyagamma(1, 0, 5)

    # don't raise error when passed non-positive h values if checks are disabled
    rng_polyagamma(-1, disable_checks=True)

    # test for reproducibility via random_state
    rng = np.random.default_rng(12345)
    expected = polyagamma(size=5, random_state=rng)
    rng2 = np.random.default_rng(12345)
    assert np.allclose(expected, polyagamma(size=5, random_state=rng2))
    assert not np.allclose(expected, polyagamma(size=5))

