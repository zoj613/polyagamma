import numpy as np
import pytest

from polyagamma import default_rng

seed = 12345

def test_default_rng():
    rng = default_rng()
    assert isinstance(rng, np.random.Generator)
    pg = 'polyagamma'
    assert (hasattr(rng, pg) and callable(getattr(rng, pg)))


def test_polyagamma():
    rng = default_rng(seed)
    assert len(rng.polyagamma(size=5)) == 5
    assert rng.polyagamma(size=(5, 2)).shape == (5, 2)
    # raise an error if shape tuple element is not an integer
    with pytest.raises(TypeError):
        rng.polyagamma(size=(5.0, 2))

    h = rng.random((4, 5, 1))
    z = rng.random(4)
    assert rng.polyagamma(h, z).shape == (4, 5, 4)
    z = np.array([1, 2, 3, 4, 6])  # test if integer arrays dont crash things
    assert rng.polyagamma(h, z).shape == (4, 5, 5)
    assert rng.polyagamma(h, 0.12345).shape == (4, 5, 1)

    # should work on list/tuple input
    h = [[1, 2, 3], [3, 3, 1]]
    out = rng.polyagamma(h)
    assert out.shape == (2, 3)
    assert not np.allclose(out, 0)
    z = (1, 2, 10, 0)
    out = rng.polyagamma(z=z)
    assert out.shape == (4,)
    assert not np.allclose(out, 0)

    out = np.zeros(5)
    rng.polyagamma(out=out)
    assert not np.allclose(out, 0)
    # test size of output array when a parameter is a sequence
    with pytest.raises(ValueError):
        rng.polyagamma(h, out=out)

    # raise an error when array when dim > 1 is passed as an arg
    with pytest.raises(ValueError):
        rng.polyagamma(out=rng.random((2, 3)))

    # h must be positive
    with pytest.raises(ValueError):
        rng.polyagamma(0)
    with pytest.raises(ValueError):
        rng.polyagamma(-rng.random())
    with pytest.raises(ValueError):
        h = [1, 2, 3, 0.00001]
        rng.polyagamma(h)

    # should work for negative values of z
    rng.polyagamma(z=rng.integers(-100000000, -0))

    # raise error on unknown method names
    with pytest.raises(ValueError):
        rng.polyagamma(method="unknown method")
    # raise error for values less than 1 with alternate method
    with pytest.raises(ValueError):
        rng.polyagamma(0.9, method="alternate")
    # raise an error when using devroye with non-integer values of h
    with pytest.raises(ValueError):
        rng.polyagamma(2.0000000001, method="devroye")
    # should work for whole numbers is 2.000000 == 2
    rng.polyagamma(2.0000000000, method="devroye")

    # raise error when passed a none-keyword args after the first 2
    with pytest.raises(TypeError, match="takes at most 3 positional arguments"):
        rng.polyagamma(1, 0, 5)

    # don't raise error when passed non-positive h values if checks are disabled
    rng.polyagamma(-1, disable_checks=True)
