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
    h = rng.random((4, 5, 1))
    z = rng.random(4)
    assert rng.polyagamma(h, z).shape == (4, 5, 4)
    z = rng.random(5)
    assert rng.polyagamma(h, z).shape == (4, 5, 5)
    assert rng.polyagamma(h, 0.12345).shape == (4, 5, 1)

    out = np.zeros(5)
    rng.polyagamma(out=out)
    assert not np.allclose(out, 0)
    # test size of output array when a parameter is a sequence
    with pytest.raises(ValueError):
        rng.polyagamma(h, out=out)

    # raise an error when nd-array is passed as an arg
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
