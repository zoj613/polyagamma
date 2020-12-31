import numpy as np
import pytest

from polyagamma import Generator, default_rng

seed = 12345

def test_default_rng():
    rng = default_rng()
    rng2 = Generator(np.random.PCG64())
    assert isinstance(rng, np.random.Generator)
    assert isinstance(rng2, np.random.Generator)


def test_polyagamma_devroye():
    rng = default_rng(seed)
    actual = rng.polyagamma()
    desired = 0.2898647553952438
    assert desired == actual

    desired = np.array([0.309637, 0.068448, 0.497534, 0.756462, 0.042912])
    actual = rng.polyagamma(size=5)
    assert len(actual) == 5
    np.testing.assert_array_almost_equal(actual, desired)

    # h must be positive
    with pytest.raises(ValueError):
        rng.polyagamma(-1)

    # should work for negative values of z
    rng.polyagamma(1, -1)

    out = np.zeros(5)
    rng.polyagamma(out=out)
    assert not np.allclose(out, 0)

