import numpy as np
import pytest

from polyagamma import default_rng

seed = 12345

def test_default_rng():
    rng = default_rng()
    assert isinstance(rng, np.random.Generator)
    pg = 'polyagamma'
    assert (hasattr(rng, pg) and callable(getattr(rng, pg)))


def test_polyagamma_devroye():
    rng = default_rng(seed)
    actual = rng.polyagamma(size=5)
    assert len(actual) == 5

    # h must be positive
    with pytest.raises(ValueError):
        rng.polyagamma(0)
    with pytest.raises(ValueError):
        rng.polyagamma(-rng.random())

    # should work for negative values of z
    rng.polyagamma(z=rng.integers(-100000000, -0))

    out = np.zeros(5)
    rng.polyagamma(out=out)
    assert not np.allclose(out, 0)

