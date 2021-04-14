import sys
from typing import overload, Union, Tuple

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np  # type: ignore
from numpy.typing import _ArrayLikeFloat_co  # type: ignore
from numpy.random import Generator, BitGenerator, SeedSequence  # type: ignore

_MethodType = Union[Literal["devroye", "alternate", "saddle", "gamma"], None]
_RNGType = Union[Generator, BitGenerator, SeedSequence, None]


@overload
def random_polyagamma(
    h: float,
    z: float,
    size: None = ...,
    out: None = ...,
    method: _MethodType = ...,
    disable_checks: bool = ...,
    random_state: _RNGType = ...
) -> float: ...
@overload
def random_polyagamma(size: Tuple[int, ...] = ...) -> np.ndarray: ...
@overload
def random_polyagamma(method: _MethodType = ...) -> float: ...
@overload
def random_polyagamma(out: _ArrayLikeFloat_co = ...) -> None: ...
@overload
def random_polyagamma(random_state: _RNGType = ...) -> float: ...
@overload
def random_polyagamma(h: float, z: _ArrayLikeFloat_co) -> np.ndarray: ...
@overload
def random_polyagamma(h: _ArrayLikeFloat_co, z: float) -> np.ndarray: ...
@overload
def random_polyagamma(h: _ArrayLikeFloat_co, disable_checks: bool) -> np.ndarray: ...
@overload
def random_polyagamma(h: _ArrayLikeFloat_co, z: _ArrayLikeFloat_co) -> np.ndarray: ...


@overload
def polyagamma_pdf(x: float) -> float: ...
@overload
def polyagamma_pdf(x: float, return_log: bool = ...) -> float: ...
@overload
def polyagamma_pdf(x: float, h: float = ..., z: float = ...) -> float: ...
@overload
def polyagamma_pdf(
    x: float, h: float = ..., z: _ArrayLikeFloat_co = ..., return_log: bool = ...,
) -> np.ndarray: ...
@overload
def polyagamma_pdf(
    x: float, h: _ArrayLikeFloat_co = ..., z: float = ...,
) -> np.ndarray: ...
@overload
def polyagamma_pdf(
    x: _ArrayLikeFloat_co, h: float = ..., z: float = ...,
) -> np.ndarray: ...
@overload
def polyagamma_pdf(
    x: float, h: _ArrayLikeFloat_co = ..., z: _ArrayLikeFloat_co = ...,
) -> np.ndarray: ...
@overload
def polyagamma_pdf(
    x: _ArrayLikeFloat_co, h: _ArrayLikeFloat_co = ..., z: float = ...,
) -> np.ndarray: ...
@overload
def polyagamma_pdf(
    x: _ArrayLikeFloat_co, h: _ArrayLikeFloat_co = ..., z: _ArrayLikeFloat_co = ...,
) -> np.ndarray: ...


@overload
def polyagamma_cdf(x: float) -> float: ...
@overload
def polyagamma_cdf(x: float, return_log: bool = ...) -> float: ...
@overload
def polyagamma_cdf(x: float, h: float = ..., z: float = ...) -> float: ...
@overload
def polyagamma_cdf(
    x: float, h: float = ..., z: _ArrayLikeFloat_co = ..., return_log: bool = ...,
) -> np.ndarray: ...
@overload
def polyagamma_cdf(
    x: float, h: _ArrayLikeFloat_co = ..., z: float = ...,
) -> np.ndarray: ...
@overload
def polyagamma_cdf(
    x: _ArrayLikeFloat_co, h: float = ..., z: float = ...,
) -> np.ndarray: ...
@overload
def polyagamma_cdf(
    x: float, h: _ArrayLikeFloat_co = ..., z: _ArrayLikeFloat_co = ...,
) -> np.ndarray: ...
@overload
def polyagamma_cdf(
    x: _ArrayLikeFloat_co, h: _ArrayLikeFloat_co = ..., z: float = ...,
) -> np.ndarray: ...
@overload
def polyagamma_cdf(
    x: _ArrayLikeFloat_co, h: _ArrayLikeFloat_co = ..., z: _ArrayLikeFloat_co = ...,
) -> np.ndarray: ...
