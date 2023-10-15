from ._polyagamma import (
    rvs as random_polyagamma,
    pdf as polyagamma_pdf,
    cdf as polyagamma_cdf
)
try:
    from ._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover
    raise RuntimeError(
        "Unable to find the version number that is generated when either building or "
        "installing from source. Please make sure that `polyagamma` has been properly "
        "installed, e.g. with\n\n  pip install -e .\n"
    )
