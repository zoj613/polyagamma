from ._polyagamma import (
    polyagamma as random_polyagamma, polyagamma_pdf, polyagamma_cdf
)
try:
    from ._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover
    raise RuntimeError(
        "Unable to find the version number that is generated when either building or "
        "installing from source. Please make sure that `polyagamma` has been properly "
        "installed, e.g. with\n\n  pip install -e .\n"
    )
