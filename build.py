
from distutils.core import Extension
from os.path import join

import numpy as np

source_files = [
    "polyagamma/_polyagamma.c",
    "src/pgm_random.c",
]

include_dirs = [np.get_include(), "./include"]
macros = [('NPY_NO_DEPRECATED_API', 0)]

# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
extensions = [
    Extension(
        "polyagamma._polyagamma",
        source_files,
        include_dirs=include_dirs,
        library_dirs=[join(np.get_include(), '..', '..', 'random', 'lib')],
        libraries=['npyrandom'],
        define_macros=macros,
    ),
]


def build(setup_kwargs):
    """Build extension modules."""
    setup_kwargs.update(ext_modules=extensions)
