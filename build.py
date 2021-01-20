from distutils.core import Extension
from os.path import join

import numpy as np


source_files = [
    "polyagamma/_polyagamma.c",
    "src/pgm_random.c",
    "src/pgm_alternate.c",
    "src/pgm_igammaq.c",
    "src/pgm_devroye.c",
    "src/pgm_common.c",
    "src/pgm_saddle.c",
]


# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
include_path = np.get_include()
extensions = [
    Extension(
        "_polyagamma",
        source_files,
        include_dirs=[include_path, "./include"],
        library_dirs=[join(include_path, '..', '..', 'random', 'lib')],
        libraries=['npyrandom'],
        define_macros=[('NPY_NO_DEPRECATED_API', 0)],
        extra_compile_args=['-std=c99']
    ),
]


def build(setup_kwargs):
    """Build extension modules."""
    setup_kwargs.update(ext_modules=extensions)
