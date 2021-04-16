from distutils.core import Extension
import os
from os.path import join

import numpy as np


source_files = [
    "polyagamma/_polyagamma.c",
    "src/pgm_random.c",
    "src/pgm_alternate.c",
    "src/pgm_devroye.c",
    "src/pgm_common.c",
    "src/pgm_saddle.c",
    "src/pgm_density.c",
]


macros = [('NPY_NO_DEPRECATED_API', 0)]
if os.getenv("BUILD_WITH_COVERAGE", None):
    macros.append(('CYTHON_TRACE_NOGIL', 1))


compile_args = ['-O2', '-std=c99', '-march=native']
if os.getenv("BUILD_WHEEL", None):
    compile_args.pop()

# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
include_path = np.get_include()
extensions = [
    Extension(
        "_polyagamma",
        source_files,
        include_dirs=[include_path, "./include"],
        library_dirs=[join(include_path, '..', '..', 'random', 'lib')],
        libraries=['npyrandom', 'm'],
        define_macros=macros,
        extra_compile_args=compile_args,
    ),
]


def build(setup_kwargs):
    """Build extension modules."""
    setup_kwargs.update(ext_modules=extensions, zip_safe=False)
